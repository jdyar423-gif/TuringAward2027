"""Time execution plans that the start-up benchmark does not try, on this machine's first device.

    python scripts/plan_sweep.py --config configs/final.py \
        --plans splash-remat1-ms64-onehot,splash-remat1-ms128-onehot --out runs/sweep_final.jsonl

The start-up benchmark of ``scripts/run_24h.py`` measures the configured plan, the
rematerialisation flip (off with half the micro-batch) and the other table gradients, inside a
time budget (``bes/train/benchmark.py``). It never tries a *larger* micro-batch, although with
rematerialisation the configured one uses a small part of the device memory
(``docs/TPU_NOTES.md``). This script measures any list of plans with the benchmark's own
measurement (``bes.train.benchmark.make_measure``: the real train step, real loader batches
regrouped for the plan, compile time, the compiled peak and tokens per second), so that a plan
can be chosen by hand and given to the run as a configuration.

A plan is written like the benchmark's key: ``<attention>-remat<0|1>-ms<micro-batch rows at the
base length>-<table gradient>``, for example ``splash-remat1-ms128-onehot``. The token shards
must be on disk (a run's background tasks tokenise them); no run may hold the device.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PLAN = re.compile(r"^(splash|xla)-remat([01])-ms(\d+)-(onehot|take|sorted)$")


def parse_plan(text: str):
    """``splash-remat1-ms128-onehot`` as a :class:`bes.train.benchmark.Plan`."""
    from bes.train import benchmark as B

    m = PLAN.match(text.strip())
    if not m:
        raise SystemExit(f"plan {text!r}: write <splash|xla>-remat<0|1>-ms<N>-<onehot|take|sorted>")
    return B.Plan(m.group(1), m.group(2) == "1", int(m.group(3)), m.group(4))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", action="append", default=[], help="section.field=value")
    ap.add_argument("--plans", required=True, help="comma-separated plan keys")
    ap.add_argument("--seq-len", type=int, default=0, help="sequence length (default: the base length)")
    ap.add_argument("--steps", type=int, default=6, help="timed steps per plan (after 2 warm-up steps)")
    ap.add_argument("--workers", type=int, default=6, help="loader workers for the batches")
    ap.add_argument("--no-limit", action="store_true", help="do not skip plans above the memory limit")
    ap.add_argument("--tok-root", default=None, help="directory that holds tok/<source>/ (default: the "
                                                     "configuration's; a rehearsal run keeps it in <run>/data)")
    ap.add_argument("--out", default=None, help="append one JSON line per measurement")
    args = ap.parse_args(argv)

    import jax

    from bes.config import LoaderCfg, load_config
    from bes.data.loader import Loader
    from bes.tokenizer import load as load_tokenizer
    from bes.train import benchmark as B
    from bes.train import schedule as S
    from bes.train import supervisor as SV

    cfg = load_config(args.config, list(args.override))
    plans = [parse_plan(p) for p in args.plans.split(",") if p.strip()]
    t = int(args.seq_len or cfg.data.seq_len)
    p_at = 0.0 if t == cfg.data.seq_len else cfg.data.late_from
    stage = dataclasses.replace(S.stage_at(p_at, cfg), library_frac_base=0.0, library_frac_chat=0.0)
    if stage.seq_len != t:
        raise SystemExit(f"the configuration has no stage at sequence length {t}")
    dev = SV.device_report()
    print(f"plan_sweep: {args.config}, T={t}, backend {jax.default_backend()}, devices "
          f"{[d['kind'] for d in dev['devices']]}, base plan {B.Plan.of(cfg).key}", flush=True)
    t0 = time.time()
    with tempfile.TemporaryDirectory(dir=str(REPO / cfg.run.data_dir)) as tmp:
        lcfg = dataclasses.replace(LoaderCfg(), workers=args.workers)
        tok_root = Path(args.tok_root).resolve() if args.tok_root else SV.tok_root(cfg, tmp)
        with Loader(cfg, load_tokenizer(), tok_root, library=None, n_workers=args.workers,
                    seed=cfg.run.seed, loader_cfg=lcfg, journal=Path(tmp) / "journal.jsonl") as loader:
            batches = [loader(i, stage) for i in range(4)]
    print(f"plan_sweep: 4 batches of shape {S.micro_shape(cfg, t)} in {time.time() - t0:.0f} s", flush=True)
    measure = B.make_measure(cfg, {t: batches}, steps=args.steps, warmup=2,
                             memory_limit=None if args.no_limit else B.memory_limit_bytes(dev),
                             consts_switches=SV.parse_switches(cfg.run.switches), seed=cfg.run.seed)
    rows = []
    for plan in plans:
        why = B.plan_valid(cfg, plan)
        m = B.Measurement(plan, t, False, why) if why else measure(plan, t)
        row = {"config": args.config, "plan": plan.key, "seq_len": t, "ok": m.ok, "reason": m.reason,
               "shape": list(m.shape or ()), "tokens_per_s": m.tokens_per_s, "step_s": m.step_s,
               "compile_s": m.compile_s, "peak_gib": (m.peak_bytes or 0) / B.GIB, "time": time.time()}
        rows.append(row)
        if m.ok:
            print(f"plan_sweep: {plan.key} T={t}: {m.tokens_per_s:,.0f} tok/s, step {m.step_s:.3f} s, compile "
                  f"{m.compile_s:.1f} s, peak {row['peak_gib']:.2f} GiB, shape {tuple(m.shape)}", flush=True)
        else:
            print(f"plan_sweep: {plan.key} T={t}: NO ({m.reason})", flush=True)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")
    best = max((r for r in rows if r["ok"]), key=lambda r: r["tokens_per_s"], default=None)
    if best:
        print(f"plan_sweep: fastest {best['plan']} at {best['tokens_per_s']:,.0f} tok/s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
