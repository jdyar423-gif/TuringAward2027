"""Which micro-batch size trains fastest at the same tokens per step: one measurement on the
idle accelerator.

    python scripts/tpu_micro.py --config configs/final.py --override model.table_grad=take

A training step is one device program: a scan over ``A`` micro-batches of ``B x T`` tokens
(``bes.train.train.accumulate``), ``A * B * T = optim.tokens_per_step``. On the TPU v6e the
passes over large tensors are memory-bound, and smaller micro-batches were faster every time the
size was halved (64 -> 32 rows: +1%, 32 -> 16: +8%; ``docs/TPU_NOTES.md``, ``docs/CHANGES.md``
19.3); sizes under 16 rows had never been timed. Against that stands a fixed cost per
micro-batch: its gradient is a full float32 parameter tree that is added to the running sum.
So this script times the whole accumulation (not one ``value_and_grad``) for several ``B`` at
both sequence lengths, with synthetic batches and freshly initialised parameters (nothing else
may hold the device), and prints tokens per second and the loss (the same micro-batch is
repeated ``A`` times, so the loss must not depend on ``B`` beyond float rounding).

The last line is ``MICRO <json>``: per variant ``ms`` per step and ``tokens_per_s``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", action="append", default=[], help="section.field=value")
    ap.add_argument("--rows", default="", help="comma-separated rows per micro-batch at the base length "
                    "(default: the configured one, then halves down to 2)")
    ap.add_argument("--calls", type=int, default=3, help="timed steps per variant")
    ap.add_argument("--budget-s", type=float, default=200.0, help="no new variant after this many seconds")
    ap.add_argument("--no-long", action="store_true", help="skip the long sequence length")
    args = ap.parse_args(argv)

    import jax
    import numpy as np

    import tpu_tune
    from bes.config import load_config
    from bes.model import model as M
    from bes.train import supervisor as SV
    from bes.train.train import accumulate

    cfg = load_config(args.config, list(args.override))
    mc = cfg.model
    t1, t2 = int(cfg.data.seq_len), int(cfg.data.seq_len_late)
    rows1, per_step = int(cfg.optim.micro_seqs), int(cfg.optim.tokens_per_step)
    dev = jax.devices()[0].device_kind
    say = lambda msg: print(msg, flush=True)                                   # noqa: E731
    say(f"tpu_micro: {args.config} {args.override}, device {dev}, {per_step} tokens per step, "
        f"configured micro-batch {rows1} x {t1}")
    consts = M.make_consts(mc, **SV.parse_switches(cfg.run.switches))
    params = M.init_params(mc, jax.random.key(0))
    if args.rows:
        sizes = [int(x) for x in args.rows.split(",") if x.strip()]
    else:
        sizes, r = [], rows1
        while r >= 2:
            sizes.append(r)
            r //= 2
    variants = [(r, t1) for r in sizes if per_step % (r * t1) == 0]
    if t2 != t1 and not args.no_long:
        variants += [(r * t1 // t2, t2) for r in sizes if r * t1 // t2 >= 1 and per_step % (r * t1) == 0]
    t_start = time.time()
    out: list[dict] = []

    def step_fn(p, b):
        g, loss, w, _ = accumulate(M.loss_fn, p, consts, b, mc)
        return g, loss, w

    def timed(rows: int, t: int, label: str = "") -> dict | None:
        a = per_step // (rows * t)
        one = tpu_tune.synth_batch(mc, rows, t)
        batch = jax.device_put({k: np.ascontiguousarray(np.broadcast_to(v, (a, *v.shape))) for k, v in one.items()})
        f = jax.jit(step_fn)
        try:
            t0 = time.perf_counter(); got = f(params, batch); jax.block_until_ready(got)
            comp = time.perf_counter() - t0
            got = f(params, batch); jax.block_until_ready(got)                   # warm
            t0 = time.perf_counter()
            for _ in range(args.calls):
                got = f(params, batch)
            jax.block_until_ready(got)
            ms = (time.perf_counter() - t0) / args.calls * 1e3
        except Exception as e:  # noqa: BLE001 - a size that does not run is reported, not chosen
            say(f"  {label or f'{a:4d} x {rows:3d} x {t}':28s} FAILED {type(e).__name__}: {str(e).splitlines()[0][:160]}")
            return None
        rec = {"a": a, "rows": rows, "t": t, "ms": ms, "tokens_per_s": per_step / ms * 1e3,
               "loss": float(got[1]), "compile_s": comp}
        say(f"  {label or f'{a:4d} x {rows:3d} x {t}':28s} {ms:9.1f} ms per step  {rec['tokens_per_s']:10,.0f} tok/s  "
            f"loss {rec['loss']:.5f}  (compile {comp:.0f} s)")
        return rec

    if dev.lower().startswith("tpu") and variants:
        timed(*variants[0], label="warm-up (discarded)")          # the first program of a process is slow
    for rows, t in variants:
        if time.time() - t_start > args.budget_s:
            say(f"  budget of {args.budget_s:.0f} s used: {rows} x {t} and later variants not timed")
            break
        rec = timed(rows, t)
        if rec is not None:
            out.append(rec)
    say("MICRO " + json.dumps({"variants": out, "seconds": round(time.time() - t_start, 1), "device": dev}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
