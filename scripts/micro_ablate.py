"""Gate E2: micro ablations of the BES model (``docs/PLAN.md`` section 7), one arm at a time.

The queue runs the arms sequentially, each in its own child process (one training process at a
time; every child pins itself to ``--threads`` cores), skips arms that already have a finished
record in ``<out>/results.jsonl`` and resumes a killed arm from its last evaluation checkpoint.
Before the queue starts, a short timing probe (one compile and a few steps of the reference)
prints the expected time per arm.

    nohup python scripts/micro_ablate.py --tokens 1000000 --out runs/e2 > runs/e2/queue.log 2>&1 &
    python scripts/micro_ablate.py --report --out runs/e2           # writes docs/E2_ABLATIONS.md
    python scripts/micro_ablate.py --list                           # arms and expected durations
    python scripts/micro_ablate.py --arms ref,no_tape --tokens 60000 --out runs/e2_smoke

Arms are declarative (:data:`E2_ARMS`): configuration overrides, training-time switches, seed
and kind (how the report judges them; ``bes.eval.microbench.decide``).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _limit_threads(argv: list[str]) -> None:
    """Before numpy / jax load: ``--threads`` threads and CPU affinity to that many cores."""
    n, cpus = 2, None
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            n = int(argv[i + 1])
        elif a.startswith("--threads="):
            n = int(a.split("=", 1)[1])
        elif a == "--cpus" and i + 1 < len(argv):
            cpus = [int(c) for c in argv[i + 1].split(",")]
        elif a.startswith("--cpus="):
            cpus = [int(c) for c in a.split("=", 1)[1].split(",")]
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(n)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if hasattr(os, "sched_setaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        chosen = [c for c in (cpus or allowed[-n:]) if c in allowed] or allowed[:n]
        os.sched_setaffinity(0, chosen)


if __name__ == "__main__":
    _limit_threads(sys.argv)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bes.config import load_config  # noqa: E402
from bes.eval import microbench as mb  # noqa: E402
from bes.eval.microbench import Arm  # noqa: E402

_BASE = load_config(str(mb.BENCH_CONFIG))
_VE, _LRF = _BASE.model.ve_dim, _BASE.optim.lr_feature

#: The E2 queue, in the order it runs (reference and its seed replicate first).
E2_ARMS: tuple[Arm, ...] = (
    Arm("ref", kind="reference", component="every path on (configs/micro/bench.py)"),
    Arm("ref_s1", seed=1, kind="replicate", component="the reference with seed 1 (data order and init)"),
    Arm("no_tape", ("model.use_cand_in=false", "model.use_cand_out=false"), kind="remove_new",
        component="candidate tape (input features and output-law masses)"),
    Arm("transfer_rule", ("optim.batch_exponent=0.558", "optim.data_exponent=-0.32"),
        kind="setting", component="reference learning-rate transfer rule at 4,096 tokens per step"),
    Arm("tape_out_only", ("model.use_cand_in=false",), kind="remove_new",
        component="candidate input features (output masses kept)"),
    Arm("tape_in_only", ("model.use_cand_out=false",), kind="remove_new",
        component="output-law masses (input features kept)"),
    Arm("no_reg", ("model.use_reg=false",), kind="remove_new", component="register embeddings"),
    Arm("no_clock", ("model.clock_planes=0",), kind="remove_new", component="structural RoPE clocks"),
    Arm("no_sink", ("model.use_sink=false",), kind="remove_evidence", component="attention sink at BOS"),
    Arm("no_conf", ("model.use_conf=false",), kind="remove_new", component="confidence value channel"),
    Arm("ve_0", ("model.ve_dim=0",), kind="remove_evidence", component="per-loop value embeddings"),
    Arm("ve_wide", (f"model.ve_dim={2 * _VE}",), kind="setting",
        component=f"value-embedding width {2 * _VE} instead of {_VE}"),
    Arm("muon", ("optim.optimizer=muon",), kind="setting",
        component="Muon + NorMuon + cautious decay instead of MuonH"),
    Arm("loops_4", ("model.n_loops=4",), kind="setting", component="4 loops (same tokens, half the core compute)"),
    Arm("loops_12", ("model.n_loops=12",), kind="setting", component="12 loops (same tokens, 1.5x the core compute)"),
    Arm("lr_feature_x0.3", (f"optim.lr_feature={0.3 * _LRF:g}",), kind="setting",
        component=f"feature-group learning rate {0.3 * _LRF:g} instead of {_LRF:g}"),
    Arm("lr_feature_x3", (f"optim.lr_feature={3 * _LRF:g}",), kind="setting",
        component=f"feature-group learning rate {3 * _LRF:g} instead of {_LRF:g}"),
    Arm("mass_offset_4", ("model.mass_offset=4",), kind="setting", component="initial tape log-mass -4 instead of -6"),
    Arm("mass_offset_8", ("model.mass_offset=8",), kind="setting", component="initial tape log-mass -8 instead of -6"),
    Arm("src5_train_025", switches=(("src_5", 0.25),), kind="setting",
        component="self-evidence masses x0.25 during training (as the decoder applies them)"),
    Arm("src5_train_0", switches=(("src_5", 0.0),), kind="setting",
        component="no masses from the assistant turn being written, during training"),
)
#: Arms the task list names that the queue cannot run, with the reason.
SKIPPED: tuple[tuple[str, str], ...] = (
    ("sink_prior", "no configuration hook initialises `sink_logit` (bes.model.model.init_params "
                   "sets it to 0 and ModelCfg has no field for it); not run"),
)


def bench_from(args) -> mb.Bench:
    return mb.Bench(seq_len=args.seq_len, micro_rows=args.micro_rows,
                    tokens_per_step=args.tokens_per_step, eval_rows=args.eval_rows,
                    eval_fracs=tuple(float(x) for x in args.eval_fracs.split(",")))


def bench_args(args) -> list[str]:
    return ["--seq-len", str(args.seq_len), "--micro-rows", str(args.micro_rows),
            "--tokens-per-step", str(args.tokens_per_step), "--eval-rows", str(args.eval_rows),
            "--eval-fracs", args.eval_fracs, "--threads", str(args.threads)] + (
        ["--cpus", args.cpus] if args.cpus else [])


def select(arms: tuple[Arm, ...], names: str | None) -> list[Arm]:
    if not names:
        return list(arms)
    by = {a.name: a for a in arms}
    unknown = [n for n in names.split(",") if n not in by]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; known: {sorted(by)}")
    return [by[n] for n in names.split(",")]


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


# ------------------------------------------------------------------------ children
def die_with_parent() -> None:
    """Child modes: ask the kernel to send SIGTERM when the queue process dies (Linux
    ``PR_SET_PDEATHSIG``), so a killed queue never leaves a training process running."""
    import ctypes
    import signal

    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, int(signal.SIGTERM))  # PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        return
    if os.getppid() == 1:            # the parent is already gone
        raise SystemExit("queue process gone; not starting")


def child(args, extra: list[str], script: str | None = None) -> int:
    """Run ``script`` (default: this file) as a child process with the queue's options."""
    cmd = [sys.executable, str(Path(script or __file__).resolve())] + extra + [
        "--tokens", str(args.tokens), "--out", args.out] + bench_args(args)
    return subprocess.call(cmd, cwd=str(REPO))


def timing_probe(args, steps: int = 3) -> dict:
    """Child mode: compile the reference step and time a few steps (no record, no files)."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from bes.model import model as M
    from bes.optim import build as build_opt
    from bes.tokenizer import load as load_tok
    from bes.train import schedule as S
    from bes.train.train import TrainState, make_train_step

    bench = bench_from(args)
    arm = E2_ARMS[0]
    cfg = mb.arm_config(arm, args.tokens, bench)
    shape = S.micro_shape(cfg, cfg.data.seq_len)
    tok = load_tok()
    pools = mb.train_pools(arm, shape[2], tok, bench, log)
    maker = mb.BatchMaker(pools, arm.mixture, 0, shape, steps + 1, tok)
    params = M.init_params(cfg.model, jax.random.key(0))
    opt = build_opt(cfg, params)
    state = TrainState(jnp.zeros((), jnp.int32), params, opt.init(params), M.make_consts(cfg.model))
    step = make_train_step(M.loss_fn, opt, cfg)
    t0 = time.time()
    state, m = step(state, maker.batch(0), np.float32(1.0), np.float32(0.95), np.float32(1.0))
    float(m["loss"])
    compile_s = time.time() - t0
    ts = []
    for s in range(1, steps + 1):
        t = time.time()
        state, m = step(state, maker.batch(s), np.float32(1.0), np.float32(0.95), np.float32(1.0))
        float(m["loss"])
        ts.append(time.time() - t)
    res = {"compile_s": compile_s, "sec_per_step": float(np.median(ts)), "steps": ts,
           "tokens_per_step": cfg.optim.tokens_per_step, "peak_rss_mb": mb.peak_rss_mb(),
           "flops_per_token": mb.flops_per_token(cfg)["train"]}
    print("TIMING " + json.dumps(res), flush=True)
    return res


def expected(args, probe: dict, arms: list[Arm]) -> list[tuple[str, int, float]]:
    """``(arm, steps, seconds)``: training scaled by analytic FLOPs per token relative to the
    probed reference, plus evaluation (forward passes ~1/3 of a training token) and compile."""
    bench = bench_from(args)
    ref_cfg = mb.arm_config(E2_ARMS[0], args.tokens, bench)
    ref_flops = mb.flops_per_token(ref_cfg)["train"]
    sec_tok = probe["sec_per_step"] / probe["tokens_per_step"]
    out = []
    for a in arms:
        cfg = mb.arm_config(a, args.tokens, bench)
        ratio = mb.flops_per_token(cfg)["train"] / ref_flops
        steps = max(1, (a.tokens or args.tokens) // cfg.optim.tokens_per_step)
        n_views = 1 + len(a.eval_views())
        eval_tok = len(bench.eval_fracs) * n_views * bench.eval_rows * bench.seq_len * len(bench.eval_domains)
        sec = steps * probe["sec_per_step"] * ratio + eval_tok * sec_tok * ratio / 3 + probe["compile_s"] + 30
        out.append((a.name, steps, sec))
    return out


# ---------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the arms and their expected durations")
    ap.add_argument("--arms", default=None, help="comma-separated subset of the arms (default: all)")
    ap.add_argument("--tokens", type=int, default=1_000_000, help="training tokens per arm")
    ap.add_argument("--out", default="runs/e2", help="run directory (results.jsonl, arms/)")
    ap.add_argument("--report", action="store_true", help="write the E2 document from <out>/results.jsonl")
    ap.add_argument("--doc", default=str(REPO / "docs" / "E2_ABLATIONS.md"))
    ap.add_argument("--threads", type=int, default=2, help="cores per training process")
    ap.add_argument("--cpus", default="", help="explicit CPU list for the affinity, e.g. 2,3")
    ap.add_argument("--seq-len", type=int, default=mb.Bench.seq_len)
    ap.add_argument("--micro-rows", type=int, default=mb.Bench.micro_rows)
    ap.add_argument("--tokens-per-step", type=int, default=mb.Bench.tokens_per_step)
    ap.add_argument("--eval-rows", type=int, default=mb.Bench.eval_rows)
    ap.add_argument("--eval-fracs", default=",".join(str(f) for f in mb.Bench.eval_fracs))
    ap.add_argument("--no-probe", action="store_true", help="skip the timing probe")
    ap.add_argument("--run-arm", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--timing-probe", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    args.out = str(out)
    bench = bench_from(args)

    if args.report:
        recs = mb.read_jsonl(out / mb.RESULTS)
        text = mb.e2_report(recs, arms=E2_ARMS, skipped=SKIPPED, source=mb._rel(out / mb.RESULTS))
        Path(args.doc).write_text(text)
        log(f"wrote {args.doc} ({len(mb.latest_done(recs))} finished arms)")
        return 0
    if args.timing_probe:
        die_with_parent()
        timing_probe(args)
        return 0
    if args.run_arm:
        die_with_parent()
        arm = select(E2_ARMS, args.run_arm)[0]
        mb.run_arm(arm, out, args.tokens, bench, log=log)
        return 0

    arms = select(E2_ARMS, args.arms)
    probe = None
    if not args.no_probe:
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--timing-probe", "--tokens",
               str(args.tokens), "--out", args.out] + bench_args(args)
        res = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
        line = next((ln for ln in res.stdout.splitlines() if ln.startswith("TIMING ")), None)
        if line is None:
            log("timing probe failed:\n" + res.stdout[-2000:] + res.stderr[-2000:])
        else:
            probe = json.loads(line[len("TIMING "):])
            log(f"timing probe: {probe['sec_per_step']:.2f} s per step of {probe['tokens_per_step']} "
                f"tokens ({probe['sec_per_step'] / probe['tokens_per_step'] * 1e6:.0f} s per 1M tokens), "
                f"compile {probe['compile_s']:.0f} s, peak {probe['peak_rss_mb']:.0f} MB")
    rows = expected(args, probe, arms) if probe else [(a.name, 0, float("nan")) for a in arms]
    total = 0.0
    for a, (name, steps, sec) in zip(arms, rows):
        state = "done" if mb.finished(out, a, args.tokens, bench) else "todo"
        what = ", ".join([*a.overrides, *(f"{k}={v:g}" for k, v in a.switches)]) or "-"
        if state == "todo" and sec == sec:
            total += sec
        eta = f"~{sec / 60:6.1f} min" if sec == sec else "  (no probe)"
        print(f"  {name:18s} {a.kind:16s} {state:5s} {eta}  {what}", flush=True)
    for name, reason in SKIPPED:
        print(f"  {name:18s} skipped: {reason}", flush=True)
    if probe:
        log(f"expected time for the remaining arms: {total / 3600:.1f} h")
    if args.list:
        return 0

    failed = []

    def runner(arm: Arm) -> None:
        log(f"start {arm.name}")
        code = child(args, ["--run-arm", arm.name])
        if code != 0:
            failed.append(arm.name)
            log(f"{arm.name} failed with exit code {code}; continuing")
        else:
            log(f"done {arm.name}")

    mb.run_queue(arms, out, args.tokens, bench, runner=runner, log=log)
    recs = mb.read_jsonl(out / mb.RESULTS)
    log(f"queue finished: {len(mb.latest_done(recs))} finished arms in {out / mb.RESULTS}"
        + (f"; failed: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
