"""Time the training step under several settings of the TPU runtime's compiler flags
(``LIBTPU_INIT_ARGS``), each in a process of its own (the flags are read when the runtime starts).

    python scripts/tpu_flags.py --config configs/final.py --override model.table_grad=take \
        --rows 4 --variant base= --variant vmem64=--xla_tpu_scoped_vmem_limit_kib=65536

Every variant runs ``scripts/tpu_micro.py`` (the whole accumulation step on synthetic batches) with
its flags in the environment; ``+`` in a variant stands for a space between flags. A variant whose
flags the runtime rejects simply fails and is reported. This script never imports JAX itself, so
the device is free for the variants. ``--long-for`` names the variants that are also timed at the
long sequence length (default: only the first); ``--rows-for NAME=ROWS`` gives a variant its own
rows per micro-batch.

The last line is ``FLAGS <json>``: per variant the flags and the per-shape milliseconds per step.
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--rows", default="4", help="rows per micro-batch at the base length")
    ap.add_argument("--variant", action="append", default=[], metavar="NAME=FLAGS",
                    help="'+' between flags stands for a space; empty FLAGS: the runtime's defaults")
    ap.add_argument("--long-for", default="", help="comma-separated variant names also timed at the long length")
    ap.add_argument("--rows-for", action="append", default=[], metavar="NAME=ROWS")
    ap.add_argument("--calls", type=int, default=3)
    ap.add_argument("--per-variant-s", type=float, default=150.0)
    ap.add_argument("--budget-s", type=float, default=330.0, help="no new variant after this many seconds")
    args = ap.parse_args()

    variants = [(v.split("=", 1)[0], v.split("=", 1)[1].replace("+", " ") if "=" in v else "") for v in args.variant]
    if not variants:
        variants = [("base", "")]
    long_for = {x for x in args.long_for.split(",") if x} or {variants[0][0]}
    rows_for = dict(x.split("=", 1) for x in args.rows_for)
    t_start = time.time()
    out = []
    print(f"tpu_flags: {len(variants)} variants, rows {args.rows}, budget {args.budget_s:.0f} s", flush=True)
    for name, flags in variants:
        if time.time() - t_start > args.budget_s:
            print(f"  {name:14s} not timed: the budget of {args.budget_s:.0f} s is used", flush=True)
            continue
        cmd = [sys.executable, str(REPO / "scripts" / "tpu_micro.py"), "--config", args.config,
               "--rows", rows_for.get(name, args.rows), "--calls", str(args.calls), "--budget-s", "100"]
        for o in args.override:
            cmd += ["--override", o]
        if name not in long_for:
            cmd.append("--no-long")
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        if flags:
            env["LIBTPU_INIT_ARGS"] = (env.get("LIBTPU_INIT_ARGS", "") + " " + flags).strip()
        t0 = time.time()
        try:
            p = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True,
                               timeout=args.per_variant_s)
            text, rc = p.stdout + p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:
            text = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            rc = -1
        rec = {"name": name, "flags": flags, "rc": rc, "seconds": round(time.time() - t0, 1), "shapes": []}
        for line in text.splitlines():
            if line.startswith("MICRO "):
                try:
                    rec["shapes"] = json.loads(line[6:])["variants"]
                except ValueError:
                    pass
        if rec["shapes"]:
            for s in rec["shapes"]:
                print(f"  {name:14s} {s['a']:4d} x {s['rows']:3d} x {s['t']:<5d} {s['ms']:9.1f} ms  "
                      f"{s['tokens_per_s']:10,.0f} tok/s  loss {s['loss']:.5f}  (compile {s['compile_s']:.0f} s)  {flags}",
                      flush=True)
        else:
            tail = " | ".join(x.strip() for x in text.strip().splitlines()[-3:])[:300]
            print(f"  {name:14s} FAILED (exit {rc}, {rec['seconds']} s): {tail}", flush=True)
            rec["error"] = tail
        out.append(rec)
    print("FLAGS " + json.dumps({"variants": out, "seconds": round(time.time() - t_start, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
