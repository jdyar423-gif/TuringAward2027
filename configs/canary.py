"""Canary run: the final model and the whole pipeline in one hour on the real chip.

Run it before the 24-hour run (it is not part of that run's budget) to see every stage work on
the TPU with the real model: first files of the main sources only, the full start-up benchmark
and canary, about half an hour of training at both sequence lengths, a short post-training,
a small proxy evaluation and the export.

    python scripts/run_24h.py --config configs/canary.py --out runs/canary
"""
import dataclasses
from pathlib import Path

from bes.config import load_config

_FINAL = load_config(str(Path(__file__).with_name("final.py")))

CFG = dataclasses.replace(
    _FINAL,
    run=dataclasses.replace(
        _FINAL.run, name="canary", out_dir="runs/canary", deadline_hours=1.0,
        post_reserve_hours=0.25, ckpt_every_min=10.0, eval_every_min=15.0,
        restart_window_hours=1.0, benchmark_min=12.0, benchmark_cap_min=18.0,
        download_sources=("climbmix:2 smol_smoltalk:1 smoltalk_magpie_ultra:1 smoltalk_constraints:1 "
                          "tulu3_sft:1 ultrachat:1 gsm8k:1 orca_math:1 squad_v2:1 ultrafeedback:1"),
        eval_limit=20),
)
