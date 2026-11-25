"""Tiny CPU rehearsal of the six-hour run (profile ``m6``; ``configs/m6.py``)::

    BES_PROFILE=m6 python scripts/run_24h.py --config configs/m6_smoke.py --allow-cpu \\
        --budget-minutes 14 --out runs/m6_smoke

The stages, the profile's mixture and sources, the answers library (exemplars in chat
samples), the single-plan benchmark, the run's speed switches, preference optimisation with one
learning rate and the export with the library are those of ``configs/m6.py``; the model, the
shapes and the data are those of ``configs/smoke.py``.
"""
import dataclasses
from pathlib import Path

from bes.config import load_config

_M6 = load_config(str(Path(__file__).with_name("m6.py")))
_SMOKE = load_config(str(Path(__file__).with_name("smoke.py")))

CFG = dataclasses.replace(
    _SMOKE,
    data=dataclasses.replace(_M6.data, seq_len=_SMOKE.data.seq_len, seq_len_late=_SMOKE.data.seq_len_late),
    run=dataclasses.replace(
        _SMOKE.run, name="m6_smoke", out_dir="runs/m6_smoke", deadline_hours=14.0 / 60.0,
        post_reserve_hours=5.0 / 60.0,
        post_minutes="sample=0 critic=0 prefopt=3 select=0 judge=0 export=1.5 buffer=0.5",
        download_sources="climbmix:1 smoltalk_constraints:1 alpaca:1 codealpaca:1 gsm8k:1 orca_math:1 "
                         "ultrafeedback:1",
        tokenize_max_records=3000, bench_plans="single", fast="relu2=1 ce=0", micro_split="2",
        tokenize_workers=2, eval_opponents="", eval_judge="", prefopt_lrs="3e-6", ship_library="always"),
)
