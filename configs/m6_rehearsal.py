"""CPU rehearsal of the six-hour run with **every** source of its download plan (one file of
each, 3,000 records per file), for a Colab CPU runtime before the TPU is taken::

    BES_PROFILE=m6 python scripts/run_24h.py --config configs/m6_rehearsal.py --allow-cpu \\
        --budget-minutes 16 --out runs/m6_rehearsal

``configs/m6_smoke.py`` with the profile's whole registry: each adapter of the six-hour
mixture reads real rows through the tokenisation path, and the loader mixes all of them with
the answers library.
"""
import dataclasses
from pathlib import Path

from bes.config import load_config
from bes.train.supervisor import download_plan

_SMOKE = load_config(str(Path(__file__).with_name("m6_smoke.py")))
_M6 = load_config(str(Path(__file__).with_name("m6.py")))

CFG = dataclasses.replace(
    _SMOKE,
    run=dataclasses.replace(
        _SMOKE.run, name="m6_rehearsal", out_dir="runs/m6_rehearsal", deadline_hours=16.0 / 60.0,
        download_sources=" ".join(dict.fromkeys(f"{name}:1" for name, *_ in download_plan(_M6))),
        tokenize_workers=6),
)
