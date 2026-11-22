"""Final configuration under the strict reading of "active parameters".

Strict reading: every stored number the next-token distribution depends on, including the table
rows read at every position of the context window (docs/ACCOUNTING.md). ``configs/final.py``
satisfies the standard reading (numbers read per generated token). This variant gives up about 6%
of the dense core (narrower feed-forward layers, smaller critic) so that the strict count also
stays under 20M at a 2048-token window with the critic and the library reads included.
Everything else is the final configuration.
"""
import dataclasses
from pathlib import Path

from bes.config import load_config

_FINAL = load_config(str(Path(__file__).with_name("final.py")))

CFG = dataclasses.replace(
    _FINAL,
    model=dataclasses.replace(_FINAL.model, ffn_width=2688, critic_hidden=128),
    run=dataclasses.replace(_FINAL.run, name="strict", out_dir="runs/strict"),
)
