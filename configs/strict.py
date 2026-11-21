"""Final configuration under the strict reading of "active parameters".

Strict reading: every stored number the next-token distribution depends on, including the table
rows read at every position of the context window (docs/ACCOUNTING.md). ``configs/final.py``
satisfies the standard reading (numbers read per generated token). This variant gives up about 5%
of the dense core (narrower feed-forward layers, smaller critic) so that the strict count also
stays under 20M at a 2048-token window with the critic and the library reads included.
"""
from bes.config import Cfg, ModelCfg, RunCfg

CFG = Cfg(
    model=ModelCfg(attn_impl="splash", act_dtype="bfloat16", remat=True,
                   ffn_width=2688, critic_hidden=128),
    run=RunCfg(name="strict", out_dir="runs/strict"),
)
