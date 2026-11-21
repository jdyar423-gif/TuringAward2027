"""Final 24-hour configuration (TPU v6e). Frozen after the evidence gates."""
import dataclasses

from bes.config import Cfg, ModelCfg, RunCfg

CFG = Cfg(
    model=ModelCfg(attn_impl="splash", act_dtype="bfloat16", remat=True),
    run=RunCfg(name="final", out_dir="runs/final"),
)
