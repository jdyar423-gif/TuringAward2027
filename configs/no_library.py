"""Alternative without the written library: its budget goes to wider per-loop value embeddings."""
import dataclasses

from bes.config import Cfg, DataCfg, DecodeCfg, ModelCfg, RunCfg

CFG = Cfg(
    model=ModelCfg(attn_impl="splash", act_dtype="bfloat16", ve_dim=512),
    data=DataCfg(use_library=False, library_frac_base=0.0, library_frac_chat=0.0),
    decode=DecodeCfg(use_library=False),
    run=RunCfg(name="no_library", out_dir="runs/no_library"),
)
