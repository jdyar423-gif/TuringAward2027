"""Proxy model for the CPU evidence gates (same code paths as the final model, d=128)."""
from bes.config import Cfg, DataCfg, ModelCfg, OptimCfg, RunCfg

CFG = Cfg(
    model=ModelCfg(d_model=128, n_heads=4, head_dim=32, ve_dim=32, head_rank=64, n_loops=8,
                   max_seq=256, clock_planes=3, cand_dim=32, critic_hidden=64),
    optim=OptimCfg(tokens_per_step=16384, micro_seqs=16),
    data=DataCfg(seq_len=256, seq_len_late=256),
    run=RunCfg(name="micro", out_dir="runs/micro"),
)
