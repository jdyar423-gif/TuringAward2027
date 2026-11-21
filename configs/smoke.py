"""Tiny CPU configuration: exercises every code path in minutes."""
from bes.config import Cfg, DataCfg, DecodeCfg, ModelCfg, OptimCfg, RunCfg

CFG = Cfg(
    model=ModelCfg(d_model=128, n_heads=4, head_dim=32, ve_dim=32, head_rank=64, n_loops=4,
                   max_seq=256, clock_planes=3, cand_dim=32, critic_hidden=64),
    optim=OptimCfg(tokens_per_step=4096, micro_seqs=8),
    data=DataCfg(seq_len=128, seq_len_late=256),
    decode=DecodeCfg(particles=4, max_new_tokens=64),
    run=RunCfg(name="smoke", deadline_hours=0.2, post_reserve_hours=0.05, ckpt_every_min=1.0,
               eval_every_min=2.0, restart_window_hours=0.02, out_dir="runs/smoke"),
)
