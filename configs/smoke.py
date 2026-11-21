"""Tiny CPU configuration: exercises every code path in minutes.

The ``run`` section also makes it the CPU rehearsal of the 24-hour supervisor::

    python scripts/run_24h.py --config configs/smoke.py --allow-cpu --budget-minutes 12 --out runs/smoke

downloads only the two small samples already in ``data_cache`` (a web-text shard and the
SmolTalk constraints file), tokenises a few thousand records of each into the run directory,
uses the built library, benchmarks a reduced set of plans, and runs the proxy evaluation on four
prompts per split against one small GGUF opponent of the harness registry and a stand-in,
judged by a stand-in judge.
"""
from bes.config import Cfg, DataCfg, DecodeCfg, ModelCfg, OptimCfg, RunCfg

CFG = Cfg(
    model=ModelCfg(d_model=128, n_heads=4, head_dim=32, ve_dim=32, head_rank=64, n_loops=4,
                   max_seq=256, clock_planes=3, cand_dim=32, critic_hidden=64),
    optim=OptimCfg(tokens_per_step=4096, micro_seqs=8),
    data=DataCfg(seq_len=128, seq_len_late=256),
    decode=DecodeCfg(particles=4, max_new_tokens=64),
    run=RunCfg(name="smoke", deadline_hours=0.2, post_reserve_hours=0.075, ckpt_every_min=1.0,
               eval_every_min=2.0, restart_window_hours=0.02, out_dir="runs/smoke",
               # rehearsal settings of the supervisor (bes/train/supervisor.py)
               tok_root="run", proxy_root="run",
               download_sources="climbmix:1 smoltalk_constraints:1", tokenize_max_records=3000,
               preflight_min=0.25, benchmark_min=2.5, benchmark_cap_min=4.0, bench_plans="reduced",
               bench_steps=3, canary_steps=9, log_every=5, guard_warmup=20,
               eval_opponents="falcon=recommended standin-plain=common",
               eval_judge="standin-longer", eval_limit=4),
)
