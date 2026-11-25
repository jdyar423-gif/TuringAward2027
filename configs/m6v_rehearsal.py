"""CPU rehearsal of the six-hour continuation with the **real model and the real weights**
(``configs/m6v.py``), for a Colab CPU runtime before the TPU is taken::

    BES_PROFILE=m6 python scripts/run_24h.py --config configs/m6v_rehearsal.py --allow-cpu \\
        --budget-minutes 50 --out runs/m6v_rehearsal

The model, the initial weights (``release/main16-v9-before-dpo``), the optimiser settings, the
mixture, the answers library and the stages are those of ``configs/m6v.py``; a step is 4,096
tokens instead of 262,144 (two 1,024-token rows per micro-batch), one file of every source is
tokenised, and the attention is the dense reference path with float32 activations (the chip's
kernels do not run on a CPU). What it shows before the chip is taken: the release parts unpack
on a fresh machine, the 64 leaves load into the run's configuration, the first training loss is
that of the trained model and not of an initialisation, a checkpoint resumes, and preference
optimisation and the export run on the continued weights.
"""
import dataclasses
from pathlib import Path

from bes.config import load_config
from bes.train.supervisor import download_plan

_M6V = load_config(str(Path(__file__).with_name("m6v.py")))
_M6 = load_config(str(Path(__file__).with_name("m6.py")))
_SMOKE = load_config(str(Path(__file__).with_name("m6_smoke.py")))

CFG = dataclasses.replace(
    _M6V,
    model=dataclasses.replace(_M6V.model, attn_impl="xla", act_dtype="float32", remat=False),
    optim=dataclasses.replace(_M6V.optim, tokens_per_step=4096, micro_seqs=2),
    data=dataclasses.replace(_M6V.data, seq_len=1024, seq_len_late=1024),
    run=dataclasses.replace(
        _SMOKE.run, name="m6v_rehearsal", out_dir="runs/m6v_rehearsal", deadline_hours=50.0 / 60.0,
        post_reserve_hours=13.0 / 60.0,
        post_minutes="sample=0 critic=0 prefopt=6 select=0 judge=0 export=6 buffer=1",
        download_sources=" ".join(dict.fromkeys(f"{name}:1" for name, *_ in download_plan(_M6))),
        tokenize_workers=6, init_from=_M6V.run.init_from, init_sha256=_M6V.run.init_sha256,
        fast="relu2=1 ce=0 flat=1 rows=1 tokn=1", micro_split="1", ship_library="select",
        benchmark_min=12.0, benchmark_cap_min=25.0, bench_steps=1, canary_steps=9, log_every=1,
        ckpt_every_min=4.0, eval_every_min=20.0),
)
