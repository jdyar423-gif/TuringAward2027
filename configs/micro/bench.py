"""Micro bench configuration for the CPU evidence gates E2, E4 and E5 (``bes/eval/microbench.py``).

``configs/micro/base.py`` (d=128, 8 loops, the same code paths as the final model) with the
settings of the bench: rows of 512 tokens, 4096 tokens per optimiser step in micro-batches of
4 rows (gradient accumulation over A=2 micro-batches inside one jitted step), recomputation of
every loop in the backward pass (``remat``) and scatter-add gradients for the vocabulary tables
(``table_grad="take"``, the fast choice on CPU; same gradients as the TPU one-hot path).
Measured on this container (2 cores, other work running): about 1.4 GB peak and 5-7 s per step.

The learning-rate transfer rule is switched OFF here (both exponents 0): its batch factor was
fitted near 262,144 tokens per step, and extrapolated to 4,096 tokens and a few hundred steps it
cuts every Adam learning rate tenfold and under-trains the model (measured at 1M tokens, held-out
loss web / chat: 5.78 / 5.19 with the rule, 5.12 / 4.32 without). The final run uses the reference
batch, where the batch factor is exactly 1; only its data factor on the matrix group applies
there. The arm ``transfer_rule`` keeps the comparison.
"""
import dataclasses

from bes.config import load_config
from pathlib import Path

_BASE = load_config(str(Path(__file__).with_name("base.py")))

CFG = dataclasses.replace(
    _BASE,
    model=dataclasses.replace(_BASE.model, max_seq=512, remat=True, table_grad="take"),
    optim=dataclasses.replace(_BASE.optim, tokens_per_step=4096, micro_seqs=4,
                              batch_exponent=0.0, data_exponent=0.0),
    data=dataclasses.replace(_BASE.data, seq_len=512, seq_len_late=512),
    run=dataclasses.replace(_BASE.run, name="micro_bench", out_dir="runs/micro_bench"),
)
