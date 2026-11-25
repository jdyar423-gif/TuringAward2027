"""Six-hour continuation of ``main16-v9-before-dpo`` (TPU v6e, profile ``m6``): ``docs/PLAN_M6V.md``.

Start every process of the run with ``BES_PROFILE=m6`` (or ``m6`` in ``.bes_profile``).

* **the model is ``configs/v9.py`` unchanged** (token pages on the dense looped core: 98,036,104
  stored numbers, 19,631,240 read per token) with the execution settings its 24-hour run ended
  with on the chip: the ``take`` table gradient, 16 rows per micro-batch cut into four pieces,
  and the speed switches ``kernel=onetile:save relu2=1 ce=0 flat=1 rows=1 tokn=1``. The compiled
  training step is the one that run tuned; nothing new is introduced on the chip;
* **it starts from the exported weights** (``run.init_from``: the parts of
  ``release/main16-v9-before-dpo``, step 51,812, 1.358e10 tokens; every one of the 64 leaves) with
  a fresh optimiser state;
* **the data stream is the six-hour one** of ``configs/m6.py``: conversations 90% (95% from
  85%), exemplars from the answers library with leave-one-out retrieval, hindsight calculator
  spans, 2,048-token rows, assistant-only loss in the last 15%;
* **the learning rate continues one schedule over both runs**: with ``T0 = 1.358e10`` tokens
  before and about ``T1 = 3.7e9`` now, the 24-hour rule (peak, then linear to 0.1 of it) for a run
  of ``T0 + T1`` tokens stands at ``1 - 0.9 T0 / (T0 + T1) = 0.29`` of its peak where this run
  begins and ends at 0.1: ``lr_mult`` 0.29, ``final_lr_frac`` 0.1 / 0.29 = 0.34, and the matrix
  group's data factor takes ``tokens_before`` into its horizon. Warm-up over the first 2% (the
  optimiser's moments start empty);
* **what ships is decided after the run**, on CPU (the water line between page rows and written
  entries, ``docs/PLAN_M6V.md`` section 3): the export of the chip holds the v9 weights without
  a library (``ship_library`` select, and no selection stage), so its ledger is v9's.
"""
import dataclasses
from pathlib import Path

from bes.config import load_config

_M6 = load_config(str(Path(__file__).with_name("m6.py")))
_V9 = load_config(str(Path(__file__).with_name("v9.py")))

TOKENS_BEFORE = 13_582_204_928          # training tokens of main16 at step 51,812
LR_MULT = 0.29
FINAL_LR_FRAC = 0.34
FAST = "kernel=onetile:save relu2=1 ce=0 flat=1 rows=1 tokn=1"
# weights16.npz of release/main16-v9-before-dpo (its README)
INIT_SHA256 = "634230225867f197ed75593680c6dce3a3828e3d404787573559f2a510a9073a"

CFG = dataclasses.replace(
    _M6,
    model=dataclasses.replace(_V9.model, table_grad="take"),
    optim=dataclasses.replace(_M6.optim, lr_mult=LR_MULT, tokens_before=float(TOKENS_BEFORE),
                              final_lr_frac=FINAL_LR_FRAC, warmup_frac=0.02),
    run=dataclasses.replace(_M6.run, name="m6v", out_dir="runs/m6v",
                            init_from="release/main16-v9-before-dpo", init_sha256=INIT_SHA256, fast=FAST,
                            ship_library="select"),
)
