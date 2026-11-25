"""Six-hour configuration (TPU v6e, profile ``m6``): ``docs/PLAN_M6.md``.

Start every process of the run with ``BES_PROFILE=m6``, or write ``m6`` into ``.bes_profile``
at the repository root (``bes.config.PROFILE``): the profile selects the answers library
(``bes.library.answers``, ``data_cache/library/answers_v1``, unpacked from
``release/m6/library``), the chat-heavy mixture (``MixCfg``) and the registry of
``bes.data.sources`` with its extra chat sources and calculator parts for arithmetic.

The core is ``configs/final.py`` unchanged (the dense looped core with the candidate tape,
registers and per-loop value embeddings: 19.2M numbers read per token), with the execution
settings measured on the chip in the 24-hour run made part of the configuration: 16 rows per
micro-batch, the ``take`` table gradient, the one-tile attention kernel and the hand-written
backward of the squared ReLU (``run.fast``), micro-batches cut into four pieces on the device
(``run.micro_split``). What differs:

* **from scratch in six hours**: about 5e9 training tokens instead of 1.36e10, so nearly all
  of them are conversations (``chat_frac`` 0.90 at the first step, 0.95 from 85%) and the whole
  run uses 2,048-token rows (``late_from`` 0): a conversation with its exemplars does not fit
  1,024;
* **exemplars**: a chat sample goes through the retrieval of inference with probability
  ``library_frac_chat`` 0.9; when the answers library holds a similar question, that entry
  (and at most one more) is shown in a ``LIBRARY`` turn before the user turn, as the
  responder will show it. Passage tokens keep the loss weight 0.25. ``distractor_frac`` 0.05
  swaps in an unrelated entry, so the model also sees exemplars it must ignore;
* **no library tasks** (``library_frac_base`` 0 and ``MixCfg.library_weight`` 0 in the
  profile): the stored entries are first exchanges of training conversations already;
* the last 15% of training has loss on assistant tokens only (``assistant_only_from``,
  ``tail_from`` 0.85);
* **a short start**: only the configured plan is measured (``bench_plans`` single), the canary
  takes 10 steps, no opponents are prepared (the comparison is made after the chip is given
  back), 16 tokenisation workers and 14 loader workers use the host's idle cores;
* **post-training on the chip**: preference optimisation (one learning rate, 3e-6: the one
  the 24-hour run accepted) and the export only; the library ships without a selection
  stage (``ship_library``): it is part of the model. Sampling, the critic, the decode
  selection and the judged comparison are done on CPU after the chip is given back.
"""
import dataclasses
from pathlib import Path

from bes.config import load_config

_BASE = load_config(str(Path(__file__).with_name("final.py")))

CFG = dataclasses.replace(
    _BASE,
    model=dataclasses.replace(_BASE.model, table_grad="take"),
    optim=dataclasses.replace(_BASE.optim, micro_seqs=16),
    data=dataclasses.replace(
        _BASE.data, profile="m6", late_from=0.0, chat_frac=0.90, chat_frac_tail=0.95,
        tail_from=0.85, assistant_only_from=0.85, library_frac_base=0.0, library_frac_chat=0.9,
        distractor_frac=0.05, use_library=True),
    run=dataclasses.replace(
        _BASE.run, name="m6", out_dir="runs/m6", deadline_hours=6.0,
        post_reserve_hours=22.0 / 60.0,
        post_minutes="sample=0 critic=0 prefopt=15 select=0 judge=0 export=4 buffer=3",
        ckpt_every_min=12.0, eval_every_min=45.0, restart_window_hours=0.25,
        preflight_min=1.0, benchmark_min=4.0, benchmark_cap_min=7.0, bench_plans="single",
        canary_steps=10, loader_workers=14, tokenize_workers=16,
        fast="kernel=onetile:save relu2=1 ce=0", micro_split="4",
        eval_opponents="", eval_judge="", prefopt_lrs="3e-6", ship_library="always"),
)
