"""v9: write-side token pages (token neurons and output gains) on the dense looped core, without
the written library.

``configs/no_library.py`` with the value embedding back at the final width (``ve_dim=128``); the
stored numbers that frees go to two kinds of token page, both read one row per token:

* token neurons (``tok_neurons=8``, ``tok_dim=192``; ``bes.model.model.token_rows``,
  ``bes.model.layers.token_neurons``): every vocabulary id owns 8 private ReLU^2 neurons per
  shared block in a 192-dimensional neuron space shared by all tokens. The dense projections
  ``page/pin`` / ``page/pout`` (MuonH) map the feed-forward input into that space and back; the
  token's keys ``page/key`` (Adam ``embed`` group, used RMS-normalised) and values ``page/val``
  (Adam ``page`` group, zero-initialised) are rows of two lookup tables;
* output gains (``gain_o``; ``bes.model.model.gain_rows``): a zero-initialised row of
  multiplicative deltas for the attention output channels of the token's own position before
  ``W_o`` (Adam ``gain`` group). No FFN or value gains (``gain_ffn=0``, ``gain_v=False``), one
  phase.

The optimiser settings are those of the base: the two page groups train with Adam at
``lr_page`` and ``lr_gain`` and, like every other per-token table of the model, without decay
(``page_decay``, ``gain_decay`` exist and are 0: docs/V9.md section 8).

Both paths add exactly zero at initialisation and have a runtime switch (``gain``, ``page``).
The head-wise attention output gate (``attn_gate``; ``bes.model.attention.head_gate``) and
selective attention (``sel_attn``) are implemented and off here: the gate was neutral in the
micro ablation (-0.02% at 1M tokens) and has no measurement of its own behind it, selective
attention needs the dense attention path (docs/V9.md sections 8 and 9). The flash kernel
(``attn_impl="splash"``) is used; the vocabulary tables use the scatter-free ``"sorted"`` table
gradient.

Ledger (``bes.budget.count``, no library and no index; stored / read per token):
core 14,118,912 / all; head 4,587,520 / all; small 306,776 / all; emb 6,291,456 / 768; ve
8,388,608 / 1,024; gain 12,582,912 / 1,536; page 50,921,472 / 595,968 (keys and values
2 x 25,165,824 / 2 x 3,072, projections 589,824 / all); tables 838,448 / 18,736. Total
98,036,104 stored, 19,631,240 read per step (standard reading; the optional gate would add
9,312 to both). The strict reading counts the
token tables' rows of every distinct token of the window, 9,536 numbers each (24.5M at T=1024
with 480 distinct tokens), so only the standard reading is under 20M.
"""
import dataclasses
from pathlib import Path

from bes.config import load_config

_BASE = load_config(str(Path(__file__).with_name("no_library.py")))

CFG = dataclasses.replace(
    _BASE,
    model=dataclasses.replace(_BASE.model, ve_dim=128, gain_ffn=0, gain_o=True, gain_v=False,
                              gain_phases=1, tok_neurons=8, tok_dim=192, attn_gate=False,
                              sel_attn=False, attn_impl="splash", table_grad="sorted"),
    run=dataclasses.replace(_BASE.run, name="v9", out_dir="runs/v9"),
)
