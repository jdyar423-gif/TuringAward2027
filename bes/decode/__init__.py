"""Feynman-Kac decoding (``docs/SPEC_MODEL_TRAIN.md`` section 8, ``docs/MATH.md`` F1-F5).

- ``bes.decode.lm``: the language-model interface (``LanguageModel``), ``OurLM`` over
  ``bes.model.generate``, ``TableLM`` / ``TinyTokenizer`` for exhaustive tests,
  ``PerSequenceLM`` for one-sequence-at-a-time models (llama.cpp).
- ``bes.decode.sampling``: per-step transforms (temperature, top-k / top-p / min-p, n-gram
  blocking, copy-run cap), counter-based random streams, resampling.
- ``bes.decode.potentials``: exact 0/1 token-local potentials with reachability pruning,
  calculator forcing and the optional look-ahead twist.
- ``bes.decode.smc``: the particle filter (``decode``, ``run``).

Submodules are imported lazily so that ``bes.decode.sampling`` stays numpy-only.
"""
from __future__ import annotations

import importlib

__all__ = ["lm", "sampling", "potentials", "smc"]


def __getattr__(name: str):
    if name in __all__:
        return importlib.import_module(f"bes.decode.{name}")
    raise AttributeError(name)
