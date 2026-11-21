"""Byte-level BPE tokenizer with 8192 ids (``docs/SPEC.md`` section 1).

``from bes.tokenizer import load`` returns the committed tokenizer; training code lives in
``bes.tokenizer.train_bpe`` and is never imported when loading.
"""
from bes.tokenizer.tok import Tokenizer, load

__all__ = ["Tokenizer", "load"]
