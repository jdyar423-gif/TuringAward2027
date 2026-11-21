"""Checks on the real built library and the real question data (skipped when they are not
in ``data_cache``; marked slow)."""
from __future__ import annotations

import os
import random

import numpy as np
import pytest

from bes.config import LibraryCfg
from bes.library import qa as Q
from bes.library.retrieve import Library, ReadLog

L = LibraryCfg()
CACHE = "data_cache/library"


def _have_lib() -> bool:
    return os.path.exists(os.path.join(L.library_dir, "library.json"))


def _ref_key(s: str) -> str:
    import unicodedata

    s = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", s))
    s = "".join(c for c in s if not unicodedata.category(c).startswith("M")).lower()
    return " ".join("".join(c if unicodedata.category(c)[0] in "LN" else " " for c in s).split())


@pytest.mark.slow
def test_no_stored_question_equals_an_evaluation_question():
    from bes.library.build import eval_questions
    from bes.library.sources import QA_SOURCES

    raw = [os.path.join(CACHE, "raw", s.dir, s.filename) for s in QA_SOURCES.values()]
    if not all(os.path.exists(p) for p in raw):
        pytest.skip("question data not downloaded")
    ev = {_ref_key(q) for q in eval_questions(CACHE)} - {""}
    if not _have_lib():
        pytest.skip("library not built")
    lib = Library.load(L.library_dir)
    bad = [q for q in (lib.passage(pid).title for pid in lib.kind_range("qa")) if _ref_key(q) in ev]
    assert not bad, bad[:5]


@pytest.mark.slow
def test_built_library_accounting_equals_saved_arrays_and_cap():
    if not _have_lib():
        pytest.skip("library not built")
    lib = Library.load(L.library_dir)
    acc = lib.accounting()
    files = sum((np.load(os.path.join(L.library_dir, f), mmap_mode="r").nbytes + 1) // 2
                for f in os.listdir(L.library_dir) if f.endswith(".npy"))
    headers = lib.index.HEADER_NUMBERS + (Q.QAStore.HEADER_NUMBERS if lib.qa is not None else 0)
    assert acc["total_numbers"] == files + headers
    assert acc["total_numbers"] <= L.total_numbers_cap
    assert lib.read_bound()["total"] <= L.max_read_numbers
    assert lib.gate is not None and 0.0 < lib.gate.threshold < 1.0


@pytest.mark.slow
def test_built_library_entries_and_kinds():
    if not _have_lib():
        pytest.skip("library not built")
    lib = Library.load(L.library_dir)
    from bes.library.clean import bpe_count_tokens

    rng = random.Random(0)
    for pid in rng.sample(range(lib.n_passages), 300):
        p = lib.passage(pid)
        assert p.kind == lib.kind_of(pid)
        assert bpe_count_tokens(p.text) <= L.max_passage_tokens
        if p.kind == "qa":
            assert p.text == f"Q: {p.title}\nA: {p.body}" and p.title_id == -1
    for q in ("Who was Alan Turing?", "What is photosynthesis?", "Who wrote Hamlet?"):
        log = ReadLog()
        lib.retrieve(q, log=log)
        assert log.total <= L.max_read_numbers
        for kinds in (("wiki",), ("qa",)):
            r = lib.retrieve(q, kinds=kinds)
            assert r is None or r.kind == kinds[0]
