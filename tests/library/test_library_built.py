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
    shipped = [os.path.join(r, f) for r, _, fs in os.walk(L.library_dir) for f in fs if f.endswith(".npy")]
    files = sum((np.load(f, mmap_mode="r").nbytes + 1) // 2 for f in shipped)   # every array in the directory
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


# ------------------------------------------------------------------ review fixes on the built library


def _stored_questions(lib):
    from bes.library.qa import parse_entry

    b = lib.qa.blocks
    out = []
    for k in range(b.n_blocks):
        raw = b.read_block(k)
        lo, hi = int(b.block_first[k]), int(b.block_first[k + 1])
        for e in range(lo, hi):
            s0 = int(b.entry_off[e])
            s1 = int(b.entry_off[e + 1]) if e + 1 < hi else len(raw)
            out.append(parse_entry(raw[s0:s1])[0])
    return out


_COUNTRIES = ("france germany italy spain england scotland ireland wales canada mexico brazil argentina china "
              "japan india russia australia egypt kenya nigeria sweden norway denmark finland poland austria "
              "hungary greece turkey iran iraq israel portugal belgium netherlands switzerland").split()


def _variants(q, keys):
    """The review's edits (its qa_perturb.py), written independently of bes.library.devsets."""
    import re

    out = []
    for a, b in (("before", "after"), ("after", "before")):
        if re.search(rf"\b{a}\b", q, re.I):
            out.append(("before/after", re.sub(rf"\b{a}\b", lambda m: b if m.group(0).islower() else b.title(),
                                               q, count=1, flags=re.I)))
    for a, b in (("first", "second"), ("first", "last"), ("last", "first")):
        if re.search(rf"\b{a}\b", q, re.I):
            out.append(("ordinal", re.sub(rf"\b{a}\b", lambda m: b if m.group(0).islower() else b.title(),
                                          q, count=1, flags=re.I)))
            break
    if re.search(r"\bnot\b", q, re.I):
        out.append(("drop not", re.sub(r"\s*\bnot\b", "", q, count=1, flags=re.I)))
    m = re.search(r"\b(19[0-9]{2}|20[0-2][0-9])\b", q)
    if m:
        for d in (4, 1, -4, -1):
            v = q[:m.start(1)] + str(int(m.group(1)) + d) + q[m.end(1):]
            if Q.question_key(v) not in keys:
                out.append(("year", v))
                break
    low = q.lower()
    for c in _COUNTRIES:
        if re.search(rf"\b{c}\b", low):
            o = [x for x in _COUNTRIES if x != c and not re.search(rf"\b{x}\b", low)][len(q) % 30]
            out.append(("country", re.sub(rf"\b{c}\b", o.title(), q, count=1, flags=re.I)))
            break
    return out


@pytest.mark.slow
def test_perturbed_questions_do_not_get_the_original_answer():
    """The review's perturbation test as a measured property: for stored questions edited so
    that their stored answer no longer fits (another year, ordinal, relation, polarity or
    country), the library returns the original entry for under 5% of the variants per class."""
    if not _have_lib():
        pytest.skip("library not built")
    lib = Library.load(L.library_dir)
    qs = _stored_questions(lib)
    keys = {Q.question_key(q) for q in qs}
    rng = random.Random(7)
    order = list(range(len(qs)))
    rng.shuffle(order)
    tested, original = {}, {}
    for e in order:
        for cls, v in _variants(qs[e], keys):
            if tested.get(cls, 0) >= 200 or Q.question_key(v) in keys:
                continue
            tested[cls] = tested.get(cls, 0) + 1
            p = lib.retrieve(v)
            original[cls] = original.get(cls, 0) + int(p is not None and p.kind == "qa" and p.pid == lib.n_wiki + e)
        if tested and min(tested.values()) >= 200 and len(tested) == 5:
            break
    assert set(tested) == {"before/after", "ordinal", "drop not", "year", "country"}
    for cls, n in tested.items():
        assert n >= 100 and original[cls] / n < 0.05, (cls, original[cls], n)


@pytest.mark.slow
def test_built_gate_meets_the_dev_floor():
    """The stored operating point: precision >= LibraryCfg.gate_precision on every dev set with
    right answers, at most 5% known-wrong returns on the contrastive and conversation sets."""
    if not _have_lib():
        pytest.skip("library not built")
    lib = Library.load(L.library_dir)
    per = lib.meta["build"]["gate_dev_point"]["per_set"]
    assert {"dev_nq", "dev_tqa", "dev_probe", "dev_names", "dev_contrast", "dev_chat"} <= set(per)
    for s, (cov, prec) in per.items():
        if s in ("dev_contrast", "dev_chat"):
            assert cov <= 0.05, s
        else:
            assert cov > 0 and prec >= L.gate_precision - 1e-9, s
    assert abs(lib.gate.threshold - lib.meta["build"]["gate_dev_point"]["t"]) < 1e-6


@pytest.mark.slow
def test_built_library_head_entities_and_follow_ups():
    if not _have_lib():
        pytest.skip("library not built")
    lib = Library.load(L.library_dir)
    for q, want in (("Who was Galileo?", "Galileo Galilei"), ("Who was Darwin?", "Charles Darwin"),
                    ("Who was Lincoln?", "Abraham Lincoln"), ("Who was Bach?", "Johann Sebastian Bach"),
                    ("Who was Einstein?", "Albert Einstein"), ("Who was Napoleon?", "Napoleon"),
                    ("What is DNA?", "DNA"), ("Tell me about Paris.", "Paris")):
        p = lib.retrieve(q)
        assert p is not None and p.title == want, (q, None if p is None else p.title)
    for q in ("Who was Gandhi?", "Who was Jefferson?", "What is a pointer?", "Why is the sky blue?",
              "merry christmas", "try again", "What does len do?", "Who wrote Romeo and Juliet?",
              "Who won the Super Bowl in 2020?", "Is 17 a prime number?", "Who is the richest person in the world?"):
        p = lib.retrieve(q)
        assert p is None or (p.kind == "wiki" and q.split()[-1].rstrip("?").lower() in p.title.lower()), (q, p.title)
    for q in ("What is its capital?", "What is the next step?", "Is it safe?", "What color is it?", "And cats?",
              "What about Mars?", "Who is the author?", "thanks, that was helpful", "ok great"):
        assert lib.retrieve(q, context=["Tell me about France."]) is None, q
