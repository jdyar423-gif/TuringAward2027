"""Gated retrieval, exclusion, read-set log and accounting (bes.library.retrieve)."""
from __future__ import annotations

import dataclasses
import os
import random

import numpy as np
import pytest

from bes.library.clean import default_count_tokens, render_passage
from bes.library.index import IndexCfg, IndexRead
from bes.library.pack import PackCfg
from bes.library.retrieve import (GateCfg, Library, ReadLog, RetrieveCfg, cap_tokens, query_type)

FILLER = ("people city river music history language country year world government school "
          "animal plant water food family art book game team sport").split()


def corpus(seed: int = 0):
    rng = random.Random(seed)
    arts = [
        ("Alan Turing", ["Alan Turing was an English mathematician and computer scientist.",
                         "Turing worked at Bletchley Park during the Second World War.",
                         "He died in 1954 at his home in Wilmslow."]),
        ("Ada Lovelace", ["Ada Lovelace was an English mathematician who wrote about the Analytical Engine."]),
        ("Paris", ["Paris is the capital and largest city of France.", "The Seine river flows through Paris."]),
        ("Rome", ["Rome is the capital city of Italy.", "Rome was founded long ago."]),
        ("Mint (candy)", ["A mint is a sweet candy with a fresh taste."]),
    ]
    for i in range(40):
        arts.append((f"Topic {i}", [" ".join(rng.choice(FILLER) for _ in range(rng.randint(8, 30))) + "."
                                    for _ in range(rng.randint(1, 4))]))
    return [(t, b) for t, bodies in arts for b in bodies]


def build(tmp_path=None, **kw):
    gate = kw.pop("gate", GateCfg(min_score=1.0, min_margin=0.3, min_margin_title=0.05, min_title_match=0.6))
    cfg = RetrieveCfg(max_tokens=kw.pop("max_tokens", 160), gate=gate)
    icfg = kw.pop("index_cfg", IndexCfg(budget_numbers=10**7))
    return Library.build(corpus(), out_dir=None if tmp_path is None else str(tmp_path),
                         index_cfg=icfg, pack_cfg=PackCfg(block_raw_bytes=512, titles_per_block=4),
                         cfg=cfg)


def test_query_types():
    assert query_type("Who was Alan Turing?") == "question"
    assert query_type("please tell me about Paris") == "question"
    assert query_type("Alan Turing") == "lookup"
    assert query_type("Write a poem about Alan Turing") == "task"
    assert query_type("Can you translate this into French: hello") == "task"
    assert query_type("What is 12 * 7?") == "math"
    assert query_type("```python\nprint(1)\n```") == "task"
    assert query_type("") == "other"
    assert query_type("hello") == "other" and query_type("thank you!") == "other"
    assert query_type("hello, who wrote Hamlet?") == "question"


def test_retrieve_gate_and_title_match():
    lib = build()
    p = lib.retrieve("Who was Alan Turing?")
    assert p is not None and p.title == "Alan Turing" and p.text.startswith("Alan Turing\n")
    assert p.body == "Alan Turing was an English mathematician and computer scientist."  # the lead
    assert lib.retrieve("Write a poem about Alan Turing") is None          # task: no retrieval
    assert lib.retrieve("xyzzy plugh") is None                              # nothing matches
    strict = dataclasses.replace(lib.cfg.gate, min_score=1e9)
    lib.cfg = dataclasses.replace(lib.cfg, gate=strict)
    assert lib.retrieve("Who was Alan Turing?") is None                     # score threshold


def test_gate_margin_logic_on_features():
    lib = build()
    g = GateCfg(min_score=5.0, min_margin=0.4, min_margin_title=0.1, min_title_match=0.6)
    base = {"s1": 10.0, "s2": 9.0, "margin": 0.1, "title_match": 0.0, "qtype": "question"}
    assert not lib.gate_pass(base, g)                                     # small margin, no title
    assert lib.gate_pass({**base, "title_match": 1.0}, g)                 # title match lowers the bar
    assert not lib.gate_pass({**base, "title_match": 1.0, "margin": 0.05}, g)
    assert lib.gate_pass({**base, "margin": 0.5}, g)                      # large margin alone
    assert not lib.gate_pass({**base, "margin": 0.5, "s1": 4.0}, g)       # score floor
    assert not lib.gate_pass({**base, "margin": 0.5, "qtype": "task"}, g)  # query type
    gc = dataclasses.replace(g, min_coverage=0.8)
    assert lib.gate_pass({**base, "margin": 0.5, "coverage": 0.9}, gc)
    assert not lib.gate_pass({**base, "margin": 0.5, "coverage": 0.7}, gc)


def test_gate_features_margin_uses_a_different_title():
    lib = build()
    pids, sc = lib.search("Alan Turing mathematician")
    f = lib.gate_features("Alan Turing mathematician", pids, sc)
    t1 = lib.title_id_of(int(pids[0]))
    r = lib.title_range(t1)
    other = [s for p, s in zip(pids, sc) if not (r.start <= p < r.stop)]
    assert f["s2"] == pytest.approx(other[0] if other else 0.0)
    assert f["margin"] == pytest.approx((f["s1"] - f["s2"]) / f["s1"])
    assert f["title_match"] == 1.0
    _, _, smax = lib.search_max("Alan Turing mathematician")
    f2 = lib.gate_features("Alan Turing mathematician", pids, sc, smax=smax)
    assert 0 < f2["coverage"] <= 1.0 and smax >= f2["s1"] - 1e-9


def test_exclusion_by_passage_and_title():
    lib = build()
    q = "Alan Turing mathematician Bletchley"
    pids, sc = lib.search(q)
    top = int(pids[0])
    p2, s2 = lib.search(q, exclude_pids=[top])
    assert list(p2) == [int(p) for p in pids if p != top]
    tid = lib.title_id_of(top)
    rng = lib.title_range(tid)
    p3, _ = lib.search(q, exclude_titles=[tid])
    assert list(p3) == [int(p) for p in pids if not (rng.start <= p < rng.stop)]
    hit = lib.retrieve("Who was Alan Turing?", exclude_titles=[tid])
    assert hit is None or hit.title != "Alan Turing"
    assert lib.find_title("Alan Turing") == tid and lib.find_title("Nope") is None


def test_read_set_log_is_exact():
    lib = build()
    q = "Who was Alan Turing?"
    log = ReadLog()
    p = lib.retrieve(q, log=log)
    assert p is not None
    ir = IndexRead()
    lib.index.score(q, ir)
    assert (log.index_df, log.index_offsets, log.index_postings) == (ir.df_numbers, ir.offset_numbers,
                                                                     ir.posting_numbers)
    b, _, _, _ = lib.packed.locate(p.pid)
    assert log.blocks == {b} and log.text_blocks == lib.packed.block_numbers(b)
    tb = p.title_id // lib.packed.cfg.titles_per_block
    assert log.title_block_ids == {tb} and log.title_blocks == lib.packed.title_block_numbers(tb)
    assert log.total == log.index + log.directory + log.text_blocks + log.title_blocks
    assert log.total == sum(v for k, v in log.as_dict().items() if k in (
        "index_df", "index_offsets", "index_postings", "directory", "text_blocks", "title_blocks"))
    before = log.total
    lib.passage(p.pid, log=log)  # reading the same passage again adds nothing
    assert log.total == before
    # a rejected query reads no text block
    log2 = ReadLog()
    assert lib.retrieve("xyzzy plugh", log=log2) is None and log2.text_blocks == 0
    # top-k: every distinct block is counted once
    log3 = ReadLog()
    ps = lib.retrieve_topk("capital city river", 5, log=log3)
    blocks = {lib.packed.locate(x.pid)[0] for x in ps}
    assert log3.blocks == blocks
    assert log3.text_blocks == sum(lib.packed.block_numbers(x) for x in blocks)


def test_accounting_equals_saved_arrays(tmp_path):
    lib = build(tmp_path)
    acc = lib.accounting()
    assert acc["total_numbers"] == acc["library_numbers"] + acc["index_numbers"]
    total = 0
    for fn in os.listdir(tmp_path):
        if fn.endswith(".npy"):
            total += (np.load(os.path.join(tmp_path, fn)).nbytes + 1) // 2
    assert total + lib.index.HEADER_NUMBERS == acc["total_numbers"]
    re = Library.load(str(tmp_path))
    for q in ("Who was Alan Turing?", "capital of France", "Rome"):
        a, b = lib.retrieve(q), re.retrieve(q)
        assert (a is None and b is None) or (a.pid == b.pid and a.text == b.text)
    assert re.accounting() == acc


def test_passages_are_capped_in_tokens():
    lib = build(max_tokens=12)
    for q in ("Who was Alan Turing?", "Paris", "Rome capital"):
        for p in lib.retrieve_topk(q, 3):
            assert default_count_tokens(p.text) <= 12
            assert p.text.startswith(p.title)


def test_cap_tokens_cuts_at_sentence_boundaries():
    title, body = "T", "First sentence here. Second sentence is longer than the first.\nThird line."
    assert cap_tokens(title, body, 1000, default_count_tokens) == render_passage(title, body)
    cut = cap_tokens(title, body, default_count_tokens("T\nFirst sentence here.") + 0.1, default_count_tokens)
    assert cut == "T\nFirst sentence here."
    words = cap_tokens(title, "aaaa bbbb cccc dddd eeee", default_count_tokens("T\naaaa bbbb"), default_count_tokens)
    assert words == "T\naaaa bbbb"


def test_retrieve_topk_matches_index_ranking():
    lib = build()
    q = "capital city of France and Italy"
    pids, sc = lib.search(q)
    top = lib.retrieve_topk(q, 4)
    assert [p.pid for p in top] == [int(x) for x in pids[:4]]
    assert [p.score for p in top] == pytest.approx([float(s) for s in sc[:4]])
    assert {p.title for p in top[:2]} == {"Paris", "Rome"}
