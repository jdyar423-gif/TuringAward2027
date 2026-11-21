"""Library v2 (bes.library.retrieve with the qa layer, title look-up and calibrated gate):
kinds, gate decision, exclusion, read-set log and bound, save/load, accounting, workers,
determinism - each against a direct computation written here."""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import pickle
import random

import numpy as np
import pytest

from bes.config import LibraryCfg
from bes.library import gate as G
from bes.library import qa as Q
from bes.library.build import dir_digest
from bes.library.clean import proxy_count_tokens
from bes.library.index import IndexCfg, IndexRead
from bes.library.pack import PackCfg
from bes.library.retrieve import (Library, ReadLog, RetrieveCfg, TitleTable, key_hash, query_title_key,
                                  query_type, title_key)
from bes.library.terms import STOPWORDS, stem, strip_disambiguator

FILLER = ("people city river music history language country year world government school "
          "animal plant water food family art book game team sport").split()


def corpus(seed: int = 0):
    rng = random.Random(seed)
    arts = [
        ("Alan Turing", ["Alan Turing was an English mathematician and computer scientist.",
                         "Turing worked at Bletchley Park during the Second World War."]),
        ("Napoleon", ["Napoleon was a French military leader and emperor."]),
        ("Napoleon Dynamite", ["Napoleon Dynamite is a 2004 comedy film."]),
        ("Mercury (planet)", ["Mercury is the smallest planet in the Solar System."]),
        ("Mercury (element)", ["Mercury is a chemical element with symbol Hg."]),
        ("Black Hole (2019 movie)", ["Black Hole is a 2019 short film."]),
        ("Black hole", ["A black hole is a region of space from which nothing can escape."]),
        ("Hamlet", ["Hamlet is a tragedy written by William Shakespeare."]),
        ("Paris", ["Paris is the capital and largest city of France."]),
    ]
    for i in range(40):
        arts.append((f"Topic {i}", [" ".join(rng.choice(FILLER) for _ in range(rng.randint(8, 30))) + "."
                                    for _ in range(rng.randint(1, 3))]))
    return [(t, b) for t, bodies in arts for b in bodies]


QA = [("who wrote hamlet?", "William Shakespeare"), ("what is the capital of france?", "Paris"),
      ("who was the first emperor of france?", "Napoleon"), ("when did alan turing die?", "1954"),
      ("who sang hey jude?", "The Beatles"), ("what is the smallest planet?", "Mercury"),
      ("who wrote the screenplay of the 1996 film hamlet?", "Kenneth Branagh")]


def gate_model(wiki_bias: float = 0.0, qa_bias: float = 0.0, threshold: float = 0.5) -> G.GateModel:
    cw = np.zeros(len(G.WIKI_FEATURES) + 1)
    cw[G.WIKI_FEATURES.index("exact_title")] = 3.0
    cw[G.WIKI_FEATURES.index("log_s1")] = 0.5
    cw[-1] = wiki_bias - 2.0
    cq = np.zeros(len(G.QA_FEATURES) + 1)
    cq[G.QA_FEATURES.index("sym")] = 6.0
    cq[-1] = qa_bias - 4.0
    return G.GateModel.stored({"wiki": cw, "qa": cq}, threshold)


def build(tmp_path=None, gate=None, **kw):
    pairs = [Q.QAPair(q, a, "s", i) for i, (q, a) in enumerate(QA)]
    store = Q.QAStore.build(pairs, Q.QACfg(min_cov_q=0.0), PackCfg(block_raw_bytes=128))
    docs = corpus()
    titles = list(dict.fromkeys(t for t, _ in docs))
    imp = [5.0 if t in ("Mercury (planet)", "Black Hole (2019 movie)") else 1.0 for t in titles]
    cfg = RetrieveCfg(max_tokens=kw.pop("max_tokens", 160), qa_min_sym=0.5)
    return Library.build(docs, out_dir=None if tmp_path is None else str(tmp_path),
                         index_cfg=IndexCfg(budget_numbers=10**7),
                         pack_cfg=PackCfg(block_raw_bytes=256, titles_per_block=4), cfg=cfg,
                         count_tokens=proxy_count_tokens, qa=store, title_importance=imp,
                         gate=gate if gate is not None else gate_model())


# ------------------------------------------------------------------ ids and kinds


def test_kind_ranges_and_entries():
    lib = build()
    w, q = lib.kind_range("wiki"), lib.kind_range("qa")
    assert w.start == 0 and w.stop == q.start and q.stop == lib.n_passages
    assert len(q) == len(QA) and lib.n_passages == len(corpus()) + len(QA)
    stored = {}
    for pid in q:
        p = lib.passage(pid)
        assert p.kind == "qa" and p.title_id == -1 and p.text == f"Q: {p.title}\nA: {p.body}"
        stored[p.title] = p.body
    assert stored == dict(QA)
    for pid in w:
        p = lib.passage(pid)
        assert p.kind == "wiki" and p.text.startswith(p.title + "\n")
        assert pid in lib.title_range(p.title_id)
    assert lib.title_range(-1) == range(0)
    with pytest.raises(ValueError):
        lib.kind_range("book")
    with pytest.raises(IndexError):
        lib.passage(lib.n_passages)


def test_kinds_filter_and_gate_decision_equal_direct_computation():
    lib = build()
    queries = ["Who wrote Hamlet?", "Who was Napoleon?", "What is the capital of France?",
               "Tell me about Alan Turing.", "Mercury", "xyzzy plugh", "When did Alan Turing die?"]
    for q in queries:
        for kinds in (None, ("wiki",), ("qa",)):
            cands = lib.candidates(q, 1, kinds=kinds)
            best = None
            for c in cands:                      # direct: gate probability of each candidate
                p = lib.gate.prob(c.kind, c.feats)
                assert c.prob == p
                if best is None or p > best[0]:
                    best = (p, c)
            got = lib.retrieve(q, kinds=kinds)
            allowed = query_type(q) in ("question", "lookup")
            if not allowed or best is None or best[0] < lib.gate.threshold:
                assert got is None
            else:
                assert got is not None and got.pid == best[1].pid and got.prob == best[0]
                assert got.kind in (kinds or ("wiki", "qa"))
            top = lib.retrieve_topk(q, 4, kinds=kinds)
            assert all(p.kind in (kinds or ("wiki", "qa")) for p in top)
            assert [p.prob for p in top] == sorted([p.prob for p in top], reverse=True)
    with pytest.raises(ValueError):
        lib.retrieve("Who wrote Hamlet?", kinds=("book",))


def test_qa_entry_wins_when_its_probability_is_larger():
    lib = build(gate=gate_model(wiki_bias=-10.0, qa_bias=5.0, threshold=0.5))
    p = lib.retrieve("Who wrote Hamlet?")
    assert p is not None and p.kind == "qa" and p.body == "William Shakespeare"
    lib = build(gate=gate_model(wiki_bias=5.0, qa_bias=-10.0, threshold=0.5))
    p = lib.retrieve("Who wrote Hamlet?")
    assert p is not None and p.kind == "wiki"
    lib = build(gate=gate_model(threshold=1.01))
    assert lib.retrieve("Who wrote Hamlet?") is None            # nothing reaches the threshold


def test_exclusion_of_qa_entries_and_wiki_titles():
    lib = build(gate=gate_model(wiki_bias=-10.0, qa_bias=5.0, threshold=0.1))
    p = lib.retrieve("Who wrote Hamlet?")
    assert p.kind == "qa"
    p2 = lib.retrieve("Who wrote Hamlet?", exclude_pids=[p.pid])
    assert p2 is None or p2.pid != p.pid
    cands = lib.candidates("Who wrote Hamlet?", 5, exclude_pids=[p.pid], kinds=("qa",))
    assert p.pid not in [c.pid for c in cands]
    lib = build(gate=gate_model(wiki_bias=5.0, qa_bias=-10.0, threshold=0.1))
    w = lib.retrieve("Who was Napoleon?")
    assert w.kind == "wiki" and w.title == "Napoleon"
    w2 = lib.retrieve("Who was Napoleon?", exclude_titles=[w.title_id])
    assert w2 is None or w2.title_id != w.title_id
    w3 = lib.retrieve("Who was Napoleon?", exclude_pids=[w.pid])
    assert w3 is None or w3.pid != w.pid
    assert lib.retrieve("Who was Napoleon?", exclude_titles=[-1]).pid == w.pid   # -1: no-op


# ------------------------------------------------------------------ title look-up


def ref_title_key(title: str) -> str:
    base = title[:title.find(" (")] if " (" in title and title.endswith(")") else title
    words = "".join(c.lower() if c.isalnum() else " " for c in base).split()
    return " ".join(stem(t) for t in words if t not in STOPWORDS)


def test_title_lookup_against_brute_force():
    lib = build()
    titles = lib.packed.all_titles()
    for q in ["Who was Napoleon?", "What is Mercury?", "Tell me about Alan Turing.", "hamlet",
              "Who was Napoleon Dynamite?", "What does Paris mean?", "Who wrote Hamlet?", "Black hole"]:
        key = query_title_key(q)
        matches = [i for i, t in enumerate(titles) if ref_title_key(t) == key and key]
        hit = lib.titles.lookup(key)
        if not matches:
            assert hit is None or ref_title_key(titles[hit[0]]) != key
            continue
        assert hit is not None
        tid, amb = hit
        plain = [i for i in matches if "(" not in titles[i]]
        pool = plain or matches                                   # the primary topic first
        assert tid in pool
        assert amb == (len(matches) > 1 and not (tid in plain and len(plain) == 1))
    tid, amb = lib.titles.lookup(query_title_key("What is Mercury?"))
    assert titles[tid] == "Mercury (planet)" and amb             # no primary topic: more important kept
    tid, amb = lib.titles.lookup(query_title_key("What is a black hole?"))
    assert titles[tid] == "Black hole" and not amb               # the plain title is the primary topic
    c = lib.candidates("Who was Napoleon?", 1, kinds=("wiki",))[0]
    assert lib.passage(c.pid).title == "Napoleon" and c.feats["exact_title"] and c.pid == lib.title_range(
        lib.title_id_of(c.pid)).start
    assert title_key("Mercury (planet)") == "mercury" and query_title_key("What does Paris mean?") == "pari"


def test_title_hash_collision_is_rejected(monkeypatch):
    lib = build()
    import bes.library.retrieve as R

    monkeypatch.setattr(R, "key_hash", lambda key: 7)       # every key collides
    titles = lib.packed.all_titles()
    table = TitleTable.build(titles)
    lib.titles = table
    tid, _ = table.lookup("anything")
    # the table answers for any key, the library verifies the stored title
    c = lib.candidates("Who was Napoleon?", 1, kinds=("wiki",))[0]
    if titles[tid] != "Napoleon":
        assert not (c.feats["exact_title"] and lib.passage(c.pid).title != "Napoleon")
    assert key_hash is not R.key_hash


# ------------------------------------------------------------------ read set


def test_read_log_is_the_sum_of_its_parts():
    lib = build()
    for q in ["Who wrote Hamlet?", "Who was Napoleon?", "capital city river", "xyzzy"]:
        log = ReadLog()
        lib.retrieve(q, log=log)
        ir = IndexRead()
        lib.index.score_with_max(q, ir)
        assert log.index == ir.total
        qr = Q.QARead()
        lib.qa.search(q, (), qr)
        assert log.qa_index == qr.total
        assert log.text_blocks == sum(lib.packed.block_numbers(b) for b in log.blocks)
        assert log.qa_blocks == sum(lib.qa.blocks.block_numbers(b) for b in log.qa_block_ids)
        assert log.title_blocks == sum(lib.packed.title_block_numbers(b) for b in log.title_block_ids)
        assert log.total == (log.index + log.directory + log.text_blocks + log.title_blocks
                             + log.title_lookup + log.qa_index + log.qa_blocks)
        assert log.as_dict()["total"] == log.total


def worst_case_library():
    """Every query term in more passages than the list cap, every qa list longer than the qa
    read budget, incompressible block text: the read set of one retrieve() is maximal."""
    L = LibraryCfg()
    rng = random.Random(0)
    terms16 = [f"term{i}" for i in range(16)]
    n = L.index_max_list + 50
    docs = []
    for i in range(n):
        noise = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(400))
        docs.append((f"T{i}", " ".join(terms16) + " " + noise))
    pairs = []
    for i in range(L.qa_read_postings + 200):
        noise = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(30))
        pairs.append(Q.QAPair(" ".join(terms16) + f" {noise}?", noise[:20], "s", i))
    store = Q.QAStore.build(pairs, Q.QACfg(min_cov_q=0.0))
    icfg = IndexCfg(budget_numbers=10**8)
    lib = Library.build(docs, index_cfg=icfg, pack_cfg=PackCfg(), cfg=RetrieveCfg(qa_min_sym=0.0),
                        count_tokens=proxy_count_tokens, qa=store, gate=gate_model(threshold=0.0))
    return lib, " ".join(terms16) + "?"


def test_retrieve_reads_at_most_the_budget():
    lib, q = worst_case_library()
    L = LibraryCfg()
    bound = lib.read_bound()
    assert bound["total"] <= L.max_read_numbers, bound              # provable worst case
    rng = random.Random(1)
    words = q.rstrip("?").split()
    worst = 0
    for i in range(40):
        qq = "what is " + " ".join(rng.sample(words, rng.randint(1, 16))) + "?"
        log = ReadLog()
        lib.retrieve(qq, log=log)
        assert log.total <= bound["total"], (log.as_dict(), bound)
        assert log.index <= bound["index"] and log.qa_index <= bound["qa_index"]
        worst = max(worst, log.total)
    log = ReadLog()
    assert lib.retrieve("what is " + q, log=log) is not None
    assert log.index_postings == 16 * 2 * L.index_max_list          # every list read in full
    assert log.total <= bound["total"]


def test_read_bound_for_qa_lists_within_the_budget():
    L = LibraryCfg()
    terms16 = [f"term{i}" for i in range(16)]
    per = L.qa_read_postings // 16
    pairs = [Q.QAPair(f"{terms16[i % 16]} filler{i}?", f"a{i}", "s", i) for i in range(16 * per)]
    pairs += [Q.QAPair(" ".join(terms16) + f" extra{j}?", f"b{j}", "s", 10**6 + j) for j in range(3)]
    store = Q.QAStore.build(pairs, Q.QACfg(min_cov_q=0.0))
    lib = Library.build(corpus(), index_cfg=IndexCfg(budget_numbers=10**7), cfg=RetrieveCfg(qa_min_sym=0.0),
                        count_tokens=proxy_count_tokens, qa=store, gate=gate_model(threshold=0.0))
    log = ReadLog()
    lib.retrieve("what is " + " ".join(terms16) + "?", log=log)
    assert log.qa_index <= lib.read_bound()["qa_index"]
    assert log.qa_index > 2 * (16 * per - per)                       # nearly the whole qa budget read


@pytest.mark.slow
def test_built_library_read_set_bound():
    path = LibraryCfg().library_dir
    if not os.path.exists(os.path.join(path, "library.json")):
        pytest.skip("library not built")
    lib = Library.load(path)
    rng = random.Random(0)
    words = ["who", "what", "when", "film", "song", "first", "world", "war", "king", "river", "name",
             "the", "of", "in", "year", "city", "country", "team", "won", "played"]
    worst = 0
    for _ in range(300):
        q = " ".join(rng.choice(words) for _ in range(rng.randint(2, 24))) + "?"
        log = ReadLog()
        lib.retrieve(q, log=log)
        worst = max(worst, log.total)
    assert worst <= LibraryCfg().max_read_numbers


# ------------------------------------------------------------------ io, accounting, workers


def test_save_load_round_trip_and_accounting(tmp_path):
    lib = build(tmp_path)
    re_ = Library.load(str(tmp_path), count_tokens=proxy_count_tokens)
    assert re_.qa is not None and re_.gate is not None and re_.titles is not None
    for q in ["Who wrote Hamlet?", "Who was Napoleon?", "What is the capital of France?", "Mercury", "xyz"]:
        a, b = lib.retrieve(q), re_.retrieve(q)
        assert (a is None and b is None) or (a.pid == b.pid and a.text == b.text and a.prob == b.prob)
        ta, tb = lib.retrieve_topk(q, 3), re_.retrieve_topk(q, 3)
        assert [(x.pid, x.kind) for x in ta] == [(x.pid, x.kind) for x in tb]
    acc = re_.accounting()
    assert acc == lib.accounting()
    files = sum((np.load(os.path.join(tmp_path, f)).nbytes + 1) // 2 for f in os.listdir(tmp_path)
                if f.endswith(".npy"))
    headers = lib.index.HEADER_NUMBERS + Q.QAStore.HEADER_NUMBERS
    assert acc["total_numbers"] == files + headers
    assert acc["gate_numbers"] == (len(G.WIKI_FEATURES) + len(G.QA_FEATURES) + 3) * 2
    parts = ("library_numbers", "index_numbers", "title_table_numbers", "qa_numbers", "gate_numbers")
    assert acc["total_numbers"] == sum(acc[k] for k in parts)


def _worker(args):
    lib, q = args
    p = lib.retrieve(q)
    return None if p is None else (p.pid, p.kind, p.text)


def test_pickled_library_works_in_a_spawned_worker(tmp_path):
    build(tmp_path)
    lib = Library.load(str(tmp_path), count_tokens=proxy_count_tokens)
    blob = pickle.dumps(lib)
    assert len(blob) < 10_000                        # a loaded library pickles as its path
    again = pickle.loads(blob)
    q = "Who wrote Hamlet?"
    want = _worker((lib, q))
    assert _worker((again, q)) == want
    with mp.get_context("spawn").Pool(1) as pool:
        assert pool.map(_worker, [(lib, q)])[0] == want


def test_block_cache_is_bounded_and_used(monkeypatch):
    lib = build()
    lib.cfg = RetrieveCfg(cache_blocks=2, max_tokens=lib.cfg.max_tokens)
    calls = []
    real = lib.packed.read_block
    monkeypatch.setattr(lib.packed, "read_block", lambda b: (calls.append(b), real(b))[1])
    lib.passage(0)
    lib.passage(0)
    assert len(calls) == 1
    for pid in range(lib.n_wiki):
        lib.passage(pid)
    assert len(lib._cache) <= 2


def test_deterministic_build(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    build(a)
    build(b)
    da, db = dir_digest(str(a)), dir_digest(str(b))
    assert da == db and len(da) > 20


# ------------------------------------------------------------------ query type


@pytest.mark.parametrize("q, t", [
    ("Write a Python function that removes all whitespace from a string.", "task"),
    ("What does this line of Python do?  squares = {n: n * n for n in range(5)}", "task"),
    ("Explain what an API is in simple terms.", "task"),
    ("How do I reverse a linked list?", "task"),
    ("What is 20% of 150?", "math"),
    ("Convert 25 degrees Celsius to degrees Fahrenheit.", "math"),
    ("Write a haiku about autumn rain.", "task"),
    ("Can you write a poem about the sea?", "task"),
    ("Tell me a story about a dragon.", "task"),
    ("Describe your ideal weekend.", "other"),
    ("What is your name?", "other"),
    ("I drove 150 km in 2 hours. What was my average speed?", "math"),
    ("Who was Alan Turing?", "question"),
    ("What is photosynthesis?", "question"),
    ("Can you tell me about the Eiffel Tower?", "question"),
    ("Tell me about the history of Rome.", "question"),
    ("What is the function of the liver?", "question"),
    ("Photosynthesis", "lookup"),
    ("What genre is Thinking of You?", "question"),                  # a title, not the asker
    ("Who is the father of David I of Scotland?", "question"),       # a regnal numeral
    ("Which London theatre used to boast 'We Never Close'?", "question"),
    ("With which game would you use a squidger?", "question"),       # generic "you"
    ("who sings you're welcome in moana credits", "question"),
    ("nearest metro station to gandhi nagar in delhi", "lookup"),
    ("Should I quit my job?", "other"),
    ("What do you think about cats?", "other"),
    ("What is 12 squared minus 5 cubed?", "math"),
    ("How many milliliters are in 2.4 liters?", "math"),
])
def test_query_types_for_non_factual_prompts(q, t):
    assert query_type(q) == t
