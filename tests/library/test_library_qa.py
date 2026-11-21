"""Question-answer layer (bes.library.qa): cleaning, evaluation-split exclusion, symmetric
matching against an independent brute force, read-set log, storage and accounting."""
from __future__ import annotations

import math
import os
import random
import re
import unicodedata

import numpy as np
import pytest

from bes.library import qa as Q
from bes.library.pack import PackCfg
from bes.library.terms import STOPWORDS, bucket, stem

# ------------------------------------------------------------------ independent helpers


def ref_tokens(s: str) -> list[str]:
    """Independent re-implementation of the normalisation (NFKD, no marks, lower, words)."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.category(c).startswith("M")).lower()
    out, cur = [], []
    for c in s:
        if unicodedata.category(c)[0] in "LN":
            cur.append(c)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


def ref_terms(s: str) -> list[str]:
    return [stem(t) for t in ref_tokens(s) if t not in STOPWORDS]


def ref_sym(query: str, questions: list[str], n_bits: int) -> np.ndarray:
    """Brute force: idf-weighted symmetric overlap of distinct term buckets."""
    sets = [{bucket(t, n_bits) for t in ref_terms(q)} for q in questions]
    n = len(questions)
    df = {}
    for s in sets:
        for b in s:
            df[b] = df.get(b, 0) + 1

    def idf(b):
        d = df.get(b, 0)
        return math.log(1.0 + (n - d + 0.5) / (d + 0.5))

    qs = {bucket(t, n_bits) for t in ref_terms(query)}
    wq = sum(idf(b) for b in qs)
    out = np.zeros(n)
    for i, s in enumerate(sets):
        m = sum(idf(b) for b in qs & s)
        we = sum(idf(b) for b in s)
        out[i] = min(m / wq, m / we) if wq > 0 and we > 0 else 0.0
    return out


WORDS = ("hamlet wrote shakespeare film 1996 screenplay paris capital france river longest nile "
         "egypt who played batman movie 1989 sang song beatles yesterday king england first "
         "president united states moon landing year war ended treaty").split()
WH = ("who", "what", "when", "where", "which", "how many")


def synth_pairs(seed: int, n: int = 300) -> list[Q.QAPair]:
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < n:
        q = f"{rng.choice(WH)} " + " ".join(rng.sample(WORDS, rng.randint(1, 6)))
        q = Q.clean_question(q)
        k = Q.question_key(q)
        if k in seen:
            continue
        seen.add(k)
        out.append(Q.QAPair(q, rng.choice(WORDS).title(), "synth", len(out)))
    return out


# ------------------------------------------------------------------ cleaning


def test_clean_question_and_answer_forms():
    assert Q.clean_question('"Who created the line, ""Happiness is a warm puppy?"') == \
        'Who created the line, "Happiness is a warm puppy?'
    assert Q.clean_question("  who got the first nobel prize in physics  ") == \
        "who got the first nobel prize in physics?"
    assert Q.clean_question("What is it ? ?") == "What is it?"
    assert Q.clean_question(" ?? ") == ""
    assert Q.clean_answer("November\xa08,\xa01972 ;") == "November 8, 1972"


@pytest.mark.parametrize("q, rules", [
    ("who is the current leader of south africa", {"current"}),
    ("when was the last time montreal canadiens won the stanley cup", {"relative_time", "last_time"}),
    ("who won the national basketball championship last year", {"relative_time"}),
    ("what is the latest season of new girl", {"latest"}),
    ("when is the next world cup", {"relative_time", "next"}),
    ("when does the next episode of lost air", {"relative_time", "next"}),
    ("how old is tom cruise", {"records_counts"}),
    ("how many super bowls have the patriots won", {"records_counts"}),
    ("who has the most grand slam titles", {"records_counts"}),
    ("who is the chief operating officer of facebook", {"present_role"}),
    ("who is taylor swift dating", {"personal_status"}),
    ("when will the olympics be held in japan", {"future"}),
    ("what is the population of canada", {"population"}),
    ("who won the world series", {"won_undated"}),
])
def test_time_sensitive_questions(q, rules):
    assert Q.time_sensitive(q) in rules


@pytest.mark.parametrize("q", [
    "who wrote hamlet", "By what name is the city of Madras now known?",
    "what is alternating current", "how many tusks does a walrus have",
    "who won the 1966 world cup", "who is the first governor general of trinidad and tobago",
    "How old was George Gershwin when he died?", "who won the battle of hastings",
    "what is next to the eiffel tower",
])
def test_stable_questions_are_kept(q):
    assert Q.time_sensitive(q) is None


def test_evaluation_questions_are_never_stored():
    ev = ["Who wrote Hamlet?", "WHAT is the capital  of France", "Ｃａｆé owner name"]
    keys = Q.eval_keys(ev)
    rows = [("a", 0, "who wrote hamlet", ["Shakespeare"]),            # case / punctuation
            ("a", 1, "What is the capital of France?!", ["Paris"]),     # spacing / punctuation
            ("a", 2, "cafe owner name", ["Bob"]),                       # accents / width forms
            ("a", 3, "who wrote macbeth", ["Shakespeare"]),
            ("a", 4, "what is the capital of spain", ["Madrid"])]
    pairs, stats = Q.clean_pairs(rows, keys)
    assert [p.row for p in pairs] == [3, 4]
    assert stats["a"]["eval_overlap"] == 3
    # independent check: no stored question normalises to an evaluation question
    ref_keys = {" ".join(ref_tokens(q)) for q in ev}
    assert all(" ".join(ref_tokens(p.question)) not in ref_keys for p in pairs)
    Q.assert_no_eval_overlap(pairs, keys)
    with pytest.raises(ValueError):
        Q.assert_no_eval_overlap([Q.QAPair("Who wrote HAMLET?", "x", "a", 9)], keys)


def test_exclusion_matches_brute_force_on_random_strings():
    rng = random.Random(3)
    words = ["alpha", "Beta", "gamma", "délta", "EPSILON", "zeta", "eta", "theta"]
    seps = [" ", "  ", " , ", "-", " ? ", "! ", "'"]

    def sentence(ws):
        return "".join(w + rng.choice(seps) for w in ws).strip()

    ev = [sentence(rng.sample(words, rng.randint(1, 4))) for _ in range(150)]
    cand = [sentence(ref_tokens(q)) for q in ev[:80]]                       # re-spelled copies
    cand += [sentence(rng.sample(words, rng.randint(1, 4))) for _ in range(300)]
    keys = Q.eval_keys(ev)
    ref = {" ".join(ref_tokens(q)) for q in ev}
    rows = [("s", i, q, ["ans"]) for i, q in enumerate(cand)]
    pairs, _ = Q.clean_pairs(rows, keys, Q.QACleanCfg(drop_time_sensitive=False))
    kept = {p.row for p in pairs}
    seen = set()
    for i, q in enumerate(cand):
        k = " ".join(ref_tokens(q))
        expect = k not in ref and k not in seen
        seen.add(k)
        assert (i in kept) == expect, (q, k)


def test_duplicates_and_ambiguous_questions():
    rows = [("a", 0, "who wrote hamlet", ["Shakespeare"]),
            ("a", 1, "who wrote hamlet", ["Shakespeare"]),            # exact duplicate
            ("b", 0, "Who wrote Hamlet?", ["William Shakespeare"]),   # same key, other answer
            ("a", 2, "who sang hey jude", ["The Beatles"]),
            ("b", 1, "Who sang Hey Jude", ["Beatles"]),               # same key, same answer key
            ("a", 3, "what is", ["x"]),                               # no content term
            ("a", 4, "who painted it", [""]),                         # empty answer
            ("a", 5, "who painted the mona lisa", ["one two three four five six seven eight nine ten "
                                                  "eleven twelve thirteen"])]
    pairs, st = Q.clean_pairs(rows, frozenset())
    assert [(p.source, p.row) for p in pairs] == [("a", 2)]
    assert st["a"]["duplicate_exact"] == 1 and st["a"]["ambiguous"] == 1 and st["b"]["ambiguous"] == 1
    assert st["b"]["duplicate_norm"] == 1 and st["a"]["no_terms"] == 1
    assert st["a"]["empty_answer"] == 1 and st["a"]["long_answer"] == 1
    assert sum(v.get("kept", 0) for v in st.values()) == len(pairs)


def test_qtype():
    assert Q.QTYPES[Q.qtype("Who wrote Hamlet?")] == "who"
    assert Q.QTYPES[Q.qtype("In which year did the war end?")] == "when"
    assert Q.QTYPES[Q.qtype("how many legs does a spider have")] == "howmany"
    assert Q.QTYPES[Q.qtype("The Battle of X was fought in which country?")] == "what"
    assert Q.QTYPES[Q.qtype("name the longest river")] == "other"


# ------------------------------------------------------------------ matching


@pytest.mark.parametrize("n_bits", [17, 5])   # 5 bits = 32 buckets forces collisions
@pytest.mark.parametrize("seed", [0, 1])
def test_symmetric_overlap_equals_brute_force(seed, n_bits):
    pairs = synth_pairs(seed)
    st = Q.QAStore.build(pairs, Q.QACfg(n_bits=n_bits, group_bits=2, read_postings=10**9, min_cov_q=0.0),
                         PackCfg(block_raw_bytes=2048))
    questions = [st.entry(e)[0] for e in range(st.n_entries)]
    rng = random.Random(seed + 10)
    queries = [p.question for p in rng.sample(pairs, 20)] + ["who wrote hamlet", "paris", "zzzz qqq"]
    for q in queries:
        ref = ref_sym(q, questions, n_bits)
        m = st.search(q)
        got = np.zeros(len(questions))
        got[m.entries] = m.sym
        # mass is quantised to 1/MASS_UNIT: relative error below 1/(2 * MASS_UNIT * min mass)
        np.testing.assert_allclose(got, ref, atol=5e-3)
        assert np.all(got[ref >= 1 - 1e-12] >= 1 - 1e-9)    # an identical bucket set scores 1
        assert set(m.entries.tolist()) == {i for i in range(len(questions)) if ref[i] > 0}
        # ranking: by sym (desc), ties by type match then id
        assert all(m.sym[i] >= m.sym[i + 1] - 1e-9 for i in range(m.size - 1))


def test_strict_matching_examples():
    qs = ["who wrote hamlet?", "who wrote the screenplay of the 1996 film hamlet?",
          "when did hamlet premiere?", "who played batman in the 1989 movie?"]
    pairs = [Q.QAPair(q, a, "s", i) for i, (q, a) in enumerate(zip(qs, ["Shakespeare", "Kenneth Branagh",
                                                                       "1609", "Michael Keaton"]))]
    st = Q.QAStore.build(pairs, Q.QACfg(min_cov_q=0.0))
    m = st.search("Who wrote Hamlet")
    best_q, best_a = st.entry(int(m.entries[0]))
    assert best_a == "Shakespeare" and m.sym[0] == pytest.approx(1.0, abs=1e-3) and m.type_match[0]
    long_e = next(e for e in range(st.n_entries) if st.entry(e)[1] == "Kenneth Branagh")
    i = list(m.entries).index(long_e)
    assert m.cov_q[i] == pytest.approx(1.0, abs=1e-3) and m.sym[i] < 0.6   # both directions matter


def test_exclusion_and_prefilter():
    pairs = synth_pairs(4, 120)
    st = Q.QAStore.build(pairs, Q.QACfg(min_cov_q=0.5))
    q = st.entry(7)[0]
    m = st.search(q)
    assert 7 in m.entries.tolist() and np.all(m.cov_q >= 0.5 - 1e-9)
    m2 = st.search(q, exclude=[7])
    assert 7 not in m2.entries.tolist()
    assert set(m2.entries.tolist()) == set(m.entries.tolist()) - {7}


def test_read_budget_skips_the_commonest_lists():
    pairs = [Q.QAPair(f"who wrote book number {i}?", f"A{i}", "s", i) for i in range(400)]
    pairs.append(Q.QAPair("who wrote hamlet?", "Shakespeare", "s", 400))
    st = Q.QAStore.build(pairs, Q.QACfg(read_postings=50, min_cov_q=0.0))
    log = Q.QARead()
    m = st.search("who wrote hamlet", log=log)
    # "wrote" (401 entries) exceeds the budget: only "hamlet" is read, Wq still counts both
    assert m.size == 1 and st.entry(int(m.entries[0]))[1] == "Shakespeare"
    assert m.cov_q[0] < 1.0
    assert log.postings == 2 * 1


def test_read_log_counts_distinct_numbers():
    pairs = synth_pairs(5, 200)
    cfg = Q.QACfg(n_bits=8, group_bits=3, min_cov_q=0.0)
    st = Q.QAStore.build(pairs, cfg)
    q = "who wrote hamlet film 1996 paris"
    log = Q.QARead()
    m = st.search(q, log=log)
    qb = list(dict.fromkeys(bucket(t, 8) for t in ref_terms(q)))
    used = [b for b in qb if int(st.counts[b]) > 0]
    count_entries = set(qb)
    for b in used:
        count_entries |= set(range((b >> 3) << 3, b + 1))
    assert log.counts == len(count_entries)
    assert log.coarse == 2 * len({b >> 3 for b in used})
    assert log.postings == 2 * sum(int(st.counts[b]) for b in used)
    assert log.per_entry == m.size + math.ceil(m.size / 2)
    assert log.total == log.counts + log.coarse + log.postings + log.per_entry


# ------------------------------------------------------------------ storage


def test_store_round_trip_and_accounting(tmp_path):
    pairs = synth_pairs(6, 250)
    st = Q.QAStore.build(pairs, Q.QACfg(), PackCfg(block_raw_bytes=4096))
    st.save(str(tmp_path))
    re_ = Q.QAStore.load(str(tmp_path))
    by_q = {p.question: p.answer for p in pairs}
    assert re_.n_entries == len(pairs)
    for e in range(re_.n_entries):
        q, a = re_.entry(e)
        assert by_q[q] == a
    for q in ("who wrote hamlet", "paris capital"):
        a, b = st.search(q), re_.search(q)
        assert np.array_equal(a.entries, b.entries) and np.allclose(a.sym, b.sym)
    acc = re_.accounting()
    files = sum((np.load(os.path.join(tmp_path, f)).nbytes + 1) // 2 for f in os.listdir(tmp_path)
                if f.endswith(".npy"))
    assert acc["qa_numbers"] == files + Q.QAStore.HEADER_NUMBERS
    assert acc == st.accounting()


def test_entry_order_by_answer_and_one_block_per_entry(monkeypatch):
    pairs = synth_pairs(7, 300)
    st = Q.QAStore.build(pairs, Q.QACfg(order="answer"), PackCfg(block_raw_bytes=1024))
    answers = [st.entry(e)[1] for e in range(st.n_entries)]
    keys = [Q.answer_key(a) for a in answers]
    assert keys == sorted(keys)
    calls = []
    real = st.blocks.read_block
    monkeypatch.setattr(st.blocks, "read_block", lambda b: (calls.append(b), real(b))[1])
    for e in random.Random(0).sample(range(st.n_entries), 30):
        calls.clear()
        st.entry(e)
        assert calls == [st.blocks.locate(e)[0]]


def test_render():
    assert Q.render_qa("Who wrote Hamlet?", "Shakespeare") == "Q: Who wrote Hamlet?\nA: Shakespeare"
    raw = Q.entry_bytes("Who wrote Hamlet?", "Shakespeare")
    assert Q.parse_entry(raw) == ("Who wrote Hamlet?", "Shakespeare")
    assert re.fullmatch(r"Q: .+\nA: .+", Q.render_qa(*Q.parse_entry(raw)))
