"""Index scoring against brute-force BM25 (bes.library.index)."""
from __future__ import annotations

import math
import random

import numpy as np
import pytest

from bes.library.index import (ExactBM25, IndexCfg, IndexRead, TruncatedIndex, bucket_map,
                               rekey_postings, term_postings)
from bes.library.terms import bucket, terms

WORDS = ("river city king queen war music art paris london rome france spain cat dog moon sun "
         "star planet ocean mountain music piano violin novel poem actor movie film 1990 2001").split()


def make_corpus(seed: int, n_titles: int = 40):
    rng = random.Random(seed)
    docs = []
    for t in range(n_titles):
        title = " ".join(rng.sample(WORDS, rng.randint(1, 2))) + f" {t}"
        for _ in range(rng.randint(1, 4)):
            body = " ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 40)))
            docs.append((title, body))
    return docs


def brute_force(docs, query, k1, b, tw, ltw, key):
    """Independent BM25 over keys ``key(term)``: returns (scores, per-doc tf dicts, df)."""
    tfs = []
    for i, (title, body) in enumerate(docs):
        lead = i == 0 or docs[i - 1][0] != title
        w = 1.0 + (ltw if lead else tw)
        tf: dict = {}
        for t in terms(title):
            tf[key(t)] = tf.get(key(t), 0.0) + w
        for t in terms(body):
            tf[key(t)] = tf.get(key(t), 0.0) + 1.0
        tfs.append(tf)
    n = len(docs)
    dl = [sum(tf.values()) for tf in tfs]
    avgdl = sum(dl) / n
    df: dict = {}
    for tf in tfs:
        for k in tf:
            df[k] = df.get(k, 0) + 1
    qk = list(dict.fromkeys(key(t) for t in terms(query)))
    scores = []
    for i in range(n):
        s = 0.0
        for k in qk:
            if k in tfs[i]:
                f = tfs[i][k]
                idf = math.log(1.0 + (n - df[k] + 0.5) / (df[k] + 0.5))
                s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl[i] / avgdl))
        scores.append(s)
    return scores, tfs, df, dl, avgdl


QUERIES = ["who was the king of france", "music and piano", "the moon and the sun star",
           "paris", "war 1990 film", "ocean mountain river city", "nothing matches xyzzy"]


@pytest.mark.parametrize("n_bits", [18, 4])  # 4 bits = 16 buckets forces hash collisions
@pytest.mark.parametrize("seed", [0, 1])
def test_unpruned_index_equals_brute_force(seed, n_bits):
    docs = make_corpus(seed)
    cfg = IndexCfg(n_bits=n_bits, group_bits=2, budget_numbers=10**8, max_list=60000)
    idx = TruncatedIndex.build(docs, cfg)
    assert idx.accounting()["index_postings"] == sum(
        len(set(bucket(t, n_bits) for t in terms(a) + terms(b2))) for a, b2 in docs)
    for q in QUERIES:
        bf, *_ = brute_force(docs, q, cfg.k1, cfg.b, cfg.title_weight, cfg.lead_title_weight,
                             lambda t: bucket(t, n_bits))
        pids, sc = idx.score(q)
        got = np.zeros(len(docs))
        got[pids] = sc * idx.scale
        nq = len(set(bucket(t, n_bits) for t in terms(q)))
        tol = nq * idx.scale + 1e-9  # rounding to 1..255 with one global scale
        assert np.all(np.abs(got - np.array(bf)) <= tol)
        assert set(pids.tolist()) == {i for i, s in enumerate(bf) if s > 0}
        # ranking agrees wherever the brute-force gap exceeds the quantisation error
        order = {p: r for r, p in enumerate(pids.tolist())}
        for i in range(len(docs)):
            for j in range(len(docs)):
                if bf[i] > bf[j] + 2 * tol and bf[i] > 0:
                    assert j not in order or order[i] < order[j]


def test_exact_bm25_equals_brute_force():
    docs = make_corpus(3)
    ex = ExactBM25.build(docs, 0.6, 0.3, 1.0, 6.0)
    for q in QUERIES:
        bf, *_ = brute_force(docs, q, 0.6, 0.3, 1.0, 6.0, lambda t: t)
        np.testing.assert_allclose(ex.scores(q), np.array(bf), rtol=1e-5, atol=1e-5)
        pids, sc = ex.topk(q, 5)
        ref = sorted(range(len(docs)), key=lambda i: (-bf[i], i))
        ref = [i for i in ref if bf[i] > 0][:5]
        # same set up to float ties
        assert len(pids) == len(ref)
        np.testing.assert_allclose(sc, [bf[i] for i in ref], rtol=1e-5, atol=1e-5)


def test_rekeyed_term_postings_equal_direct_build():
    docs = make_corpus(4)
    cfg = IndexCfg(budget_numbers=600_000)
    p, tl = term_postings(docs, cfg.title_weight, cfg.lead_title_weight)
    a = TruncatedIndex.from_postings(rekey_postings(p, bucket_map(tl, cfg.n_bits)), cfg)
    b = TruncatedIndex.build(docs, cfg)
    for name in TruncatedIndex.ARRAYS:
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert a.scale == b.scale and a.avgdl == pytest.approx(b.avgdl)


@pytest.mark.parametrize("seed", [0, 5])
def test_pruning_keeps_highest_utility_postings(seed):
    docs = make_corpus(seed, n_titles=60)
    rng = np.random.default_rng(seed)
    prior = rng.uniform(0.5, 3.0, size=len(docs))
    cfg = IndexCfg(n_bits=6, group_bits=2, max_list=25)
    overhead = cfg.overhead_numbers()
    cfg = IndexCfg(n_bits=6, group_bits=2, max_list=25, budget_numbers=overhead + 2 * 300)
    idx = TruncatedIndex.build(docs, cfg, prior)
    key = lambda t: bucket(t, 6)  # noqa: E731
    _, tfs, df, dl, avgdl = brute_force(docs, "", cfg.k1, cfg.b, cfg.title_weight,
                                        cfg.lead_title_weight, key)
    n = len(docs)
    posts = []
    for d, tf in enumerate(tfs):
        title_keys = {key(t) for t in terms(docs[d][0])}
        imps = {}
        for k, f in tf.items():
            idf = math.log(1.0 + (n - df[k] + 0.5) / (df[k] + 0.5))
            imps[k] = idf * f * (cfg.k1 + 1) / (f + cfg.k1 * (1 - cfg.b + cfg.b * dl[d] / avgdl))
        mx = max(imps.values())
        for k, v in imps.items():
            u = (v / mx) ** cfg.rel_exponent * prior[d] ** cfg.prior_exponent
            u *= cfg.title_bonus if k in title_keys else 1.0
            posts.append((u, d, k))
    by_bucket: dict = {}
    for u, d, k in posts:
        by_bucket.setdefault(k, []).append((u, d, k))
    cand = []
    for k, lst in by_bucket.items():
        lst.sort(key=lambda x: (-x[0], x[1]))
        cand.extend(lst[: cfg.max_list])
    cand.sort(key=lambda x: (-x[0], x[1], x[2]))
    expect = {(k, d) for _, d, k in cand[:300]}
    got = set()
    for b in range(1 << 6):
        pids, q = idx.read_bucket(b)
        assert len(pids) <= cfg.max_list
        assert list(q) == sorted(q, reverse=True)  # impact order inside a bucket
        got |= {(b, int(p)) for p in pids}
    assert got == expect
    assert idx.accounting()["index_numbers"] <= cfg.budget_numbers


def test_offsets_counts_and_read_log():
    docs = make_corpus(6)
    cfg = IndexCfg(n_bits=8, group_bits=3, budget_numbers=10**7)
    idx = TruncatedIndex.build(docs, cfg)
    counts = np.asarray(idx.counts, dtype=np.int64)
    start = np.concatenate([[0], np.cumsum(counts)])
    for b in range(1 << 8):
        assert idx.bucket_range(b) == (int(start[b]), int(start[b + 1]))
    q = "music and piano river city 1990"
    log = IndexRead()
    idx.score(q, log)
    buckets = list(dict.fromkeys(bucket(t, 8) for t in terms(q)))
    used = [b for b in buckets if idx.df[b] > 0]
    assert log.df_numbers == len(buckets)
    groups = {b >> 3 for b in used}
    count_entries = set()
    for b in used:
        count_entries |= set(range((b >> 3) << 3, b + 1))
    assert log.offset_numbers == 2 * len(groups) + len(count_entries)
    assert log.posting_numbers == sum(2 * int(counts[b]) for b in used)
    assert log.total == log.df_numbers + log.offset_numbers + log.posting_numbers


def test_query_reads_at_most_max_query_terms_lowest_df():
    docs = make_corpus(7)
    cfg = IndexCfg(budget_numbers=10**7, max_query_terms=3)
    idx = TruncatedIndex.build(docs, cfg)
    q = " ".join(WORDS)
    bs = idx.query_buckets(q)
    assert len(bs) == 3
    dfs = sorted((int(idx.df[bucket(t, cfg.n_bits)]), bucket(t, cfg.n_bits)) for t in set(terms(q))
                 if idx.df[bucket(t, cfg.n_bits)] > 0)
    assert bs == [b for _, b in dfs[:3]]


def test_save_load_round_trip(tmp_path):
    docs = make_corpus(8)
    idx = TruncatedIndex.build(docs, IndexCfg(budget_numbers=700_000))
    idx.save(str(tmp_path))
    re = TruncatedIndex.load(str(tmp_path))
    for name in TruncatedIndex.ARRAYS:
        assert np.array_equal(np.asarray(getattr(re, name)), getattr(idx, name))
    for q in QUERIES:
        a, b = idx.score(q), re.score(q)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    acc = re.accounting()
    assert acc["index_numbers"] == sum((np.asarray(getattr(re, n)).nbytes + 1) // 2
                                       for n in TruncatedIndex.ARRAYS) + TruncatedIndex.HEADER_NUMBERS


def test_budget_accounting_is_exact():
    docs = make_corpus(9)
    for budget in (540_000, 600_000, 10**7):
        cfg = IndexCfg(budget_numbers=budget)
        idx = TruncatedIndex.build(docs, cfg)
        acc = idx.accounting()
        n = 1 << cfg.n_bits
        assert acc["index_counts_numbers"] == n and acc["index_df_numbers"] == n
        assert acc["index_coarse_numbers"] == 2 * ((n >> cfg.group_bits) + 1)
        assert acc["index_postings_numbers"] == 2 * acc["index_postings"]
        assert acc["index_numbers"] <= max(budget, cfg.overhead_numbers())
