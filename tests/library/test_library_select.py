"""Cost-benefit selection (bes.library.select)."""
from __future__ import annotations

import random

import numpy as np
import pytest

from bes.library.pack import PackCfg
from bes.library.select import (SelectCfg, article_order, benefits, duplicate_mask, estimate_costs,
                                greedy_mask, importance, mention_counts, passage_order,
                                passage_prior, select_for_budget, title_key)
from bes.library.sources import Pool


def make_pool(seed: int, n_articles: int = 120) -> Pool:
    rng = random.Random(seed)
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi".split()
    titles, cats, aw, pa, pp, pw, pl, bodies = [], [], [], [], [], [], [], []
    for a in range(n_articles):
        titles.append(f"Article {a}")
        cats.append(rng.choice(["Cats A", "Cats B", ""]))
        n_p = rng.randint(1, 5)
        tot = 0
        for k in range(n_p):
            n = rng.randint(5, 80)
            body = " ".join(rng.choice(words) for _ in range(n)) + "."
            if rng.random() < 0.03 and bodies:
                body = bodies[-1]  # an exact duplicate
            bodies.append(body)
            pa.append(a)
            pp.append(k)
            pw.append(len(body.split()))
            pl.append(0.0)
            tot += len(body.split())
        aw.append(tot)
    blob = "".join(bodies).encode()
    off = np.concatenate([[0], np.cumsum([len(b.encode()) for b in bodies])]).astype(np.int64)
    return Pool(titles=titles, category=cats, article_words=np.asarray(aw, np.int32),
                article_src=np.arange(n_articles, dtype=np.int32), p_article=np.asarray(pa, np.int32),
                p_pos=np.asarray(pp, np.int16), p_words=np.asarray(pw, np.int32),
                p_list=np.asarray(pl, np.float32), body_blob=blob, body_off=off, drops={}, meta={})


def test_greedy_mask_is_ratio_prefix_with_fill():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(1, 40))
        ratio = rng.uniform(0, 5, n)
        cost = rng.uniform(1, 10, n)
        elig = rng.uniform(size=n) < 0.9
        target = float(rng.uniform(0, cost.sum()))
        m = greedy_mask(ratio, cost, target, elig)
        # brute force: walk by ratio desc (ties by index), take what fits
        order = sorted(np.flatnonzero(elig), key=lambda i: (-ratio[i], i))
        used, ref = 0.0, np.zeros(n, bool)
        prefix = True
        for i in order:
            if prefix and used + cost[i] <= target:
                ref[i] = True
                used += cost[i]
            else:
                prefix = False
                if used + cost[i] <= target:
                    ref[i] = True
                    used += cost[i]
        assert np.array_equal(m, ref)
        assert cost[m].sum() <= target + 1e-9 and not m[~elig].any()


def test_mention_counts_brute_force():
    titles = ["Paris", "Alan Turing", "Mint (candy)", "Mint (money)", "Rome"]
    texts = [(0, "Paris is in France. Alan Turing visited Paris."),
             (1, "Alan Turing was born in London. He liked Mint and Rome."),
             (-1, "Rome and Paris and Alan Turing."),
             (4, "rome is lower case here; Paris too.")]
    got = mention_counts(titles, texts)
    # Paris: mentioned by texts 1? no; by -1 and 4 -> 2 (self-mention in 0 excluded)
    assert got.tolist() == [2.0, 2.0, 0.5, 0.5, 2.0]
    assert title_key("Mint (candy)") == "Mint"


def test_duplicates_and_benefits():
    pool = make_pool(1)
    dup = duplicate_mask(pool)
    seen = set()
    for i in range(pool.n_passages):
        norm = " ".join(pool.body(i).lower().split())
        assert dup[i] == (norm in seen)
        seen.add(norm)
    imp = importance(pool, np.zeros(pool.n_articles))
    ben = benefits(pool, imp, dup)
    prior = passage_prior(pool, imp)
    assert np.all(ben[dup] == 0) and np.all(prior > 0)
    np.testing.assert_allclose(ben[~dup], (prior * pool.p_words)[~dup])
    # lead passages weigh more than later ones of the same article (same words)
    cfg = SelectCfg()
    assert (1 + 0) ** -cfg.pos_decay > (1 + 1) ** -cfg.pos_decay


def test_pack_order_groups_articles_and_sorts_by_category():
    pool = make_pool(2)
    aord = article_order(pool)
    keys = [(pool.category[a] or "￿", pool.titles[a]) for a in aord]
    assert keys == sorted(keys)
    order = passage_order(pool, aord)
    arts = pool.p_article[order]
    # contiguous articles, positions increasing inside an article
    change = np.flatnonzero(np.diff(arts) != 0) + 1
    runs = np.split(arts, change)
    assert len(runs) == pool.n_articles
    for seg in np.split(order, change):
        assert list(pool.p_pos[seg]) == sorted(pool.p_pos[seg])


@pytest.mark.parametrize("budget_frac", [0.3, 0.6, 2.0])
def test_select_for_budget_fits_exactly(budget_frac):
    pool = make_pool(3, n_articles=300)
    order = passage_order(pool, article_order(pool))
    cfg = PackCfg(block_raw_bytes=4096, threads=1)
    cost, packed_all = estimate_costs(pool, order, cfg)
    total = packed_all.accounting()["library_numbers"]
    dup = duplicate_mask(pool)
    ben = benefits(pool, importance(pool, np.zeros(pool.n_articles)), dup)
    budget = int(total * budget_frac)
    sel = select_for_budget(pool, budget, ben, cost, order, cfg, SelectCfg(tol=0.02, max_iter=10))
    acc = sel.packed.accounting()
    assert acc["library_numbers"] <= budget
    if budget_frac >= 1.0:
        assert sel.mask.sum() == (~dup).sum()
    else:
        assert acc["library_numbers"] >= budget * 0.9
    # the packed library holds exactly the selected passages, in pack order
    assert sorted(sel.pool_idx.tolist()) == np.flatnonzero(sel.mask).tolist()
    for k, i in enumerate(sel.pool_idx[:50]):
        assert sel.packed.passage_body(k) == pool.body(int(i))
