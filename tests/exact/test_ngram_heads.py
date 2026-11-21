"""Tests of bes.exact.suffix and bes.exact.ngram_heads against brute force written here.

The brute force builds every history as a Python tuple (scanning back token by token) and
compares tuples directly; the n-gram statistics are checked at full resolution (raw ``n``,
``n_b``, ``K``), not only through the buckets of the tape.
"""
from __future__ import annotations

import numpy as np
import pytest

from bes import config
from bes.config import ASSISTANT, BOS, CALC, END, LIBRARY, PAD, ROLE_TOKENS, SYSTEM, USER
from bes.exact import ngram_heads, suffix

FIRST_SPECIAL = config.VOCAB_SIZE - config.N_SPECIAL
ROLES = tuple(ROLE_TOKENS)
SPECIALS = [BOS, SYSTEM, USER, ASSISTANT, LIBRARY, END, CALC, PAD]


def random_batch(seed, n_rows=3, n_t=90, alphabet=3, p_special=0.06, p_seg=0.02):
    rng = np.random.default_rng(seed)
    toks = rng.integers(0, alphabet, (n_rows, n_t))
    mask = rng.random((n_rows, n_t)) < p_special
    toks[mask] = rng.choice(SPECIALS, int(mask.sum()))
    seg = np.cumsum(rng.random((n_rows, n_t)) < p_seg, axis=1)
    return toks.astype(np.int32), seg.astype(np.int32)


def periodic_batch(seed, n_rows=2, n_t=120):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_rows):
        motif = rng.integers(0, 9, int(rng.integers(1, 5)))
        row = motif[np.arange(n_t) % len(motif)]
        noise = rng.random(n_t) < 0.04
        row[noise] = rng.integers(0, 9, int(noise.sum()))
        rows.append(row)
    return np.stack(rows).astype(np.int32), np.zeros((n_rows, n_t), np.int32)


BATCHES = [random_batch(s, alphabet=a) for s, a in ((1, 2), (2, 3), (3, 6))] + \
          [random_batch(4, p_special=0.25, p_seg=0.05), periodic_batch(5), periodic_batch(6)]


# ------------------------------------------------------------------------ brute force
class Brute:
    """Flat brute-force view of a batch: segments, turn starts, histories."""

    def __init__(self, toks, seg):
        self.n_rows, self.n_t = toks.shape
        self.x = [int(v) for v in toks.reshape(-1)]
        s = [int(v) for v in seg.reshape(-1)]
        n = len(self.x)
        self.seg_start, self.seg_id, self.ts = [], [], []
        sid = -1
        for g in range(n):
            new = g % self.n_t == 0 or s[g] != s[g - 1] or self.x[g] == BOS
            sid += new
            self.seg_start.append(g if new else self.seg_start[-1])
            self.seg_id.append(sid)
            prev_ts = None if new else self.ts[-1]
            self.ts.append(g if new or self.x[g] in ROLES else prev_ts)

    def history(self, g, cap=None):
        out, p = [], g
        while p >= self.ts[g] and (cap is None or len(out) < cap):
            out.append(self.x[p])
            p -= 1
        return tuple(out)

    def lcs(self, a, b, cap=32):
        if self.seg_id[a] != self.seg_id[b]:
            return 0
        ha, hb = self.history(a, cap), self.history(b, cap)
        n = 0
        while n < min(len(ha), len(hb)) and ha[n] == hb[n]:
            n += 1
        return n

    def context(self, g, L):
        """The L-token context ending at g as a tuple, or None when it would cross a boundary."""
        if g - L + 1 < self.ts[g]:
            return None
        return tuple(self.x[g - L + 1:g + 1])

    def succ(self, g):
        return self.x[g + 1] if (g + 1) % self.n_t else -1


def brute_head(br: Brute, t, match_set, window=16, n_cand=4):
    """Raw statistics of one head: list of (slot, token, n_b, n, K, source)."""
    succ = [br.x[j + 1] for j in match_set]
    picked = []
    for j in sorted(match_set, reverse=True)[:window]:
        b = br.x[j + 1]
        if b >= FIRST_SPECIAL or b in [p[0] for p in picked]:
            continue
        picked.append((b, j))
        if len(picked) == n_cand:
            break
    return [(slot, b, succ.count(b), len(match_set), len(set(succ)), j)
            for slot, (b, j) in enumerate(picked)]


def head_rows(h: ngram_heads.HeadOut, with_level=False):
    rows = zip(h.query.tolist(), h.slot.tolist(), h.token.tolist(), h.count.tolist(),
               h.total.tolist(), h.types.tolist(), h.source.tolist(), h.level.tolist())
    return sorted(r if with_level else r[:-1] for r in rows)


# ---------------------------------------------------------------------- suffix tests
@pytest.mark.parametrize("k", range(len(BATCHES)))
def test_layout(k):
    toks, seg = BATCHES[k]
    lay = suffix.layout(toks, seg)
    br = Brute(toks, seg)
    assert lay.seg_id.tolist() == br.seg_id
    assert lay.ts.tolist() == br.ts
    assert lay.avail.tolist() == [g - br.ts[g] + 1 for g in range(len(br.x))]
    roles = [ROLES.index(br.x[br.ts[g]]) + 1 if br.x[br.ts[g]] in ROLES else 0 for g in range(len(br.x))]
    assert lay.role.tolist() == roles
    assert lay.succ.tolist() == [br.succ(g) for g in range(len(br.x))]


@pytest.mark.parametrize("k", range(len(BATCHES)))
def test_history_ranks_are_lexicographic(k):
    toks, seg = BATCHES[k]
    lay = suffix.layout(toks, seg)
    ranks = suffix.history_ranks(lay)
    br = Brute(toks, seg)
    n = len(br.x)
    for level, r in enumerate(ranks):
        keys = [(br.seg_id[g], br.history(g, 1 << level)) for g in range(n)]
        distinct = sorted(set(keys))
        want = [distinct.index(key) for key in keys]       # dense lexicographic rank
        assert r.tolist() == want, level


@pytest.mark.parametrize("k", range(len(BATCHES)))
def test_sorted_histories_and_neighbour_lcp(k):
    toks, seg = BATCHES[k]
    sh = suffix.sort_histories(suffix.layout(toks, seg))
    br = Brute(toks, seg)
    n = len(br.x)
    want_order = sorted(range(n), key=lambda g: ((br.seg_id[g], br.history(g, 32)), g))
    assert sh.order.tolist() == want_order
    assert sh.adj[0] == -1
    want_adj = [br.lcs(want_order[r - 1], want_order[r]) for r in range(1, n)]
    assert sh.adj[1:].tolist() == want_adj


@pytest.mark.parametrize("k", range(len(BATCHES)))
@pytest.mark.parametrize("min_len", [1, 4])
def test_longest_match(k, min_len):
    toks, seg = BATCHES[k]
    sh = suffix.sort_histories(suffix.layout(toks, seg))
    br = Brute(toks, seg)
    want = []
    for t in range(len(br.x)):
        best = max((br.lcs(t, j) for j in range(br.seg_start[t], t)), default=0)
        want.append(best if best >= min_len else 0)
    assert suffix.longest_match(sh, min_len).tolist() == want


@pytest.mark.parametrize("k", range(len(BATCHES)))
def test_level_blocks_are_context_classes(k):
    toks, seg = BATCHES[k]
    sh = suffix.sort_histories(suffix.layout(toks, seg))
    br = Brute(toks, seg)
    levels = [1, 2, 3, 5, 8, 13, 32]
    got = {lb.level: lb for lb in suffix.iter_level_blocks(sh, levels)}
    for level in levels:
        classes: dict = {}
        for g in range(len(br.x)):
            ctx = br.context(g, level)
            if ctx is not None:
                classes.setdefault((br.seg_id[g], ctx), []).append(g)
        want = sorted(sorted(v) for v in classes.values() if len(v) >= 2)
        if level not in got:
            assert not want
            continue
        lb = got[level]
        blocks: dict = {}
        for p, b in zip(lb.pos.tolist(), lb.block.tolist()):
            blocks.setdefault(b, []).append(p)
        assert sorted(sorted(v) for v in blocks.values()) == want
        assert lb.n_blocks == len(blocks)


# ------------------------------------------------------------------------ head tests
@pytest.mark.parametrize("seed", range(8))
def test_group_candidates_random_groups(seed):
    """Arbitrary groups (not derived from text): positions, successors and query flags random."""
    rng = np.random.default_rng(seed)
    n_flat = 400
    x = rng.integers(0, 5, n_flat)
    x[rng.random(n_flat) < 0.15] = END                 # special successors
    succ = np.append(x[1:], -1).astype(np.int64)
    pos, group = [], []
    for g in range(int(rng.integers(1, 12))):
        members = rng.choice(n_flat, int(rng.integers(1, 40)), replace=False)  # 399: succ -1
        pos += members.tolist()
        group += [g * 7 + 3] * len(members)
    pos, group = np.asarray(pos, np.int64), np.asarray(group, np.int64)
    perm = rng.permutation(pos.size)
    pos, group = pos[perm], group[perm]
    is_q = rng.random(pos.size) < 0.7
    level = rng.integers(1, 33, pos.size)
    window, n_cand = int(rng.choice([1, 3, 16])), int(rng.choice([1, 2, 4]))
    got = head_rows(ngram_heads.group_candidates(pos, group, is_q, succ, level, window, n_cand),
                    with_level=True)
    want = []
    for i in np.flatnonzero(is_q):
        t, g = int(pos[i]), int(group[i])
        earlier = [int(p) for p, gg in zip(pos, group) if gg == g and p < t]
        for slot, b, nb, n, kk, j in brute_head_raw(succ, earlier, window, n_cand):
            want.append((t, slot, b, nb, n, kk, j, int(level[i])))
    assert got == sorted(want)


def brute_head_raw(succ, earlier, window, n_cand):
    s = [int(succ[j]) for j in earlier]
    picked = []
    for j in sorted(earlier, reverse=True)[:window]:
        b = int(succ[j])
        if b < 0 or b >= FIRST_SPECIAL or b in [p[0] for p in picked]:
            continue
        picked.append((b, j))
        if len(picked) == n_cand:
            break
    return [(slot, b, s.count(b), len(earlier), len(set(s)), j) for slot, (b, j) in enumerate(picked)]


@pytest.mark.parametrize("k", range(len(BATCHES)))
def test_order_heads_raw_statistics(k):
    toks, seg = BATCHES[k]
    lay = suffix.layout(toks, seg)
    heads = ngram_heads.order_heads(suffix.sort_histories(lay), lay.succ)
    br = Brute(toks, seg)
    for order in (1, 2, 3):
        want = []
        for t in range(len(br.x)):
            ctx = br.context(t, order)
            if ctx is None:
                continue
            m = [j for j in range(br.seg_start[t], t) if br.context(j, order) == ctx]
            want += [(t, *row) for row in brute_head(br, t, m)]
        assert head_rows(heads[order]) == sorted(want), order
        assert (heads[order].level == order).all()


@pytest.mark.parametrize("k", range(len(BATCHES)))
def test_long_head_raw_statistics(k):
    toks, seg = BATCHES[k]
    lay = suffix.layout(toks, seg)
    sh = suffix.sort_histories(lay)
    lstar = suffix.longest_match(sh, ngram_heads.MIN_LONG)
    head = ngram_heads.long_head(sh, lay.succ, lstar)
    br = Brute(toks, seg)
    want = []
    for t in range(len(br.x)):
        lens = {j: br.lcs(t, j) for j in range(br.seg_start[t], t)}
        best = max(lens.values(), default=0)
        if best < 4:
            continue
        m = [j for j, ml in lens.items() if ml == best]
        want += [(t, *row, best) for row in brute_head(br, t, m)]
    assert head_rows(head, with_level=True) == sorted(want)


def test_empty_inputs():
    lay = suffix.layout(np.zeros((0, 4), np.int32))
    sh = suffix.sort_histories(lay)
    assert sh.order.size == 0 and suffix.longest_match(sh).size == 0
    heads = ngram_heads.order_heads(sh, lay.succ)
    assert all(h.query.size == 0 for h in heads.values())
    empty = ngram_heads.group_candidates(np.zeros(0, np.int64), np.zeros(0, np.int64),
                                         np.zeros(0, bool), np.zeros(0, np.int64), 1)
    assert empty.query.size == 0
    with pytest.raises(ValueError):
        list(suffix.iter_level_blocks(sh, [3, 2]))


def test_large_flat_batches_are_rejected():
    lay = suffix.layout(np.zeros((1, suffix.MAX_FLAT + 1), np.int32))
    with pytest.raises(ValueError):
        suffix.history_ranks(lay)
