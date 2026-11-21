"""Sampling transforms of the particle decoder (bes.decode.sampling).

Every transform is compared with a reference written here in plain Python / numpy loops
(sorting, cumulative sums, n-gram search by brute force); random streams are checked for
determinism, batching invariance and uniformity.
"""
from __future__ import annotations

import math
import random

import numpy as np
import pytest

from bes.decode import sampling as S


def _rand_logp(rng, k, v, scale=2.0, ties=False):
    z = rng.standard_normal((k, v)) * scale
    if ties:
        z = np.round(z)
    z = z - np.log(np.exp(z).sum(1, keepdims=True))
    return z.astype(np.float32)


# ------------------------------------------------------------------------- random streams
def test_counter_uniform_is_pure_batch_invariant_and_uniform():
    idx = np.arange(64)
    a = S.counter_uniform(7, S.STREAM_TOKEN, idx, 5)
    b = np.array([S.counter_uniform(7, S.STREAM_TOKEN, int(i), 5)[()] for i in idx])
    np.testing.assert_array_equal(a, b)                       # batching never changes a draw
    np.testing.assert_array_equal(a, S.counter_uniform(7, S.STREAM_TOKEN, idx, 5))
    assert np.all((a >= 0) & (a < 1))
    assert not np.array_equal(a, S.counter_uniform(8, S.STREAM_TOKEN, idx, 5))
    assert not np.array_equal(a, S.counter_uniform(7, S.STREAM_RESAMPLE, idx, 5))
    assert not np.array_equal(a, S.counter_uniform(7, S.STREAM_TOKEN, idx, 6))
    # negative and huge seeds are fine
    S.counter_uniform(-3, 1, 0, 0)
    S.counter_uniform(2 ** 70 + 5, 1, 0, 0)
    # uniformity: 20000 draws, Kolmogorov-Smirnov distance against U(0, 1)
    u = np.sort(S.counter_uniform(1, 1, np.arange(200), np.arange(100)[:, None]).ravel())
    n = u.size
    ks = np.max(np.maximum(np.arange(1, n + 1) / n - u, u - np.arange(n) / n))
    assert ks < 1.63 / math.sqrt(n)        # 1% critical value
    # neighbouring counters are not correlated
    c = np.corrcoef(u[:-1], S.counter_uniform(1, 1, 0, np.arange(1, n)))[0, 1]
    assert abs(c) < 0.05


# --------------------------------------------------------------------------- transforms
def test_log_softmax_and_temperature_match_reference():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((3, 11)) * 3
    z[1, 4] = -np.inf
    for tau in (1.0, 0.6, 1.7):
        out = S.tempered(z, tau)
        for r in range(3):
            ref = [x / tau for x in z[r]]
            m = max(ref)
            lse = m + math.log(sum(math.exp(x - m) for x in ref if x > -math.inf))
            np.testing.assert_allclose(out[r], [x - lse for x in ref], atol=1e-12)
    allneg = S.log_softmax(np.full((1, 4), -np.inf))
    assert np.all(allneg == -np.inf)
    with pytest.raises(ValueError):
        S.tempered(z, 0.0)


def _ref_keep(lq, top_k, top_p, min_p, within):
    """Reference truncation of one row."""
    v = len(lq)
    cand = [i for i in range(v) if within[i] and lq[i] > -math.inf]
    if top_k > 0:
        cand = sorted(cand, key=lambda i: (-lq[i], i))[:top_k]
    if top_p < 1.0 and cand:
        m = max(lq[i] for i in cand)
        tot = sum(math.exp(lq[i] - m) for i in cand)
        order = sorted(cand, key=lambda i: (-lq[i], i))
        kept, cum = [], 0.0
        for i in order:
            if cum >= top_p:
                break
            kept.append(i)
            cum += math.exp(lq[i] - m) / tot
        cand = kept
    if min_p > 0 and cand:
        m = max(lq[i] for i in cand)
        cand = [i for i in cand if lq[i] >= m + math.log(min_p)]
    out = np.zeros(v, bool)
    out[cand] = True
    return out


@pytest.mark.parametrize("top_k,top_p,min_p", [(0, 0.9, 0.0), (3, 1.0, 0.0), (4, 0.7, 0.0),
                                               (0, 1.0, 0.2), (5, 0.8, 0.1), (1, 0.5, 0.0),
                                               (0, 1.0, 0.0)])
def test_truncation_matches_reference(top_k, top_p, min_p):
    rng = np.random.default_rng(1)
    for ties in (False, True):
        lq = S.log_softmax(_rand_logp(rng, 6, 13, ties=ties))
        within = rng.random((6, 13)) < 0.8
        within[0] = True
        got = S.truncation_keep(lq, top_k, top_p, min_p, within=within)
        for r in range(6):
            np.testing.assert_array_equal(got[r], _ref_keep(lq[r], top_k, top_p, min_p, within[r]))


def test_step_increment_before_and_after_the_mask():
    rng = np.random.default_rng(2)
    k, v = 5, 17
    logp = _rand_logp(rng, k, v)
    allowed = rng.random((k, v)) < 0.5
    allowed[3] = False                       # nothing allowed: weight 0
    allowed[4, :] = False
    allowed[4, 2] = True
    keep = rng.random((k, v)) < 0.85
    for after in (False, True):
        st = S.SamplingSettings(temperature=0.7, top_p=0.8, top_k=9, truncate_after_mask=after)
        lg, lw = S.step_increment(logp, st, allowed, keep)
        for r in range(k):
            p_tau = np.exp(S.tempered(logp[r:r + 1], 0.7)[0])
            if after:
                alpha = p_tau[allowed[r]].sum()
                within = allowed[r] & keep[r] if (allowed[r] & keep[r]).any() else allowed[r]
                kept = _ref_keep(np.log(p_tau), 9, 0.8, 0.0, within)
                gamma = np.where(kept, p_tau, 0.0)
                gamma = alpha * gamma / gamma.sum() if gamma.sum() > 0 else gamma
            else:
                base = keep[r] if keep[r].any() else np.ones(v, bool)
                kept = _ref_keep(np.log(p_tau), 9, 0.8, 0.0, base)
                p_tr = np.where(kept, p_tau, 0.0)
                p_tr /= p_tr.sum()
                gamma = np.where(allowed[r], p_tr, 0.0)
                alpha = gamma.sum()
            if alpha == 0:
                assert lw[r] == -np.inf and np.all(lg[r] == -np.inf)
                continue
            np.testing.assert_allclose(np.exp(lg[r]), gamma, atol=1e-12)
            np.testing.assert_allclose(lw[r], math.log(alpha), atol=1e-12)   # weight = allowed mass
    # no mask, no truncation: the increment is the tempered model itself, weight 1
    lg, lw = S.step_increment(logp, S.SamplingSettings(temperature=1.3))
    np.testing.assert_allclose(lg, S.tempered(logp, 1.3), atol=1e-12)
    np.testing.assert_allclose(lw, 0.0, atol=1e-12)


def test_draw_is_inverse_cdf_and_never_picks_excluded_tokens():
    rng = np.random.default_rng(3)
    lg = np.log(rng.random((4, 9)))
    lg[:, [1, 5]] = -np.inf
    lg[2] = -np.inf
    lg[2, 7] = 0.0
    u = np.array([0.0, 0.3, 0.999999, 0.5])
    got = S.draw(lg, u)
    for r in range(4):
        p = np.exp(lg[r] - lg[r].max())
        c = np.cumsum(p) / p.sum()
        assert got[r] == int(np.searchsorted(c, u[r], side="right"))
        assert np.isfinite(lg[r, got[r]])
    assert S.draw(np.full((1, 3), -np.inf), np.array([0.5]))[0] == -1
    # frequencies: chi-square over 40000 draws
    with np.errstate(divide="ignore"):
        lg1 = np.log(np.array([[0.5, 0.0, 0.2, 0.3]]))
    uu = S.counter_uniform(5, 1, 0, np.arange(40000))
    toks = S.draw(np.repeat(lg1, 40000, 0), uu)
    counts = np.bincount(toks, minlength=4)
    assert counts[1] == 0
    exp = np.array([0.5, 0.2, 0.3]) * 40000
    chi2 = ((counts[[0, 2, 3]] - exp) ** 2 / exp).sum()
    assert chi2 < 13.8                     # 0.1% critical value, 2 degrees of freedom
    assert list(S.greedy(np.array([[0.1, 0.3, 0.3], [-np.inf] * 3]))) == [1, -1]


def test_systematic_resampling_and_ess_match_reference():
    rng = np.random.default_rng(4)
    for _ in range(50):
        k = int(rng.integers(1, 9))
        lw = np.log(rng.random(k)) * 3
        if k > 2:
            lw[rng.integers(k)] = -np.inf
        u = float(rng.random())
        w = np.exp(lw - lw.max())
        w = w / w.sum()
        ref = []
        for i in range(k):
            pos, cum = (u + i) / k, 0.0
            for j in range(k):
                cum += w[j]
                if pos < cum or j == k - 1:
                    ref.append(j)
                    break
        assert list(S.systematic(lw, u)) == ref
        assert S.ess(lw) == pytest.approx(1.0 / np.sum(w ** 2))
        counts = np.bincount(S.systematic(lw, u), minlength=k)
        assert np.all(np.abs(counts - k * w) < 1 + 1e-9)       # systematic: |count - K w| < 1
    with pytest.raises(ValueError):
        S.systematic(np.full(3, -np.inf), 0.5)
    assert S.ess(np.full(3, -np.inf)) == 0.0


# --------------------------------------------------------------------- degeneracy guards
def _ref_blocked(hist, n, lib, special, word):
    if len(hist) < n - 1:
        return set()
    ctx = tuple(hist[len(hist) - n + 1:]) if n > 1 else ()
    out = set()
    for i in range(len(hist) - n + 1):
        gram = tuple(hist[i:i + n])
        if gram[:-1] != ctx:
            continue
        if any(special[t] for t in gram) or not any(word[t] for t in gram):
            continue
        if gram in lib:
            continue
        out.add(gram[-1])
    return out


def test_ngram_blocker_matches_brute_force():
    rng = random.Random(5)
    v = 9
    special = np.zeros(v, bool)
    special[8] = True
    word = np.array([True, True, False, True, False, True, True, False, False])
    library = [[0, 1, 2, 0, 8, 3, 4], [5, 6, 7, 0, 1]]
    for n in (1, 2, 3):
        lib = set()
        for p in library:
            runs, cur = [], []
            for t in p:
                if special[t]:
                    runs.append(cur)
                    cur = []
                else:
                    cur.append(t)
            runs.append(cur)
            for run in runs:
                for i in range(len(run) - n + 1):
                    lib.add(tuple(run[i:i + n]))
        bl = S.NgramBlocker(n, 2, word=word, special=special, library=library)
        hists = [[], []]
        for _ in range(60):
            for j in range(2):
                assert set(bl.blocked(j)) == _ref_blocked(hists[j], n, lib, special, word)
                t = rng.randrange(v) if rng.random() < 0.6 else rng.choice([0, 1, 2])
                bl.push(j, t)
                hists[j].append(t)
        bl.select([1, 1])
        hists = [list(hists[1]), list(hists[1])]
        bl.push(0, 3)
        hists[0].append(3)
        for j in range(2):
            assert set(bl.blocked(j)) == _ref_blocked(hists[j], n, lib, special, word)


def test_copy_cap_matches_brute_force():
    rng = random.Random(6)
    v = 7
    special = np.zeros(v, bool)
    special[6] = True
    exempt = np.array([False, False, True, False, False, False, True])   # 2: a name or number
    source = [[0, 1, 2, 3, 0, 1, 4, 6, 0, 1, 2, 5, 3]]
    for c in (1, 2, 3):
        cap = S.CopyCap(c, 1, source, exempt=exempt, special=special)
        hist = []
        for _ in range(80):
            ref = set()
            if len(hist) >= c and not any(special[t] for t in hist[-c:]):
                run_tokens = []
                for src in source:
                    cur = []
                    for t in src:
                        if special[t]:
                            run_tokens.append(cur)
                            cur = []
                        else:
                            cur.append(t)
                    run_tokens.append(cur)
                for run in run_tokens:
                    for i in range(len(run) - c):
                        if run[i:i + c] == hist[-c:] and not exempt[run[i + c]]:
                            ref.add(run[i + c])
            assert set(cap.blocked(0)) == ref
            t = rng.choice([0, 1, 2, 3, 4, 5]) if rng.random() < 0.8 else rng.randrange(v)
            cap.push(0, t)
            hist.append(t)


def test_token_classes():
    special = np.array([False] * 6 + [True])
    tb = [b" the", b"Paris", b" 2024", b",", b"\n\n", b"\xc3\xa9", b""]
    word, exempt = S.token_classes(tb, special)
    assert list(word) == [True, True, False, False, False, True, False]
    assert list(exempt) == [False, True, True, False, False, False, True]
