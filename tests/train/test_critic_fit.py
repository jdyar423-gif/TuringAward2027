"""Critic fitting (``bes/train/critic_fit.py``): hidden-state collection, the fit, the report.

Independent references in the tests: Spearman's rho from the definition of average ranks
(``rank_i = 1 + #{x_j < x_i} + #{j != i: x_j == x_i} / 2``) and a float64 numpy critic
``relu(h @ W1) @ w2``; hidden states from ``forward_hidden`` on one padded row at a time.
"""
from __future__ import annotations

import dataclasses
import functools
import os

import jax
import numpy as np
import pytest

from bes.config import BOS, END, PAD, USER, CriticCfg, load_config
from bes.data import prefs as P
from bes.exact.features import exact_features
from bes.model import model as M
from bes.train import critic_fit as C

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D_MODEL = 32


@pytest.fixture(scope="module")
def cfg():
    base = load_config(os.path.join(ROOT, "configs", "smoke.py")).model
    return dataclasses.replace(base, d_model=D_MODEL, n_heads=1, head_dim=32, ve_dim=32,
                               head_rank=16, n_loops=2, max_seq=64, clock_planes=3, cand_dim=8,
                               critic_hidden=32, loss_chunk=64)


@pytest.fixture(scope="module")
def params(cfg):
    return jax.jit(functools.partial(M.init_params, cfg))(jax.random.PRNGKey(0))


@pytest.fixture(scope="module")
def tok():
    from bes.tokenizer.tok import load

    return load()


# ------------------------------------------------------------------------ references
def ref_ranks(x):
    x = np.asarray(x, np.float64)
    return np.array([1 + np.sum(x < v) + 0.5 * (np.sum(x == v) - 1) for v in x])


def ref_spearman(x, y):
    rx, ry = ref_ranks(x), ref_ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float(np.sum(rx * ry) / np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))


def ref_critic(params, h):
    w1 = np.asarray(params["critic"]["w1"], np.float64)
    w2 = np.asarray(params["critic"]["w2"], np.float64)
    return np.maximum(np.asarray(h, np.float64) @ w1, 0.0) @ w2


def planted(n_groups=160, per=8, seed=0):
    """Synthetic hidden states with a planted linear signal ``z = h . u``."""
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((n_groups * per, D_MODEL)).astype(np.float32)
    u = rng.standard_normal(D_MODEL)
    u /= np.linalg.norm(u)
    z = h.astype(np.float64) @ u
    groups = np.repeat(np.arange(n_groups), per)
    return h, z, groups


# ------------------------------------------------------------------------- statistics
def test_spearman_matches_the_rank_definition_with_ties():
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.integers(0, 5, size=12).astype(float)
        y = rng.integers(0, 4, size=12).astype(float) + 0.1 * x
        np.testing.assert_allclose(C.average_ranks(x), ref_ranks(x))
        if np.ptp(x) and np.ptp(y):
            assert C.spearman(x, y) == pytest.approx(ref_spearman(x, y), abs=1e-12)
    assert np.isnan(C.spearman([1, 1, 1], [1, 2, 3]))
    assert C.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_within_prompt_spearman_rules():
    scores = np.array([1, 2, 3, 5, 5, 5, 1, 2, 9, 8, 7.0])
    labels = np.array([1, 2, 3, 0, 1, 2, 4, 4, np.nan, 1, 2.0])
    groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3])
    mean, n, per = C.within_prompt_spearman(scores, labels, groups)
    # group 0: rho 1; group 1: constant scores -> 0; group 2: constant labels -> skipped;
    # group 3: the NaN label is unlabelled, two samples left with rho -1
    assert per == {0: pytest.approx(1.0), 1: 0.0, 3: pytest.approx(-1.0)} and n == 3
    assert mean == pytest.approx(0.0)
    mean, n, per = C.within_prompt_spearman(scores, labels, groups, groups == 0)
    assert n == 1 and mean == pytest.approx(1.0)
    assert np.isnan(C.within_prompt_spearman(scores, labels, groups, groups == 2)[0])


def test_win_rates_and_heldout_groups():
    pairs = np.array([[0, 1], [0, 2], [2, 1]])
    np.testing.assert_allclose(C.win_rates(pairs, 4)[:3], [1.0, 0.0, 0.5])
    assert np.isnan(C.win_rates(pairs, 4)[3])
    g = np.repeat(np.arange(400), 2)
    held = C.heldout_groups(g, 0.25, seed=1)
    assert held == C.heldout_groups(g[::-1], 0.25, seed=1)          # order independent
    assert 0.18 < len(held) / 400 < 0.32 and held != C.heldout_groups(g, 0.25, seed=2)
    for frac in (1e-12, 0.9999999):
        h2 = C.heldout_groups(np.array([5, 7]), frac, seed=0)
        assert len(h2) == 1


# ------------------------------------------------------------------------------ fit
def test_scalar_fit_recovers_a_planted_signal(params):
    h, z, groups = planted()
    q = 1.0 / (1.0 + np.exp(-2.0 * z))
    ccfg = CriticCfg(steps=300, lr=3e-3, batch=256, heldout_frac=0.25, seed=0)
    new, rep = C.fit_critic(params, h, groups=groups, quality=q, ccfg=ccfg)
    assert rep["labels"] == "scalar" and rep["rho_labels"] == "quality"
    assert rep["rho_heldout"] > 0.85 and rep["use_critic"]
    assert rep["loss_end"] < rep["loss_start"]
    # the reported correlation equals an independent recomputation on the held-out prompts
    scores = ref_critic(new, h)
    held = rep["heldout_groups"]
    assert len(held) == rep["n_heldout_groups"] >= 20
    rhos = [ref_spearman(scores[groups == gid], q[groups == gid]) for gid in held]
    assert rep["rho_heldout"] == pytest.approx(float(np.mean(rhos)), abs=1e-6)
    assert set(held) <= C.heldout_groups(groups, 0.25, ccfg.seed)   # the documented split
    scipy_stats = pytest.importorskip("scipy.stats")
    rho_scipy = np.mean([scipy_stats.spearmanr(scores[groups == g], q[groups == g])[0]
                         for g in held])
    assert rep["rho_heldout"] == pytest.approx(float(rho_scipy), abs=1e-6)


def test_pairwise_fit_recovers_a_planted_signal(params):
    h, z, groups = planted(seed=1)
    rng = np.random.default_rng(3)
    pairs = []
    for g in range(160):
        idx = np.flatnonzero(groups == g)
        for _ in range(6):
            i, j = rng.choice(idx, size=2, replace=False)
            pairs.append((i, j) if z[i] > z[j] else (j, i))
    pairs = np.array(pairs)
    ccfg = CriticCfg(steps=300, lr=3e-3, pair_batch=256, heldout_frac=0.25, seed=0)
    new, rep = C.fit_critic(params, h, groups=groups, pairs=pairs, ccfg=ccfg)
    assert rep["labels"] == "pairs" and rep["rho_labels"] == "win_rate"
    assert rep["pair_acc_heldout"] > 0.88 and rep["n_heldout_pairs"] > 100 and rep["use_critic"]
    scores = ref_critic(new, h)
    ho = np.isin(groups, list(C.heldout_groups(groups, 0.25, ccfg.seed)))
    sel = pairs[ho[pairs[:, 0]]]
    acc = np.mean(scores[sel[:, 0]] > scores[sel[:, 1]])
    assert rep["pair_acc_heldout"] == pytest.approx(acc, abs=1e-9)
    # win rates from 6 comparisons per prompt are noisy labels: even the planted score itself
    # does not reach rho = 1 against them. The critic must come close to that ceiling and rank
    # the held-out samples almost like the planted score.
    wins = np.bincount(pairs[:, 0], minlength=len(z)).astype(float)
    games = wins + np.bincount(pairs[:, 1], minlength=len(z))
    rate = np.where(games > 0, wins / np.maximum(games, 1), np.nan)
    def mean_rho(s, y):
        out = []
        for gid in np.unique(groups[ho]):
            m = (groups == gid) & np.isfinite(y)
            if m.sum() >= 2 and np.ptp(y[m]) > 0:
                out.append(0.0 if np.ptp(s[m]) == 0 else ref_spearman(s[m], y[m]))
        return float(np.mean(out))
    assert rep["rho_heldout"] == pytest.approx(mean_rho(scores, rate), abs=1e-6)
    assert rep["rho_heldout"] > 0.85 * mean_rho(z, rate)
    assert mean_rho(scores, z) > 0.9


def test_both_label_kinds_and_only_critic_leaves_change(params):
    h, z, groups = planted(n_groups=60, seed=2)
    q = (z > 0).astype(float)
    q[::5] = np.nan                                            # some samples unlabelled
    pairs = np.array([(i, i + 1) if z[i] > z[i + 1] else (i + 1, i) for i in range(0, h.shape[0], 2)])
    pairs = np.concatenate([pairs, [[0, 8]]])                  # different prompts: dropped
    ccfg = CriticCfg(steps=50, lr=3e-3, batch=0, pair_batch=0, heldout_frac=0.3)
    new, rep = C.fit_critic(params, h, groups=groups, quality=q, pairs=pairs, ccfg=ccfg)
    assert rep["labels"] == "scalar+pairs" and rep["n_dropped_pairs"] == 1
    assert rep["n_scalar_train"] == int(np.sum(np.isfinite(q) & ~np.isin(groups, list(
        C.heldout_groups(groups, 0.3, ccfg.seed)))))
    for key in params:
        if key == "critic":
            continue
        assert new[key] is params[key]                         # the same objects, untouched
    for key in ("w1", "w2"):
        assert not np.array_equal(np.asarray(new["critic"][key]), np.asarray(params["critic"][key]))
        assert new["critic"][key].shape == params["critic"][key].shape
    w1_before = np.asarray(params["critic"]["w1"]).copy()
    C.fit_critic(params, h, groups=groups, quality=q, ccfg=ccfg)
    np.testing.assert_array_equal(np.asarray(params["critic"]["w1"]), w1_before)


def test_random_labels_do_not_pass_the_gate(params):
    h, _, groups = planted(n_groups=120, seed=4)
    q = np.random.default_rng(9).random(h.shape[0])
    _, rep = C.fit_critic(params, h, groups=groups, quality=q,
                          ccfg=CriticCfg(steps=100, lr=3e-3, batch=256, heldout_frac=0.3))
    assert abs(rep["rho_heldout"]) < 0.2 and not rep["use_critic"]


def test_untrained_critic_reports_zero_correlation(params):
    h, z, groups = planted(n_groups=30, seed=5)
    _, rep = C.fit_critic(params, h, groups=groups, quality=(z > 0).astype(float),
                          ccfg=CriticCfg(steps=0, heldout_frac=0.5))
    assert rep["rho_heldout"] == 0.0 and not rep["use_critic"]     # w2 = 0: constant scores


def test_fit_rejects_bad_inputs(params):
    h, z, groups = planted(n_groups=4)
    with pytest.raises(ValueError):
        C.fit_critic(params, h, groups=groups, quality=z)          # outside [0, 1]
    with pytest.raises(ValueError):
        C.fit_critic(params, h, groups=groups)                     # no labels
    with pytest.raises(ValueError):
        C.fit_critic(params, h[:, :8], groups=groups, quality=np.zeros(h.shape[0]))


# ------------------------------------------------------------------ hidden states
def sample_rows(n=7, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        a, b = rng.integers(200, 260, size=2).tolist()
        prompt = [BOS, USER, a, b, a, END]
        resp = rng.integers(65, 120, size=int(rng.integers(1, 12))).tolist()
        rows.append(P.row_tokens(prompt, resp))
    return rows


@functools.lru_cache(maxsize=2)
def _hidden(cfg):
    return jax.jit(lambda p, c, f: M.forward_hidden(p, c, f, cfg))


def test_collect_hidden_matches_a_row_by_row_forward(params, cfg, tok, tmp_path):
    consts = M.make_consts(cfg)
    rows = sample_rows()
    rows.append(np.concatenate([rows[0], [65] * 30]))         # longer than seq_len: invalid
    seq_len = 24
    cache = C.collect_hidden(params, consts, rows, cfg, seq_len=seq_len, batch_rows=3, tok=tok)
    assert cache.h.shape == (8, D_MODEL) and cache.h.dtype == np.float32
    assert cache.valid.tolist() == [True] * 7 + [False] and not cache.h[7].any()
    for i, row in enumerate(rows[:7]):
        L = len(row)
        tokens = np.full((1, seq_len), PAD, np.int32)
        tokens[0, :L] = row
        seg = np.array([[0 if t < L else 1 for t in range(seq_len)]], np.int32)
        feats = {"tokens": tokens, "seg": seg, **exact_features(tokens, seg, tok)}
        want = np.asarray(_hidden(cfg)(params, consts, feats))[0, L - 1]
        np.testing.assert_allclose(cache.h[i], want, rtol=1e-5, atol=1e-5)
    # explicit positions
    pos = [1] * 7 + [0]
    other = C.collect_hidden(params, consts, rows, cfg, seq_len=seq_len, batch_rows=3, tok=tok,
                             positions=pos)
    assert other.fingerprint != cache.fingerprint and not np.allclose(other.h[0], cache.h[0])
    # the cache file is reused only while rows, positions and frozen parameters match
    path = tmp_path / "hidden.npz"
    first = C.collect_hidden(params, consts, rows, cfg, seq_len=seq_len, batch_rows=3, tok=tok,
                             cache_path=path)
    assert path.exists() and first.fingerprint == cache.fingerprint
    again = C.collect_hidden(params, consts, rows, cfg, seq_len=seq_len, batch_rows=3, tok=tok,
                             cache_path=path)
    np.testing.assert_array_equal(again.h, first.h)
    changed = dict(params)
    changed["head"] = dict(params["head"], g=params["head"]["g"] * 1.5)
    assert C.params_fingerprint(changed, consts) != C.params_fingerprint(params, consts)
    critic_only = dict(params, critic={"w1": params["critic"]["w1"], "w2": params["critic"]["w2"] + 1})
    assert C.params_fingerprint(critic_only, consts) == C.params_fingerprint(params, consts)
    stale = C.collect_hidden(changed, consts, rows, cfg, seq_len=seq_len, batch_rows=3, tok=tok,
                             cache_path=path)
    assert stale.fingerprint != first.fingerprint
    assert C.HiddenCache.load(path).fingerprint == stale.fingerprint


def test_run_critic_fit_end_to_end(params, cfg, tok):
    consts = M.make_consts(cfg)
    rows = sample_rows(n=24, seed=1)
    groups = np.repeat(np.arange(6), 4)
    quality = np.array([float(len(r) % 3 == 0) for r in rows])
    new, rep = C.run_critic_fit(params, consts, rows, cfg, groups=groups, quality=quality,
                                ccfg=CriticCfg(steps=5, seq_len=24, batch_rows=8,
                                               heldout_frac=0.4), tok=tok)
    assert rep["n_invalid_rows"] == 0 and rep["steps"] == 5 and rep["finite"]
    assert all(new[k] is params[k] for k in params if k != "critic")
