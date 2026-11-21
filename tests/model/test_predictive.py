"""Output law (``bes/model/predictive.py``): p(v) = (q(v) + sum_k m_k [cand_k = v]) / (1 + sum_k m_k).

Every check compares against an independent numpy float64 implementation written here: the full
log-probabilities and the per-token cross-entropy from the direct formula, repeated and invalid
candidate ids, the masses from the trust tables (cap, kind switches, ``src`` multipliers), and
the zero-initialised tables at the model level.
"""
from __future__ import annotations

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _model_testlib import init, jitted, make_batch, random_params, tiny_cfg, to_numpy
from bes.config import CAND_FEATS
from bes.model import model as M
from bes.model import predictive

FEATS = [name for name, _ in CAND_FEATS]
KIND, SRC = FEATS.index("kind"), FEATS.index("src")


# ----------------------------------------------------------------------- references
def np_softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, np.float64)
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def np_law(z: np.ndarray, cand: np.ndarray, m: np.ndarray) -> np.ndarray:
    """The direct formula for one row (float64): z [V], cand [K], m [K] -> p [V]."""
    q = np_softmax(z)
    p, total = q.copy(), 0.0
    for c, mk in zip(np.asarray(cand).tolist(), np.asarray(m, np.float64).tolist()):
        if 0 <= c < q.size:                        # empty slots and ids >= V carry nothing
            p[c] += mk
            total += mk
    return p / (1.0 + total)


def random_case(seed: int, n: int = 9, V: int = 29, K: int = 6, zero_row: bool = True):
    """Rows of logits, candidates (empty, out-of-vocabulary and repeated ids), masses and
    targets; row 0 has no mass (``zero_row``), rows 1-2 have targets held by two slots."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 2.0, (n, V)).astype(np.float32)
    cand = rng.integers(-1, V + 3, (n, K)).astype(np.int32)
    m = (rng.gamma(0.7, 1.0, (n, K)) + 1e-3).astype(np.float32)
    y = rng.integers(0, V, n).astype(np.int32)
    if zero_row:
        m[0] = 0.0
    for i in (1, 2):
        cand[i, :2] = y[i]
    cand[3, :3] = (-1, V, V + 2)
    m[3, :3] = 50.0
    return z, cand, m, y


_mix_logprobs = jax.jit(predictive.mix_logprobs)
_mix_ce = jax.jit(predictive.mix_ce, static_argnums=(4,))


def _ce_base(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``-log q(y)`` in float32 (what ``head.chunked_ce`` delivers)."""
    lq = jax.nn.log_softmax(jnp.asarray(z), axis=-1)
    return np.asarray(-jnp.take_along_axis(lq, jnp.asarray(y)[:, None], -1)[:, 0])


# ----------------------------------------------------------------- the law itself
@pytest.mark.parametrize("seed", [0, 1])
def test_mix_logprobs_is_the_direct_formula_and_normalised(seed):
    z, cand, m, _ = random_case(seed)
    lp = np.asarray(_mix_logprobs(z, cand, m), np.float64)
    assert lp.shape == z.shape
    for i in range(z.shape[0]):
        p = np_law(z[i], cand[i], m[i])
        np.testing.assert_allclose(lp[i], np.log(p), atol=2e-5, rtol=0)
        assert abs(np.exp(lp[i]).sum() - 1.0) < 1e-6
    # the row without mass is the network's own distribution
    np.testing.assert_allclose(np.exp(lp[0]), np_softmax(z[0]), atol=1e-7)


@pytest.mark.parametrize("seed", [0, 1])
def test_mix_ce_is_the_direct_formula(seed):
    z, cand, m, y = random_case(seed)
    V = z.shape[1]
    ce = np.asarray(_mix_ce(_ce_base(z, y), y, cand, m, V), np.float64)
    lp = np.asarray(_mix_logprobs(z, cand, m), np.float64)
    for i in range(z.shape[0]):
        p_y = np_law(z[i], cand[i], m[i])[y[i]]
        assert abs(ce[i] + math.log(p_y)) < 2e-5
        assert abs(math.exp(-ce[i]) - p_y) <= 2e-5 * p_y
    # the per-token loss is the full law's log-probability at the target
    np.testing.assert_allclose(ce, -lp[np.arange(z.shape[0]), y], atol=2e-5)


def test_repeated_ids_add_and_invalid_slots_add_nothing():
    V = 11
    z = np.random.default_rng(3).normal(size=(2, V)).astype(np.float32)
    cand = np.array([[3, 3, 3, -1, V, V + 7, 8], [3, -1, -1, -1, -1, -1, -1]], np.int32)
    m = np.array([[0.5, 0.25, 0.125, 9.0, 9.0, 9.0, 0.0],
                  [0.875, 0, 0, 0, 0, 0, 0]], np.float32)
    lp = np.asarray(_mix_logprobs(z, cand, m), np.float64)
    q = np_softmax(z[0])
    want = q.copy()
    want[3] += 0.875                               # three slots on id 3; -1, V, V+7 add nothing
    want /= 1.875
    np.testing.assert_allclose(np.exp(lp[0]), want, atol=1e-6, rtol=0)
    # one slot holding the summed mass gives the same law
    single = np.asarray(_mix_logprobs(z[:1], cand[1:], m[1:]), np.float64)[0]
    np.testing.assert_allclose(lp[0], single, atol=1e-6, rtol=0)
    # and the same per-token losses (id 3 hit three times, id 8 with zero mass, id 0 untouched)
    y = np.array([3, 8], np.int32)
    ce = np.asarray(_mix_ce(_ce_base(z[[0, 0]], y), y, cand[[0, 0]], m[[0, 0]], V), np.float64)
    np.testing.assert_allclose(ce, -np.log(want[y]), atol=2e-6, rtol=0)


def test_zero_masses_give_the_network_bit_for_bit():
    z, cand, m, y = random_case(4)
    zero = np.zeros_like(m)
    base = _ce_base(z, y)
    ce = np.asarray(_mix_ce(base, y, cand, zero, z.shape[1]))
    assert ce.tobytes() == base.tobytes()
    lp = np.asarray(_mix_logprobs(z, cand, zero), np.float64)
    np.testing.assert_allclose(lp, np.log(np_softmax(z)), atol=2e-6, rtol=0)


def test_gradients_of_mix_ce_against_the_closed_form():
    """d ce / d m_k = -[cand_k = y] / (q_y + M_y) + 1 / (1 + M); d ce / d ce_base =
    q_y / (q_y + M_y); invalid slots get zero gradient. Row 0 has no mass at all: there the
    plain branch is taken (d ce / d ce_base = 1, d ce / d m_k = 1 / (1 + M) = 1) -- the
    derivative at a zero target mass is one-sided, which is harmless in the model, where a
    mass is zero only when its constant slot weight is zero."""
    z, cand, m, y = random_case(5)
    V = z.shape[1]
    base = _ce_base(z, y)
    g_base, g_m = jax.grad(lambda b, mm: jnp.sum(predictive.mix_ce(b, y, cand, mm, V)),
                           argnums=(0, 1))(base, m)
    for i in range(z.shape[0]):
        valid = (cand[i] >= 0) & (cand[i] < V)
        mm = np.where(valid, m[i].astype(np.float64), 0.0)
        q_y = math.exp(-float(base[i]))
        m_y = float(mm[cand[i] == y[i]].sum())
        hit = (cand[i] == y[i]) / (q_y + m_y) if m_y > 0 else 0.0
        want_m = np.where(valid, -hit + 1.0 / (1.0 + mm.sum()), 0.0)
        np.testing.assert_allclose(np.asarray(g_m[i]), want_m, rtol=2e-5, atol=1e-7)
        want_b = q_y / (q_y + m_y) if m_y > 0 else 1.0
        assert abs(float(g_base[i]) - want_b) < 2e-6
    assert np.all(np.isfinite(np.asarray(g_base))) and np.all(np.isfinite(np.asarray(g_m)))


def test_gradients_of_mix_logprobs_against_dense_autodiff():
    """The compare/select accumulation differentiates like an explicit one-hot product
    (strictly positive masses; see the closed-form test for a zero mass)."""
    z, cand, m, _ = random_case(6, zero_row=False)
    V = z.shape[1]
    g = np.random.default_rng(7).normal(size=z.shape).astype(np.float32)

    def dense(zz, mm):                               # one-hot product, explicit normaliser
        valid = ((cand >= 0) & (cand < V)).astype(jnp.float32)
        mm = mm * valid
        add = jnp.einsum("nk,nkv->nv", mm, jax.nn.one_hot(cand, V, dtype=jnp.float32))
        return jnp.log(jax.nn.softmax(zz, -1) + add) - jnp.log1p(jnp.sum(mm, -1))[:, None]

    ga = jax.grad(lambda zz, mm: jnp.sum(g * predictive.mix_logprobs(zz, cand, mm)), (0, 1))(z, m)
    gd = jax.grad(lambda zz, mm: jnp.sum(g * dense(zz, mm)), (0, 1))(z, m)
    for a, d in zip(ga, gd):
        np.testing.assert_allclose(np.asarray(a), np.asarray(d), atol=2e-5, rtol=1e-4)


def test_loss_rises_at_most_by_log1p_of_the_total_mass():
    """Bounded downside: -log p(y) <= -log q(y) + log(1 + M) for every target."""
    z, cand, m, y = random_case(8, n=40)
    V = z.shape[1]
    base = _ce_base(z, y).astype(np.float64)
    ce = np.asarray(_mix_ce(_ce_base(z, y), y, cand, m * 30.0, V), np.float64)
    total = np.where((cand >= 0) & (cand < V), m.astype(np.float64) * 30.0, 0.0).sum(-1)
    assert np.all(ce <= base + np.log1p(total) + 1e-5)
    assert np.any(ce > base + 1.0) and np.any(ce < base - 1.0)   # both directions occur


# --------------------------------------------------------------- masses in the model
def _hidden_and_tape(cfg, params, batch):
    """Features and final hidden states (the trunk does not depend on the mass fields, so the
    default tiny configuration's compiled forward serves every variant)."""
    feats, _, _ = M.split_batch(batch)
    base = dataclasses.replace(cfg, mass_offset=tiny_cfg().mass_offset,
                               mass_log_cap=tiny_cfg().mass_log_cap)
    h = jitted("forward_hidden", base)(params, M.make_consts(base), feats)
    return feats, h


@functools.lru_cache(maxsize=None)
def _tape_fn(cfg):
    return jax.jit(functools.partial(M.tape_masses, cfg=cfg))


def np_masses(P: dict, h: np.ndarray, cand: np.ndarray, feat: np.ndarray, cfg, kind_on=None,
              src_scale=None, cand_out: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """``(m, a)`` of the law from the trust tables, float64 (gathered rows, explicit sums)."""
    feat = feat.astype(np.int64)
    kind_on = np.ones(8) if kind_on is None else np.asarray(kind_on, np.float64)
    src_scale = np.ones(8) if src_scale is None else np.asarray(src_scale, np.float64)
    c = sum(P["trust"]["feat"][name][feat[..., i]] for i, name in enumerate(FEATS))
    beta = sum(P["trust"]["bias"][name][feat[..., i]] for i, name in enumerate(FEATS))
    a = beta + np.einsum("...c,...kc->...k", h @ P["trust"]["proj"], c) - cfg.mass_offset
    w = ((cand >= 0) & (cand < cfg.vocab_size)) * kind_on[feat[..., KIND]] * cand_out \
        * src_scale[feat[..., SRC]]
    return np.exp(np.minimum(a, cfg.mass_log_cap)) * w, a


def test_masses_follow_the_formula():
    """Masses from random trust tables against the float64 formula, with a kind switched
    off, a src multiplier, ids beyond the vocabulary and the cap reached by some slots."""
    cfg = tiny_cfg(mass_offset=2.0, mass_log_cap=1.5)
    params = random_params(cfg, seed=3)
    batch = make_batch(cfg, B=2, T=12, seed=4)
    batch["cand_tok"][:, ::4, 2] = cfg.vocab_size + 3
    batch["cand_feat"][:, ::4, 2, KIND] = 1
    feats, h = _hidden_and_tape(cfg, params, batch)
    consts = M.make_consts(cfg, kind_2=0.0, src_3=0.5)
    cand, m = _tape_fn(cfg)(params, consts, h, feats)
    kind_on, src = np.ones(8), np.ones(8)
    kind_on[2], src[3] = 0.0, 0.5
    want, a = np_masses(to_numpy(params), np.asarray(h, np.float64), batch["cand_tok"],
                        batch["cand_feat"], cfg, kind_on, src)
    np.testing.assert_array_equal(np.asarray(cand), batch["cand_tok"])
    np.testing.assert_allclose(np.asarray(m, np.float64), want, rtol=2e-5, atol=1e-7)
    live = want > 0
    assert np.any(live & (a > cfg.mass_log_cap)) and np.any(live & (a < cfg.mass_log_cap))
    assert np.all(np.asarray(m)[batch["cand_tok"] >= cfg.vocab_size] == 0)


def test_src_multipliers_scale_exactly_the_masses_of_that_source():
    cfg = tiny_cfg()
    params = random_params(cfg, seed=5)
    batch = make_batch(cfg, B=2, T=16, seed=6)
    feats, h = _hidden_and_tape(cfg, params, batch)
    fn = _tape_fn(cfg)
    m1 = np.asarray(fn(params, M.make_consts(cfg), h, feats)[1])
    m0 = np.asarray(fn(params, M.make_consts(cfg, src_5=0.0), h, feats)[1])
    mq = np.asarray(fn(params, M.make_consts(cfg, src_5=0.25), h, feats)[1])
    is5 = batch["cand_feat"][..., SRC] == 5
    assert np.any(m1[is5] > 0) and np.any(m1[~is5] > 0)
    np.testing.assert_array_equal(m0[is5], 0.0)
    np.testing.assert_array_equal(m0[~is5], m1[~is5])
    np.testing.assert_array_equal(mq[is5], 0.25 * m1[is5])       # a power of two: exact
    np.testing.assert_array_equal(mq[~is5], m1[~is5])
    # in the loss, src_5 = 0 equals emptying those slots (candidate input off in both runs)
    consts_in_off = {"cand_in": 0.0}
    loss_a, _ = jitted("loss_fn", cfg)(params, M.make_consts(cfg, src_5=0.0, **consts_in_off), batch)
    emptied = dict(batch, cand_tok=np.where(is5, -1, batch["cand_tok"]))
    loss_b, _ = jitted("loss_fn", cfg)(params, M.make_consts(cfg, **consts_in_off), emptied)
    np.testing.assert_allclose(float(loss_a), float(loss_b), rtol=1e-6)
    loss_c, _ = jitted("loss_fn", cfg)(params, M.make_consts(cfg, **consts_in_off), batch)
    assert abs(float(loss_c) - float(loss_a)) > 1e-4


def test_make_consts_src_multipliers():
    cfg = tiny_cfg()
    assert np.asarray(M.make_consts(cfg)["enable"]["src"]).tolist() == [1.0] * 8
    c = M.make_consts(cfg, src=0.5, src_5=0.25)
    assert np.asarray(c["enable"]["src"]).tolist() == [0.5] * 5 + [0.25] + [0.5] * 2
    with pytest.raises(KeyError):
        M.make_consts(cfg, src_8=1.0)
    with pytest.raises(ValueError):
        M.make_consts(cfg, src_5=-0.1)


def test_zero_initialised_trust_tables_follow_the_formula():
    """At initialisation every active slot has mass exp(-mass_offset), so
    ce = ce_base + log1p(n_active exp(-off)) - log1p(n_hit exp(-off) / q(y))."""
    cfg = tiny_cfg()
    params = init(cfg, seed=3)
    batch = make_batch(cfg, B=2, T=24, seed=9)
    y = batch["tokens"][:, 1:]
    batch["cand_tok"][:, ::3, 0] = y[:, ::3]          # plant hits: the target is a candidate
    batch["cand_tok"][:, ::6, 5] = y[:, ::6]          # twice at some positions
    batch["cand_feat"][:, ::3, 0, KIND] = 1
    batch["cand_feat"][:, ::6, 5, KIND] = 2
    consts = M.make_consts(cfg)
    feats, yy, w = M.split_batch(batch)
    h = jitted("forward_hidden", cfg)(params, consts, feats)
    tok_ce = jax.jit(functools.partial(M.token_ce, cfg=cfg, parts=True))
    ce, ce_base, mass = (np.asarray(a, np.float64) for a in tok_ce(params, consts, h, feats, yy))
    plain = jax.jit(functools.partial(M.token_ce, cfg=dataclasses.replace(cfg, use_cand_out=False)))
    np.testing.assert_array_equal(np.asarray(plain(params, consts, h, feats, yy)), ce_base)
    e = math.exp(-cfg.mass_offset)
    active = (batch["cand_tok"] >= 0).reshape(-1, 16)                # every kind is on
    hits = (active & (batch["cand_tok"].reshape(-1, 16) == y.reshape(-1, 1))).sum(-1)
    assert hits.max() >= 2
    q_y = np.exp(-ce_base)
    want = ce_base + np.log1p(active.sum(-1) * e) - np.log1p(hits * e / q_y)
    np.testing.assert_allclose(ce, want, atol=5e-6, rtol=0)
    np.testing.assert_allclose(mass, active.sum(-1) * e, rtol=1e-6)
    loss, met = jitted("loss_fn", cfg)(params, consts, batch)
    wn = np.asarray(w, np.float64).reshape(-1)
    np.testing.assert_allclose(float(loss), (wn * want).sum() / wn.sum(), rtol=1e-6)
    np.testing.assert_allclose(float(met["tape_mass"]), (wn * active.sum(-1) * e).sum() / wn.sum(),
                               rtol=1e-5)
    np.testing.assert_allclose(float(met["tape_gain"]), (wn * (ce_base - want)).sum() / wn.sum(),
                               rtol=1e-4, atol=2e-6)
    assert float(met["tape_gain"]) != 0.0
