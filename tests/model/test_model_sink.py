"""Calibrated sink and confidence dim: identity B1 and an independent single-head reference."""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from _model_testlib import jitted, make_batch, random_params, tiny_cfg
from bes.model import attention, model as M, sinks


def _identity_ctx(tokens: np.ndarray, cfg, conf=1.0) -> attention.Ctx:
    """Context with zero rotary angles (cos 1, sin 0) and the sink at BOS."""
    B, T = tokens.shape
    n_planes = (cfg.head_dim - cfg.bias_dims) // 2
    return attention.Ctx(cos=jnp.ones((B, T, 1, n_planes)), sin=jnp.zeros((B, T, 1, n_planes)),
                         sink=sinks.sink_indicator(jnp.asarray(tokens), M.bos_id(cfg), 1.0),
                         conf=jnp.asarray(conf, jnp.float32),
                         seg=jnp.zeros((B, T), jnp.int32),
                         pos=jnp.broadcast_to(jnp.arange(T, dtype=jnp.int32), (B, T)))


def test_sink_head_is_dirichlet_posterior_predictive():
    """B1: one-hot context keys, gain beta, sink logit beta + log(lam), uniform prior value:
    the head returns (n_ab + lam/A) / (n_a + lam) and its confidence dim n_a / (n_a + lam)."""
    cfg = tiny_cfg(d_model=32, n_heads=2, head_dim=16)
    A, lam, beta, T = 4, 2.5, 30.0, 40
    d, H, hd = cfg.d_model, cfg.n_heads, cfg.head_dim
    dc = hd - cfg.bias_dims
    rng = np.random.default_rng(0)
    sym = rng.integers(0, A, size=T)          # sym[0] unused (BOS position)
    tokens = sym.copy()[None]
    tokens[0, 0] = M.bos_id(cfg)
    # block input: [onehot(current symbol) | onehot(previous symbol)] (as built by earlier layers)
    x = np.zeros((1, T, d), np.float32)
    for t in range(1, T):
        x[0, t, sym[t]] = 1.0
        if t >= 2:
            x[0, t, A + sym[t - 1]] = 1.0
    wq = np.zeros((d, H * dc), np.float32)
    wk = np.zeros((d, H * dc), np.float32)
    wv = np.zeros((d, H * hd), np.float32)
    wq[np.arange(A), np.arange(A)] = 1.0           # head 0: q = onehot(x_t)
    wk[A + np.arange(A), np.arange(A)] = 1.0       # head 0: k = onehot(x_{t-1})
    wv[np.arange(A), np.arange(A)] = 1.0           # head 0: v = onehot(x_t)
    sink_value = np.zeros((H, hd - 1), np.float32)
    sink_value[0, :A] = 1.0 / A                    # uniform prior mean
    pa = {"wq": jnp.asarray(wq), "wk": jnp.asarray(wk), "wv": jnp.asarray(wv),
          "q_gain": jnp.asarray([beta, 1.0], jnp.float32),
          "sink_logit": jnp.asarray([sinks.dirichlet_sink_logit(beta, lam), 0.0], jnp.float32),
          "sink_value": jnp.asarray(sink_value)}
    y, _ = attention.heads(pa, jnp.asarray(x), _identity_ctx(tokens, cfg), None, cfg)
    y = np.asarray(y)[0, :, 0]                     # head 0: [T, hd]
    for t in range(T):
        a = sym[t] if t >= 1 else None
        succ = [sym[i] for i in range(2, t + 1) if a is not None and sym[i - 1] == a]
        n_a = len(succ)
        want = np.array([(succ.count(b) + lam / A) / (n_a + lam) for b in range(A)])
        np.testing.assert_allclose(y[t, :A], want, atol=1e-6, rtol=0)
        np.testing.assert_allclose(y[t, hd - 1], n_a / (n_a + lam), atol=1e-6, rtol=0)
        np.testing.assert_allclose(y[t, A:hd - 1], 0.0, atol=1e-6)


def _head_reference(q_c, k_c, v, gain, is_sink, sink_logit, sink_value, conf):
    """One head, float64: logits gain*cos for ordinary keys and sink_logit for sinks; values
    v (last dim 1 if conf) for ordinary keys and (sink_value, 0) for sinks; causal mask."""
    T = q_c.shape[0]
    qn = q_c / np.linalg.norm(q_c, axis=-1, keepdims=True)
    kn = k_c / np.linalg.norm(k_c, axis=-1, keepdims=True)
    logits = gain * qn @ kn.T
    logits[:, is_sink] = sink_logit
    logits[np.triu_indices(T, 1)] = -np.inf
    p = np.exp(logits - logits.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    vals = v.copy()
    if conf:
        vals[:, -1] = 1.0
    vals[is_sink, :-1] = sink_value
    vals[is_sink, -1] = 0.0
    return p @ vals, p[:, is_sink].sum(-1)


def test_sink_head_matches_independent_reference():
    cfg = tiny_cfg(d_model=32, n_heads=2, head_dim=16)
    d, H, hd = cfg.d_model, cfg.n_heads, cfg.head_dim
    dc = hd - cfg.bias_dims
    T = 12
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1, T, d)).astype(np.float32)
    tokens = rng.integers(0, 100, size=(1, T)).astype(np.int32)
    tokens[0, 0] = M.bos_id(cfg)
    pa = {"wq": rng.normal(size=(d, H * dc)).astype(np.float32) / 6,
          "wk": rng.normal(size=(d, H * dc)).astype(np.float32) / 6,
          "wv": rng.normal(size=(d, H * hd)).astype(np.float32) / 6,
          "q_gain": np.asarray([3.0, 5.0], np.float32),
          "sink_logit": np.asarray([0.7, -1.3], np.float32),
          "sink_value": rng.normal(size=(H, hd - 1)).astype(np.float32)}
    pa = {k: jnp.asarray(v) for k, v in pa.items()}
    for conf in (0.0, 1.0):
        y, mass = attention.heads(pa, jnp.asarray(x), _identity_ctx(tokens, cfg, conf), None, cfg)
        y = np.asarray(y)[0]
        masses = []
        for h in range(H):
            q_c = x[0] @ np.asarray(pa["wq"])[:, h * dc:(h + 1) * dc]
            k_c = x[0] @ np.asarray(pa["wk"])[:, h * dc:(h + 1) * dc]
            v = (x[0] @ np.asarray(pa["wv"])[:, h * hd:(h + 1) * hd]).astype(np.float64)
            want, sink_mass = _head_reference(
                q_c.astype(np.float64), k_c.astype(np.float64), v, float(pa["q_gain"][h]),
                tokens[0] == M.bos_id(cfg), float(pa["sink_logit"][h]),
                np.asarray(pa["sink_value"][h], np.float64), conf)
            np.testing.assert_allclose(y[:, h], want, atol=2e-5, rtol=1e-5)
            if conf:
                np.testing.assert_allclose(y[:, h, -1], 1.0 - sink_mass, atol=1e-6)
            masses.append(sink_mass)
        np.testing.assert_allclose(float(mass), np.mean(masses), atol=1e-6)


def test_sink_hides_bos_content_from_other_positions():
    """With the sink on, other positions see BOS only through the learned sink, so changing
    the BOS embedding row changes the logits at the BOS position only; with the sink off it
    changes every later position of the segment."""
    cfg = tiny_cfg()
    params = random_params(cfg, seed=5)
    feats, _, _ = M.split_batch(make_batch(cfg, B=1, T=16, seed=6, seg_starts=((0,),)))
    bos = M.bos_id(cfg)
    moved = jax.tree.map(lambda a: a, params)
    moved["emb"] = dict(params["emb"])
    moved["emb"]["tok"] = params["emb"]["tok"].at[bos].add(1.0)
    f = jitted("logits_fn", cfg)
    for sink, expect_rest_changed in ((1.0, False), (0.0, True)):
        consts = M.make_consts(cfg, sink=sink)
        a = np.asarray(f(params, consts, feats))
        b = np.asarray(f(moved, consts, feats))
        assert np.max(np.abs(a[:, 0] - b[:, 0])) > 1e-3
        rest = np.max(np.abs(a[:, 1:] - b[:, 1:]))
        if expect_rest_changed:
            assert rest > 1e-3
        else:
            assert rest == 0.0


def test_dirichlet_sink_logit():
    assert sinks.dirichlet_sink_logit(5.0, 1.0) == 5.0
    assert math.isclose(sinks.dirichlet_sink_logit(2.0, math.e), 3.0)
