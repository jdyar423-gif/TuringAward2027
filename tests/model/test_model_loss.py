"""Loss: chunked float32 cross-entropy, the scatter-free tilt and the trust head."""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from _model_testlib import (init, jitted, make_batch, random_params, tiny_cfg, to_numpy,
                            value_and_grad_fn)
from bes.config import CAND_FEATS
from bes.model import head, model as M, oracle_io


def test_tilt_off_is_bit_identical_to_no_tilt_code_path():
    cfg = tiny_cfg()
    cfg_none = dataclasses.replace(cfg, use_cand_out=False)
    batch = make_batch(cfg, seed=11)
    # (a) trained-looking trust tables, tilt switched off at run time
    params = random_params(cfg, seed=12)
    loss_off, _ = jitted("loss_fn", cfg)(params, M.make_consts(cfg, cand_out=0.0), batch)
    loss_none, _ = jitted("loss_fn", cfg_none)(params, M.make_consts(cfg_none), batch)
    assert np.asarray(loss_off).tobytes() == np.asarray(loss_none).tobytes()
    # (b) zero-initialised trust tables (b_k = 0) with the tilt switched on
    p0 = init(cfg, seed=3)
    loss_b0, _ = jitted("loss_fn", cfg)(p0, M.make_consts(cfg), batch)
    loss_none0, _ = jitted("loss_fn", cfg_none)(p0, M.make_consts(cfg_none), batch)
    assert np.asarray(loss_b0).tobytes() == np.asarray(loss_none0).tobytes()
    # and the tilt does act when it is on and non-zero
    loss_on, met = jitted("loss_fn", cfg)(params, M.make_consts(cfg), batch)
    assert abs(float(loss_on) - float(loss_none)) > 1e-4
    assert float(met["trust_abs"]) > 0.0


def test_chunked_loss_equals_unchunked():
    cfg = tiny_cfg()
    params = random_params(cfg, seed=13)
    consts = M.make_consts(cfg)
    batch = make_batch(cfg, B=2, T=24, seed=14)
    assert cfg.loss_chunk >= 2 * 24                     # the reference runs as a single chunk
    (l_big, _), g_big = value_and_grad_fn(cfg)(params, consts, batch)
    # 7 rows per chunk: 7 chunks, the last one padded
    (l_c, _), g_c = value_and_grad_fn(dataclasses.replace(cfg, loss_chunk=7))(params, consts, batch)
    np.testing.assert_allclose(float(l_c), float(l_big), rtol=1e-6)
    for a, b in zip(jax.tree.leaves(g_c), jax.tree.leaves(g_big)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-6, rtol=1e-5)


def _np_log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def test_loss_matches_independent_cross_entropy_with_tilt():
    cfg = tiny_cfg()
    params = random_params(cfg, seed=15)
    consts = M.make_consts(cfg, kind_2=0.0)          # one candidate kind switched off
    batch = make_batch(cfg, B=2, T=24, seed=16)
    feats, y, w = M.split_batch(batch)
    loss, met = jitted("loss_fn", cfg)(params, consts, batch)
    hidden = np.asarray(jitted("forward_hidden", cfg)(params, consts, feats), np.float64)
    P = to_numpy(params)
    # untilted logits from the hidden state
    z = (hidden @ P["head"]["proj"]) @ P["head"]["out"].T
    z = cfg.softcap * np.tanh(z / cfg.softcap)
    # trust: b_k = 8 tanh(<h A, c_k> / 8) on active slots
    cand, feat = batch["cand_tok"], batch["cand_feat"].astype(np.int64)
    c = sum(P["trust"]["feat"][name][feat[..., i]] for i, (name, _) in enumerate(CAND_FEATS))
    score = np.einsum("btc,btkc->btk", hidden @ P["trust"]["proj"], c)
    kind_on = np.ones(8)
    kind_on[2] = 0.0
    m = (cand >= 0) * kind_on[feat[..., 0]]
    b = 8.0 * np.tanh(score / 8.0) * m
    zt = z.copy()
    for bi in range(cand.shape[0]):
        for t in range(cand.shape[1]):
            for k in range(cand.shape[2]):
                if cand[bi, t, k] >= 0:
                    zt[bi, t, cand[bi, t, k]] += b[bi, t, k]
    ce = -np.take_along_axis(_np_log_softmax(zt), np.asarray(y)[..., None], -1)[..., 0]
    want = (ce * batch["w"]).sum() / batch["w"].sum()
    np.testing.assert_allclose(float(loss), want, rtol=2e-5)
    np.testing.assert_allclose(float(met["weight"]), batch["w"].sum())
    np.testing.assert_allclose(float(met["tokens"]), (batch["w"] > 0).sum())
    np.testing.assert_allclose(float(met["trust_abs"]), np.abs(b).sum() / m.sum(), rtol=1e-4)
    # logits_from_hidden gives the same tilted logits
    got_z = np.asarray(jitted("logits_fn", cfg)(params, consts, feats))
    np.testing.assert_allclose(got_z, zt, atol=1e-4)
    # loss share by role sums to one and matches the per-role sums
    role = batch["reg"][..., 0]
    by_role = np.array([(ce * batch["w"])[role == r].sum() for r in range(6)])
    np.testing.assert_allclose(np.asarray(met["ce_by_role"]), by_role, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(met["loss_share_by_role"]).sum(), 1.0, rtol=1e-5)


def test_tilt_custom_vjp_matches_dense_autodiff():
    rng = np.random.default_rng(0)
    n, V, K = 6, 37, 5
    z = jnp.asarray(rng.normal(size=(n, V)), jnp.float32)
    cand = jnp.asarray(rng.integers(-1, V, size=(n, K)), jnp.int32)
    b = jnp.asarray(rng.normal(size=(n, K)), jnp.float32)
    g = jnp.asarray(rng.normal(size=(n, V)), jnp.float32)

    def dense(z, b):   # explicit one-hot product (reference)
        oh = jax.nn.one_hot(cand, V, dtype=jnp.float32)       # -1 -> all zeros
        return z + jnp.einsum("nk,nkv->nv", b, oh)

    out = oracle_io.add_tilt(z, cand, b)
    np.testing.assert_allclose(np.asarray(out), np.asarray(dense(z, b)), atol=1e-6)
    ga = jax.grad(lambda z, b: jnp.sum(oracle_io.add_tilt(z, cand, b) * g), argnums=(0, 1))(z, b)
    gd = jax.grad(lambda z, b: jnp.sum(dense(z, b) * g), argnums=(0, 1))(z, b)
    for a, d in zip(ga, gd):
        np.testing.assert_allclose(np.asarray(a), np.asarray(d), atol=1e-6)


def test_chunked_ce_memory_stays_per_chunk():
    """The compiled loss keeps O(chunk * V) temporaries, never [n, K, V] or [n, V] for all n."""
    cfg = tiny_cfg(vocab_size=4096, loss_chunk=256)
    n, d, K = 4096, cfg.d_model, 16
    ph = {"proj": jnp.zeros((d, cfg.head_rank)), "out": jnp.zeros((cfg.vocab_size, cfg.head_rank))}

    def f(ph, h, y, cand, b):
        return jnp.sum(head.chunked_ce(ph, h, y, cand, b, cfg))

    args = (ph, jnp.zeros((n, d)), jnp.zeros((n,), jnp.int32), jnp.zeros((n, K), jnp.int32),
            jnp.zeros((n, K)))
    compiled = jax.jit(jax.grad(f, argnums=(0, 1, 4))).lower(*args).compile()
    temp = compiled.memory_analysis().temp_size_in_bytes
    full_logits = n * cfg.vocab_size * 4
    assert temp < full_logits / 2, (temp, full_logits)
