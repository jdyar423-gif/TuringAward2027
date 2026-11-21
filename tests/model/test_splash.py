"""Splash attention path (TPU kernels run in Pallas interpret mode on CPU) against the dense path:
forward, gradients (separate and fused backward kernels), the whole model's loss and gradients,
padding of lengths that are not a multiple of 128, and the block sizes of the final shapes."""
from __future__ import annotations

import dataclasses
import functools
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _model_testlib import config_model, make_batch, random_params, tiny_cfg
from bes.model import attention, sinks
from bes.model import attention_splash as S
from bes.model import model as M

pytestmark = pytest.mark.skipif(not S.available(), reason="splash attention not installed")


def _qkv(b: int, t: int, h: int, hd: int, seg_starts, seed: int = 0, conf: bool = True):
    """Model-like inputs: rotated/normalised content dims with a per-head gain, a sink key with
    its bias dim at every segment start, sink values and the confidence dim."""
    rng = np.random.default_rng(seed)
    dc = hd - 2
    qc = rng.standard_normal((b, t, h, dc)).astype(np.float32)
    kc = rng.standard_normal((b, t, h, dc)).astype(np.float32)
    qc = qc / np.linalg.norm(qc, axis=-1, keepdims=True) * 4.0
    kc = kc / np.linalg.norm(kc, axis=-1, keepdims=True)
    v = rng.standard_normal((b, t, h, hd)).astype(np.float32)
    seg = np.zeros((b, t), np.int32)
    sink = np.zeros((b, t), np.float32)
    for r in range(b):
        for i, s in enumerate(seg_starts[r % len(seg_starts)]):
            seg[r, s:] = i
            sink[r, s] = 1.0
    logit = jnp.asarray(rng.standard_normal(h), jnp.float32)
    value = jnp.asarray(rng.standard_normal((h, hd - 1)), jnp.float32)
    q, k, v = sinks.assemble(jnp.asarray(qc), jnp.asarray(kc), jnp.asarray(v), jnp.asarray(sink),
                             logit, value, 1.0 if conf else None, 2, jnp.float32)
    return q, k, v, jnp.asarray(seg), jnp.asarray(sink)


def _dense(q, k, v, seg):
    pos = jnp.broadcast_to(jnp.arange(q.shape[1], dtype=jnp.int32), seg.shape)
    return attention.attend_xla(q, k, v, seg, seg, pos, pos)[0]


@pytest.mark.parametrize("t", [128, 200])
def test_forward_matches_dense_and_float64(t):
    cfg = tiny_cfg(attn_impl="splash")
    q, k, v, seg, sink = _qkv(2, t, 2, 32, ((0,), (0, 37, 150 if t > 150 else 90)), seed=t)
    y = np.asarray(S.attend(q, k, v, seg, cfg))
    assert y.shape == (2, t, 2, 32) and y.dtype == np.float32
    np.testing.assert_allclose(y, np.asarray(_dense(q, k, v, seg)), atol=1e-5)
    np.testing.assert_allclose(y, S.reference(q, k, v, seg), atol=1e-5)
    # confidence dim = attention mass outside the sink keys (dense masses as reference)
    pos = jnp.broadcast_to(jnp.arange(t, dtype=jnp.int32), seg.shape)
    _, mass = attention.attend_xla(q, k, v, seg, seg, pos, pos, sink)
    np.testing.assert_allclose(1.0 - y[..., -1], np.swapaxes(np.asarray(mass), 1, 2), atol=1e-5)
    y2, m2 = S.attend_like_xla(q, k, v, seg, seg, pos, pos, sink, cfg=cfg)
    assert m2 is None
    np.testing.assert_array_equal(np.asarray(y2), y)


@pytest.mark.parametrize("fused", [False, True])
def test_gradients_match_dense(fused):
    cfg = tiny_cfg(attn_impl="splash", splash_fused_bwd=fused)
    q, k, v, seg, _ = _qkv(2, 128, 2, 32, ((0, 50), (0,)), seed=3)
    r = jnp.asarray(np.random.default_rng(4).standard_normal(q.shape), jnp.float32)
    g_s = jax.grad(lambda *a: jnp.sum(r * S.attend(*a, seg, cfg)), argnums=(0, 1, 2))(q, k, v)
    g_d = jax.grad(lambda *a: jnp.sum(r * _dense(*a, seg)), argnums=(0, 1, 2))(q, k, v)
    for a, b in zip(g_s, g_d):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=2e-4, rtol=1e-4)


def test_kernel_cache_survives_traces():
    """A kernel first built inside a scan trace is reused by later, separate traces."""
    cfg = tiny_cfg(attn_impl="splash")
    q, k, v, seg, _ = _qkv(1, 128, 2, 32, ((0, 40),), seed=10)
    S.make_kernel.cache_clear()

    def scanned(scale):
        def body(c, s):
            return c, S.attend(q * s, k, v, seg, cfg)
        return jax.lax.scan(body, 0.0, jnp.asarray([1.0, scale]))[1]
    y1 = jax.jit(scanned)(1.0)                     # builds the kernel inside the scan trace
    y2 = jax.jit(lambda: S.attend(q, k, v, seg, cfg))()  # a new trace reuses the cached kernel
    np.testing.assert_allclose(np.asarray(y1[0]), np.asarray(y2), atol=1e-6)
    assert all(not isinstance(a, jax.core.Tracer)
               for a in jax.tree.leaves(S.make_kernel(128, 2, S.block_sizes(cfg, 128), True)))


def test_bfloat16_inputs():
    cfg = tiny_cfg(attn_impl="splash")
    q, k, v, seg, _ = _qkv(1, 128, 2, 32, ((0, 64),), seed=5)
    qb, kb, vb = (a.astype(jnp.bfloat16) for a in (q, k, v))
    y = S.attend(qb, kb, vb, seg, cfg)
    assert y.dtype == jnp.bfloat16
    ref = _dense(qb, kb, vb, seg)
    assert float(jnp.max(jnp.abs(y.astype(jnp.float32) - ref))) < 0.05


@pytest.fixture
def splash_module(monkeypatch):
    """The splash module, also registered under its former alias ``bes.model.splash``."""
    monkeypatch.setitem(sys.modules, "bes.model.splash", S)
    return S


def _model_runs(grad: bool):
    base = tiny_cfg(n_loops=2)
    params = random_params(base, seed=6)
    consts = M.make_consts(base)
    batch = make_batch(base, B=2, T=40, seed=7, seg_starts=((0,), (0, 17)))
    out = {}
    for impl in ("xla", "splash"):
        cfg = dataclasses.replace(base, attn_impl=impl)
        loss = functools.partial(M.loss_fn, cfg=cfg)
        f = jax.jit(jax.value_and_grad(loss, has_aux=True) if grad else loss)
        out[impl] = f(params, consts, batch)
    return out["xla"], out["splash"]


@pytest.mark.slow
def test_model_loss_with_the_splash_path(splash_module):
    (l_x, m_x), (l_s, m_s) = _model_runs(grad=False)
    np.testing.assert_allclose(float(l_s), float(l_x), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(m_s["sink_mass"]), np.asarray(m_x["sink_mass"]), atol=1e-5)


@pytest.mark.slow
def test_model_gradients_with_the_splash_path(splash_module):
    ((l_x, _), g_x), ((l_s, _), g_s) = _model_runs(grad=True)
    np.testing.assert_allclose(float(l_s), float(l_x), rtol=1e-5)
    for a, b in zip(jax.tree.leaves(g_s), jax.tree.leaves(g_x)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=2e-4, rtol=2e-3)


def test_core_selects_the_splash_module(monkeypatch):
    import bes.model
    monkeypatch.delitem(sys.modules, "bes.model.splash", raising=False)
    monkeypatch.delattr(bes.model, "splash", raising=False)
    cfg = tiny_cfg(attn_impl="splash")
    params = jax.eval_shape(functools.partial(M.init_params, cfg), jax.random.PRNGKey(0))
    feats, _, _ = M.split_batch(make_batch(cfg, B=1, T=16, seed=9))
    h = jax.eval_shape(functools.partial(M.forward_hidden, cfg=cfg), params, M.make_consts(cfg), feats)
    assert h.shape == (1, 16, cfg.d_model)


def test_block_sizes_and_kernels_of_the_final_shapes():
    cfg = config_model("final")
    assert cfg.attn_impl == "splash"
    for t in (1024, 2048):
        bs = S.block_sizes(cfg, t)
        assert (bs.block_q, bs.block_kv, bs.block_kv_compute) == (512, 512, 512)
        assert bs.has_backward_blocks
        kern = S.make_kernel(t, cfg.n_heads, bs, False)
        assert type(kern).__name__ == "SplashAttentionKernel"
    assert S.padded_length(200) == 256 and S.block_sizes(cfg, 384).block_q == 384
    assert S.block_sizes(dataclasses.replace(cfg, splash_block_q=256), 1024).block_q == 256
    assert S.use_interpret(dataclasses.replace(cfg, splash_interpret="false")) is False
    with pytest.raises(ValueError):
        S.use_interpret(dataclasses.replace(cfg, splash_interpret="sometimes"))
