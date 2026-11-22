"""Selective attention (dense path): an independent numpy implementation with several segments
per row, BOS keys and padding; switch off = plain attention bit for bit; BOS keys are never
masked; gradients flow through the accumulated selection; chunked queries with a carried prior
(the key-value cache recurrence) reproduce the full computation; causality."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from _model_testlib import jitted, make_batch, random_params, tiny_cfg
from bes.model import attention, model as M


def _inputs(seed: int, b=2, t=14, h=3, hd=8, seg_rows=((0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2),
                                                       (-1, -1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1))):
    """Random q, k, v; row 0 has three segments, row 1 two segments after two padding slots
    (segment -1). BOS at every segment start, one more BOS inside a segment of row 0."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(b, t, h, hd)) * 1.5
    k = rng.normal(size=(b, t, h, hd))
    v = rng.normal(size=(b, t, h, hd))
    seg = np.asarray(seg_rows[:b], np.int32)
    pos = np.broadcast_to(np.arange(t, dtype=np.int32), (b, t)).copy()
    bos = np.zeros((b, t), bool)
    for r in range(b):
        for i in range(t):
            bos[r, i] = seg[r, i] >= 0 and (i == 0 or seg[r, i] != seg[r, i - 1])
    bos[0, 3] = True
    return q, k, v, seg, pos, bos


def numpy_selective(q, k, v, seg, pos, bos, sel=1.0):
    """Loops over rows, queries and keys (float64), from the formulas of the specification:
    S[i, j] = relu(head-0 logit) for visible keys other than the query's own slot, BOS keys and
    padding; F[i, j] = sum over earlier queries; every head's logits minus sel * F."""
    b_n, t_n, h_n, _ = q.shape
    logits = np.einsum("bqhd,bkhd->bhqk", q, k)
    vis = np.zeros((b_n, t_n, t_n), bool)
    s = np.zeros((b_n, t_n, t_n))
    for b in range(b_n):
        for i in range(t_n):
            for j in range(t_n):
                vis[b, i, j] = pos[b, j] <= pos[b, i] and seg[b, j] == seg[b, i]
                if vis[b, i, j] and j != i and not bos[b, j] and seg[b, j] >= 0:
                    s[b, i, j] = max(logits[b, 0, i, j], 0.0)
    y = np.zeros(q.shape)
    for b in range(b_n):
        for i in range(t_n):
            f = s[b, :i].sum(axis=0)                                    # [Tk]
            keys = np.flatnonzero(vis[b, i])
            for h in range(h_n):
                row = logits[b, h, i, keys] - sel * f[keys]
                p = np.exp(row - row.max())
                y[b, i, h] = (p / p.sum()) @ v[b, keys, h]
    return y


@pytest.mark.parametrize("sel", [1.0, 0.5])
def test_attend_xla_matches_the_numpy_implementation(sel):
    q, k, v, seg, pos, bos = _inputs(0)
    j = lambda a: jnp.asarray(a, jnp.float32)  # noqa: E731
    y, _ = attention.attend_xla(j(q), j(k), j(v), jnp.asarray(seg), jnp.asarray(seg), jnp.asarray(pos),
                                jnp.asarray(pos), sel=jnp.float32(sel), bos_k=jnp.asarray(bos))
    want = numpy_selective(q, k, v, seg, pos, bos, sel)
    np.testing.assert_allclose(np.asarray(y), want, atol=2e-5, rtol=2e-5)
    plain = numpy_selective(q, k, v, seg, pos, bos, 0.0)
    assert np.max(np.abs(want - plain)) > 1e-2


def test_switch_off_is_plain_attention_bit_for_bit():
    q, k, v, seg, pos, bos = (jnp.asarray(a) for a in _inputs(1))
    q, k, v = (a.astype(jnp.float32) for a in (q, k, v))
    sink = jnp.asarray(bos, jnp.float32)
    plain = attention.attend_xla(q, k, v, seg, seg, pos, pos, sink)
    off = attention.attend_xla(q, k, v, seg, seg, pos, pos, sink, sel=jnp.float32(0.0), bos_k=bos)
    for a, b in zip(plain, off):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # model level: sel_attn with the switch at 0 equals the model without selective attention
    cfg, base = tiny_cfg(sel_attn=True), tiny_cfg()
    params = random_params(cfg, seed=2)
    feats, _, _ = M.split_batch(make_batch(cfg, seed=3))
    a = np.asarray(jitted("logits_fn", cfg)(params, M.make_consts(cfg, sel=0.0), feats))
    b = np.asarray(jitted("logits_fn", base)(params, M.make_consts(base), feats))
    np.testing.assert_array_equal(a, b)
    assert np.max(np.abs(np.asarray(jitted("logits_fn", cfg)(params, M.make_consts(cfg), feats)) - b)) > 1e-3


def test_bos_keys_and_own_slots_are_never_masked():
    """However strongly head 0 attends to them, the BOS columns and the diagonal keep their
    logits; every other visible key loses exactly the selection of the earlier queries."""
    rng = np.random.default_rng(4)
    b_n, h_n, t_n = 1, 2, 6
    logits = rng.normal(size=(b_n, h_n, t_n, t_n)).astype(np.float32) + 3.0     # mostly positive
    visible = np.tril(np.ones((t_n, t_n), bool))[None]
    logits = np.where(visible[:, None], logits, attention.NEG_INF)
    bos = np.zeros((b_n, t_n), bool)
    bos[0, [0, 3]] = True
    out, s_sum = attention.selective_logits(jnp.asarray(logits), jnp.asarray(visible),
                                            jnp.asarray(~bos), jnp.arange(t_n), jnp.float32(1.0))
    out, s_sum = np.asarray(out), np.asarray(s_sum)
    np.testing.assert_array_equal(out[..., [0, 3]], logits[..., [0, 3]])
    for i in range(t_n):
        assert out[0, 0, i, i] == logits[0, 0, i, i]
    s = np.where(visible[0] & ~bos[0][None] & ~np.eye(t_n, dtype=bool), np.maximum(logits[0, 0], 0), 0)
    f = np.concatenate([np.zeros((1, t_n)), np.cumsum(s, axis=0)[:-1]])
    np.testing.assert_allclose(out[0], np.where(visible[0], logits[0] - f[None], attention.NEG_INF),
                               rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(s_sum[0], s.sum(0), rtol=1e-6)
    assert np.all(out[~np.broadcast_to(visible[:, None], out.shape)] == attention.NEG_INF)


def test_chunked_queries_with_a_carried_prior_equal_the_full_computation():
    """The key-value cache recurrence: queries fed in chunks at slots ``t0 ..`` over all keys,
    ``F = prior + selection of the earlier queries of the chunk``, then ``prior += s_sum``."""
    rng = np.random.default_rng(5)
    b_n, h_n, t_n = 2, 2, 11
    raw = rng.normal(size=(b_n, h_n, t_n, t_n)).astype(np.float32)
    seg = np.asarray([[0] * 5 + [1] * 6, [-1, -1] + [0] * 9])
    pos = np.broadcast_to(np.arange(t_n), (b_n, t_n))
    visible = (pos[:, None, :] <= pos[:, :, None]) & (seg[:, None, :] == seg[:, :, None])
    logits = np.where(visible[:, None], raw, attention.NEG_INF)
    maskable = np.ones((b_n, t_n), bool)
    maskable[0, [0, 5]] = False
    maskable[1, :3] = False
    full, total = attention.selective_logits(jnp.asarray(logits), jnp.asarray(visible), jnp.asarray(maskable),
                                             jnp.arange(t_n), jnp.float32(1.0))
    prior = jnp.zeros((b_n, t_n), jnp.float32)
    parts = []
    for lo, hi in ((0, 4), (4, 5), (5, 9), (9, 10), (10, 11)):
        out, s_sum = attention.selective_logits(
            jnp.asarray(logits[:, :, lo:hi]), jnp.asarray(visible[:, lo:hi]), jnp.asarray(maskable),
            jnp.arange(lo, hi), jnp.float32(1.0), prior)
        parts.append(np.asarray(out))
        prior = prior + s_sum
    np.testing.assert_allclose(np.concatenate(parts, axis=2), np.asarray(full), rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(np.asarray(prior), np.asarray(total), rtol=1e-6, atol=1e-5)


def test_gradients_flow_through_the_selection_float64():
    q, k, v, seg, pos, bos = _inputs(6, t=9, seg_rows=((0,) * 4 + (1,) * 5, (0,) * 9))
    with jax.enable_x64(True):
        args = tuple(jnp.asarray(a, jnp.float64) for a in (q, k, v))

        def f(q, k, v, sel):
            y, _ = attention.attend_xla(q, k, v, jnp.asarray(seg), jnp.asarray(seg), jnp.asarray(pos),
                                        jnp.asarray(pos), sel=sel, bos_k=jnp.asarray(bos))
            return jnp.sum(jnp.sin(y))

        check_grads(lambda q, k, v: f(q, k, v, jnp.float64(1.0)), args, order=1, modes=["rev"], eps=1e-6)
        # the selection depends on head 0's queries and keys only: their gradient changes
        g_on = jax.grad(f, argnums=(0, 1))(*args, jnp.float64(1.0))
        g_off = jax.grad(f, argnums=(0, 1))(*args, jnp.float64(0.0))
        assert float(jnp.max(jnp.abs(g_on[0][:, :, 0] - g_off[0][:, :, 0]))) > 1e-3


def test_causality_of_selective_attention():
    """Perturbing a future token leaves every earlier position's logits unchanged."""
    cfg = tiny_cfg(sel_attn=True)
    params = random_params(cfg, seed=7)
    consts = M.make_consts(cfg)
    feats, _, _ = M.split_batch(make_batch(cfg, B=2, T=24, seed=8))
    feats = {kk: np.asarray(a) for kk, a in feats.items()}
    f = jitted("logits_fn", cfg)
    base = np.asarray(f(params, consts, feats))
    for j in (2, 11, 17):
        pert = {kk: a.copy() for kk, a in feats.items()}
        pert["tokens"][:, j] = (pert["tokens"][:, j] + 9) % (cfg.vocab_size - 16)
        out = np.asarray(f(params, consts, pert))
        np.testing.assert_array_equal(out[:, :j], base[:, :j])
        assert np.max(np.abs(out[:, j:] - base[:, j:])) > 1e-4


def test_selective_attention_needs_the_dense_path():
    with pytest.raises(ValueError, match="attn_impl"):
        M.check_cfg(tiny_cfg(sel_attn=True, attn_impl="splash"))
    q = jnp.zeros((1, 4, 2, 8))
    seg = pos = jnp.zeros((1, 4), jnp.int32)
    with pytest.raises(ValueError, match="Tq == Tk"):
        attention.attend_xla(q[:, :2], q, q, seg[:, :2], seg, pos[:, :2], pos, sel=jnp.float32(1.0),
                             bos_k=jnp.zeros((1, 4), bool))
    cfg = tiny_cfg(sel_attn=True)
    ctx = M.make_ctx(M.make_consts(cfg), {"tokens": jnp.zeros((1, 4), jnp.int32)}, cfg)
    assert ctx.bos.dtype == jnp.bool_ and ctx.bos.shape == (1, 4) and float(ctx.sel) == 1.0
    plain = M.make_ctx(M.make_consts(tiny_cfg()), {"tokens": jnp.zeros((1, 4), jnp.int32)}, tiny_cfg())
    assert plain.bos is None and plain.sel is None
