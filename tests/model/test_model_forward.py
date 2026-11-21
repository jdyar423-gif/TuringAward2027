"""Forward pass: shapes and dtypes, causality, segment isolation, inert initial feature paths,
remat, activation dtypes, critic and runtime switches."""
from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _model_testlib import init, jitted, make_batch, random_params, tiny_cfg, value_and_grad_fn
from bes.config import CAND_SLOTS, N_CAND_FEAT, N_CLOCKS, N_REG
from bes.model import model as M


def test_shapes_and_dtypes():
    cfg = tiny_cfg()
    params = random_params(cfg, seed=0)
    consts = M.make_consts(cfg)
    batch = make_batch(cfg, B=2, T=24, seed=1)
    feats, _, _ = M.split_batch(batch)
    h = jitted("forward_hidden", cfg)(params, consts, feats)
    assert h.shape == (2, 24, cfg.d_model) and h.dtype == jnp.float32
    z = jitted("logits_fn", cfg)(params, consts, feats)
    assert z.shape == (2, 24, cfg.vocab_size) and z.dtype == jnp.float32
    loss, met = jitted("loss_fn", cfg)(params, consts, batch)
    assert loss.shape == () and loss.dtype == jnp.float32 and np.isfinite(float(loss))
    assert met["sink_mass"].shape == (cfg.n_blocks,)
    assert met["loss_share_by_role"].shape == (6,)
    assert all(v.dtype == jnp.float32 for v in met.values())
    # one decoding-style position: hidden [B, d] with a tape [B, K]
    one = jax.jit(functools.partial(M.logits_from_hidden, cfg=cfg))
    zs = one(params, consts, h[:, 5], {"cand_tok": batch["cand_tok"][:, 5],
                                       "cand_feat": batch["cand_feat"][:, 5]})
    np.testing.assert_allclose(np.asarray(zs), np.asarray(z[:, 5]), atol=1e-5)
    assert M.critic(params, h[:, -1]).shape == (2,)
    assert M.critic(params, h).shape == (2, 24)
    assert M.critic(params, h).dtype == jnp.float32


def test_batch_feature_dtypes_of_the_spec_are_accepted():
    """int16 registers and int8 candidate features as delivered by the loader."""
    cfg = tiny_cfg()
    batch = make_batch(cfg, seed=2)
    assert batch["reg"].dtype == np.int16 and batch["cand_feat"].dtype == np.int8
    assert batch["reg"].shape[-1] == N_REG and batch["clock"].shape[-1] == N_CLOCKS
    assert batch["cand_tok"].shape[-1] == CAND_SLOTS and batch["cand_feat"].shape[-1] == N_CAND_FEAT
    loss, _ = jitted("loss_fn", cfg)(random_params(cfg, seed=0), M.make_consts(cfg), batch)
    assert np.isfinite(float(loss))


def _perturb_position(feats: dict, cfg, j: int, rng) -> dict:
    out = {k: np.array(v) for k, v in feats.items()}
    out["tokens"][:, j] = (out["tokens"][:, j] + 1 + rng.integers(0, 50)) % (cfg.vocab_size - 16)
    out["reg"][:, j] = (out["reg"][:, j] + 1) % 2
    out["clock"][:, j] += 7
    out["cand_tok"][:, j] = rng.integers(0, cfg.vocab_size - 16, size=out["cand_tok"][:, j].shape)
    out["cand_feat"][:, j, :, 1:] = (out["cand_feat"][:, j, :, 1:] + 1) % 8
    return out


def test_causality_with_all_features_on():
    cfg = tiny_cfg()
    params = random_params(cfg, seed=3)
    consts = M.make_consts(cfg)
    feats, _, _ = M.split_batch(make_batch(cfg, B=2, T=24, seed=4))
    feats = {k: np.asarray(v) for k, v in feats.items()}
    f = jitted("logits_fn", cfg)
    base = np.asarray(f(params, consts, feats))
    rng = np.random.default_rng(0)
    for j in (1, 6, 13, 23):
        pert = np.asarray(f(params, consts, _perturb_position(feats, cfg, j, rng)))
        np.testing.assert_array_equal(pert[:, :j], base[:, :j])
        assert np.max(np.abs(pert[:, j] - base[:, j])) > 1e-4


def test_segment_isolation_and_shift_invariance():
    cfg = tiny_cfg()
    params = random_params(cfg, seed=5)
    consts = M.make_consts(cfg)
    L0, T = 9, 24
    batch = make_batch(cfg, B=1, T=T, seed=6, seg_starts=((0, L0),))
    feats, _, _ = M.split_batch(batch)
    feats = {k: np.asarray(v) for k, v in feats.items()}
    f = jitted("logits_fn", cfg)
    base = np.asarray(f(params, consts, feats))
    # perturb everything in segment 0 except its BOS: segment 1 is unchanged (bit-exact)
    pert = {k: v.copy() for k, v in feats.items()}
    pert["tokens"][0, 1:L0] = (pert["tokens"][0, 1:L0] + 3) % (cfg.vocab_size - 16)
    pert["reg"][0, 1:L0] = 0
    pert["cand_tok"][0, 1:L0] = 5
    out = np.asarray(f(params, consts, pert))
    np.testing.assert_array_equal(out[:, L0:], base[:, L0:])
    assert np.max(np.abs(out[:, 1:L0] - base[:, 1:L0])) > 1e-4
    # segment 1 alone at the start of a row gives the same logits (relative positions only)
    alone = {k: v[:, L0:] for k, v in feats.items()}
    alone["seg"] = np.zeros_like(alone["seg"])
    np.testing.assert_allclose(np.asarray(jitted("logits_fn", cfg)(params, consts, alone)),
                               base[:, L0:], atol=2e-4, rtol=1e-4)


def test_zero_initialised_paths_at_init():
    """At initialisation the register and candidate-input paths add exactly zero, and the
    output law gives every active slot the mass exp(-mass_offset)."""
    cfg = tiny_cfg()
    params = init(cfg, seed=7)
    batch = make_batch(cfg, seed=8)
    feats, _, _ = M.split_batch(batch)
    f = jitted("logits_fn", cfg)
    on = np.asarray(f(params, M.make_consts(cfg), feats))
    off = np.asarray(f(params, M.make_consts(cfg, reg=0.0, cand_in=0.0), feats))
    np.testing.assert_array_equal(on, off)
    # the output law at init: p = (q + exp(-off) * #slots holding v) / (1 + exp(-off) * #active)
    logq = np.asarray(f(params, M.make_consts(cfg, cand_out=0.0), feats), np.float64)
    q = np.exp(logq - np.log(np.exp(logq).sum(-1, keepdims=True)))
    e = np.exp(-cfg.mass_offset)
    cand = batch["cand_tok"]
    p = q.copy()
    for b, t, k in zip(*np.nonzero(cand >= 0)):
        p[b, t, cand[b, t, k]] += e
    p /= (1.0 + e * (cand >= 0).sum(-1))[..., None]
    np.testing.assert_allclose(on, np.log(p), atol=2e-5, rtol=0)
    assert np.max(np.abs(on - logq)) > 1e-3
    # the sink and confidence paths are live from the start and switch off exactly
    plain = np.asarray(f(params, M.make_consts(cfg, **M.switches_off()), feats))
    assert np.max(np.abs(plain - on)) > 0.0


def test_remat_gives_the_same_loss_and_gradients():
    cfg = tiny_cfg()
    params = random_params(cfg, seed=9)
    consts = M.make_consts(cfg)
    batch = make_batch(cfg, seed=10)
    out = {remat: value_and_grad_fn(dataclasses.replace(cfg, remat=remat))(params, consts, batch)
           for remat in (False, True)}
    np.testing.assert_allclose(float(out[True][0][0]), float(out[False][0][0]), rtol=1e-6)
    for a, b in zip(jax.tree.leaves(out[True][1]), jax.tree.leaves(out[False][1])):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_activation_dtype_paths_run(dtype):
    cfg = tiny_cfg(act_dtype=dtype)
    params = random_params(cfg, seed=11, scale=0.1)
    consts = M.make_consts(cfg)
    batch = make_batch(cfg, seed=12)
    (loss, met), grads = value_and_grad_fn(cfg)(params, consts, batch)
    assert np.isfinite(float(loss))
    assert all(np.all(np.isfinite(np.asarray(g))) for g in jax.tree.leaves(grads))
    assert all(g.dtype == jnp.float32 for g in jax.tree.leaves(grads))
    if dtype == "bfloat16":
        ref, _ = jitted("loss_fn", tiny_cfg())(params, consts, batch)
        assert abs(float(loss) - float(ref)) < 0.05 * float(ref)


def test_make_consts_switches():
    cfg = tiny_cfg()
    c = M.make_consts(cfg)
    assert set(c["enable"]) == set(M.SWITCHES) | {"kind", "src"}
    assert all(float(c["enable"][s]) == 1.0 for s in M.SWITCHES)
    assert np.asarray(c["enable"]["src"]).tolist() == [1.0] * 8
    c = M.make_consts(cfg, kind_4=0, sink=False, src_5=0.25)
    assert np.asarray(c["enable"]["kind"]).tolist() == [1, 1, 1, 1, 0, 1, 1, 1]
    assert np.asarray(c["enable"]["src"]).tolist() == [1, 1, 1, 1, 1, 0.25, 1, 1]
    assert float(c["enable"]["sink"]) == 0.0
    off = M.make_consts(cfg, **M.switches_off())
    assert all(float(off["enable"][s]) == 0.0 for s in M.SWITCHES)
    assert float(M.make_consts(tiny_cfg(clock_planes=0))["enable"]["clock"]) == 0.0
    assert float(M.make_consts(dataclasses.replace(cfg, use_reg=False))["enable"]["reg"]) == 0.0
    with pytest.raises(KeyError):
        M.make_consts(cfg, bogus=1)
    assert c["rope_freq"].shape == (1 + N_CLOCKS, (cfg.head_dim - cfg.bias_dims) // 2)


def test_attend_hook_and_cache_style_queries():
    """``attend_fn`` reproduces the default attention, and queries of a suffix attending over
    all keys (as a KV-cache decode step does) give the same rows as the full sequence."""
    from bes.model import attention
    cfg = tiny_cfg()
    params = random_params(cfg, seed=15)
    consts = M.make_consts(cfg)
    feats, _, _ = M.split_batch(make_batch(cfg, B=2, T=16, seed=16))
    ctx = M.make_ctx(consts, feats, cfg)
    pb = M.block_params(params, cfg)[1]
    u = M.embed(params, consts, feats, cfg)
    ref, _ = M.block(pb, u, ctx, None, cfg)

    def hook(q, k, v):
        y, _ = attention.attend_xla(q, k, v, ctx.seg, ctx.seg, ctx.pos, ctx.pos, ctx.sink)
        return y, (k, v)

    out, (k, v) = M.block(pb, u, ctx, None, cfg, attend_fn=hook)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))
    assert k.shape == v.shape == (2, 16, cfg.n_heads, cfg.head_dim)
    q, k, v = attention.qkv(pb["attn"], M.rmsnorm(u, pb["g1"]), ctx, None, cfg)
    full, _ = attention.attend_xla(q, k, v, ctx.seg, ctx.seg, ctx.pos, ctx.pos)
    sl = slice(10, 16)
    part, _ = attention.attend_xla(q[:, sl], k, v, ctx.seg[:, sl], ctx.seg, ctx.pos[:, sl], ctx.pos)
    np.testing.assert_allclose(np.asarray(part), np.asarray(full[:, sl]), atol=1e-6)


def test_kind_switch_removes_that_candidate_kind():
    """Disabling a candidate kind equals emptying the slots of that kind."""
    cfg = tiny_cfg()
    params = random_params(cfg, seed=13)
    batch = make_batch(cfg, seed=14)
    feats, _, _ = M.split_batch(batch)
    f = jitted("logits_fn", cfg)
    a = np.asarray(f(params, M.make_consts(cfg, kind_3=0.0), feats))
    emptied = dict(feats)
    ct = np.array(feats["cand_tok"])
    ct[np.asarray(feats["cand_feat"])[..., 0] == 3] = -1
    emptied["cand_tok"] = ct
    b = np.asarray(f(params, M.make_consts(cfg), emptied))
    np.testing.assert_allclose(a, b, atol=1e-5)
