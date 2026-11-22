"""Head-wise attention output gate (``attn_gate``, part B of the v9 specification): zero leaves
and the switch at 0 are bit-identical to the model without the gate, an independent numpy
reference (per-loop biases; the token output gain after the gate), the gate on the ``attend_fn``
hook path and after the sink-mass metric, float64 gradients of both leaves."""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _model_testlib import init, jitted, make_batch, random_params, tiny_cfg, to_numpy
from _v9_testlib import check_directional_grads, reference_logits, sigmoid, switches_on, tree_f64
from bes.model import attention, model as M
from bes.model.layers import rmsnorm


def _without_gate(params: dict) -> dict:
    attn = {k: v for k, v in params["core"]["attn"].items() if not k.startswith("gate_")}
    return {**params, "core": {**params["core"], "attn": attn}}


def _zero_gate(params: dict) -> dict:
    attn = dict(params["core"]["attn"])
    attn.update(gate_w=jnp.zeros_like(attn["gate_w"]), gate_b=jnp.zeros_like(attn["gate_b"]))
    return {**params, "core": {**params["core"], "attn": attn}}


def _feats(cfg, seed, **kw):
    feats, _, _ = M.split_batch(make_batch(cfg, seed=seed, **kw))
    return {k: np.asarray(v) for k, v in feats.items()}


# ------------------------------------------------------------------------- parameters
def test_gate_leaves_roles_and_switch_defaults():
    cfg = tiny_cfg(attn_gate=True)
    params = init(cfg)
    at = params["core"]["attn"]
    assert at["gate_w"].shape == (cfg.n_blocks, cfg.d_model, cfg.n_heads)
    assert at["gate_b"].shape == (cfg.n_loops, cfg.n_blocks, cfg.n_heads)
    assert all(a.dtype == jnp.float32 and float(jnp.max(jnp.abs(a))) == 0.0 for a in (at["gate_w"], at["gate_b"]))
    plain = init(tiny_cfg())
    assert "gate_w" not in plain["core"]["attn"] and "gate_b" not in plain["core"]["attn"]
    stripped = _without_gate(params)                  # every other leaf as without the flag
    assert jax.tree.structure(stripped) == jax.tree.structure(plain)
    for a, b in zip(jax.tree.leaves(stripped), jax.tree.leaves(plain)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert float(M.make_consts(cfg)["enable"]["gate"]) == 1.0
    assert float(M.make_consts(tiny_cfg())["enable"]["gate"]) == 0.0
    assert float(M.make_consts(cfg, **M.switches_off())["enable"]["gate"]) == 0.0
    feats = {"tokens": np.zeros((1, 4), np.int32)}
    assert float(M.make_ctx(M.make_consts(cfg, gate=0.5), feats, cfg).gate) == 0.5
    assert M.make_ctx(M.make_consts(tiny_cfg()), feats, tiny_cfg()).gate is None
    pa = M.block_params(params, cfg)[1]["attn"]
    np.testing.assert_array_equal(np.asarray(pa["gate_b"]), np.asarray(at["gate_b"][:, 1]))
    assert pa["gate_w"].dtype == jnp.float32 and "gate_w" not in M.block_params(plain, tiny_cfg())[1]["attn"]


# ------------------------------------------------------------------------ exactness
@pytest.mark.parametrize("flags", [dict(), dict(gain_o=True, gain_phases=2)])
def test_zero_gate_is_bit_identical_to_the_model_without_it(flags):
    """Same other leaves, gate parameters at their zero initialisation: logits and loss equal
    the model without ``attn_gate`` bit for bit."""
    cfg = tiny_cfg(attn_gate=True, **flags)
    base = tiny_cfg(**flags)
    params = _zero_gate(random_params(cfg, seed=1))
    batch = make_batch(cfg, seed=2)
    feats, _, _ = M.split_batch(batch)
    a = np.asarray(jitted("logits_fn", cfg)(params, M.make_consts(cfg), feats))
    b = np.asarray(jitted("logits_fn", base)(_without_gate(params), M.make_consts(base), feats))
    np.testing.assert_array_equal(a, b)
    la, _ = jitted("loss_fn", cfg)(params, M.make_consts(cfg), batch)
    lb, _ = jitted("loss_fn", base)(_without_gate(params), M.make_consts(base), batch)
    assert float(la) == float(lb)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_gate_switch_off_is_bit_identical_to_the_gate_ignored(dtype):
    cfg = tiny_cfg(act_dtype=dtype, attn_gate=True)
    base = tiny_cfg(act_dtype=dtype)
    params = random_params(cfg, seed=3, scale=0.1 if dtype == "bfloat16" else 0.3)
    feats = _feats(cfg, 4)
    off = np.asarray(jitted("logits_fn", cfg)(params, M.make_consts(cfg, gate=0.0), feats))
    ign = np.asarray(jitted("logits_fn", base)(_without_gate(params), M.make_consts(base), feats))
    np.testing.assert_array_equal(off, ign)
    on = np.asarray(jitted("logits_fn", cfg)(params, M.make_consts(cfg), feats))
    assert np.max(np.abs(on - off)) > 1e-2


# ------------------------------------------------------------------- numpy reference
@pytest.mark.parametrize("flags", [dict(), dict(gain_o=True, gain_v=True, gain_phases=2)])
def test_gate_matches_the_numpy_reference(flags):
    """Random gate weights and per-loop biases (three loops), alone and in front of the output
    token gain (two phases), against the float64 numpy reference."""
    cfg = tiny_cfg(clock_planes=0, attn_gate=True, **flags)
    params = random_params(cfg, seed=5)
    batch = make_batch(cfg, B=2, T=20, seed=6, seg_starts=((0, 9), (0,)))
    tokens, seg = batch["tokens"][:, :-1], batch["seg"]
    got = np.asarray(jitted("logits_fn", cfg)(params, switches_on(cfg), {"tokens": tokens, "seg": seg}))
    want = reference_logits(to_numpy(params), tokens, seg, cfg)
    np.testing.assert_allclose(got, want, atol=1e-4, rtol=1e-4)
    plain = reference_logits(to_numpy(_without_gate(params)), tokens, seg, cfg)
    assert np.max(np.abs(want - plain)) > 1e-2
    # the bias of the last loop alone matters (the loop index reaches the block)
    moved = to_numpy(params)
    moved["core"]["attn"]["gate_b"][-1] += 1.0
    other = reference_logits(moved, tokens, seg, cfg)
    assert np.max(np.abs(other - want)) > 1e-3
    got_moved = np.asarray(jitted("logits_fn", cfg)(jax.tree.map(jnp.asarray, moved), switches_on(cfg),
                                                    {"tokens": tokens, "seg": seg}))
    np.testing.assert_allclose(got_moved, other, atol=1e-4, rtol=1e-4)


def test_gate_on_the_hook_path_and_after_the_sink_mass_metric():
    """``attention.attention`` with the kernel replaced by an ``attend_fn`` hook is gated the same
    way; the gated output is ``W_o (y * gate)`` with ``y`` the kernel output; the sink-mass
    metric is that of the ungated output; the loop index is required."""
    cfg = tiny_cfg(attn_gate=True)
    params = random_params(cfg, seed=7)
    consts = M.make_consts(cfg, gate=0.75)                # sink on: the metric is not trivial
    feats = _feats(cfg, 8)
    ctx = M.make_ctx(consts, feats, cfg)
    pb = M.block_params(params, cfg)[1]
    pa, wo = pb["attn"], pb["wo"]
    rng = np.random.default_rng(9)
    x = rmsnorm(jnp.asarray(rng.standard_normal(feats["tokens"].shape + (cfg.d_model,)), jnp.float32))
    loop = 2
    out, mass = attention.attention(pa, wo, x, ctx, None, cfg, loop=loop)

    def hook(q, k, v):
        return attention.attend_xla(q, k, v, ctx.seg, ctx.seg, ctx.pos, ctx.pos, ctx.sink)

    out_hook, _ = attention.attention(pa, wo, x, ctx, None, cfg, attend_fn=hook, loop=jnp.int32(loop))
    np.testing.assert_allclose(np.asarray(out_hook), np.asarray(out), rtol=1e-6, atol=1e-6)
    y, _ = attention.heads(pa, x, ctx, None, cfg)
    b_, t_ = x.shape[:2]
    xn, w, bias = np.asarray(x, np.float64), np.asarray(pa["gate_w"], np.float64), np.asarray(pa["gate_b"], np.float64)
    gate = 1.0 + 0.75 * (2.0 * sigmoid(xn @ w / xn.shape[-1] + bias[loop]) - 1.0)   # [B, T, H]
    want = (np.asarray(y, np.float64) * gate[..., None]).reshape(b_, t_, -1) @ np.asarray(wo, np.float64)
    np.testing.assert_allclose(np.asarray(out), want, rtol=1e-5, atol=1e-5)
    g = np.asarray(attention.head_gate(pa, x, loop, 0.75))
    np.testing.assert_allclose(g, gate, rtol=1e-6, atol=1e-6)
    assert np.all((g > 0) & (g < 2))
    plain = {k: v for k, v in pa.items() if not k.startswith("gate_")}
    out_plain, mass_plain = attention.attention(plain, wo, x, ctx, None, cfg)
    assert float(mass) == float(mass_plain) and float(mass) > 0
    assert np.max(np.abs(np.asarray(out) - np.asarray(out_plain))) > 1e-3
    with pytest.raises(ValueError, match="loop"):
        attention.attention(pa, wo, x, ctx, None, cfg)


# ------------------------------------------------------------------------- gradients
def test_gate_gradients_float64():
    """Reverse-mode gradients of the loss with respect to ``gate_w`` and ``gate_b`` (random
    non-zero point, output token gain on) against central differences in float64; the gradients
    are non-zero at the zero initialisation too (the gate's slope at 0 is 1/2)."""
    cfg32 = tiny_cfg(vocab_size=64, d_model=32, n_heads=2, head_dim=16, ve_dim=16, head_rank=16,
                     n_loops=2, cand_dim=8, critic_hidden=8, loss_chunk=7, attn_gate=True, gain_o=True)
    params32 = random_params(cfg32, seed=11)
    batch = make_batch(cfg32, B=2, T=10, seed=12, seg_starts=((0,), (0, 6)))
    with jax.enable_x64(True):
        cfg = dataclasses.replace(cfg32, act_dtype="float64")
        params = tree_f64(params32)
        consts = M.make_consts(cfg)
        rest = params["core"]["attn"]

        def with_gate(g):
            return {**params, "core": {**params["core"], "attn": {**rest, **g}}}

        f = jax.jit(lambda g: M.loss_fn(with_gate(g), consts, batch, cfg)[0])
        grad_f = jax.jit(jax.grad(f))
        point = {"gate_w": rest["gate_w"], "gate_b": rest["gate_b"]}
        assert f(point).dtype == jnp.float64
        d = check_directional_grads(f, grad_f, point, seed=13)
        assert all(abs(v) > 0 for v in d.values())
        zero = grad_f(jax.tree.map(jnp.zeros_like, point))
        assert all(float(jnp.max(jnp.abs(g))) > 0 for g in zero.values())
