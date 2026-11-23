"""The head-major attention path of the splash implementation (``attention.attention_major``)
and the other TPU-speed rewrites compute the values of the reference composition: queries, keys
and values against ``attention.qkv`` transposed; the whole sub-layer (kernel, sink-mass metric,
head gate, token gains, ``W_o``) and its gradients against the reference path on the same
kernels; the model's loss against the dense path with every gain form; the feed-forward layer
with stored pre-activations; lanes carried as a tuple."""
from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _model_testlib import make_batch, random_params, tiny_cfg
from bes.model import attention, controller, head, layers, model as M
from bes.model.layers import rmsnorm


def _setup(cfg, seed=3, T=24, **consts_kw):
    params = random_params(cfg, seed=seed)
    consts = M.make_consts(cfg, **consts_kw)
    feats = {k: jnp.asarray(v) for k, v in M.split_batch(make_batch(cfg, T=T, seed=seed + 1))[0].items()}
    ctx = M.make_ctx(consts, feats, cfg)
    rng = np.random.default_rng(seed + 2)
    x = rmsnorm(jnp.asarray(rng.standard_normal(feats["tokens"].shape + (cfg.d_model,)), jnp.float32))
    return params, consts, feats, ctx, x.astype(layers.act_dtype(cfg))


def _ve(params, feats, cfg, loop=1):
    ve = M.value_embeddings(params, feats["tokens"], cfg)
    return None if ve is None else ve[loop]


CASES = {
    "default": {},
    "no sink": dict(use_sink=False),
    "no confidence": dict(use_conf=False),
    "neither, plain rotary": dict(use_sink=False, use_conf=False, clock_planes=0),
    "three reserved dims": dict(bias_dims=4, clock_planes=0),
    "value embedding on every head": dict(ve_dim=64),
    "no value embedding": dict(ve_dim=0),
}


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("act", ["float32", "bfloat16"])
def test_queries_keys_values_are_the_reference_transposed(case, act):
    cfg = tiny_cfg(attn_impl="splash", act_dtype=act, **CASES[case])
    params, consts, feats, ctx, x = _setup(cfg)
    assert ctx.rot_c is not None and ctx.rot_c.shape == (2, 1, 24, cfg.head_dim)
    tol = 1e-5 if act == "float32" else 2e-2          # one bfloat16 rounding of values of order 1
    for i, pb in enumerate(M.block_params(params, cfg)):
        pa = pb["attn"]
        assert pa["wq_a"].shape == (cfg.d_model, cfg.n_heads, cfg.head_dim)
        ve = _ve(params, feats, cfg) if i == 0 else None
        want = attention.qkv(pa, x, ctx, ve, cfg)
        got = attention.qkv_major(pa, x, ctx, ve, cfg)
        # the same without the matrices and tables made outside the loop
        bare = {k: v for k, v in pa.items() if k not in ("wq_a", "wq_b", "wk_a", "wk_b", "wv_m")}
        lazy = attention.qkv_major(bare, x, ctx._replace(rot_c=None, rot_s=None), ve, cfg)
        for name, w, g, lz in zip("qkv", want, got, lazy):
            assert g.dtype == w.dtype == layers.act_dtype(cfg) and g.shape == (2, cfg.n_heads, 24, cfg.head_dim)
            np.testing.assert_allclose(np.asarray(jnp.swapaxes(g, 1, 2), np.float32), np.asarray(w, np.float32),
                                       atol=tol, rtol=tol, err_msg=f"{case} block {i} {name}")
            np.testing.assert_array_equal(np.asarray(lz, np.float32), np.asarray(g, np.float32))
    if act == "float32" and cfg.use_sink:             # the sink and the reserved dims are exact
        q, k, v = attention.qkv_major(M.block_params(params, cfg)[1]["attn"], x, ctx, None, cfg)
        dc = cfg.head_dim - cfg.bias_dims
        on = np.asarray(ctx.sink)[:, None, :] > 0
        assert on.any()
        np.testing.assert_array_equal(np.asarray(q[..., dc]), 1.0)
        np.testing.assert_array_equal(np.asarray(q[..., dc + 1:]), 0.0)
        np.testing.assert_array_equal(np.asarray(k[..., :dc])[np.broadcast_to(on, k.shape[:3])], 0.0)
        np.testing.assert_array_equal(np.asarray(k[..., dc + 1:]), 0.0)


def _reference(pa, wo, x, ctx, ve, cfg, **kw):
    """The reference composition on the same kernels (the ``attend_fn`` hook selects it)."""
    sink = ctx.sink if "sink_logit" in pa else None
    return attention.attention(pa, wo, x, ctx, ve, cfg,
                               attend_fn=lambda q, k, v: attention.self_attend(q, k, v, ctx, sink, cfg), **kw)


@pytest.mark.parametrize("gain", ["none", "rows", "major", "phase"])
@pytest.mark.parametrize("gate", [False, True])
def test_the_sub_layer_matches_the_reference_composition(gain, gate):
    phases = 2 if gain == "phase" else 1
    on = gain != "none"
    cfg = tiny_cfg(attn_impl="splash", attn_gate=gate, gain_o=on, gain_v=on, gain_phases=phases, n_loops=4)
    params, consts, feats, ctx, x = _setup(cfg, seed=5, **({"gate": 0.75} if gate else {}))
    b, t = feats["tokens"].shape
    gains = M.gain_rows(params, consts, feats["tokens"], cfg)
    loop = jnp.int32(2) if gate else None
    for i, pb in enumerate(M.block_params(params, cfg)):
        ref_gain = M.block_gain(gains, i, jnp.int32(1) if gain == "phase" else None) or {}
        new_gain = dict(ref_gain)
        if gain == "major":
            major = M.major_gains(gains, cfg)
            assert major["o"].shape == (cfg.n_blocks, b, cfg.n_heads, t, cfg.head_dim)
            new_gain = {name: attention.MajorGain(rows[i]) for name, rows in major.items()}
        ve = _ve(params, feats, cfg) if i == 0 else None

        def run(fn, g, pa, wo, xx):
            out, aux = fn(pa, wo, xx, ctx, ve, cfg, gain_v=g.get("v"), gain_o=g.get("o"), loop=loop)
            return jnp.sum(jnp.sin(out)), (out, aux)

        new = jax.value_and_grad(functools.partial(run, attention.attention, new_gain), argnums=(0, 1, 2),
                                 has_aux=True)(pb["attn"], pb["wo"], x)
        ref = jax.value_and_grad(functools.partial(run, _reference, ref_gain), argnums=(0, 1, 2),
                                 has_aux=True)(pb["attn"], pb["wo"], x)
        (_, (out, aux)), g_new = new
        (_, (out_ref, aux_ref)), g_ref = ref
        np.testing.assert_allclose(np.asarray(out), np.asarray(out_ref), atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(float(aux), float(aux_ref), atol=1e-6)
        assert float(aux) > 0.0                        # sink and confidence on: the metric is live
        # gradients with respect to the input, W_o and the parameters the reference reads
        np.testing.assert_allclose(np.asarray(g_new[2]), np.asarray(g_ref[2]), atol=2e-5, rtol=2e-4)
        np.testing.assert_allclose(np.asarray(g_new[1]), np.asarray(g_ref[1]), atol=2e-5, rtol=2e-4)
        for name in ("wq", "wk", "wv", "q_gain", "sink_logit", "sink_value") + (("ve_gate",) if i == 0 else ()) \
                + (("gate_w", "gate_b") if gate else ()):
            total = np.asarray(g_new[0][name], np.float64)
            if name in ("wq", "wk"):                   # the path reads the derived matrices
                h, hd, dc = cfg.n_heads, cfg.head_dim, cfg.head_dim - cfg.bias_dims
                a = np.asarray(g_new[0][name + "_a"], np.float64)[..., :dc]
                bb = np.asarray(g_new[0][name + "_b"], np.float64)[..., :dc]
                p = dc // 2
                total = total + (a + np.concatenate([bb[..., p:], bb[..., :p]], -1)).reshape(cfg.d_model, h * dc)
            elif name == "wv":
                total = total + np.asarray(g_new[0]["wv_m"], np.float64).reshape(total.shape)
            np.testing.assert_allclose(total, np.asarray(g_ref[0][name], np.float64), atol=2e-5, rtol=2e-4,
                                       err_msg=f"block {i} {name}")


def test_selective_attention_is_refused_and_the_hook_keeps_the_reference_layout():
    cfg = tiny_cfg(attn_impl="splash")
    params, consts, feats, ctx, x = _setup(cfg)
    pb = M.block_params(params, cfg)[0]
    with pytest.raises(ValueError, match="dense path"):
        attention.attention(pb["attn"], pb["wo"], x, ctx._replace(sel=jnp.float32(1.0)), None, cfg)
    seen = {}

    def hook(q, k, v):
        seen["shape"] = q.shape
        return v.astype(jnp.float32), jnp.zeros((), jnp.float32)
    attention.attention(pb["attn"], pb["wo"], x, ctx, None, cfg, attend_fn=hook)
    assert seen["shape"] == (2, 24, cfg.n_heads, cfg.head_dim)


@pytest.mark.slow
@pytest.mark.parametrize("extras", [dict(), dict(gain_o=True, tok_neurons=2, tok_dim=16),
                                    dict(gain_o=True, gain_v=True, gain_ffn=8, gain_phases=2, attn_gate=True)])
def test_model_loss_and_gradients_match_the_dense_path(extras):
    base = tiny_cfg(n_loops=4, **extras)
    params = random_params(base, seed=11)
    consts = M.make_consts(base)
    batch = make_batch(base, B=2, T=40, seed=12, seg_starts=((0,), (0, 17)))
    out = {}
    for impl in ("xla", "splash"):
        cfg = dataclasses.replace(base, attn_impl=impl, remat=impl == "splash")
        out[impl] = jax.jit(jax.value_and_grad(functools.partial(M.loss_fn, cfg=cfg), has_aux=True))(
            params, consts, batch)
    (l_x, m_x), g_x = out["xla"]
    (l_s, m_s), g_s = out["splash"]
    np.testing.assert_allclose(float(l_s), float(l_x), rtol=2e-5)
    for a, b in zip(jax.tree.leaves(g_s), jax.tree.leaves(g_x)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=3e-5, rtol=2e-3)


def test_feed_forward_stores_pre_activations_in_the_activation_dtype():
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((2, 5, 16)), jnp.float32)
    w_up = jnp.asarray(rng.standard_normal((16, 64)) * 0.25, jnp.float32)
    w_down = jnp.asarray(rng.standard_normal((64, 16)) * 0.125, jnp.float32)
    gain = jnp.asarray(rng.standard_normal((2, 5, 8)) * 0.1, jnp.float32)
    z = np.square(np.maximum(np.asarray(x, np.float64) @ np.asarray(w_up, np.float64), 0.0))
    want = z @ np.asarray(w_down, np.float64)
    np.testing.assert_allclose(np.asarray(layers.mlp(x, w_up, w_down, jnp.float32)), want, rtol=1e-4, atol=1e-5)
    zg = z.copy()
    zg[..., :8] *= 1.0 + np.asarray(gain, np.float64)
    np.testing.assert_allclose(np.asarray(layers.mlp(x, w_up, w_down, jnp.float32, gain=gain)),
                               zg @ np.asarray(w_down, np.float64), rtol=1e-4, atol=1e-5)
    # bfloat16: the float32 result of the same bfloat16 operands, to the rounding of the
    # stored pre-activation (2^-9 per neuron, about 2^-9 / sqrt(d_ffn) of the output scale)
    xb, ub, db = (a.astype(jnp.bfloat16) for a in (x, w_up, w_down))
    got = layers.mlp(xb, ub, db, jnp.bfloat16)
    assert got.dtype == jnp.float32
    zb = np.square(np.maximum(np.asarray(xb, np.float64) @ np.asarray(ub, np.float64), 0.0))
    ref = zb @ np.asarray(db, np.float64)
    assert np.max(np.abs(np.asarray(got, np.float64) - ref)) < 0.02 * np.sqrt(np.mean(ref ** 2))


def test_lanes_as_a_tuple_and_as_a_stack_agree():
    rng = np.random.default_rng(1)
    stack = jnp.asarray(rng.standard_normal((3, 2, 4, 8)), jnp.float32)
    lanes = tuple(stack[n] for n in range(3))
    alpha = jnp.asarray(rng.standard_normal(3), jnp.float32)
    mix = jnp.asarray(rng.standard_normal((3, 3)), jnp.float32)
    beta = jnp.asarray(rng.standard_normal(3), jnp.float32)
    s = jnp.asarray(rng.standard_normal((2, 4, 8)), jnp.float32)
    g = jnp.asarray(rng.standard_normal(8), jnp.float32)
    np.testing.assert_array_equal(np.asarray(controller.combine(alpha, lanes, jnp.float32)),
                                  np.asarray(controller.combine(alpha, stack, jnp.float32)))
    up_t, up_s = controller.update(mix, beta, lanes, s, jnp.float32), controller.update(mix, beta, stack, s, jnp.float32)
    assert isinstance(up_t, tuple) and up_s.shape == stack.shape
    np.testing.assert_array_equal(np.asarray(jnp.stack(up_t)), np.asarray(up_s))
    np.testing.assert_array_equal(np.asarray(head.final_hidden(lanes, g)), np.asarray(head.final_hidden(stack, g)))
    out, ys = controller.run_loops(lambda ln, x: (controller.update(mix, beta, ln, x * s, jnp.float32), x),
                                   s, jnp.arange(3, dtype=jnp.float32), 3, remat=True)
    assert isinstance(out, tuple) and len(out) == 3 and out[0].shape == s.shape and ys.shape == (3,)


def test_keeping_the_kernel_outputs_changes_no_value_and_runs_no_forward_kernel_twice():
    """``remat_keep="kernel"``: the same loss and gradients as full rematerialisation; the
    gradient program holds the forward kernels once (the forward pass) instead of twice (again
    in the backward pass of the rematerialised loop step)."""
    base = tiny_cfg(n_loops=3, attn_impl="splash", remat=True)
    params = random_params(base, seed=21)
    consts = M.make_consts(base)
    batch = make_batch(base, B=2, T=40, seed=22, seg_starts=((0,), (0, 17)))
    out, kernels = {}, {}
    for keep in ("", "kernel"):
        cfg = dataclasses.replace(base, remat_keep=keep)
        fn = jax.value_and_grad(lambda p, c=cfg: M.loss_fn(p, consts, batch, c)[0])
        out[keep] = jax.jit(fn)(params)
        kernels[keep] = str(jax.make_jaxpr(fn)(params)).count("pallas_call")
    np.testing.assert_allclose(float(out["kernel"][0]), float(out[""][0]), rtol=1e-6)
    for a, b in zip(jax.tree.leaves(out["kernel"][1]), jax.tree.leaves(out[""][1])):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-6, rtol=1e-4)
    # the printed program names the forward kernel in the forward body and, without the
    # policy, again in the backward body next to the backward kernel
    assert kernels["kernel"] < kernels[""], kernels
    with pytest.raises(ValueError, match="remat_keep"):
        M.check_cfg(dataclasses.replace(base, remat_keep="everything"))
