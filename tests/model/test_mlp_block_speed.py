"""Speed switches of the feed-forward sub-layer, the norms and the residual plumbing
(``bes.model.layers.FAST_*``): each one computes the values of the reference code path.

Every switch is compared off against on for the loss, the metrics and every gradient leaf of
``value_and_grad(loss_fn)`` on a tiny configuration with the live run's features (splash
attention in interpret mode, bfloat16 activations, rematerialised loops, sink, candidate tape in
and out, registers, value embeddings, ``table_grad="take"``), and in float32. The tolerance is
zero: ``assert_array_equal`` (which counts ``-0.0 == 0.0``, the one difference a switch may
make)."""
from __future__ import annotations

import contextlib
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _model_testlib import make_batch, random_params, tiny_cfg
from bes.model import layers, model as M

#: the live run's model (configs/final.py with table_grad=take) at a tiny size
LIVE = dict(attn_impl="splash", act_dtype="bfloat16", remat=True, clock_planes=0, use_conf=False,
            table_grad="take")


@contextlib.contextmanager
def switch(name: str, value: bool):
    old = getattr(layers, name)
    setattr(layers, name, value)
    try:
        yield
    finally:
        setattr(layers, name, old)


def _value_and_grad(cfg, params, consts, batch):
    """A fresh trace every call (the switches are read at trace time); also returns the
    printed program, to check that the switch reached it."""
    def loss(p, c, b):
        return M.loss_fn(p, c, b, cfg)
    vg = jax.value_and_grad(loss, has_aux=True)
    return jax.jit(vg)(params, consts, batch), str(jax.make_jaxpr(vg)(params, consts, batch))


def _assert_same(a, b, what):
    (la, ma), ga = a
    (lb, mb), gb = b
    np.testing.assert_array_equal(np.asarray(la), np.asarray(lb), err_msg=f"{what}: loss")
    for k in ma:
        np.testing.assert_array_equal(np.asarray(ma[k]), np.asarray(mb[k]), err_msg=f"{what}: metric {k}")
    fa, fb = jax.tree_util.tree_flatten_with_path(ga)[0], jax.tree_util.tree_flatten_with_path(gb)[0]
    assert [p for p, _ in fa] == [p for p, _ in fb]
    for (path, x), (_, y) in zip(fa, fb):
        assert x.dtype == y.dtype and x.shape == y.shape
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y),
                                      err_msg=f"{what}: gradient {jax.tree_util.keystr(path)}")


def _model_case(name, cfg, seed=31):
    params = random_params(cfg, seed=seed)
    consts = M.make_consts(cfg)
    batch = make_batch(cfg, B=2, T=40, seed=seed + 1, seg_starts=((0,), (0, 17)))
    with switch(name, False):
        ref, ref_text = _value_and_grad(cfg, params, consts, batch)
    with switch(name, True):
        got, got_text = _value_and_grad(cfg, params, consts, batch)
    assert np.isfinite(float(ref[0][0]))
    assert got_text != ref_text, f"{name} did not change the traced program"
    return ref, got


# ------------------------------------------------------------------- FAST_RELU2
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_relu2_has_the_values_and_the_gradient_of_the_reference_expression(dtype):
    rng = np.random.default_rng(0)
    h = rng.standard_normal((3, 7, 64)).astype(np.float32)
    h[0, 0, :8] = 0.0
    h[0, 1, :8] = -0.0
    h = jnp.asarray(h, dtype)
    ct = jnp.asarray(rng.standard_normal(h.shape), layers.hi(dtype))

    def ref(x):
        return jnp.square(jax.nn.relu(x.astype(layers.hi(x.dtype))))

    y_ref, vjp_ref = jax.vjp(ref, h)
    y_got, vjp_got = jax.vjp(layers.relu2, h)
    assert y_got.dtype == y_ref.dtype == layers.hi(dtype)
    np.testing.assert_array_equal(np.asarray(y_got), np.asarray(y_ref))
    (d_ref,), (d_got,) = vjp_ref(ct), vjp_got(ct)
    assert d_got.dtype == d_ref.dtype == dtype
    np.testing.assert_array_equal(np.asarray(d_got, np.float32), np.asarray(d_ref, np.float32))
    assert np.all(np.asarray(d_got, np.float32)[np.asarray(h, np.float32) <= 0] == 0.0)
    # under jit and inside a checkpointed scan (the model's setting), too
    def total(x, f):
        return jnp.sum(jax.lax.scan(jax.checkpoint(lambda c, xx: (c, jnp.sum(f(xx) * ct[0]))), 0.0, x)[1])

    g_ref = jax.jit(jax.grad(lambda x: total(x, ref)))(h)
    g_got = jax.jit(jax.grad(lambda x: total(x, layers.relu2)))(h)
    np.testing.assert_array_equal(np.asarray(g_got, np.float32), np.asarray(g_ref, np.float32))


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_fast_relu2_feed_forward_values_and_gradients(dtype):
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal((2, 9, 32)), dtype)
    w_up = jnp.asarray(rng.standard_normal((32, 128)) * 0.2, dtype)
    w_down = jnp.asarray(rng.standard_normal((128, 32)) * 0.1, dtype)
    gain = jnp.asarray(rng.standard_normal((2, 9, 16)) * 0.1, jnp.float32)
    for g in (None, gain):
        def f(x, u, d, g=g):
            return jnp.sum(jnp.sin(layers.mlp(x, u, d, dtype, gain=g)))
        out, text = {}, {}
        for on in (False, True):
            with switch("FAST_RELU2", on):
                vg = jax.value_and_grad(f, argnums=(0, 1, 2))
                text[on] = str(jax.make_jaxpr(vg)(x, w_up, w_down))
                out[on] = jax.jit(vg)(x, w_up, w_down)
        assert "custom_vjp" in text[True] and "custom_vjp" not in text[False]
        ref, got = out[False], out[True]
        np.testing.assert_array_equal(np.asarray(got[0]), np.asarray(ref[0]))
        for a, b in zip(got[1], ref[1]):
            np.testing.assert_array_equal(np.asarray(a, np.float32), np.asarray(b, np.float32))


@pytest.mark.slow
@pytest.mark.parametrize("extras", [dict(LIVE), dict(LIVE, act_dtype="float32"),
                                    dict(LIVE, gain_ffn=8, gain_phases=2, remat=False)])
def test_fast_relu2_model_loss_and_gradients_are_unchanged(extras):
    cfg = tiny_cfg(n_loops=3, **extras)
    ref, got = _model_case("FAST_RELU2", cfg)
    _assert_same(ref, got, f"FAST_RELU2 {extras}")
