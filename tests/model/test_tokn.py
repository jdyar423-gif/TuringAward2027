"""bes.model.layers.FAST_TOKN (execution-level speed switch ``tokn``, off by default): the token
neurons' rows are handed to a block with the tokens in lanes (``[B, rho, dk, T]``) and the two
per-token contractions are float32 products and sums in that layout. The products are the same;
the float32 sums over ``dk`` (pre-activations; cotangent of the neurons' weights ``s``) and over
``rho`` (the neurons' output; cotangent of the projected input ``z``) may add in another order.

Compared with the switch off and on: the trainer's accumulation (bes.train.train.accumulate, loss,
weight, every gradient leaf) on the tiny form of configs/v9.py (token neurons, output gains, value
embeddings, remat, non-zero pages and gains), micro_split 1 and 2, in float32 (tight) and bfloat16
(one bfloat16 rounding step of the leaf's largest entry: a float32 reordering can flip one
bfloat16 rounding of an activation or a cotangent, which the backward pass carries on). The
measured differences are printed (pytest -s) and quoted in the switch's report.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _model_testlib import make_batch, tiny_cfg
from bes.model import fast
from bes.model import layers as L
from bes.model import model as M
from bes.train import train as TR


def _batch(cfg, a: int = 2, rows: int = 2, t: int = 16) -> dict:
    parts = [make_batch(cfg, B=rows, T=t, seed=11 + i) for i in range(a)]
    return {k: jnp.asarray(np.stack([p[k] for p in parts])) for k in parts[0]}


def _run(mcfg, params, consts, batch, on: bool, split: int):
    before = fast.apply({"tokn": on})
    try:
        fn = jax.jit(lambda p, c, b: TR.accumulate(M.loss_fn, p, c, b, mcfg, split)[:3])
        return jax.device_get(fn(params, consts, batch))
    finally:
        fast.apply(before)


def _setup(act_dtype: str):
    mcfg = tiny_cfg(gain_o=True, tok_neurons=4, tok_dim=16, table_grad="take", remat=True,
                    act_dtype=act_dtype)
    params = M.init_params(mcfg, jax.random.key(3))
    keys = iter(jax.random.split(jax.random.key(5), 8))
    params["page"]["val"] = 0.1 * jax.random.normal(next(keys), params["page"]["val"].shape)
    params["gain"]["o"] = 0.1 * jax.random.normal(next(keys), params["gain"]["o"].shape)
    return mcfg, params, M.make_consts(mcfg), _batch(mcfg)


def test_switch_is_registered_and_off():
    assert L.FAST_TOKN is False
    assert fast.FLAGS["tokn"] == ("bes.model.layers", "FAST_TOKN")
    assert fast.parse("tokn=1") == {"tokn": True} and fast.current()["tokn"] is False


def test_token_neurons_lanes_layer_float32():
    """The layer alone (forward and every input's cotangent), float32: rounding only."""
    k = iter(jax.random.split(jax.random.key(0), 8))
    b, t, d, rho, dk = 2, 24, 32, 4, 16
    x = jax.random.normal(next(k), (b, t, d))
    pin = jax.random.normal(next(k), (d, dk)) / np.sqrt(d)
    pout = jax.random.normal(next(k), (dk, d)) / np.sqrt(dk)
    key = L.rmsnorm(jax.random.normal(next(k), (b, t, rho, dk)))
    val = jax.random.normal(next(k), (b, t, rho, dk))
    ct = jax.random.normal(next(k), (b, t, d))

    def ref(x, pin, pout, key, val):
        return jnp.sum(ct * L.token_neurons(x, pin, pout, key, val, 1.0, jnp.float32))

    def lanes(x, pin, pout, key, val):
        kt, vt = jnp.moveaxis(key, 1, -1), jnp.moveaxis(val, 1, -1)
        return jnp.sum(ct * L._token_neurons_lanes(x, pin, pout, kt, vt, 1.0, jnp.float32))

    y_r = L.token_neurons(x, pin, pout, key, val, 1.0, jnp.float32)
    y_g = L._token_neurons_lanes(x, pin, pout, jnp.moveaxis(key, 1, -1), jnp.moveaxis(val, 1, -1),
                                 1.0, jnp.float32)
    assert float(jnp.max(jnp.abs(y_g - y_r))) <= 1e-6 * float(jnp.max(jnp.abs(y_r)))
    r = jax.grad(ref, argnums=(0, 1, 2, 3, 4))(x, pin, pout, key, val)
    g = jax.grad(lanes, argnums=(0, 1, 2, 3, 4))(x, pin, pout, key, val)
    for a, c in zip(r, g):
        assert a.shape == c.shape
        assert float(jnp.max(jnp.abs(a - c))) <= 1e-5 * float(jnp.max(jnp.abs(a)))


@pytest.mark.parametrize("act_dtype", ["float32", "bfloat16"])
def test_tokn_accumulate_matches(act_dtype):
    mcfg, params, consts, batch = _setup(act_dtype)
    for split in (1, 2):
        ref = _run(mcfg, params, consts, batch, False, split)
        got = _run(mcfg, params, consts, batch, True, split)
        assert jax.tree.structure(ref) == jax.tree.structure(got)
        (g_r, l_r, w_r), (g_g, l_g, w_g) = ref, got
        np.testing.assert_array_equal(np.asarray(w_r), np.asarray(w_g))
        dl = abs(float(l_g) - float(l_r)) / abs(float(l_r))
        worst, n_exact, n = 0.0, 0, 0
        for (path, a), b in zip(jax.tree_util.tree_leaves_with_path(g_r), jax.tree.leaves(g_g)):
            a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
            assert a.shape == b.shape
            scale = float(np.max(np.abs(a)))
            rel = float(np.max(np.abs(a - b))) / scale if scale > 0 else float(np.max(np.abs(b)))
            n += 1
            n_exact += rel == 0.0
            if rel > worst:
                worst, worst_path = rel, jax.tree_util.keystr(path)
            # float32: reordered float32 sums only; bfloat16: one rounding step (2^-7) of the leaf
            tol = 2e-5 if act_dtype == "float32" else 2.0 ** -7
            assert rel <= tol, (jax.tree_util.keystr(path), rel)
        print(f"\n[tokn {act_dtype} micro_split={split}] loss rel {dl:.2e}; {n_exact}/{n} gradient "
              f"leaves bit for bit; worst leaf {worst:.2e}" + (f" ({worst_path})" if worst else ""))
        assert dl <= (1e-6 if act_dtype == "float32" else 1e-4)
        assert float(np.max(np.abs(g_r["page"]["key"]))) > 0.0
        assert float(np.max(np.abs(g_r["page"]["val"]))) > 0.0
