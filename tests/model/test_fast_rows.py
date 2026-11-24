"""bes.model.model.FAST_ROWS (execution-level speed switch ``rows``, off by default): (a) the key
preparation of the token-neuron rows keeps its residuals instead of being rematerialised under
remat, (b) the loop scan closes over lane-dense views of the rows and each step reshapes them
back. Same operations on the same values, so the trainer's accumulation (bes.train.train.accumulate:
loss, every gradient leaf, weight and metrics) is compared with the switch off and on, with
remat, micro_split 1 and 2, in float32 and bfloat16, alone and with FLAT_TABLES. The tiny model
has rho * dk = 4 * 32 = 128 (dk not a multiple of 128 lanes, the product is: part (b) applies,
as for the 8 x 192 rows of configs/v9.py)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _model_testlib import make_batch, tiny_cfg
from bes.model import model as M
from bes.train import train as TR


def _cfg(dtype: str):
    return tiny_cfg(gain_o=True, tok_neurons=4, tok_dim=32, table_grad="take", remat=True, act_dtype=dtype)


def _params(mcfg):
    params = M.init_params(mcfg, jax.random.key(3))
    # non-zero pages and gains, so that every path carries a gradient
    keys = iter(jax.random.split(jax.random.key(5), 8))
    params["page"]["val"] = 0.1 * jax.random.normal(next(keys), params["page"]["val"].shape)
    params["gain"]["o"] = 0.1 * jax.random.normal(next(keys), params["gain"]["o"].shape)
    return params


def _batch(cfg, a: int = 2, rows: int = 2, t: int = 16) -> dict:
    parts = [make_batch(cfg, B=rows, T=t, seed=7 + i) for i in range(a)]
    return {k: jnp.asarray(np.stack([p[k] for p in parts])) for k in parts[0]}


def _run(mcfg, params, consts, batch, rows: bool, split: int, flat: bool = False):
    old = (M.FAST_ROWS, TR.FLAT_TABLES)
    M.FAST_ROWS, TR.FLAT_TABLES = rows, flat
    try:
        fn = jax.jit(lambda p, c, b: TR.accumulate(M.loss_fn, p, c, b, mcfg, split))
        return jax.device_get(fn(params, consts, batch))
    finally:
        M.FAST_ROWS, TR.FLAT_TABLES = old


def _max_diff(ref, got) -> float:
    assert jax.tree.structure(ref) == jax.tree.structure(got)
    worst = 0.0
    for a, b in zip(jax.tree.leaves(ref), jax.tree.leaves(got)):
        assert np.shape(a) == np.shape(b) and np.asarray(a).dtype == np.asarray(b).dtype
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        worst = max(worst, float(np.max(np.abs(a - b), initial=0.0)))
    return worst


def test_lane_dense_carrier_applies_to_the_tiny_rows():
    assert M.FAST_ROWS is False                          # off by default
    assert M._lane_dense(jnp.zeros((2, 3, 4, 32))).shape == (2, 3, 128)
    assert M._lane_dense(jnp.zeros((2, 3, 8, 192))).shape == (2, 3, 1536)    # v9's rows
    assert M._lane_dense(jnp.zeros((2, 3, 1536))).shape == (2, 3, 1536)     # already lane-dense
    assert M._lane_dense(jnp.zeros((2, 3, 4, 16))).shape == (2, 3, 4, 16)   # 64 lanes: kept
    assert M._lane_dense(jnp.zeros(())).shape == ()
    mcfg = _cfg("float32")
    params, consts = _params(mcfg), M.make_consts(mcfg)
    feats, _, _ = M.split_batch({k: v[0] for k, v in _batch(mcfg).items()})

    def scan_inputs(rows: bool) -> list:
        old = M.FAST_ROWS
        M.FAST_ROWS = rows
        try:
            jaxpr = jax.make_jaxpr(lambda p: M.forward(p, consts, feats, mcfg))(params)
        finally:
            M.FAST_ROWS = old
        scan = [e for e in jaxpr.jaxpr.eqns if e.primitive.name == "scan"]
        assert len(scan) == 1
        return [tuple(v.aval.shape) for v in scan[0].invars]      # closed-over values, carry, xs

    off, on = scan_inputs(False), scan_inputs(True)
    assert (2, 16, 4, 32) in off and (2, 16, 4, 32) not in on
    assert on.count((2, 16, 128)) == off.count((2, 16, 128)) + 2 * mcfg.n_blocks   # key, val per block


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_fast_rows_change_no_value(dtype):
    mcfg = _cfg(dtype)
    params, consts = _params(mcfg), M.make_consts(mcfg)
    batch = _batch(mcfg)
    for split in (1, 2):
        ref = _run(mcfg, params, consts, batch, False, split)
        got = _run(mcfg, params, consts, batch, True, split)
        diff = _max_diff(ref, got)
        print(f"{dtype} micro_split={split}: largest difference over loss, gradients, weight, metrics {diff:.3g}")
        assert diff == 0.0
        assert float(np.max(np.abs(ref[0]["page"]["key"]))) > 0.0
        assert float(np.max(np.abs(ref[0]["page"]["val"]))) > 0.0
        assert float(np.max(np.abs(ref[0]["page"]["pin"]))) > 0.0


def test_fast_rows_with_flat_tables_change_no_value():
    mcfg = _cfg("bfloat16")
    params, consts = _params(mcfg), M.make_consts(mcfg)
    batch = _batch(mcfg)
    ref = _run(mcfg, params, consts, batch, False, 2, flat=False)
    got = _run(mcfg, params, consts, batch, True, 2, flat=True)
    assert _max_diff(ref, got) == 0.0
