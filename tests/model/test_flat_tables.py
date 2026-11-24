"""bes.train.train.FLAT_TABLES (execution-level speed switch, off by default): the token tables
are handed to the loss function as reshapes of [V, W] matrices and their gradients accumulated
in that shape. Only reshapes, so the accumulation's gradients, loss, weight and metrics are the
same bit for bit, with and without pieces (micro_split), for the tiny form of configs/v9.py
(token neurons, output gains, value embeddings) in every table-gradient mode."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _model_testlib import make_batch, tiny_cfg
from bes.model import model as M
from bes.train import train as TR


def _batch(cfg, a: int = 2, rows: int = 2, t: int = 16) -> dict:
    parts = [make_batch(cfg, B=rows, T=t, seed=7 + i) for i in range(a)]
    return {k: jnp.asarray(np.stack([p[k] for p in parts])) for k in parts[0]}


def _run(mcfg, params, consts, batch, flat: bool, split: int):
    old = TR.FLAT_TABLES
    TR.FLAT_TABLES = flat
    try:
        fn = jax.jit(lambda p, c, b: TR.accumulate(M.loss_fn, p, c, b, mcfg, split))
        return jax.device_get(fn(params, consts, batch))
    finally:
        TR.FLAT_TABLES = old


@pytest.mark.parametrize("table_grad", ["take", "sorted", "onehot"])
def test_flat_tables_change_no_value(table_grad):
    mcfg = tiny_cfg(gain_o=True, tok_neurons=4, tok_dim=16, table_grad=table_grad, sorted_chunk=8, remat=True)
    assert TR.FLAT_TABLES is False                       # off by default
    params = M.init_params(mcfg, jax.random.key(3))
    # non-zero pages and gains, so that every path carries a gradient
    keys = iter(jax.random.split(jax.random.key(5), 8))
    params["page"]["val"] = 0.1 * jax.random.normal(next(keys), params["page"]["val"].shape)
    params["gain"]["o"] = 0.1 * jax.random.normal(next(keys), params["gain"]["o"].shape)
    flat_leaves = [p for p in jax.tree.leaves(params) if p.ndim >= 3 and p.shape[0] == mcfg.vocab_size]
    assert len(flat_leaves) >= 3                         # page/key, page/val, gain/o (and emb/ve)
    consts = M.make_consts(mcfg)
    batch = _batch(mcfg)
    for split in (1, 2):
        ref = _run(mcfg, params, consts, batch, False, split)
        got = _run(mcfg, params, consts, batch, True, split)
        assert jax.tree.structure(ref) == jax.tree.structure(got)
        for a, b in zip(jax.tree.leaves(ref), jax.tree.leaves(got)):
            assert np.shape(a) == np.shape(b)
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        assert float(np.max(np.abs(ref[0]["page"]["key"]))) > 0.0
        assert float(np.max(np.abs(ref[0]["gain"]["o"]))) > 0.0
