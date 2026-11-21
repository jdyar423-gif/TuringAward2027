"""Scatter-free lookup tables against numpy references (values and gradients)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bes.config import REG_FIELDS
from bes.model import tables


def _np_gather_sum(table, idx, w):
    rows = table.shape[0]
    idx = np.clip(idx, 0, rows - 1)
    return np.einsum("...k,...kw->...w", w, table[idx])


def _np_table_grad(table, idx, w, g):
    """Gradient of sum(g * gather_sum(table)) w.r.t. the table, by explicit scatter-add."""
    rows = table.shape[0]
    idx = np.clip(idx, 0, rows - 1).reshape(-1, idx.shape[-1])
    w = w.reshape(-1, w.shape[-1])
    g = g.reshape(-1, g.shape[-1])
    out = np.zeros_like(table)
    for n in range(idx.shape[0]):
        for k in range(idx.shape[1]):
            out[idx[n, k]] += w[n, k] * g[n]
    return out


def test_multi_hot_matches_numpy():
    rng = np.random.default_rng(0)
    idx = rng.integers(-1, 9, size=(3, 4, 5))
    w = rng.integers(0, 2, size=(3, 4, 5)).astype(np.float32)
    want = np.zeros((3, 4, 7), np.float32)
    for i in np.ndindex(3, 4):
        for k in range(5):
            if 0 <= idx[i][k] < 7:
                want[i][idx[i][k]] += w[i][k]
    np.testing.assert_array_equal(np.asarray(tables.multi_hot(jnp.asarray(idx), jnp.asarray(w), 7)), want)


@pytest.mark.parametrize("mode", ["onehot", "take"])
@pytest.mark.parametrize("chunk_elems", [1 << 24, 64])   # 64 -> many chunks with padding
def test_big_lookup_values_and_gradients(mode, chunk_elems, monkeypatch):
    monkeypatch.setattr(tables, "ONEHOT_CHUNK_ELEMS", chunk_elems)
    rng = np.random.default_rng(1)
    rows, width = 13, 5
    table = rng.normal(size=(rows, width)).astype(np.float32)
    idx = rng.integers(-1, rows + 2, size=(2, 7, 3))          # duplicates and out-of-range
    w = rng.integers(0, 2, size=(2, 7, 3)).astype(np.float32)
    g = rng.normal(size=(2, 7, width)).astype(np.float32)
    f = lambda t: tables.big_lookup(t, jnp.asarray(idx), jnp.asarray(w), mode=mode)
    np.testing.assert_allclose(np.asarray(f(jnp.asarray(table))), _np_gather_sum(table, idx, w), atol=1e-6)
    grad = jax.grad(lambda t: jnp.sum(f(t) * g))(jnp.asarray(table))
    np.testing.assert_allclose(np.asarray(grad), _np_table_grad(table, idx, w, g), atol=1e-5)
    # single-row form
    i1 = rng.integers(0, rows, size=(4, 6))
    g1 = rng.normal(size=(4, 6, width)).astype(np.float32)
    f1 = lambda t: tables.big_lookup(t, jnp.asarray(i1), mode=mode)
    np.testing.assert_allclose(np.asarray(f1(jnp.asarray(table))), table[i1], atol=0)
    grad1 = jax.grad(lambda t: jnp.sum(f1(t) * g1))(jnp.asarray(table))
    np.testing.assert_allclose(np.asarray(grad1), _np_table_grad(table, i1[..., None], np.ones(i1.shape + (1,)), g1),
                               atol=1e-5)


def test_small_lookup_values_and_gradients():
    rng = np.random.default_rng(2)
    table = rng.normal(size=(11, 4)).astype(np.float32)
    idx = rng.integers(0, 11, size=(3, 5, 6))
    w = rng.integers(0, 2, size=(3, 5, 6)).astype(np.float32)
    g = rng.normal(size=(3, 5, 4)).astype(np.float32)
    f = lambda t: tables.small_lookup(t, jnp.asarray(idx), jnp.asarray(w))
    np.testing.assert_allclose(np.asarray(f(jnp.asarray(table))), _np_gather_sum(table, idx, w), atol=1e-5)
    grad = jax.grad(lambda t: jnp.sum(f(t) * g))(jnp.asarray(table))
    np.testing.assert_allclose(np.asarray(grad), _np_table_grad(table, idx, w, g), atol=1e-5)


def test_reg_embed_matches_per_field_sum():
    rng = np.random.default_rng(3)
    d = 6
    p_reg = {name: rng.normal(size=(card, d)).astype(np.float32) for name, card in REG_FIELDS}
    reg = np.stack([rng.integers(0, card + 3, size=(2, 5)) for _, card in REG_FIELDS], -1)  # some out of range
    want = sum(p_reg[name][np.clip(reg[..., f], 0, card - 1)] for f, (name, card) in enumerate(REG_FIELDS))
    got = tables.reg_embed({k: jnp.asarray(v) for k, v in p_reg.items()}, jnp.asarray(reg, jnp.int16), 1.0)
    np.testing.assert_allclose(np.asarray(got), want, atol=1e-5)
    off = tables.reg_embed({k: jnp.asarray(v) for k, v in p_reg.items()}, jnp.asarray(reg), 0.0)
    assert float(jnp.max(jnp.abs(off))) == 0.0


def test_onehot_backward_memory_is_chunked():
    """The table gradient never materialises the full [n, rows] multi-hot matrix."""
    rows, width, n = 8192, 16, 16384
    tbl = jnp.zeros((rows, width))
    idx = jnp.zeros((n,), jnp.int32)
    compiled = jax.jit(jax.grad(lambda t: jnp.sum(tables.big_lookup(t, idx)))).lower(tbl).compile()
    temp = compiled.memory_analysis().temp_size_in_bytes
    assert temp < n * rows * 4 / 4, temp
