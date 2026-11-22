"""The ``"sorted"`` table gradient: equal to the scatter-add (``"take"``) gradient in every edge
case of the specification, scatter-free in the lowered program, and equal to the one-hot mode
inside the model."""
from __future__ import annotations

import dataclasses
import functools
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _model_testlib import make_batch, random_params, tiny_cfg, value_and_grad_fn
from bes.model import model as M, tables
from bes.roles import leaves_with_paths


def _np_grad(rows: int, idx: np.ndarray, w: np.ndarray, g: np.ndarray) -> np.ndarray:
    """``G[v] = sum_{n,k: clip(idx[n,k]) = v} w[n,k] g[n]`` by an explicit loop (float64)."""
    idx = np.clip(idx, 0, rows - 1).reshape(-1, idx.shape[-1])
    w = w.reshape(-1, w.shape[-1])
    g = g.reshape(-1, g.shape[-1]).astype(np.float64)
    out = np.zeros((rows, g.shape[-1]))
    for n in range(idx.shape[0]):
        for k in range(idx.shape[1]):
            out[idx[n, k]] += w[n, k] * g[n]
    return out


@functools.lru_cache(maxsize=None)
def _grad_fn(mode: str, chunk: int, weighted: bool):
    """Jitted table gradient of ``sum(g * big_lookup(table, idx[, w]))``."""
    def loss(table, idx, w, g):
        out = tables.big_lookup(table, idx, w if weighted else None, mode=mode, chunk=chunk)
        return jnp.sum(out * g)
    return jax.jit(jax.grad(loss))


def _cases(rows: int, chunk: int):
    """Index patterns ``(name, idx [n, K])`` for the section 5 edge cases (``chunk`` small)."""
    rng = np.random.default_rng(rows * 31 + chunk)
    out = []
    for k in (1, 3):
        n = 5 * chunk + 3                                   # not a multiple of the chunk
        out.append((f"random K={k}", rng.integers(0, rows, size=(n, k))))
        out.append((f"duplicates K={k}", rng.choice([2, 5], size=(n, k))))
        out.append((f"out of range K={k}", rng.integers(-3, rows + 4, size=(n, k))))
        # one id fills several chunks: a run spanning three or more chunks, neighbours around it
        long = np.concatenate([np.full(3, 1), np.full(4 * chunk + 1, 4), np.full(2, rows - 1)])
        out.append((f"long run K={k}", rng.permutation(np.repeat(long, k)).reshape(-1, k)))
        out.append((f"absent ids K={k}", rng.choice([0, rows - 1], size=(n, k))))
        out.append((f"n < chunk K={k}", rng.integers(0, rows, size=(max(1, chunk // 2), k))))
        out.append((f"one row K={k}", np.full((1, k), rows // 2)))
        out.append((f"run on a chunk boundary K={k}",
                    np.repeat(np.arange(rows), chunk)[: 3 * chunk * k].reshape(-1, k)))
    return out


@pytest.mark.parametrize("chunk", [1, 4, 7])
@pytest.mark.parametrize("weighted", [False, True])
def test_sorted_gradient_equals_take_in_the_edge_cases(chunk, weighted):
    rows, width = 9, 5
    rng = np.random.default_rng(chunk)
    table = jnp.asarray(rng.normal(size=(rows, width)), jnp.float32)
    for name, idx in _cases(rows, chunk):
        idx = np.asarray(idx)
        k = idx.shape[-1]
        if not weighted and k > 1:
            continue
        lookup_idx = idx if weighted else idx[:, 0]
        w = (rng.random(idx.shape) < 0.7).astype(np.float32) if weighted else np.ones(idx.shape, np.float32)
        g = rng.normal(size=(idx.shape[0], width)).astype(np.float32)
        args = (table, jnp.asarray(lookup_idx), jnp.asarray(w), jnp.asarray(g))
        got = np.asarray(_grad_fn("sorted", chunk, weighted)(*args))
        want = np.asarray(_grad_fn("take", chunk, weighted)(*args))
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5, err_msg=name)
        np.testing.assert_allclose(got, _np_grad(rows, idx, w, g), rtol=1e-5, atol=1e-5, err_msg=name)
        if "absent" in name:
            assert np.all(got[1:rows - 1] == 0.0), name


def test_sorted_gradient_with_groups_of_chunks(monkeypatch):
    """More chunks than one one-hot group holds: the ``lax.map`` path gives the same gradient."""
    rows, width, chunk = 11, 3, 4
    rng = np.random.default_rng(5)
    idx = rng.integers(0, rows, size=(61, 3))
    w = (rng.random(idx.shape) < 0.8).astype(np.float32)
    g = rng.normal(size=(61, width)).astype(np.float32)
    want = _np_grad(rows, idx, w, g)
    for elems in (1 << 24, 2 * chunk * chunk, 3 * chunk * chunk):      # 1, 2 and 3 chunks a group
        monkeypatch.setattr(tables, "ONEHOT_CHUNK_ELEMS", elems)
        got = tables.sorted_table_grad(jnp.asarray(idx, jnp.int32), jnp.asarray(w), jnp.asarray(g),
                                       rows, chunk)
        np.testing.assert_allclose(np.asarray(got), want, rtol=1e-5, atol=1e-5, err_msg=str(elems))


def test_sorted_forward_is_the_gather_and_bfloat16_operands_round_only_the_cotangent():
    rows, width = 13, 6
    rng = np.random.default_rng(6)
    table = jnp.asarray(rng.normal(size=(rows, width)), jnp.float32)
    idx = jnp.asarray(rng.integers(-1, rows + 1, size=(4, 7)))
    out = tables.big_lookup(table, idx, mode="sorted", chunk=4)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(table)[np.clip(np.asarray(idx), 0, rows - 1)])
    g = jnp.asarray(rng.normal(size=(4, 7, width)), jnp.float32)
    f = lambda t, mode, dt: jnp.sum(tables.big_lookup(t, idx, mode=mode, dtype=dt, chunk=4) * g)  # noqa: E731
    ref = jax.grad(f)(table, "take", jnp.float32)
    bf = jax.grad(f)(table, "sorted", jnp.bfloat16)
    np.testing.assert_allclose(np.asarray(bf), np.asarray(ref), rtol=2e-2, atol=2e-2)
    assert bf.dtype == jnp.float32


def _scatter_ops(text: str) -> int:
    """Scatter operations in StableHLO (``stablehlo.scatter``) or compiled HLO (``scatter(``)
    text; the metadata of either may name Python functions, so op syntax is matched."""
    return len(re.findall(r"stablehlo\.scatter|\sscatter\(", text))


def test_lowered_sorted_gradient_has_no_scatter():
    """The lowered (StableHLO) and compiled (HLO) table gradient of the sorted mode contain no
    scatter; positive control: the take mode does."""
    rows, width = 16, 4
    idx = jnp.asarray(np.random.default_rng(7).integers(0, rows, size=(37, 3)), jnp.int32)
    w = jnp.ones((37, 3), jnp.float32)
    g = jnp.ones((37, width), jnp.float32)
    table = jnp.zeros((rows, width), jnp.float32)
    for mode, want_scatter in (("sorted", False), ("take", True)):
        fn = jax.jit(jax.grad(lambda t, mode=mode: jnp.sum(
            tables.big_lookup(t, idx, w, mode=mode, chunk=8) * g)))
        lowered = fn.lower(table)
        n_lowered, n_compiled = _scatter_ops(lowered.as_text()), _scatter_ops(lowered.compile().as_text())
        if want_scatter:
            assert n_lowered > 0
        else:
            assert n_lowered == 0 and n_compiled == 0, (n_lowered, n_compiled)


def test_model_gradients_sorted_equal_onehot_and_the_loss_gradient_has_no_scatter():
    """Every leaf's gradient with ``table_grad="sorted"`` (a tiny chunk, so the runs of the
    token, value-embedding, candidate and gain tables span several chunks) equals the one-hot
    mode; the compiled loss gradient of the sorted mode (gains with two phases and selective
    attention on) has no scatter."""
    cfg = tiny_cfg(gain_ffn=40, gain_o=True, gain_v=True, gain_phases=2, sel_attn=True)
    params = random_params(cfg, seed=31)
    consts = M.make_consts(cfg)
    batch = make_batch(cfg, seed=32)
    batch["tokens"][:, 2:10] = 7                               # one id over a long run
    (l1, _), g1 = value_and_grad_fn(cfg)(params, consts, batch)
    sorted_cfg = dataclasses.replace(cfg, table_grad="sorted", sorted_chunk=4)
    (l2, _), g2 = value_and_grad_fn(sorted_cfg)(params, consts, batch)
    assert float(l1) == float(l2)
    for (path, a), b in zip(leaves_with_paths(g1), jax.tree.leaves(g2)):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), atol=1e-6, rtol=1e-5, err_msg=path)
    hlo = jax.jit(jax.grad(lambda p: M.loss_fn(p, consts, batch, sorted_cfg)[0])).lower(
        params).compile().as_text()
    assert hlo.count(" scatter(") == 0


def test_unknown_table_grad_mode_is_refused():
    with pytest.raises(ValueError, match="table_grad"):
        tables.big_lookup(jnp.zeros((4, 2)), jnp.zeros((3,), jnp.int32), mode="sort")
    with pytest.raises(ValueError, match="table_grad"):
        M.check_cfg(tiny_cfg(table_grad="sort"))
    with pytest.raises(ValueError, match="sorted_chunk"):
        M.check_cfg(tiny_cfg(table_grad="sorted", sorted_chunk=0))
    assert tables.SORTED_CHUNK == tiny_cfg().sorted_chunk == 512
