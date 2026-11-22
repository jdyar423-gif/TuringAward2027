"""Lookup tables without scatter.

Two primitives, both returning ``sum_k w[..., k] * table[idx[..., k]]``:

* :func:`small_lookup` (tables with few rows: registers, candidate features): the multi-hot
  matrix ``M[..., R]`` is built by compare/select and multiplied with the table, so the forward
  is a matmul and the gradient ``M^T @ g`` is a matmul as well.
* :func:`big_lookup` (vocabulary-sized tables): forward is a gather; the gradient of the table
  is ``M^T @ g`` computed in row chunks (``mode="onehot"``, a custom VJP, no scatter), the
  plain autodiff of the gather (``mode="take"``: XLA scatter-add) for benchmarking, or a
  matmul over chunks of the index-sorted rows (``mode="sorted"``, a custom VJP, no scatter;
  :func:`sorted_table_grad`) whose matmul cost is ``chunk / R`` of the one-hot mode.

Weights ``w`` are treated as constants (no gradient flows into them) and are meant to be 0/1
switches or masks; they are applied exactly (``x * 1 = x``, ``x * 0 = 0``). Indices are clipped
into ``[0, R)``. The multi-hot matrix is accumulated slot by slot with a static Python loop so
that XLA fuses it into one element-wise pass instead of materialising ``[..., K, R]``.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from bes.config import REG_FIELDS
from bes.model.layers import hi

F32 = jnp.float32
#: element budget of one multi-hot chunk in the one-hot backward (2^24 -> 2048 rows of 8192).
ONEHOT_CHUNK_ELEMS = 1 << 24
#: rows per chunk of the sorted backward when ``big_lookup`` gets no ``chunk``
#: (``ModelCfg.sorted_chunk`` has the same default)
SORTED_CHUNK = 512


def multi_hot(idx: jax.Array, w: jax.Array, rows: int, dtype=F32) -> jax.Array:
    """``M[..., r] = sum_k w[..., k] * (idx[..., k] == r)``.

    idx: int ``[..., K]`` (values outside ``[0, rows)`` never match); w: ``[..., K]``.
    Returns ``[..., rows]`` in ``dtype``.
    """
    iota = jnp.arange(rows, dtype=jnp.int32)
    idx = idx.astype(jnp.int32)
    w = w.astype(dtype)
    zero = jnp.zeros((), dtype)
    acc = jnp.where(idx[..., 0:1] == iota, w[..., 0:1], zero)
    for k in range(1, idx.shape[-1]):
        acc = acc + jnp.where(idx[..., k:k + 1] == iota, w[..., k:k + 1], zero)
    return acc


def small_lookup(table: jax.Array, idx: jax.Array, w: jax.Array, dtype=F32) -> jax.Array:
    """Weighted multi-row lookup of a small table as a multi-hot matmul.

    table: ``[R, W]``; idx: int ``[..., K]``; w: ``[..., K]`` (constant). Operands are cast
    to ``dtype`` with float32 accumulation; returns ``float32[..., W]``.
    """
    rows = table.shape[0]
    m = multi_hot(jnp.clip(idx.astype(jnp.int32), 0, rows - 1), jax.lax.stop_gradient(w), rows, dtype)
    return jnp.einsum("...r,rw->...w", m, table.astype(dtype), preferred_element_type=hi(dtype))


def _gather_sum(table: jax.Array, idx: jax.Array, w: jax.Array) -> jax.Array:
    """``[n, K]`` indices and weights -> ``[n, W]`` (forward of the big lookup)."""
    rows = jnp.take(table, idx, axis=0)          # [n, K, W]
    if idx.shape[1] == 1:
        return rows[:, 0] * w.astype(table.dtype)
    return jnp.einsum("nk,nkw->nw", w.astype(table.dtype), rows)


@functools.lru_cache(maxsize=None)
def _onehot_gather_sum(dtype_name: str):
    """Custom-VJP big lookup whose table gradient is a chunked multi-hot matmul."""
    dtype = jnp.dtype(dtype_name)

    @jax.custom_vjp
    def f(table, idx, w):
        return _gather_sum(table, idx, w)

    def fwd(table, idx, w):
        return _gather_sum(table, idx, w), (idx, w, jnp.zeros((table.shape[0], 0), table.dtype))

    def bwd(res, g):
        idx, w, shape_carrier = res
        rows = shape_carrier.shape[0]
        n, k = idx.shape
        chunk = max(1, min(n, ONEHOT_CHUNK_ELEMS // max(rows, 1)))
        n_chunks = -(-n // chunk)
        pad = n_chunks * chunk - n
        if pad:
            idx = jnp.pad(idx, ((0, pad), (0, 0)))
            w = jnp.pad(w, ((0, pad), (0, 0)))
            g = jnp.pad(g, ((0, pad), (0, 0)))
        width = g.shape[-1]

        def chunk_grad(i_c, w_c, g_c):
            m = multi_hot(i_c, w_c, rows, dtype)
            return jnp.einsum("cr,cw->rw", m, g_c.astype(dtype), preferred_element_type=hi(dtype))

        if n_chunks == 1:
            grad = chunk_grad(idx, w, g)
        else:
            def step(acc, xs):
                return acc + chunk_grad(*xs), None
            grad, _ = jax.lax.scan(
                step, jnp.zeros((rows, width), hi(dtype)),
                (idx.reshape(n_chunks, chunk, k), w.reshape(n_chunks, chunk, k),
                 g.reshape(n_chunks, chunk, width)))
        return grad.astype(shape_carrier.dtype), None, None

    f.defvjp(fwd, bwd)
    return f


def sorted_table_grad(idx: jax.Array, w: jax.Array, g: jax.Array, rows: int,
                      chunk: int = SORTED_CHUNK, dtype=F32) -> jax.Array:
    """Table gradient ``G[v] = sum_{n, k: idx[n, k] = v} w[n, k] g[n]`` without scatter.

    idx: int ``[n, K]`` row indices already clipped into ``[0, rows)``; w: float32 ``[n, K]``;
    g: ``[n, W]`` cotangent of the lookup; chunk: rows ``C`` per chunk; dtype: operand dtype
    of the matmul (``hi(dtype)`` accumulation).

    The ``m = n K`` pairs are sorted by index (``gs [m, W]``: the weighted cotangent rows in
    that order) and padded to ``n_chunks * C`` rows with the index ``rows`` (a run of its own)
    and zero rows. ``rank`` is the dense rank of the run of equal indices and ``local`` the rank
    relative to the first row of its chunk (``[0, C)``); ``part [n_chunks, C, W]`` is
    ``onehot(local)^T @ gs`` per chunk (groups of chunks in a ``lax.map`` so that the one-hot
    operand stays near ``ONEHOT_CHUNK_ELEMS`` elements), ``pref`` the prefix sums of the chunk
    totals. A run spanning chunks ``c1 < c2`` is the last run of ``c1``, every chunk in between
    and the first run of ``c2``:
    ``G[v] = part[c1, l1] + [c2 > c1] (part[c2, 0] + pref[c2 - 1] - pref[c1])`` with ``c1, c2``
    the chunks of the run's first and last row (``searchsorted``) and ``l1`` its local rank in
    ``c1``; rows that do not occur get 0. Sort, gathers, compares, matmuls and cumulative sums
    only. Returns ``hi(dtype)`` ``[rows, W]``.
    """
    ft = hi(dtype)
    n, k = idx.shape
    width = g.shape[-1]
    m = n * k
    c = max(1, int(chunk))
    n_chunks = -(-m // c)
    pad = n_chunks * c - m
    ids = idx.reshape(m).astype(jnp.int32)
    order = jnp.argsort(ids)
    ids_s = ids[order]
    gs = (g[order // k].astype(ft) * w.reshape(m)[order].astype(ft)[:, None]).astype(dtype)
    if pad:
        ids_s = jnp.concatenate([ids_s, jnp.full((pad,), rows, jnp.int32)])
        gs = jnp.concatenate([gs, jnp.zeros((pad, width), dtype)])
    new_run = jnp.concatenate([jnp.zeros((1,), jnp.int32),
                               (ids_s[1:] != ids_s[:-1]).astype(jnp.int32)])
    rank = jnp.cumsum(new_run, dtype=jnp.int32)
    rank_c = rank.reshape(n_chunks, c)
    local = rank_c - rank_c[:, :1]
    iota = jnp.arange(c, dtype=jnp.int32)

    def chunk_part(local_c, gs_c):          # [..., C], [..., C, W] -> [..., C, W]
        onehot = (local_c[..., None] == iota).astype(dtype)
        return jnp.einsum("...cr,...cw->...rw", onehot, gs_c, preferred_element_type=ft)

    gs_c = gs.reshape(n_chunks, c, width)
    per = max(1, ONEHOT_CHUNK_ELEMS // (c * c))
    if n_chunks <= per:
        part = chunk_part(local, gs_c)
    else:
        part = jax.lax.map(lambda a: chunk_part(*a), (local, gs_c), batch_size=per)
    pref = jnp.cumsum(jnp.sum(part, axis=1), axis=0)                        # [n_chunks, W]
    v = jnp.arange(rows, dtype=jnp.int32)
    first = jnp.searchsorted(ids_s, v, side="left").astype(jnp.int32)
    last = jnp.searchsorted(ids_s, v, side="right").astype(jnp.int32) - 1
    c1 = jnp.minimum(first // c, n_chunks - 1)          # clipped: only absent rows are clipped
    c2 = jnp.clip(last // c, 0, n_chunks - 1)
    l1 = rank[jnp.minimum(first, n_chunks * c - 1)] - rank[c1 * c]
    flat = part.reshape(n_chunks * c, width)
    head = flat[c1 * c + l1]
    tail = flat[c2 * c] + (pref[jnp.maximum(c2 - 1, 0)] - pref[c1])
    grad = head + jnp.where((c2 > c1)[:, None], tail, jnp.zeros((), ft))
    return jnp.where((last >= first)[:, None], grad, jnp.zeros((), ft))


@functools.lru_cache(maxsize=None)
def _sorted_gather_sum(dtype_name: str, chunk: int):
    """Custom-VJP big lookup whose table gradient is :func:`sorted_table_grad`."""
    dtype = jnp.dtype(dtype_name)

    @jax.custom_vjp
    def f(table, idx, w):
        return _gather_sum(table, idx, w)

    def fwd(table, idx, w):
        return _gather_sum(table, idx, w), (idx, w, jnp.zeros((table.shape[0], 0), table.dtype))

    def bwd(res, g):
        idx, w, shape_carrier = res
        grad = sorted_table_grad(idx, w, g, shape_carrier.shape[0], chunk, dtype)
        return grad.astype(shape_carrier.dtype), None, None

    f.defvjp(fwd, bwd)
    return f


def big_lookup(table: jax.Array, idx: jax.Array, w: jax.Array | None = None, *,
               mode: str = "onehot", dtype=F32, chunk: int | None = None) -> jax.Array:
    """Lookup in a vocabulary-sized table.

    table: ``[R, W]`` (float32 master weights); idx: int ``[...]`` when ``w`` is None (one row
    per index, result ``[..., W]``), else int ``[..., K]`` with weights ``w [..., K]`` (weighted
    sum over K, result ``[..., W]``). ``dtype`` is the operand dtype of the backward matmul.
    mode: ``"onehot"``, ``"take"`` or ``"sorted"`` (module docstring); chunk: rows per chunk of
    the ``"sorted"`` backward (default :data:`SORTED_CHUNK`; ignored by the other modes).
    Returns rows in the table dtype.
    """
    single = w is None
    if single:
        idx = idx[..., None]
        w = jnp.ones(idx.shape, F32)
    lead = idx.shape[:-1]
    k = idx.shape[-1]
    rows = table.shape[0]
    i2 = jnp.clip(idx.astype(jnp.int32), 0, rows - 1).reshape(-1, k)
    w2 = jax.lax.stop_gradient(w).astype(F32).reshape(-1, k)
    if mode == "onehot":
        out = _onehot_gather_sum(jnp.dtype(dtype).name)(table, i2, w2)
    elif mode == "take":
        out = _gather_sum(table, i2, w2)
    elif mode == "sorted":
        out = _sorted_gather_sum(jnp.dtype(dtype).name, int(chunk or SORTED_CHUNK))(table, i2, w2)
    else:
        raise ValueError(f"unknown table_grad mode {mode!r}")
    return out.reshape(*lead, table.shape[1])


def concat_tables(tables: list[jax.Array]) -> tuple[jax.Array, np.ndarray]:
    """Stack small tables ``[R_i, W]`` into ``[sum R_i, W]``; returns the table and row offsets."""
    sizes = [t.shape[0] for t in tables]
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int32)
    return jnp.concatenate(tables, axis=0), offsets


def field_index(values: jax.Array, cards: tuple[int, ...] | list[int],
                offsets: np.ndarray) -> jax.Array:
    """Clip per-field values ``[..., F]`` into their cardinalities and add the row offsets of
    the concatenated table. Returns int32 ``[..., F]``."""
    v = values.astype(jnp.int32)
    hi = jnp.asarray(np.asarray(cards, np.int32) - 1)
    return jnp.clip(v, 0, hi) + jnp.asarray(offsets)


REG_NAMES = tuple(name for name, _ in REG_FIELDS)
REG_CARDS = tuple(card for _, card in REG_FIELDS)


def reg_embed(p_reg: dict, reg: jax.Array, on: jax.Array | float, dtype=F32) -> jax.Array:
    """``on * sum_f E_reg_f[reg_f]``: p_reg ``{field: [card_f, d]}``, reg int ``[B, T, N_REG]``.

    The per-field tables are concatenated and read with one multi-hot matmul. Returns
    ``float32[B, T, d]``.
    """
    table, offsets = concat_tables([p_reg[name] for name in REG_NAMES])
    idx = field_index(reg, REG_CARDS, offsets)
    w = jnp.broadcast_to(jnp.asarray(on, F32), idx.shape)
    return small_lookup(table, idx, w, dtype)
