"""Writes to the rows of a few tokens of a trained model (``docs/PLAN_M6V.md``, theorem 1).

Every vocabulary id owns one row in each token table (:data:`TOKEN_TABLES`): its input
embedding, its per-loop value embeddings, its output gains and the keys and values of its
private neurons. A forward pass reads the rows of the tokens at its input positions and no
other row of these tables (``bes.readset``). So a change that touches only the rows of a set
``S`` of tokens leaves the model's output on every input without a token of ``S`` **bit for
bit** what it was, whatever the change is; and two such changes commute exactly when neither
taught text contains a token whose rows the other one writes.

:class:`Writer` makes such changes by gradient steps on one batch (the taught text): only the
tables it was built for, only the rows of the tokens it is given; every step moves a row by
``lr`` times the larger of that row's norm and the table's reference norm (:func:`reference_norms`:
the median row norm, so that a row trained to nearly zero can still be written).

Which table. Measured on the 24-hour model (``lab/m6v/theorem4.py``, eight invented facts
written one after the other into the rows of tokens rarer than rank 512): writing all five
tables fits a paraphrased question to 0.58 nats per token and moves the loss of held-out texts
that contain a written token by 0.080; writing the **page values only** fits it to 0.65 and
moves them by 0.027, and unrelated answers by 0.002 instead of 0.026. The page value is the one
place where a token's row is both private (theorem 1) and gated by the context (its private
neurons), so it is the default here. The recall of a written fact in free generation stays weak
with every variant (at most 4 of 8): what must be recalled word for word is written as text
(``bes.library.overlay``); a row write is a consolidation, not the memory of record.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from bes.model import model as M
from bes.train.checkpoint import flatten_with_paths

#: tables with one row per vocabulary id, read by the token of an input position
TOKEN_TABLES = ("emb/tok", "emb/ve", "gain/o", "page/key", "page/val")
#: the default place of a write
WRITE_TABLES = ("page/val",)


def input_tokens(batch: Any) -> np.ndarray:
    """The distinct token ids at the input positions of a batch (``tokens[..., :T]`` with ``T``
    the length of ``w``; a ``feats`` dict has no ``w`` and all its tokens are inputs)."""
    tok = np.asarray(batch["tokens"])
    if "w" in batch:
        tok = tok[..., : np.shape(batch["w"])[-1]]
    return np.unique(tok)


def rows_of(batch: Any, vocab: int, *, rank: np.ndarray | None = None, min_rank: int = 0,
            n_content: int | None = None) -> np.ndarray:
    """``bool[V]``: the tokens of ``batch`` whose rows a write may change: ordinary tokens
    (ids below ``n_content``; the role and control tokens of the chat format are in every
    conversation) of frequency rank at least ``min_rank`` (``rank[v]``: 0 is the most frequent)."""
    mask = np.zeros(vocab, bool)
    ids = input_tokens(batch)
    mask[ids[(ids >= 0) & (ids < vocab)]] = True
    if n_content is not None:
        mask[int(n_content):] = False
    if rank is not None and min_rank > 0:
        mask &= np.asarray(rank) >= int(min_rank)
    return mask


def reference_norms(params: Any, tables: Sequence[str] = WRITE_TABLES, n_content: int | None = None) -> dict[str, float]:
    """The median row norm of each table of ``tables`` (over the ids below ``n_content``)."""
    paths, leaves, _ = flatten_with_paths(params)
    out = {}
    for p, a in zip(paths, leaves):
        if p in tables:
            a = np.asarray(jax.device_get(a), np.float32)
            n = np.linalg.norm(a.reshape(a.shape[0], -1), axis=-1)
            out[p] = float(np.median(n[:n_content] if n_content else n))
    return out


class Writer:
    """Row writes for one model configuration (the gradient and the step are compiled once per
    batch shape). ``tables``: the token tables that may change (a subset of the model's
    :data:`TOKEN_TABLES`); ``ref``: the reference norm per table (``{}``: the row's own norm
    only)."""

    def __init__(self, cfg: Any, consts: dict, tables: Sequence[str] = WRITE_TABLES,
                 ref: dict[str, float] | None = None):
        bad = [t for t in tables if t not in TOKEN_TABLES]
        if bad:
            raise ValueError(f"not token tables: {bad}")
        self.cfg, self.consts, self.tables, self.ref = cfg, consts, tuple(tables), dict(ref or {})
        self._grad = jax.jit(lambda p, b: jax.value_and_grad(lambda q: M.loss_fn(q, consts, b, cfg)[0])(p))
        self._step = None

    def _make_step(self, params: Any):
        paths, _, treedef = flatten_with_paths(params)
        missing = [t for t in self.tables if t not in paths]
        if missing:
            raise ValueError(f"the model has no table {missing}")
        tables, ref = self.tables, self.ref

        def step(p, g, lr, mask):
            out = []
            for path, a, ga in zip(paths, jax.tree_util.tree_leaves(p), jax.tree_util.tree_leaves(g)):
                if path not in tables:
                    out.append(a)
                    continue
                v = a.shape[0]
                a2 = a.astype(jnp.float32).reshape(v, -1)
                g2 = ga.astype(jnp.float32).reshape(v, -1)
                gn = jnp.sqrt(jnp.sum(g2 * g2, -1))
                an = jnp.maximum(jnp.sqrt(jnp.sum(a2 * a2, -1)), ref.get(path, 0.0))
                sc = jnp.where((gn > 0) & mask, lr * an / jnp.maximum(gn, 1e-30), 0.0)
                new = (a2 - sc[:, None] * g2).reshape(a.shape).astype(a.dtype)
                # a row outside the mask is the stored row itself, not a recomputed copy of it
                keep = jnp.reshape(mask, (v,) + (1,) * (a.ndim - 1))
                out.append(jnp.where(keep, new, a))
            return jax.tree_util.tree_unflatten(treedef, out)

        return jax.jit(step)

    def loss(self, params: Any, batch: Any) -> float:
        return float(self._grad(params, batch)[0])

    def write(self, params: Any, batch: Any, rows: Iterable[int] | np.ndarray, *, steps: int = 8,
              lr: float = 0.25) -> tuple[Any, dict[str, Any]]:
        """``steps`` gradient steps on ``batch`` in the rows of ``rows`` (``bool[V]`` or ids).
        Returns the new parameters and ``{"loss_before", "loss_after", "rows", "tables"}``."""
        v = int(self.cfg.vocab_size)
        mask = np.zeros(v, bool)
        rows = np.asarray(rows)
        if rows.dtype == bool:
            if rows.shape != (v,):
                raise ValueError(f"the mask has shape {rows.shape}, the vocabulary {v} ids")
            mask = rows.copy()
        else:
            mask[rows.astype(np.int64)] = True
        if self._step is None:
            self._step = self._make_step(params)
        m = jnp.asarray(mask)
        first = last = None
        for _ in range(int(steps)):
            loss, g = self._grad(params, batch)
            first = float(loss) if first is None else first
            params = self._step(params, g, float(lr), m)
        if steps:
            last = self.loss(params, batch)
        return params, {"loss_before": first, "loss_after": last, "rows": int(mask.sum()),
                        "tables": list(self.tables), "steps": int(steps), "lr": float(lr)}


__all__ = ["TOKEN_TABLES", "WRITE_TABLES", "Writer", "input_tokens", "rows_of", "reference_norms"]
