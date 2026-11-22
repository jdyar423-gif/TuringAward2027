"""Interface to the exact layer: the candidate tape as input features and its slot mask.

Input  (``cand_in``):  ``W_cin @ sum_k mask_k (E_ctok[cand_k] + sum_f E_cfeat_f[feat_kf])``.
Output (``cand_out``): the posterior-predictive law of ``bes.model.predictive``, which weights
slot ``k`` by ``mask_k`` as well.

``mask_k`` is 1 when slot ``k`` holds a candidate (``cand_tok >= 0``), its kind switch
``enable["kind"][kind_k]`` is on and the path switch (``cand_in`` / ``cand_out``) is on.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from bes.config import CAND_FEATS
from bes.model.layers import F32, act_dtype, mm
from bes.model.tables import big_lookup, concat_tables, field_index, small_lookup

FEAT_NAMES = tuple(name for name, _ in CAND_FEATS)
FEAT_CARDS = tuple(card for _, card in CAND_FEATS)
KIND_FIELD = FEAT_NAMES.index("kind")
SRC_FIELD = FEAT_NAMES.index("src")


def cand_mask(cand_tok: jax.Array, cand_feat: jax.Array, kind_on: jax.Array,
              path_on: jax.Array | float) -> jax.Array:
    """``float32[..., K]`` slot mask ``mask_k``.

    cand_tok: int ``[..., K]`` (-1 empty); cand_feat: int ``[..., K, F]``;
    kind_on: ``float32[card(kind)]`` per-kind switches; path_on: 0/1 path switch.
    """
    kind = jnp.clip(cand_feat[..., KIND_FIELD].astype(jnp.int32), 0, FEAT_CARDS[KIND_FIELD] - 1)
    valid = (cand_tok >= 0).astype(F32)
    return valid * jnp.take(jnp.asarray(kind_on, F32), kind) * jnp.asarray(path_on, F32)


def feat_index(cand_feat: jax.Array) -> jax.Array:
    """Per-slot row indices into the concatenation of the per-field tables (fields in
    ``CAND_FEATS`` order, values clipped into their cardinalities): int32 ``[..., K, F]``."""
    offsets = np.concatenate([[0], np.cumsum(FEAT_CARDS)[:-1]]).astype(np.int32)
    return field_index(cand_feat, FEAT_CARDS, offsets)


def cand_input(pc: dict, cand_tok: jax.Array, cand_feat: jax.Array, m: jax.Array, cfg) -> jax.Array:
    """Candidate-tape input features ``float32[B, T, d]``.

    pc: ``{"in_tok": [V, cd], "in_feat": {f: [card_f, cd]}, "in_proj": [cd, d]}``;
    cand_tok ``[B, T, K]``, cand_feat ``[B, T, K, F]``, m ``[B, T, K]`` the slot mask from
    :func:`cand_mask`.
    """
    dt = act_dtype(cfg)
    tok = big_lookup(pc["in_tok"], cand_tok, m, mode=cfg.table_grad, dtype=dt,
                     chunk=cfg.sorted_chunk)                                          # [B,T,cd]
    table, _ = concat_tables([pc["in_feat"][f] for f in FEAT_NAMES])
    fidx = feat_index(cand_feat)                                                      # [B,T,K,F]
    lead = fidx.shape[:-2]
    fw = jnp.broadcast_to(m[..., None], fidx.shape)
    feat = small_lookup(table, fidx.reshape(*lead, -1), fw.reshape(*lead, -1), dt)    # [B,T,cd]
    return mm(tok + feat, pc["in_proj"], dt)
