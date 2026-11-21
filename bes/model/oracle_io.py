"""Interface to the exact layer: candidate tape as input features and as an output tilt.

Input  (``cand_in``):  ``W_cin @ sum_k m_k (E_ctok[cand_k] + sum_f E_cfeat_f[feat_kf])``.
Output (``cand_out``): ``z += sum_k b_k onehot(cand_k)``, ``b_k = 8 tanh(<h @ A_t, c_k> / 8)``,
``c_k = sum_f E_tfeat_f[feat_kf]``.

``m_k`` is 1 when slot ``k`` holds a candidate (``cand_tok >= 0``), its kind switch
``enable["kind"][kind_k]`` is on and the path switch (``cand_in`` / ``cand_out``) is on.

The tilt is added by compare/select against an iota (no scatter); its backward gathers the
logit gradient at the candidate ids (custom VJP), which avoids the ``[n, K, V]`` temporary that
plain autodiff of the compare/select would create.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from bes.config import CAND_FEATS
from bes.model.layers import F32, act_dtype, hi, mm
from bes.model.tables import big_lookup, concat_tables, field_index, small_lookup

TRUST_CAP = 8.0
FEAT_NAMES = tuple(name for name, _ in CAND_FEATS)
FEAT_CARDS = tuple(card for _, card in CAND_FEATS)
KIND_FIELD = FEAT_NAMES.index("kind")


def cand_mask(cand_tok: jax.Array, cand_feat: jax.Array, kind_on: jax.Array,
              path_on: jax.Array | float) -> jax.Array:
    """``float32[..., K]`` slot mask ``m_k``.

    cand_tok: int ``[..., K]`` (-1 empty); cand_feat: int ``[..., K, F]``;
    kind_on: ``float32[card(kind)]`` per-kind switches; path_on: 0/1 path switch.
    """
    kind = jnp.clip(cand_feat[..., KIND_FIELD].astype(jnp.int32), 0, FEAT_CARDS[KIND_FIELD] - 1)
    valid = (cand_tok >= 0).astype(F32)
    return valid * jnp.take(jnp.asarray(kind_on, F32), kind) * jnp.asarray(path_on, F32)


def _feat_index(cand_feat: jax.Array) -> jax.Array:
    """Per-slot indices into the concatenated feature table: int32 ``[..., K, F]``."""
    offsets = np.concatenate([[0], np.cumsum(FEAT_CARDS)[:-1]]).astype(np.int32)
    return field_index(cand_feat, FEAT_CARDS, offsets)


def cand_input(pc: dict, cand_tok: jax.Array, cand_feat: jax.Array, m: jax.Array, cfg) -> jax.Array:
    """Candidate-tape input features ``float32[B, T, d]``.

    pc: ``{"in_tok": [V, cd], "in_feat": {f: [card_f, cd]}, "in_proj": [cd, d]}``;
    cand_tok ``[B, T, K]``, cand_feat ``[B, T, K, F]``, m ``[B, T, K]`` from :func:`cand_mask`.
    """
    dt = act_dtype(cfg)
    tok = big_lookup(pc["in_tok"], cand_tok, m, mode=cfg.table_grad, dtype=dt)       # [B,T,cd]
    table, _ = concat_tables([pc["in_feat"][f] for f in FEAT_NAMES])
    fidx = _feat_index(cand_feat)                                                     # [B,T,K,F]
    lead = fidx.shape[:-2]
    fw = jnp.broadcast_to(m[..., None], fidx.shape)
    feat = small_lookup(table, fidx.reshape(*lead, -1), fw.reshape(*lead, -1), dt)    # [B,T,cd]
    return mm(tok + feat, pc["in_proj"], dt)


def trust(pt: dict, h: jax.Array, cand_feat: jax.Array, m: jax.Array) -> jax.Array:
    """Tilt magnitudes ``b float32[..., K]`` (zero where ``m`` is 0).

    pt: ``{"proj": [d, cd], "feat": {f: [card_f, cd]}}``; h: ``[..., d]`` final hidden state;
    cand_feat: int ``[..., K, F]``; m: ``[..., K]``. Computed in float32.
    """
    table, _ = concat_tables([pt["feat"][f] for f in FEAT_NAMES])
    ft = hi(h.dtype)
    c = small_lookup(table, _feat_index(cand_feat), jnp.ones(cand_feat.shape, F32), ft)  # [...,K,cd]
    ht = jnp.einsum("...d,dc->...c", h.astype(ft), pt["proj"].astype(ft))
    score = jnp.einsum("...c,...kc->...k", ht, c)
    return TRUST_CAP * jnp.tanh(score / TRUST_CAP) * m


def _tilt_fwd_value(z: jax.Array, cand: jax.Array, b: jax.Array) -> jax.Array:
    iota = jnp.arange(z.shape[-1], dtype=jnp.int32)
    cand = cand.astype(jnp.int32)
    b = b.astype(z.dtype)
    zero = jnp.zeros((), z.dtype)
    for k in range(cand.shape[-1]):
        z = z + jnp.where(cand[..., k:k + 1] == iota, b[..., k:k + 1], zero)
    return z


@jax.custom_vjp
def add_tilt(z: jax.Array, cand: jax.Array, b: jax.Array) -> jax.Array:
    """``z [..., V] + sum_k b_k onehot(cand_k)`` by compare/select (``cand = -1`` never matches).

    z: float32 ``[..., V]``; cand: int ``[..., K]``; b: float32 ``[..., K]``.
    """
    return _tilt_fwd_value(z, cand, b)


def _add_tilt_fwd(z, cand, b):
    return _tilt_fwd_value(z, cand, b), (cand,)


def _add_tilt_bwd(res, g):
    (cand,) = res
    v = g.shape[-1]
    ci = cand.astype(jnp.int32)
    valid = ((ci >= 0) & (ci < v)).astype(g.dtype)
    gb = jnp.take_along_axis(g, jnp.clip(ci, 0, v - 1), axis=-1) * valid
    return g, None, gb


add_tilt.defvjp(_add_tilt_fwd, _add_tilt_bwd)
