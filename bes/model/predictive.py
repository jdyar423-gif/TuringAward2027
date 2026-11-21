"""Output law: the candidate tape's evidence on top of the network as base measure.

For one position with hidden state ``h``, network distribution ``q = softmax(z)`` (``z`` the
softcapped logits of ``bes.model.head``) and candidate slots ``k`` (token ``cand_k``, features
``feat_kf``)::

    a_k  = sum_f beta_f[feat_kf] + <h A, c_k> - mass_offset,     c_k = sum_f E_tfeat_f[feat_kf]
    m_k  = exp(soft_cap(a_k, mass_log_cap)) * mask_k * src_scale[src_k]
    p(v) = (q(v) + sum_k m_k [cand_k = v]) / (1 + sum_k m_k)

:func:`soft_cap` leaves a log-mass up to ``mass_log_cap - MASS_CAP_KNEE`` unchanged (bit for
bit) and bends it smoothly towards the cap above that, so a slot at or above the cap still has a
gradient (a hard ``min`` has none there, and a wrong candidate stuck above it could never be
corrected).

Parameters: ``trust/bias/<f>`` (``beta_f``, a scalar per feature value), ``trust/feat/<f>``
(``E_tfeat_f``, ``[card_f, cand_dim]``) and ``trust/proj`` (``A``, ``[d, cand_dim]``); the two
tables start at zero, so at initialisation every active slot carries ``exp(-mass_offset)``.
``mask_k`` is ``oracle_io.cand_mask`` (a valid slot, its kind switch, the ``cand_out``
switch); ``src_scale = consts["enable"]["src"]`` multiplies the masses of each value of the
``src`` feature (the decoder down-weights evidence from the text being generated, ``src = 5``).
Empty slots (-1) and ids outside the vocabulary carry no mass; a token held by several slots
receives the sum of their masses. ``mass_offset`` and ``mass_log_cap`` are ``ModelCfg`` fields.

Properties (``docs/MATH.md`` T1; ``tests/math/test_claims.py``,
``test_t1b_additive_mass_law_is_the_pitman_yor_predictive_with_the_network_as_base``):

* normalised in closed form by ``1 + M`` (``M = sum_k m_k``): no second pass over the vocabulary
  and no float32 cancellation (the log-sum-exp is the network's own);
* it is the Pitman-Yor posterior predictive with base measure ``q``: counts ``n_v``, one table
  per seen type ``t_v``, discount ``d`` and concentration ``theta`` give
  ``(n_v - d t_v + (theta + d T) q(v)) / (theta + n)`` with ``m_v = (n_v - d t_v) / (theta + d T)``;
  it also equals the logit tilt ``g_k = log(1 + m_k / q(cand_k))``;
* bounded downside: ``-log p(y) <= -log q(y) + log(1 + M)`` for every target ``y``;
* with every mass zero it is ``q`` itself, and :func:`mix_ce` returns the plain cross-entropy
  bit for bit.

Training needs per-token terms only. With ``ce_base = -log q(y)`` from the chunked
cross-entropy, ``M_y = sum_k m_k [cand_k = y]``::

    ce = -log(q(y) + M_y) + log1p(M)        (= ce_base + log1p(M) where M_y = 0)

(:func:`mix_ce`, ``O(n K)`` work on top of the cross-entropy, float32). :func:`mix_logprobs`
gives the whole ``log p [..., V]`` for generation by compare/select against an iota (no
scatter).

Sampling temperature ``T`` (:func:`mix_logprobs`): it acts on the network's distribution only,

    p_T(v) = (softmax(z / T)(v) + sum_k m_k [cand_k = v]) / (1 + sum_k m_k),

because the tape's share ``M / (1 + M)`` is a calibrated probability (the weight of the evidence
in the Pitman-Yor predictive) and must not be sharpened: tempering the whole mixture,
``p^(1/T)``, multiplies the odds of whatever the tape proposes and makes copy loops absorbing
(measured on a micro model trained with the tape, document continuation at ``T = 0.5``:
repeated 4-gram rate 0.68 with 5/24 samples in a loop when the mixture is tempered, 0.25 and
1/24 when only the network is). ``T = 1`` is the law itself, bit for bit.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from bes.model.layers import F32, hi
from bes.model.oracle_io import FEAT_CARDS, FEAT_NAMES, SRC_FIELD, feat_index
from bes.model.tables import concat_tables, small_lookup


#: Width of the smooth knee of :func:`soft_cap`: log-masses up to ``mass_log_cap - MASS_CAP_KNEE``
#: are used unchanged (bit for bit).
MASS_CAP_KNEE = 2.0


def _in_vocab(cand: jax.Array, vocab: int) -> jax.Array:
    """bool ``[..., K]``: the slot holds an id in ``[0, vocab)``."""
    ci = cand.astype(jnp.int32)
    return (ci >= 0) & (ci < vocab)


def soft_cap(a: jax.Array, cap: float, knee: float = MASS_CAP_KNEE) -> jax.Array:
    """Smooth upper cap of log-masses (same shape and dtype as ``a``).

    With ``lo = cap - knee``: ``a`` itself where ``a <= lo`` (bit for bit), else
    ``lo + knee * tanh((a - lo) / knee)``. Increasing and below ``cap`` everywhere; at ``lo`` the
    value, the slope (1) and the curvature (0) of both pieces agree; the derivative
    ``sech^2((a - lo) / knee)`` is positive everywhere (it decays like
    ``4 exp(-2 (a - lo) / knee)`` far above the cap), so a slot pushed past the cap still
    receives a gradient. ``knee > 0``.
    """
    lo = cap - knee
    return jnp.where(a <= lo, a, lo + knee * jnp.tanh((a - lo) / knee))


def log_mass(pt: dict, h: jax.Array, cand_feat: jax.Array, offset: float) -> jax.Array:
    """``a_k = sum_f beta_f[feat_kf] + <h A, c_k> - offset``: ``[..., K]`` in ``hi(h.dtype)``.

    pt: ``{"proj": [d, cd], "feat": {f: [card_f, cd]}, "bias": {f: [card_f]}}``; h: ``[..., d]``
    final hidden state; cand_feat: int ``[..., K, F]``. ``c_k`` and the bias sum are read with
    one multi-hot matmul over the concatenated ``[sum_f card_f, cd + 1]`` table.
    """
    feat, _ = concat_tables([pt["feat"][f] for f in FEAT_NAMES])                 # [R, cd]
    bias, _ = concat_tables([pt["bias"][f][:, None] for f in FEAT_NAMES])        # [R, 1]
    table = jnp.concatenate([feat, bias.astype(feat.dtype)], axis=1)              # [R, cd + 1]
    ft = hi(h.dtype)
    rows = small_lookup(table, feat_index(cand_feat), jnp.ones(cand_feat.shape, F32), ft)
    ht = jnp.einsum("...d,dc->...c", h.astype(ft), pt["proj"].astype(ft))
    score = jnp.einsum("...c,...kc->...k", ht, rows[..., :-1])
    return rows[..., -1] + score - offset


def masses(pt: dict, h: jax.Array, cand_tok: jax.Array, cand_feat: jax.Array, mask: jax.Array,
           src_scale: jax.Array, cfg) -> jax.Array:
    """``m_k = exp(soft_cap(a_k, cfg.mass_log_cap)) * mask_k * src_scale[src_k]``: ``[..., K]``
    in ``hi(h.dtype)`` (float32), zero for empty slots and ids ``>= cfg.vocab_size``, at most
    ``exp(cfg.mass_log_cap)``.

    cand_tok: int ``[..., K]``; cand_feat: int ``[..., K, F]``; mask: ``float32[..., K]`` from
    ``oracle_io.cand_mask``; src_scale: ``float32[card(src)]`` (non-negative constants: no
    gradient flows into them).
    """
    a = log_mass(pt, h, cand_feat, cfg.mass_offset)
    src = jnp.clip(cand_feat[..., SRC_FIELD].astype(jnp.int32), 0, FEAT_CARDS[SRC_FIELD] - 1)
    w = (jnp.asarray(mask, F32) * jnp.take(jnp.asarray(src_scale, F32), src)
         * _in_vocab(cand_tok, cfg.vocab_size).astype(F32))
    return jnp.exp(soft_cap(a, cfg.mass_log_cap)) * jax.lax.stop_gradient(w).astype(a.dtype)


def mix_ce(ce_base: jax.Array, y: jax.Array, cand: jax.Array, m: jax.Array, vocab: int) -> jax.Array:
    """Per-token cross-entropy of the output law: ``float32[n]``.

    ce_base: ``float32[n]`` = ``-log q(y)`` (``head.chunked_ce``); y: int ``[n]`` targets;
    cand: int ``[n, K]``; m: ``[n, K]`` masses (:func:`masses`; slots outside ``[0, vocab)``
    are ignored). Computes ``-where(M_y > 0, log(exp(-ce_base) + M_y), -ce_base) + log1p(M)``,
    so with every mass zero the result is ``ce_base`` bit for bit; the unused branch of the
    ``where`` sees a constant, so no NaN enters the gradient. Where ``M_y = 0`` the derivative
    is that of the plain branch (one-sided in the target masses); in the model a mass is zero
    only where its constant slot weight is, which multiplies that derivative by zero anyway.
    """
    m = jnp.where(_in_vocab(cand, vocab), m, jnp.zeros((), m.dtype))
    hit = cand.astype(jnp.int32) == y.astype(jnp.int32)[..., None]
    m_y = jnp.sum(jnp.where(hit, m, jnp.zeros((), m.dtype)), axis=-1)
    total = jnp.sum(m, axis=-1)
    on = m_y > 0
    mixed = jnp.log(jnp.where(on, jnp.exp(-ce_base) + m_y, jnp.ones((), ce_base.dtype)))
    return -jnp.where(on, mixed, -ce_base) + jnp.log1p(total)


def mix_logprobs(z: jax.Array, cand: jax.Array, m: jax.Array, temperature=1.0) -> jax.Array:
    """``log p_T [..., V]`` of the output law at sampling temperature ``T`` (module docstring):
    ``log((softmax(z / T) + masses) / (1 + M))``, the temperature acting on the network only.

    z: ``float32[..., V]`` softcapped logits; cand: int ``[..., K]``; m: ``[..., K]`` masses
    (slots outside ``[0, V)`` are ignored); temperature: ``T > 0``, a Python float or a float32
    scalar (traced under ``jit``: one compiled program serves every temperature).
    ``add_v = sum_k where(v == cand_k, m_k, 0)`` is accumulated slot by slot against an iota
    (compare/select, no scatter), then ``log p_T = where(add > 0, log(q_T + add), log q_T) -
    log1p(M)`` with ``log q_T = log_softmax(z / T)``. ``sum_v exp(log p_T) = 1`` up to rounding;
    ``T = 1`` gives the law ``p`` bit for bit. Meant for scoring and generation; its gradient is
    plain autodiff (as in :func:`mix_ce`, one-sided at a zero mass).
    """
    vocab = z.shape[-1]
    zero = jnp.zeros((), m.dtype)
    m = jnp.where(_in_vocab(cand, vocab), m, zero)
    logq = jax.nn.log_softmax(z / jnp.asarray(temperature, z.dtype), axis=-1)
    iota = jnp.arange(vocab, dtype=jnp.int32)
    ci = cand.astype(jnp.int32)
    add = jnp.zeros(logq.shape, m.dtype)
    for k in range(ci.shape[-1]):
        add = add + jnp.where(ci[..., k:k + 1] == iota, m[..., k:k + 1], zero)
    on = add > 0
    mixed = jnp.log(jnp.where(on, jnp.exp(logq) + add, jnp.ones((), logq.dtype)))
    return jnp.where(on, mixed, logq) - jnp.log1p(jnp.sum(m, axis=-1))[..., None]


__all__ = ["log_mass", "masses", "mix_ce", "mix_logprobs", "soft_cap", "MASS_CAP_KNEE"]
