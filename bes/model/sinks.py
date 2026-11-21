"""Calibrated attention sink, exact per-key logit biases and the confidence value dim.

Per head the ``head_dim`` dims are laid out as ``[content (dc) | reserved (bias_dims)]`` for
queries and keys, ``dc = head_dim - bias_dims``:

* queries: ``q[..., dc:] = (1, 0, ..., 0)``;
* keys:    ``k[..., dc:] = (bias_j, 0, ..., 0)``;

so ``q . k = gain * cos(q_c, k_c) + bias_j`` with any attention kernel (dense or flash).

Sink (``s_j = 1`` at BOS positions when the sink is enabled): the content key is zeroed,
``bias_j = sink_logit[head]`` and the value is ``(sink_value[head], 0)``; everywhere else
``bias_j = 0`` and the value is the projected value.

Confidence (``conf = 1``): the last value dim is the constant 1 at ordinary positions and 0 at
sink positions, so that dim of the head output is the attention mass outside the sink.

Identity B1 of the plan: with one-hot content keys of a context symbol, a query gain ``beta``
and ``sink_logit = beta + log(lam)`` (see :func:`dirichlet_sink_logit`), the head output tends,
as ``beta`` grows, to the Dirichlet posterior predictive ``(n_ab + lam * m_b) / (n_a + lam)``
(``m`` the sink value, e.g. uniform ``1/V``) and its confidence dim to ``n_a / (n_a + lam)``.

All switches are float 0/1 multipliers so that a disabled path is bit-exact: ``x * 1 = x`` and
``x + 0 = x``.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp

F32 = jnp.float32


def dirichlet_sink_logit(beta: float, lam: float) -> float:
    """Sink logit that makes a head with query gain ``beta`` a Dirichlet estimator of strength
    ``lam`` (identity B1 of the plan)."""
    return beta + math.log(lam)


def sink_indicator(tokens: jax.Array, bos_id: int, enable_sink: jax.Array | float) -> jax.Array:
    """``float32[B, T]``: 1 at BOS positions when the sink switch is on, else 0."""
    return (tokens == bos_id).astype(F32) * jnp.asarray(enable_sink, F32)


def assemble(q_c: jax.Array, k_c: jax.Array, v: jax.Array, s: jax.Array | None,
             sink_logit: jax.Array | None, sink_value: jax.Array | None,
             conf_on: jax.Array | float | None, bias_dims: int, dtype):
    """Build full-width queries, keys and values.

    q_c, k_c: ``[B, T, H, dc]`` float32 (the working precision), already rotated,
    L2-normalised, q scaled by its gain.
    v: ``[B, T, H, head_dim]`` projected values (value embedding already added).
    s: ``float32[B, T]`` sink indicator, or None when sinks are statically disabled.
    sink_logit: ``[H]``; sink_value: ``[H, head_dim - 1]`` (ignored when ``s`` is None).
    conf_on: 0/1 confidence switch, or None when confidence is statically disabled.
    Returns ``(q, k, v)``, each ``[B, T, H, head_dim]`` in ``dtype``.
    """
    if bias_dims < 1:
        raise ValueError("the reserved bias dims are required (bias_dims >= 1)")
    b, t, h, _ = q_c.shape
    ft = q_c.dtype
    pad = jnp.zeros((b, t, h, bias_dims - 1), ft)
    q = jnp.concatenate([q_c, jnp.ones((b, t, h, 1), ft), pad], axis=-1)
    v_body, v_last = v[..., :-1], v[..., -1:]
    if conf_on is not None:
        c = jnp.asarray(conf_on, ft)
        v_last = c + (1.0 - c) * v_last
    if s is None:
        k_bias = jnp.zeros((b, t, h, 1), ft)
    else:
        s = s.astype(ft)
        keep = (1.0 - s)[:, :, None, None]
        on = s[:, :, None, None]
        k_c = k_c * keep
        k_bias = on * sink_logit.astype(ft)[None, None, :, None]
        v_body = v_body * keep + on * sink_value.astype(ft)[None, None]
        v_last = v_last * keep
    k = jnp.concatenate([k_c, k_bias, pad], axis=-1)
    v = jnp.concatenate([v_body, v_last], axis=-1)
    return q.astype(dtype), k.astype(dtype), v.astype(dtype)
