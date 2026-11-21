"""Critic: a 2-layer value head on the final hidden state.

``critic(h) = relu(h @ W1) @ w2`` (float32, no biases). ``w2`` starts at zero so the critic
outputs 0 until it is fitted (it is trained in a separate stage on the frozen core; the main
optimiser treats its leaves as frozen, see ``bes.roles``).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from bes.model.layers import hi


def critic(params: dict, hidden: jax.Array) -> jax.Array:
    """hidden: ``[..., d]`` (e.g. ``[B, d]``) -> ``float32[...]`` (e.g. ``[B]``).

    params: the full parameter tree (uses ``params["critic"]``: ``w1 [d, H]``, ``w2 [H]``).
    """
    pc = params["critic"] if "critic" in params else params
    ft = hi(hidden.dtype)
    z = jax.nn.relu(jnp.einsum("...d,dh->...h", hidden.astype(ft), pc["w1"].astype(ft)))
    return jnp.einsum("...h,h->...", z, pc["w2"].astype(ft))
