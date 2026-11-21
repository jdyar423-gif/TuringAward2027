"""Shared helpers for the model tests: tiny configurations, random batches, random parameters."""
from __future__ import annotations

import dataclasses
import functools
import os

import jax
import jax.numpy as jnp
import numpy as np

from bes.config import CAND_FEATS, CAND_SLOTS, N_CLOCKS, REG_FIELDS, load_config, special_id

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def config_model(name: str):
    return load_config(os.path.join(ROOT, "configs", f"{name}.py")).model


def tiny_cfg(**overrides):
    """A small variant of the smoke model (fast to compile)."""
    base = dataclasses.replace(
        config_model("smoke"), vocab_size=512, d_model=64, n_heads=2, head_dim=32, ve_dim=32,
        head_rank=32, n_loops=3, n_blocks=2, max_seq=64, clock_planes=3, cand_dim=16,
        critic_hidden=16, loss_chunk=64)
    return dataclasses.replace(base, **overrides)


def make_batch(cfg, B: int = 2, T: int = 24, seed: int = 0, seg_starts=((0,), (0, 11))) -> dict:
    """Random micro-batch in the SPEC section 3 format (without the A axis).

    Every segment starts with BOS; candidate slots hold earlier tokens or -1 (empty) with
    valid kinds; registers and clocks are random within their ranges.
    """
    rng = np.random.default_rng(seed)
    bos = special_id("<|bos|>", cfg.vocab_size)
    n_ordinary = cfg.vocab_size - 16
    tokens = rng.integers(0, n_ordinary, size=(B, T + 1)).astype(np.int32)
    seg = np.zeros((B, T), np.int32)
    for b in range(B):
        starts = seg_starts[b % len(seg_starts)]
        for s_i, s in enumerate(starts):
            tokens[b, s] = bos
            seg[b, s:] = s_i
    reg = np.stack([rng.integers(0, card, size=(B, T)) for _, card in REG_FIELDS], -1).astype(np.int16)
    clock = rng.integers(0, 40, size=(B, T, N_CLOCKS)).astype(np.int32)
    cand_tok = rng.integers(0, n_ordinary, size=(B, T, CAND_SLOTS)).astype(np.int32)
    empty = rng.random((B, T, CAND_SLOTS)) < 0.3
    cand_tok[empty] = -1
    feats = []
    for name, card in CAND_FEATS:
        if name == "kind":
            f = rng.integers(1, 5, size=(B, T, CAND_SLOTS))
            f[empty] = 0
        else:
            f = rng.integers(0, card, size=(B, T, CAND_SLOTS))
        feats.append(f)
    cand_feat = np.stack(feats, -1).astype(np.int8)
    w = (rng.random((B, T)) < 0.85).astype(np.float32)
    return dict(tokens=tokens, w=w, seg=seg, reg=reg, clock=clock, cand_tok=cand_tok,
                cand_feat=cand_feat)


@functools.lru_cache(maxsize=None)
def _init_fn(cfg):
    from bes.model import model as M
    return jax.jit(functools.partial(M.init_params, cfg))


def init(cfg, seed: int = 0) -> dict:
    """``init_params`` compiled once per configuration (runtime-only fields do not matter)."""
    key_cfg = dataclasses.replace(cfg, act_dtype="float32", remat=False, loss_chunk=8192,
                                  table_grad="onehot", attn_impl="xla")
    return _init_fn(key_cfg)(jax.random.PRNGKey(seed))


def random_params(cfg, seed: int = 0, scale: float = 0.3) -> dict:
    """Initial parameters with every leaf perturbed (non-zero feature tables, broken lane
    identities, non-unit gains, visible sink parameters, a non-trivial unembedding)."""
    params = init(cfg, seed)
    flat, treedef = jax.tree_util.tree_flatten_with_path(params)
    rng = np.random.default_rng(seed + 1000)
    out = []
    for path, leaf in flat:
        name = "/".join(p.key for p in path)
        leaf = np.asarray(leaf)
        noise = rng.standard_normal(leaf.shape).astype(np.float32)
        if name == "head/out":
            leaf = leaf + 0.5 * noise
        elif name == "core/attn/q_gain":
            leaf = leaf * (1.0 + 0.2 * noise)
        elif name.startswith(("core/attn/w", "core/mlp/", "head/proj", "cand/in_proj",
                              "trust/proj", "critic/w1")):
            leaf = leaf * (1.0 + 0.1 * noise)
        else:
            leaf = leaf + scale * noise
        out.append(jnp.asarray(leaf, jnp.float32))
    return jax.tree_util.tree_unflatten(treedef, out)


@functools.lru_cache(maxsize=None)
def value_and_grad_fn(cfg):
    """Jitted ``value_and_grad`` of ``loss_fn`` (with metrics), compiled once per configuration."""
    from bes.model import model as M
    return jax.jit(jax.value_and_grad(functools.partial(M.loss_fn, cfg=cfg), has_aux=True))


def to_numpy(tree):
    return jax.tree.map(lambda a: np.asarray(a, np.float64), tree)


@functools.lru_cache(maxsize=None)
def jitted(fn_name: str, cfg):
    """Cached jitted model functions keyed by name and config."""
    from bes.model import model as M
    fn = getattr(M, fn_name)
    return jax.jit(functools.partial(fn, cfg=cfg))
