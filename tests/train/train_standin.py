"""Stand-in model, configuration and data for the tests of ``bes.train`` (no dependency on
``bes.model``): a two-layer network over the SPEC section 3 batch dictionary."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from bes.config import Cfg, DataCfg, ModelCfg, OptimCfg, RunCfg

V, D, H = 64, 16, 32
ROLES = {"emb": "embed", "w1": "matrix", "w2": "matrix", "g": "norm", "b": "scalar"}


def init_fn(model_cfg, key):
    """Embedding [V, D] (embed), w1 [D, H] and w2 [H, V] (matrix), gain g [D] (norm), bias b [V]
    (scalar). Matrices are non-zero (MuonH keeps them on their initial sphere)."""
    k1, k2, k3 = jax.random.split(key, 3)
    return {"emb": jax.random.normal(k1, (V, D), jnp.float32),
            "w1": jax.random.normal(k2, (D, H), jnp.float32) / np.sqrt(D),
            "w2": jax.random.normal(k3, (H, V), jnp.float32) / np.sqrt(H),
            "g": jnp.ones((D,), jnp.float32), "b": jnp.zeros((V,), jnp.float32)}


def role_fn(path: str) -> str:
    return ROLES[path]


def consts_fn(model_cfg):
    return {"scale": jnp.ones((), jnp.float32)}


def logits(params, consts, x):
    h = params["emb"][x] * params["g"]
    h = jax.nn.relu(h @ params["w1"])
    z = h @ params["w2"] + params["b"]
    return z * consts.get("scale", 1.0) if consts else z


def loss_fn(params, consts, batch, model_cfg):
    """Weighted mean next-token cross-entropy of one micro-batch ``tokens [B, T+1]``, ``w [B, T]``."""
    tok = batch["tokens"]
    x, y = tok[:, :-1], tok[:, 1:]
    z = logits(params, consts, x)
    ce = jax.nn.logsumexp(z, axis=-1) - jnp.take_along_axis(z, y[..., None], axis=-1)[..., 0]
    w = batch["w"]
    weight = jnp.sum(w)
    ce_sum = jnp.sum(w * ce)
    return ce_sum / jnp.maximum(weight, 1e-9), {"ce_sum": ce_sum, "weight": weight,
                                                "ce_rows": jnp.sum(w * ce, axis=-1)}


def tiny_cfg(optimizer: str = "muonh", *, seq_len: int = 16, seq_len_late: int = 32,
             tokens_per_step: int = 256, micro_seqs: int = 4, **run) -> Cfg:
    """Shapes (A, B, T) = (4, 4, 16) and (4, 2, 32)."""
    base = dict(deadline_hours=0.1, post_reserve_hours=0.02, ckpt_every_min=0.5, log_every=10,
                guard_warmup=5)
    base.update(run)
    return Cfg(model=ModelCfg(d_model=D, act_dtype="float32"),
               # pinned optimiser settings: these tests exercise the trainer's control flow on a
               # stand-in whose dynamics were calibrated with them (no transfer rule at 256 tokens)
               optim=OptimCfg(optimizer=optimizer, tokens_per_step=tokens_per_step,
                              micro_seqs=micro_seqs, lr_width_ref=512, warmup_frac=0.01,
                              warmdown_frac=0.65, batch_exponent=0.0, data_exponent=0.0),
               data=DataCfg(seq_len=seq_len, seq_len_late=seq_len_late),
               run=RunCfg(**base))


#: successor law of the stand-in data: x[t+1] = x[t] + 1 w.p. 0.7, x[t] + 2 w.p. 0.3 (mod V)
P_STEP1 = 0.7
LOSS_FLOOR = -(P_STEP1 * np.log(P_STEP1) + (1 - P_STEP1) * np.log(1 - P_STEP1))  # 0.611 nats


def pattern_tokens(rng: np.random.Generator, shape: tuple[int, ...], noise: float = 0.0):
    """Rows of a stochastic successor law (irreducible loss ``LOSS_FLOOR``, about 4% batch-to-batch
    noise at 256 targets), with an extra fraction ``noise`` of uniformly random tokens."""
    a, b, n = shape
    steps = np.where(rng.random((a, b, n)) < P_STEP1, 1, 2)
    steps[..., 0] = rng.integers(0, V, (a, b))
    tok = np.cumsum(steps, axis=-1) % V
    mask = rng.random((a, b, n)) < noise
    tok = np.where(mask, rng.integers(0, V, (a, b, n)), tok)
    return tok.astype(np.int32)


class Data:
    """``data(index, stage)``: deterministic batches; advances a fake clock by ``dt`` per batch.

    poison: indices whose tokens are uniformly random (a loss spike once the pattern is learnt);
    nan: indices whose weights are NaN (a non-finite loss); crash_at: raise ``Crash`` at that
    index (a simulated kill)."""

    def __init__(self, clock=None, dt: float = 0.0, poison=(), nan=(), crash_at=None,
                 keys=("tokens", "w")):
        self.clock, self.dt = clock, dt
        self.poison, self.nan, self.crash_at = set(poison), set(nan), crash_at
        self.keys = keys
        self.calls: list[tuple[int, tuple[int, int, int]]] = []

    def __call__(self, index, stage):
        if self.crash_at is not None and index == self.crash_at:
            raise Crash(index)
        if self.clock is not None and self.dt:
            self.clock.advance(self.dt)
        self.calls.append((index, stage.shape))
        a, b, t = stage.shape
        rng = np.random.default_rng(10_000 + index)
        if index in self.poison:
            tokens = rng.integers(0, V, (a, b, t + 1)).astype(np.int32)
        else:
            tokens = pattern_tokens(rng, (a, b, t + 1))
        w = np.ones((a, b, t), np.float32)
        if index in self.nan:
            w[:] = np.nan
        batch = {"tokens": tokens, "w": w, "seg": np.zeros((a, b, t), np.int32)}
        return {k: batch[k] for k in self.keys}


class Crash(RuntimeError):
    """Raised by :class:`Data` to simulate a process being killed."""
