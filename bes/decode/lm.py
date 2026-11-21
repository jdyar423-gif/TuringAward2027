"""Language models as the particle decoder sees them.

The decoder drives ``K`` particles of one prompt through one interface (:class:`LanguageModel`)::

    vocab_size: int
    prefill(prompt_ids, n_particles) -> float32[K, V]
        K identical particles after the prompt; log-probabilities of the first new token.
    step(tokens int[K], active bool[K]) -> float32[K, V]
        feed tokens[j] to every active particle j and return the log-probabilities of its next
        token. Rows of inactive particles are unspecified and their state does not change.
    reorder(ancestors int[K])
        resampling: particle j continues the history (cache, features) of old ancestors[j].
    close()

The log-probabilities are the model's final next-token distribution (``log_softmax`` of the
final scores, i.e. after everything the model does with its own features such as the candidate
tape); the decoder never looks inside.

Implementations:

- :class:`OurLM`: the project's JAX model through ``bes.model.generate`` (one batched KV cache,
  jitted once per ``(K, Tmax)``, one ``IncrementalFeatures`` provider forked per particle);
- :class:`TableLM`: an explicit Markov / n-gram table over a small vocabulary (tests: every
  sequence can be enumerated);
- :class:`PerSequenceLM`: base class for models that run one sequence at a time (for example a
  llama.cpp model): implement ``start_sequence``, ``advance`` and ``copy_state``.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from bes.decode.sampling import log_softmax


@runtime_checkable
class LanguageModel(Protocol):
    vocab_size: int

    def prefill(self, prompt_ids: Sequence[int], n_particles: int) -> np.ndarray:
        ...

    def step(self, tokens: np.ndarray, active: np.ndarray) -> np.ndarray:
        ...

    def reorder(self, ancestors: np.ndarray) -> None:
        ...

    def close(self) -> None:
        ...


def _as_tokens(tokens, k: int) -> np.ndarray:
    tok = np.asarray(tokens, np.int64).reshape(-1)
    if tok.shape[0] != k:
        raise ValueError(f"expected {k} tokens, got {tok.shape[0]}")
    return tok


def _as_active(active, k: int) -> np.ndarray:
    if active is None:
        return np.ones(k, bool)
    act = np.asarray(active, bool).reshape(-1)
    if act.shape[0] != k:
        raise ValueError(f"expected {k} active flags, got {act.shape[0]}")
    return act


def _check_ancestors(ancestors, k: int) -> np.ndarray:
    anc = np.asarray(ancestors, np.int64).reshape(-1)
    if anc.shape[0] != k or (k and (anc.min() < 0 or anc.max() >= k)):
        raise ValueError("ancestors must hold K indices in [0, K)")
    return anc


# ----------------------------------------------------------------- per-sequence adapter
class PerSequenceLM:
    """Batch interface on top of a model that evaluates one sequence at a time.

    Subclasses implement:

    - ``start_sequence(prompt_ids) -> (state, logp float[V])``: state after the prompt and the
      log-probabilities of the first new token;
    - ``advance(state, token) -> (state, logp float[V])``: feed one token;
    - ``copy_state(state) -> state``: an independent copy (resampling duplicates particles).

    The prompt is evaluated once and its state copied ``K`` times. A llama.cpp model can keep
    the token history as its state and re-evaluate (or save and restore its KV state) inside
    ``advance``.
    """

    vocab_size: int = 0

    def start_sequence(self, prompt_ids: Sequence[int]) -> tuple[Any, np.ndarray]:
        raise NotImplementedError

    def advance(self, state: Any, token: int) -> tuple[Any, np.ndarray]:
        raise NotImplementedError

    def copy_state(self, state: Any) -> Any:
        raise NotImplementedError

    def prefill(self, prompt_ids: Sequence[int], n_particles: int) -> np.ndarray:
        if not len(prompt_ids):
            raise ValueError("the prompt must hold at least one token")
        state, logp = self.start_sequence(list(prompt_ids))
        k = int(n_particles)
        self._states = [state] + [self.copy_state(state) for _ in range(k - 1)]
        self._k = k
        return np.repeat(np.asarray(logp, np.float32)[None], k, axis=0)

    def step(self, tokens: np.ndarray, active: np.ndarray | None = None) -> np.ndarray:
        tok = _as_tokens(tokens, self._k)
        act = _as_active(active, self._k)
        out = np.zeros((self._k, self.vocab_size), np.float32)
        for j in np.flatnonzero(act):
            self._states[j], logp = self.advance(self._states[j], int(tok[j]))
            out[j] = logp
        return out

    def reorder(self, ancestors: np.ndarray) -> None:
        anc = _check_ancestors(ancestors, self._k)
        if np.array_equal(anc, np.arange(self._k)):
            return
        self._states = [self.copy_state(self._states[a]) for a in anc]

    def close(self) -> None:
        self._states = []


# ------------------------------------------------------------------------ table model
class TableLM(PerSequenceLM):
    """An explicit n-gram model: ``log p(next | last order-1 tokens)``.

    ``logp_fn(context: tuple[int, ...]) -> float[V]`` returns normalised log-probabilities for
    a context of exactly ``order - 1`` tokens (the last ones of prompt + response; the prompt
    must hold at least ``order - 1`` tokens). ``TableLM.from_table(table)`` wraps a dense array
    of shape ``[V] * order`` (``table[c_1, ..., c_{order-1}]`` is the next-token row).
    ``calls`` counts ``advance`` calls and ``reorders`` the non-trivial reorders (tests).
    """

    def __init__(self, logp_fn: Callable[[tuple[int, ...]], np.ndarray], vocab_size: int,
                 order: int = 2):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.logp_fn = logp_fn
        self.vocab_size = int(vocab_size)
        self.order = int(order)
        self.calls = 0
        self.reorders = 0

    @classmethod
    def from_table(cls, table: np.ndarray) -> "TableLM":
        table = np.asarray(table, np.float64)
        order = table.ndim
        v = table.shape[-1]
        if any(s != v for s in table.shape):
            raise ValueError("table must have shape [V] * order")
        lt = log_softmax(table)
        return cls(lambda ctx: lt[tuple(ctx)] if ctx else lt, v, order)

    @classmethod
    def random(cls, vocab_size: int, order: int = 2, seed: int = 0, scale: float = 1.5,
               boost: dict[int, float] | None = None) -> "TableLM":
        """Random table with Gaussian logits of standard deviation ``scale`` (``boost[v]`` is
        added to the logit of ``v`` in every row)."""
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((vocab_size,) * order) * scale
        for v, b in (boost or {}).items():
            z[..., v] += b
        return cls.from_table(z)

    def context(self, history: Sequence[int]) -> tuple[int, ...]:
        n = self.order - 1
        if n == 0:
            return ()
        if len(history) < n:
            raise ValueError(f"the history needs at least {n} tokens")
        return tuple(int(t) for t in history[len(history) - n:])

    def logp(self, history: Sequence[int]) -> np.ndarray:
        return np.asarray(self.logp_fn(self.context(history)), np.float64)

    def start_sequence(self, prompt_ids):
        hist = tuple(int(t) for t in prompt_ids)
        return hist, self.logp(hist).astype(np.float32)

    def advance(self, state, token):
        self.calls += 1
        hist = state + (int(token),)
        return hist, self.logp(hist).astype(np.float32)

    def copy_state(self, state):
        return state

    def reorder(self, ancestors):
        anc = _check_ancestors(ancestors, self._k)
        if not np.array_equal(anc, np.arange(self._k)):
            self.reorders += 1
        super().reorder(anc)

    def sequence_logprob(self, prompt_ids: Sequence[int], tokens: Sequence[int]) -> np.ndarray:
        """``float64[n]``: log-probability of every token of ``tokens`` after ``prompt_ids``
        (float32 rounding as in :meth:`step`)."""
        hist = list(prompt_ids)
        out = []
        for t in tokens:
            out.append(float(self.logp(hist).astype(np.float32)[int(t)]))
            hist.append(int(t))
        return np.asarray(out, np.float64)


# ------------------------------------------------------------------- tiny tokenizer
class TinyTokenizer:
    """A tokenizer over a handful of byte strings (tests and exhaustive enumeration).

    Ordinary ids ``0..n-1`` are ``pieces``; the 16 special tokens of ``bes.config.SPECIAL_TOKENS``
    follow (ids ``n..n+15``), as in the real layout. ``encode`` is greedy longest match (raises
    ``ValueError`` when a text cannot be spelled). Provides what ``bes.exact.instr_table.build``
    and the decoder need: ``vocab_size``, ``special_tokens``, ``sha256``, ``all_token_bytes``,
    ``token_bytes``, ``decode_bytes``, ``encode``.
    """

    def __init__(self, pieces: Sequence[bytes | str]):
        import hashlib

        from bes.config import SPECIAL_TOKENS

        self.pieces = [p.encode("utf-8") if isinstance(p, str) else bytes(p) for p in pieces]
        if len(set(self.pieces)) != len(self.pieces) or any(not p for p in self.pieces):
            raise ValueError("pieces must be distinct non-empty byte strings")
        n = len(self.pieces)
        self.n_ordinary = n
        self.vocab_size = n + len(SPECIAL_TOKENS)
        self.special_tokens = {name: n + i for i, name in enumerate(SPECIAL_TOKENS)}
        self.special_ids = frozenset(self.special_tokens.values())
        self.sha256 = hashlib.sha256(b"tiny:" + b"\x00".join(self.pieces)).hexdigest()
        self._bytes = tuple(self.pieces) + (b"",) * len(SPECIAL_TOKENS)

    def all_token_bytes(self) -> list[bytes]:
        return list(self._bytes)

    def token_bytes(self, token_id: int) -> bytes:
        return self._bytes[int(token_id)]

    def decode_bytes(self, ids) -> bytes:
        return b"".join(self._bytes[int(i)] for i in ids)

    def encode(self, text: str) -> list[int]:
        data, out, pos = text.encode("utf-8"), [], 0
        order = sorted(range(self.n_ordinary), key=lambda i: -len(self.pieces[i]))
        while pos < len(data):
            for i in order:
                p = self.pieces[i]
                if data.startswith(p, pos):
                    out.append(i)
                    pos += len(p)
                    break
            else:
                raise ValueError(f"cannot encode {data[pos:]!r}")
        return out


# ---------------------------------------------------------------------- our JAX model
def _round_up(n: int, m: int) -> int:
    return -(-n // m) * m


class OurLM:
    """The project's model (``bes.model``) behind :class:`LanguageModel`.

    params, consts, cfg: as for ``bes.model.generate``; tok: tokenizer for the feature
    providers (``None``: the committed one); features: ``True`` to compute the exact-layer
    rows with ``bes.exact.features.IncrementalFeatures`` (registers, clocks, candidate tape),
    ``False`` for all-zero rows, or a zero-argument factory returning a provider. max_steps:
    the number of decoding steps the cache must hold after the prompt (the decoder's
    synchronous steps, waiting included); tmax: fixed cache length (default: padded prompt +
    max_steps rounded up to 128, so prompts of similar length share one compilation);
    chunk: prefill chunk length.

    The prompt is prefilled once in a one-row cache (left-padded to a multiple of ``chunk``),
    duplicated into ``K`` rows with ``reorder_cache`` and its feature provider forked ``K``
    times. ``step`` feeds inactive particles as padding (``seg = -1``): their cache rows,
    positions and providers do not change. ``last_hidden`` is the ``float32[K, d]`` hidden
    state of the last step (for a critic).
    """

    def __init__(self, params, consts, cfg, tok=None, *, features: Any = True,
                 max_steps: int = 512, tmax: int | None = None, chunk: int = 64,
                 decoder=None):
        from bes.model import generate as G

        self._G = G
        self.params, self.consts, self.cfg = params, consts, cfg
        self.vocab_size = int(cfg.vocab_size)
        self.tok = tok
        self.features = features
        self.max_steps = int(max_steps)
        self.tmax = tmax
        self.chunk = int(chunk)
        self.dec = decoder if decoder is not None else G.jitted(cfg)
        from bes.config import special_id

        self.pad_id = special_id("<|pad|>", self.vocab_size)
        self.bos_id = special_id("<|bos|>", self.vocab_size)
        self.cache = None
        self.last_hidden = None
        self.steps = 0

    def _provider(self):
        if callable(self.features) and not isinstance(self.features, bool):
            return self.features()
        if self.features:
            from bes.exact.features import IncrementalFeatures

            return IncrementalFeatures(self.tok)
        return self._G.ZeroFeatures()

    def prefill(self, prompt_ids: Sequence[int], n_particles: int) -> np.ndarray:
        G = self._G
        prompt = [int(t) for t in prompt_ids]
        if not prompt:
            raise ValueError("the prompt must hold at least one token (normally BOS)")
        k = int(n_particles)
        provider = self._provider()
        rows = [provider.step(int(t)) for t in prompt]
        rows = {key: np.stack([np.asarray(r[key]) for r in rows]) for key in G.FEATURE_KEYS}
        n = len(prompt)
        n_pad = (-n) % self.chunk
        need = n_pad + n + self.max_steps
        tmax = _round_up(need, 128) if self.tmax is None else int(self.tmax)
        if need > tmax:
            raise ValueError(f"cache of {tmax} slots cannot hold {need} (padding + prompt + steps)")
        self.capacity = tmax - n_pad - n
        tokens = np.concatenate([np.full(n_pad, self.pad_id, np.int32), np.asarray(prompt, np.int32)])
        seg = np.concatenate([np.full(n_pad, -1, np.int32), G.prompt_segments(prompt, self.bos_id)])
        pos = np.concatenate([np.zeros(n_pad, np.int32), np.arange(n, dtype=np.int32)])
        empty = G.empty_rows(n_pad)
        feat = {key: np.concatenate([empty[key], rows[key]]) for key in G.FEATURE_KEYS}
        cache = G.init_cache(self.cfg, 1, tmax)
        logits = hidden = None
        for lo in range(0, n_pad + n, self.chunk):
            sl = slice(lo, lo + self.chunk)
            f = {"tokens": tokens[None, sl], "seg": seg[None, sl], "pos": pos[None, sl],
                 **{key: a[None, sl] for key, a in feat.items()}}
            logits, hidden, cache = self.dec.prefill(self.params, self.consts, cache, f)
        self.cache = self.dec.reorder(cache, np.zeros(k, np.int32))
        self.providers = [G.fork_provider(provider) for _ in range(k)]
        self.seg = np.full(k, int(seg[-1]), np.int32)
        self.k = k
        self.steps = 0
        self.last_hidden = np.repeat(np.asarray(hidden, np.float32), k, axis=0)
        return np.repeat(log_softmax(np.asarray(logits)).astype(np.float32), k, axis=0)

    def step(self, tokens: np.ndarray, active: np.ndarray | None = None) -> np.ndarray:
        G = self._G
        tok = _as_tokens(tokens, self.k)
        act = _as_active(active, self.k)
        if self.steps >= self.capacity:
            raise RuntimeError(f"the cache holds {self.capacity} decoding steps; raise max_steps")
        empty = G.empty_rows(1)
        rows = []
        for j in range(self.k):
            if act[j]:
                out = self.providers[j].step(int(tok[j]))
                rows.append({key: np.asarray(out[key])[None] for key in G.FEATURE_KEYS})
            else:
                rows.append(empty)
        f = {key: np.concatenate([r[key] for r in rows]) for key in G.FEATURE_KEYS}
        self.seg = self.seg + ((tok == self.bos_id) & act).astype(np.int32)
        f["seg"] = np.where(act, self.seg, -1).astype(np.int32)
        feed = np.where(act, tok, self.pad_id).astype(np.int32)
        logits, hidden, self.cache = self.dec.decode_step(self.params, self.consts, self.cache,
                                                          feed, f)
        self.steps += 1
        self.last_hidden = np.asarray(hidden, np.float32)
        return log_softmax(np.asarray(logits)).astype(np.float32)

    def reorder(self, ancestors: np.ndarray) -> None:
        anc = _check_ancestors(ancestors, self.k)
        if np.array_equal(anc, np.arange(self.k)):
            return
        self.cache = self.dec.reorder(self.cache, anc.astype(np.int32))
        self.providers = [self._G.fork_provider(self.providers[a]) for a in anc]
        self.seg = self.seg[anc]
        if self.last_hidden is not None:
            self.last_hidden = self.last_hidden[anc]

    def close(self) -> None:
        self.cache = None
        self.providers = []


__all__ = ["LanguageModel", "PerSequenceLM", "TableLM", "TinyTokenizer", "OurLM"]
