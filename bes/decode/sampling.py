"""Per-step sampling transforms of the particle decoder (pure numpy).

Everything works on the model's log-probabilities ``logp float32[K, V]`` (one row per particle,
softmax of the model's final scores). One decoding step of particle ``j`` turns its row into an
*unnormalised target increment* ``gamma[j] float64[V]`` (log space, ``-inf`` = excluded):

- the **proposal** is ``gamma[j]`` renormalised (the locally optimal proposal, docs/MATH.md F2),
- the **incremental weight** is ``sum(gamma[j])`` (:func:`step_increment` returns its log).

What each operation does to the target ``pi`` (``docs/MATH.md`` F1-F5)
---------------------------------------------------------------------
``pi(x)`` is proportional to the product over steps of ``gamma_t(x_t)``. The operations of this
module therefore *define* the target; they are not proposal tricks:

- **temperature** ``tau > 0``: the target is built from the tempered model ``p_tau``, the
  distribution the decoder samples from: for a language model that tempers itself
  (``LanguageModel.set_temperature``: ``bes.decode.lm.OurLM`` tempers the network's distribution
  and leaves the candidate tape's masses alone) its rows already are ``log p_tau`` and the
  decoder uses ``tau = 1`` on them; for any other model ``p_tau = softmax(log p / tau)``
  (:func:`tempered`; ``bes.decode.smc.decode`` decides). ``K = 1`` without potentials is ordinary
  sampling at ``tau``.
- **truncation** (top-k, then top-p on the renormalised remainder, then min-p relative to the
  largest kept probability), **n-gram blocking** and the **copy-run cap** remove tokens. Where
  they act is set by ``truncate_after_mask``:

  - ``False`` ("before the mask"): the truncated, renormalised model ``p~`` replaces ``p_tau``:
    ``gamma(v) = p~(v) * m(v)`` with ``m`` the potential mask, weight = allowed mass under ``p~``.
    The filter then samples the *truncated model* conditioned on the potentials exactly.
  - ``True`` (default, "after the mask"): ``alpha = sum_v p_tau(v) m(v)`` (the allowed mass of
    the untruncated tempered model) is the weight and ``gamma = alpha * q`` where ``q`` is the
    truncation of ``p_tau * m`` renormalised inside the allowed set. Truncation then only
    reshapes the choice among allowed tokens and never enters a weight; it can never kill a
    particle (a non-empty allowed set always keeps its most likely token). The target is the
    product of these increments (a well-defined function of every prefix), documented as the
    "truncated-after-mask" target.

  N-gram blocking and the copy cap are skipped in a row where they would remove every token
  that is still possible (they are degeneracy guards, never a reason to kill a particle).
- **potential masks** ``m`` (``bes.decode.potentials``) are 0/1 factors of the target.
- **twist** ``psi`` (``bes.decode.potentials.Twist``) multiplies ``gamma(v)`` by
  ``psi_t(next state) / psi_{t-1}(state)``: proposal only, it cancels in the product.
- **random streams** and **resampling** never change the target.

Random numbers are counter based (:func:`counter_uniform`): the draw of particle slot ``j`` at its
own token position ``n`` is ``U(seed, STREAM_TOKEN, j, n)``, so batching, waiting and the other
particles never change a particle's draws.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

NEG_INF = -np.inf
STREAM_TOKEN = 1
STREAM_RESAMPLE = 2

_M64 = (1 << 64) - 1
_C = (0x9E3779B97F4A7C15, 0xD1B54A32D192ED03, 0xABC98388FB8FAC03, 0x8CB92BA72F3D8DD7)


# ------------------------------------------------------------------------- random streams
def _mix64(z: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser on ``uint64`` arrays (wrapping arithmetic)."""
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def _u64(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype.kind in "iu":
        return np.asarray(np.asarray(arr, dtype=np.int64).astype(np.uint64))
    return np.asarray([int(v) & _M64 for v in np.ravel(arr)], np.uint64).reshape(arr.shape)


def counter_uniform(seed: int, stream: int, index, counter) -> np.ndarray:
    """Uniforms in ``[0, 1)`` (float64, 53 random bits) that depend only on the four integers.

    seed, stream: ints; index, counter: ints or int arrays (broadcast). A pure function: the
    same arguments always give the same value, whatever else is drawn.
    """
    with np.errstate(over="ignore"):
        idx, cnt = np.broadcast_arrays(_u64(index), _u64(counter))
        h = _mix64(np.full(idx.shape, (int(seed) & _M64), np.uint64) + np.uint64(_C[0]))
        h = _mix64(h + np.uint64((int(stream) & _M64)) * np.uint64(_C[1]))
        h = _mix64(h + idx * np.uint64(_C[2]))
        h = _mix64(h + cnt * np.uint64(_C[3]))
    return (h >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


# ---------------------------------------------------------------------- distributions
def log_softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise log-softmax in float64 (``-inf`` entries stay ``-inf``; an all ``-inf`` row
    stays all ``-inf``)."""
    z = np.asarray(z, dtype=np.float64)
    m = np.max(z, axis=-1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        lse = np.log(np.sum(np.exp(z - m), axis=-1, keepdims=True)) + m
    return z - np.where(np.isfinite(lse), lse, 0.0)


def logsumexp(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable log-sum-exp in float64 (``-inf`` for an all ``-inf`` slice)."""
    z = np.asarray(z, dtype=np.float64)
    m = np.max(z, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True)) + m
    return np.squeeze(out, axis=axis)


def tempered(logp: np.ndarray, temperature: float) -> np.ndarray:
    """``log softmax(logp / temperature)`` (float64 ``[K, V]``); ``temperature`` must be > 0."""
    if not temperature > 0:
        raise ValueError("temperature must be > 0 (use greedy decoding for 0)")
    z = np.asarray(logp, dtype=np.float64)
    return log_softmax(z if temperature == 1.0 else z / float(temperature))


TOP_P_CANDIDATES = 256      # rows whose nucleus lies inside their this-many largest entries skip the full sort


def _top_p_keep(lq: np.ndarray, keep: np.ndarray, top_p: float,
                m: int = TOP_P_CANDIDATES) -> np.ndarray | None:
    """The top-p filter of :func:`truncation_keep` without sorting the vocabulary: exactly the
    set the full stable sort gives, or None when some row's nucleus is not inside its
    candidates (the caller then sorts).

    A row's candidates are every entry at least as large as its ``m``-th largest (all ties at
    that value included), in the order of the full stable sort (larger first, lower id first
    among equals): they are a prefix of that sort, so the cumulative masses are the same
    numbers. A full sort of ``[6, 8192]`` float64 rows was a sixth of the host's time per
    decode step.
    """
    n_rows, n_v = lq.shape
    if n_v <= 4 * m:
        return None
    x = np.where(keep, lq, NEG_INF)
    p = np.exp(log_softmax(x))
    part = np.argpartition(-lq, m - 1, axis=-1)[:, :m]
    thresh = np.min(np.take_along_axis(lq, part, axis=-1), axis=-1)
    out = np.zeros_like(keep)
    for r in range(n_rows):
        if not np.isfinite(thresh[r]):
            return None
        cand = np.flatnonzero(lq[r] >= thresh[r])                 # ascending ids
        order = cand[np.argsort(-lq[r, cand], kind="stable")]
        ps = p[r, order]
        kept = (np.cumsum(ps) - ps) < top_p
        if kept[-1]:                                             # the nucleus may go on beyond the candidates
            return None
        out[r, order[kept]] = True
    return keep & out


def truncation_keep(logq: np.ndarray, top_k: int = 0, top_p: float = 1.0, min_p: float = 0.0,
                    within: np.ndarray | None = None) -> np.ndarray:
    """Tokens kept by top-k, top-p and min-p (``bool[K, V]``).

    logq: float ``[K, V]`` log-probabilities; within: optional ``bool[K, V]``, the set the
    truncation acts inside (its complement is never kept). Order: top-k (ties broken towards
    the lower id), then top-p on the distribution renormalised over the survivors (keep the
    shortest most-likely prefix whose mass reaches ``top_p``; the token that crosses it is
    kept), then min-p (keep ``p >= min_p * max p``). ``top_k <= 0``, ``top_p >= 1`` and
    ``min_p <= 0`` switch the respective filter off.
    """
    lq = np.asarray(logq, dtype=np.float64)
    if within is not None:
        lq = np.where(within, lq, NEG_INF)
    keep = np.isfinite(lq)
    n_v = lq.shape[-1]
    use_k, use_p = 0 < top_k < n_v, top_p < 1.0
    if use_p and not use_k and lq.ndim == 2:
        fast = _top_p_keep(lq, keep, top_p)
        if fast is not None:
            keep = fast
            use_p = False
    # one stable sort per row serves both filters: top-k keeps a prefix of this order, so the
    # survivors keep their relative order (ties: lower id first) for top-p
    order = np.argsort(-lq, axis=-1, kind="stable") if (use_k or use_p) else None
    if use_k:
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.broadcast_to(np.arange(n_v), order.shape), axis=-1)
        keep &= rank < top_k
    if use_p:
        x = np.where(keep, lq, NEG_INF)
        p = np.exp(log_softmax(x))
        ps = np.take_along_axis(p, order, axis=-1)
        before = np.cumsum(ps, axis=-1) - ps
        kept_sorted = before < top_p
        kp = np.zeros_like(keep)
        np.put_along_axis(kp, order, kept_sorted, axis=-1)
        keep &= kp
    if min_p > 0.0:
        x = np.where(keep, lq, NEG_INF)
        m = np.max(x, axis=-1, keepdims=True)
        keep &= x >= m + np.log(min_p)
    return keep


@dataclass(frozen=True)
class SamplingSettings:
    """The per-step transforms (see the module docstring for what each does to the target)."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    truncate_after_mask: bool = True

    @property
    def truncates(self) -> bool:
        return (0 < self.top_k) or self.top_p < 1.0 or self.min_p > 0.0

    @classmethod
    def from_cfg(cls, cfg, **overrides) -> "SamplingSettings":
        """From a ``bes.config.DecodeCfg`` (fields that it lacks keep their defaults)."""
        values = {name: getattr(cfg, name) for name in ("temperature", "top_p", "top_k", "min_p",
                                                         "truncate_after_mask") if hasattr(cfg, name)}
        values.update(overrides)
        return cls(**values)


def _renorm(lq: np.ndarray, keep: np.ndarray) -> np.ndarray:
    return log_softmax(np.where(keep, lq, NEG_INF))


def step_increment(logp: np.ndarray, settings: SamplingSettings, allowed: np.ndarray | None = None,
                   keep: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Target increment of one step for every row.

    logp: float ``[K, V]`` model log-probabilities; allowed: ``bool[K, V]`` potential mask (None:
    everything allowed); keep: ``bool[K, V]`` tokens left by n-gram blocking and the copy cap
    (None: all). Returns ``(log_gamma float64[K, V], log_weight float64[K])`` with
    ``log_weight = logsumexp(log_gamma)`` the log incremental weight (``-inf``: nothing allowed).
    Semantics of ``settings.truncate_after_mask`` in the module docstring.
    """
    lq = tempered(logp, settings.temperature)
    n_k, _ = lq.shape
    finite = np.isfinite(lq)
    if allowed is None:
        allowed = np.ones(lq.shape, bool)
    if settings.truncate_after_mask:
        log_alpha = logsumexp(np.where(allowed, lq, NEG_INF))
        within = allowed & finite
        if keep is not None:
            both = within & keep
            ok = both.any(axis=-1)
            within = np.where(ok[:, None], both, within)
        kept = truncation_keep(lq, settings.top_k, settings.top_p, settings.min_p, within=within) \
            if settings.truncates else within
        q = _renorm(lq, kept)
        alive = np.isfinite(log_alpha)
        log_gamma = np.where(alive[:, None], q + np.where(alive, log_alpha, 0.0)[:, None], NEG_INF)
        return log_gamma, np.where(alive, log_alpha, NEG_INF)
    base = finite
    if keep is not None:
        both = base & keep
        ok = both.any(axis=-1)
        base = np.where(ok[:, None], both, base)
    kept = truncation_keep(lq, settings.top_k, settings.top_p, settings.min_p, within=base) \
        if settings.truncates else base
    p_trunc = _renorm(lq, kept)
    log_gamma = np.where(allowed, p_trunc, NEG_INF)
    return log_gamma, logsumexp(log_gamma)


def draw(log_gamma: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Inverse-CDF draw from each row of ``exp(log_gamma)`` (renormalised); ``u float64[K]`` in
    ``[0, 1)``. Returns ``int64[K]`` (-1 for a row with nothing allowed)."""
    lg = np.asarray(log_gamma, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    m = np.max(lg, axis=-1, keepdims=True)
    dead = ~np.isfinite(m[:, 0])
    p = np.exp(lg - np.where(dead[:, None], 0.0, m))
    c = np.cumsum(p, axis=-1)
    t = u[:, None] * c[:, -1:]
    out = np.argmax(c > t, axis=-1).astype(np.int64)
    out[dead] = -1
    return out


def greedy(log_gamma: np.ndarray) -> np.ndarray:
    """Arg-max of every row (lowest id on ties), -1 for a row with nothing allowed."""
    lg = np.asarray(log_gamma, dtype=np.float64)
    out = np.argmax(lg, axis=-1).astype(np.int64)
    out[~np.isfinite(np.max(lg, axis=-1))] = -1
    return out


# ------------------------------------------------------------------------- resampling
def ess(log_w: np.ndarray) -> float:
    """Effective sample size ``(sum w)^2 / sum w^2`` of log-weights (0 when all are -inf)."""
    lw = np.asarray(log_w, dtype=np.float64)
    m = np.max(lw)
    if not np.isfinite(m):
        return 0.0
    w = np.exp(lw - m)
    return float(w.sum() ** 2 / np.sum(w * w))


def systematic(log_w: np.ndarray, u: float) -> np.ndarray:
    """Systematic resampling: ancestors ``int64[K]`` for normalised weights ``exp(log_w)``
    and one uniform ``u`` in ``[0, 1)``. Raises ``ValueError`` when every weight is zero."""
    lw = np.asarray(log_w, dtype=np.float64)
    m = np.max(lw)
    if not np.isfinite(m):
        raise ValueError("cannot resample: every weight is zero")
    w = np.exp(lw - m)
    c = np.cumsum(w / w.sum())
    c[-1] = 1.0
    k = lw.shape[0]
    pos = (float(u) + np.arange(k)) / k
    return np.minimum(np.searchsorted(c, pos, side="right"), k - 1).astype(np.int64)


# ---------------------------------------------------------------- degeneracy guards
def _runs(tokens: Iterable[int], special: np.ndarray | None) -> list[list[int]]:
    """Split a token list at special tokens."""
    out, cur = [], []
    for t in tokens:
        t = int(t)
        if special is not None and 0 <= t < special.shape[0] and special[t]:
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        out.append(cur)
    return out


class NgramBlocker:
    """No-repeat n-gram blocking over each particle's response.

    Token ``v`` is blocked for particle ``j`` when ``(last n-1 response tokens, v)`` already
    occurred in that particle's response, unless

    - the n-gram contains a special token or no word token (``word[v]``: a token with a letter;
      punctuation and layout runs such as table rules are never blocked),
    - the n-gram occurs in a LIBRARY passage (copying the passage is never blocked),
    - the caller exempts the particle at this step (inside a code fence, inline code or a
      double quote, from the registers).

    ``blocked(j)`` lists the blocked ids; ``push(j, token)`` records a token; ``select(anc)``
    follows resampling. n-grams are recorded everywhere, code and quotes included.
    """

    def __init__(self, n: int, n_particles: int, *, word: np.ndarray | None = None,
                 special: np.ndarray | None = None, library: Sequence[Sequence[int]] = ()):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = int(n)
        self.word = word
        self.special = special
        self.library: set[tuple[int, ...]] = set()
        for passage in library:
            for run in _runs(passage, special):
                for i in range(len(run) - self.n + 1):
                    self.library.add(tuple(run[i:i + self.n]))
        self.hist: list[list[int]] = [[] for _ in range(n_particles)]
        self.succ: list[dict[tuple[int, ...], set[int]]] = [{} for _ in range(n_particles)]

    def _blockable(self, gram: tuple[int, ...]) -> bool:
        sp, wd = self.special, self.word
        if sp is not None and any(sp[t] for t in gram):
            return False
        return wd is None or any(wd[t] for t in gram)

    def push(self, j: int, token: int) -> None:
        h = self.hist[j]
        h.append(int(token))
        if len(h) >= self.n:
            gram = tuple(h[-self.n:])
            if self._blockable(gram):
                self.succ[j].setdefault(gram[:-1], set()).add(gram[-1])

    def blocked(self, j: int) -> list[int]:
        h = self.hist[j]
        if len(h) < self.n - 1:
            return []
        ctx = tuple(h[len(h) - self.n + 1:]) if self.n > 1 else ()
        cands = self.succ[j].get(ctx)
        if not cands:
            return []
        return sorted(v for v in cands if ctx + (v,) not in self.library)

    def select(self, ancestors: Sequence[int]) -> None:
        self.hist = [list(self.hist[a]) for a in ancestors]
        self.succ = [{k: set(v) for k, v in self.succ[a].items()} for a in ancestors]


class CopyCap:
    """Cap on verbatim copy runs from source texts: the decoder passes the prompt's LIBRARY
    passages (``bes.decode.smc.decode``), because the cap exists to stop passage
    regurgitation; the user's own text is never a source (repeating, quoting or editing it is
    often the task).

    Operational definition: let ``C = max_run``. Token ``v`` is blocked for particle ``j`` when
    the last ``C`` response tokens followed by ``v`` occur contiguously in a source run (a
    source's tokens split at special tokens), i.e. when ``v`` would extend a verbatim copy
    beyond ``C`` tokens. Exempt (never blocked): special tokens, *names and numbers*
    (``exempt[v]``: tokens with a digit, or whose first letter is an upper-case ASCII letter),
    and whole steps the caller exempts (*quoted text and code*: inside a double quote, inline
    code or a code fence, from the registers). A copy can therefore continue through names and
    numbers, but its ordinary words stop after ``C`` tokens.
    """

    def __init__(self, max_run: int, n_particles: int, source: Sequence[Sequence[int]], *,
                 exempt: np.ndarray | None = None, special: np.ndarray | None = None):
        if max_run < 1:
            raise ValueError("max_run must be >= 1")
        self.c = int(max_run)
        self.exempt = exempt
        self.special = special
        self.next: dict[tuple[int, ...], set[int]] = {}
        for src in source:
            for run in _runs(src, special):
                for i in range(len(run) - self.c):
                    self.next.setdefault(tuple(run[i:i + self.c]), set()).add(run[i + self.c])
        self.hist: list[list[int]] = [[] for _ in range(n_particles)]

    def push(self, j: int, token: int) -> None:
        self.hist[j].append(int(token))

    def blocked(self, j: int) -> list[int]:
        h = self.hist[j]
        if len(h) < self.c or not self.next:
            return []
        cands = self.next.get(tuple(h[-self.c:]))
        if not cands:
            return []
        ex = self.exempt
        return sorted(v for v in cands if ex is None or not ex[v])

    def select(self, ancestors: Sequence[int]) -> None:
        self.hist = [list(self.hist[a]) for a in ancestors]


def token_classes(token_bytes: Sequence[bytes], special: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(word bool[V], exempt bool[V])`` for :class:`NgramBlocker` and :class:`CopyCap`:
    ``word``: the token has an ASCII letter or a non-ASCII byte; ``exempt``: special, or has a
    digit, or its first letter is an upper-case ASCII letter (names, numbers)."""
    n = len(token_bytes)
    word = np.zeros(n, bool)
    exempt = np.zeros(n, bool)
    for v, b in enumerate(token_bytes):
        if special[v]:
            exempt[v] = True
            continue
        letters = [c for c in b if (65 <= c <= 90) or (97 <= c <= 122)]
        word[v] = bool(letters) or any(c >= 0x80 for c in b)
        exempt[v] = any(48 <= c <= 57 for c in b) or (bool(letters) and 65 <= letters[0] <= 90)
    return word, exempt


__all__ = ["counter_uniform", "log_softmax", "logsumexp", "tempered", "truncation_keep",
           "SamplingSettings", "step_increment", "draw", "greedy", "ess", "systematic",
           "NgramBlocker", "CopyCap", "token_classes", "STREAM_TOKEN", "STREAM_RESAMPLE",
           "NEG_INF"]
