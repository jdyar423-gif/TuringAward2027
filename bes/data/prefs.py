"""Preference pairs as fixed-shape batches (``docs/SPEC_MODEL_TRAIN.md`` sections 5 and 6).

A pair is a shared **prompt** -- a conversation ``BOS [turns ...] USER content END`` that ends
with a user turn -- and two assistant **responses**, ``chosen`` and ``rejected``: the token ids
of the assistant content, without the ``ASSISTANT`` role token and without the closing ``END``.

Sources
-------
* pair shards of the data pipeline (``tok/<source>/``, written by ``bes.data.tokenize_worker``):
  the chosen and the rejected conversation are consecutive segments with the same ``rec``; each
  is split at its last ``ASSISTANT`` token and the two prompts must be equal
  (:func:`pairs_from_shards`);
* in-memory token lists, for example particle-decoder winners against rejected samples or
  verifier-labelled samples (:func:`pairs_from_tokens`, :func:`pairs_from_scores`). A trailing
  ``ASSISTANT`` on the prompt and a trailing ``END`` on a response are stripped.

Rows
----
With ``T = seq_len`` every response becomes one row in the batch format of ``docs/SPEC.md``
section 3::

    content = prompt + [ASSISTANT] + response + [END]          L = len(content) <= T + 1
    tokens  int32[T + 1] = content + [PAD] * (T + 1 - L)       inputs tokens[:T], targets tokens[1:]
    w       float32[T]   = 1 where the target is a response token or the closing END, else 0
    seg     int32[T]     = 0 on the conversation (inputs t < L), 1 on the padding

so ``sum(w * log p(target))`` is the log-probability of the response given the prompt (the
final ``END`` is only ever a target, hence ``L <= T + 1``; padding is its own segment, as in
``docs/SPEC.md`` section 1). With ``mask_forced`` the result tokens of calculator spans (after
``EQ`` up to and including ``CALC_END``), which the host forces during generation, get weight 0
as well: the policy never samples them. The exact-layer features ``reg``, ``clock``,
``cand_tok`` and ``cand_feat`` are those of ``bes.exact.features.exact_features`` on the inputs
``tokens[:T]`` with ``seg``.

Filters (:func:`filter_pairs`, applied in this order)
------------------------------------------------------
``invalid``      malformed prompt (no leading ``BOS``, a ``BOS`` inside, not ending with the
                 ``END`` of a user turn), a response with special tokens other than the
                 calculator tokens, or an empty chosen response;
``too_long``     a row does not fit (``L > T + 1``);
``identical``    the two responses are equal;
``length_ratio`` length guard against preferences that length alone explains: the chosen
                 response (counting its ``END``) is more than ``max_len_ratio`` times longer than
                 the rejected one. Judge-scored data such as UltraFeedback favour long answers,
                 and DPO turns that bias into verbosity. Pass ``max_len_ratio=0`` for pairs
                 labelled by exact verifiers, where a length difference can be the point (for
                 example "answer in one word");
``duplicate``    the same prompt and responses were seen before.

Splitting and order
-------------------
:func:`split_pairs` holds out pairs by a hash of the prompt and the seed, so all pairs of one
prompt land on the same side and the split does not depend on the input order.
:class:`PairBatches` visits pairs in a fixed order (``shuffled_order(n, seed)`` for training).

Batches
-------
A batch has ``A = n_micro`` micro-batches of ``P = pairs_per_micro`` pairs, i.e. ``B = 2P`` rows
with the chosen row of pair ``p`` at row ``2p`` and its rejected row at ``2p + 1``:
``tokens int32[A, B, T+1]``, ``w float32[A, B, T]``, ``seg int32[A, B, T]``, the features
``[A, B, T, ...]`` (SPEC section 3 dtypes), ``pair_mask float32[A, P]`` (0 for the filler pairs
that complete the last batch) and ``ref float32[A, B]`` (the stored reference log-probability
of every row; zeros until :meth:`PairBatches.set_ref`). Numpy only; no JAX import.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bes.config import (ASSISTANT, BOS, CALC, CALC_END, CAND_SLOTS, END, EQ, N_CAND_FEAT,
                        N_CLOCKS, N_REG, N_SPECIAL, PAD, ROLE_TOKENS, USER, VOCAB_SIZE, PrefCfg)
from bes.data import chat_format, shards

FIRST_SPECIAL = VOCAB_SIZE - N_SPECIAL
#: Special ids allowed inside a response: the calculator span tokens.
RESPONSE_SPECIALS = (CALC, EQ, CALC_END)
#: Reasons of :func:`filter_pairs`, in the order they are tested.
DROP_REASONS = ("invalid", "too_long", "identical", "length_ratio", "duplicate")
#: Exact-layer feature keys of a batch (``docs/SPEC.md`` section 3).
FEATURE_KEYS = ("reg", "clock", "cand_tok", "cand_feat")

FeatureFn = Callable[[np.ndarray, np.ndarray, Any], Mapping[str, np.ndarray]]


# ----------------------------------------------------------------------------- pairs
@dataclass(frozen=True, eq=False)
class Pair:
    """One preference pair. ``prompt``: int32 ``[n_p]`` (``BOS ... USER content END``);
    ``chosen`` / ``rejected``: int32 ``[n]`` assistant contents (no role token, no ``END``).
    The arrays are read-only. ``source`` labels the origin (statistics only)."""

    prompt: np.ndarray
    chosen: np.ndarray
    rejected: np.ndarray
    source: str = ""

    def key(self) -> bytes:
        """16-byte digest of (prompt, chosen, rejected), used to drop duplicates."""
        h = hashlib.blake2b(digest_size=16)
        for a in (self.prompt, self.chosen, self.rejected):
            h.update(np.int64(a.size).tobytes())
            h.update(a.astype("<i4").tobytes())
        return h.digest()


def _ids(x: Any, what: str) -> np.ndarray:
    """Token ids as a read-only int32 array; raises ``ValueError`` on bad values."""
    a = np.asarray(x)
    if a.size == 0:
        a = np.zeros(0, np.int64)
    if a.ndim != 1:
        raise ValueError(f"{what}: token ids must be a 1-d sequence, got shape {a.shape}")
    if not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"{what}: token ids must be integers, got {a.dtype}")
    if a.size and (int(a.min()) < 0 or int(a.max()) >= VOCAB_SIZE):
        raise ValueError(f"{what}: token id outside [0, {VOCAB_SIZE})")
    out = a.astype(np.int32)
    out.setflags(write=False)
    return out


def _strip_end(a: np.ndarray) -> np.ndarray:
    return a[:-1] if a.size and int(a[-1]) == END else a


def make_pair(prompt: Any, chosen: Any, rejected: Any, source: str = "") -> Pair:
    """A :class:`Pair` from token lists; a trailing ``ASSISTANT`` on the prompt (a generation
    prompt) and one trailing ``END`` on each response are removed."""
    p = _ids(prompt, "prompt")
    if p.size and int(p[-1]) == ASSISTANT:
        p = p[:-1]
    return Pair(p, _strip_end(_ids(chosen, "chosen")), _strip_end(_ids(rejected, "rejected")),
                source)


def prompt_problem(prompt: np.ndarray) -> str | None:
    """Why ``prompt`` is not ``BOS ... USER content END`` (``None`` when it is)."""
    p = np.asarray(prompt)
    if p.size < 3:
        return "prompt shorter than BOS USER END"
    if int(p[0]) != BOS:
        return "prompt does not start with BOS"
    if bool((p[1:] == BOS).any()):
        return "BOS inside the prompt"
    if int(p[-1]) != END:
        return "prompt does not end with END"
    roles = np.flatnonzero(np.isin(p, ROLE_TOKENS))
    if roles.size == 0 or int(p[roles[-1]]) != USER:
        return "the last turn of the prompt is not a user turn"
    if bool((p[roles[-1] + 1:-1] == END).any()):
        return "the last turn of the prompt is closed twice"
    return None


def response_problem(resp: np.ndarray) -> str | None:
    """Why ``resp`` cannot be an assistant content (``None`` when it can)."""
    r = np.asarray(resp)
    special = r[r >= FIRST_SPECIAL]
    if special.size and not bool(np.isin(special, RESPONSE_SPECIALS).all()):
        return "special token other than a calculator token in the response"
    return None


def row_length(prompt: np.ndarray, resp: np.ndarray) -> int:
    """``L = len(prompt + [ASSISTANT] + resp + [END])``; the row fits when ``L <= T + 1``."""
    return int(len(prompt)) + int(len(resp)) + 2


def forced_mask(resp: np.ndarray) -> np.ndarray:
    """``bool[len(resp)]``: True on calculator results forced by the host (after ``EQ`` up to
    and including ``CALC_END``)."""
    r = np.asarray(resp)
    mask = np.zeros(r.size, bool)
    if not bool(np.isin(r, RESPONSE_SPECIALS).any()):
        return mask
    for span in chat_format.find_calc_spans(r):
        if span.eq >= 0:
            mask[span.eq + 1:(span.end + 1 if span.end >= 0 else span.stop)] = True
    return mask


def target_count(resp: np.ndarray, mask_forced: bool = True) -> int:
    """Number of weighted targets of a response row (its sampled tokens plus ``END``)."""
    n = int(len(resp)) + 1
    return n - int(forced_mask(resp).sum()) if mask_forced else n


def response_row(prompt: np.ndarray, resp: np.ndarray, seq_len: int,
                 mask_forced: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One row (module docstring): ``(tokens int32[T+1], w float32[T], seg int32[T])``."""
    n_p, n_r = int(len(prompt)), int(len(resp))
    length = n_p + n_r + 2
    if length > seq_len + 1:
        raise ValueError(f"row of {length} tokens does not fit seq_len={seq_len} (needs <= T + 1)")
    tokens = np.full(seq_len + 1, PAD, np.int32)
    tokens[:n_p] = prompt
    tokens[n_p] = ASSISTANT
    tokens[n_p + 1:n_p + 1 + n_r] = resp
    tokens[length - 1] = END
    tw = np.ones(n_r + 1, np.float32)          # targets: the response tokens, then END
    if mask_forced and n_r:
        tw[:n_r][forced_mask(resp)] = 0.0
    w = np.zeros(seq_len, np.float32)
    w[n_p:length - 1] = tw                     # input t predicts tokens[t + 1]
    seg = np.zeros(seq_len, np.int32)
    seg[length:] = 1                           # inputs t >= L are padding
    return tokens, w, seg


def pair_rows(pair: Pair, seq_len: int, mask_forced: bool = True) -> dict[str, np.ndarray]:
    """Chosen and rejected rows of ``pair``: ``tokens int32[2, T+1]``, ``w float32[2, T]``,
    ``seg int32[2, T]`` (row 0 chosen, row 1 rejected)."""
    rows = [response_row(pair.prompt, r, seq_len, mask_forced) for r in (pair.chosen, pair.rejected)]
    return {"tokens": np.stack([r[0] for r in rows]), "w": np.stack([r[1] for r in rows]),
            "seg": np.stack([r[2] for r in rows])}


def row_tokens(prompt: Any, resp: Any) -> np.ndarray:
    """``int32`` content ``prompt + [ASSISTANT] + resp + [END]`` (prompt and response normalised
    as in :func:`make_pair`); used for scoring sampled responses, e.g. by the critic."""
    p = make_pair(prompt, resp, [])
    return np.concatenate([p.prompt, [ASSISTANT], p.chosen, [END]]).astype(np.int32)


# ---------------------------------------------------------------------------- filters
def filter_pairs(pairs: Iterable[Pair], seq_len: int, *,
                 max_len_ratio: float = PrefCfg.max_len_ratio,
                 dedupe: bool = True) -> tuple[list[Pair], dict[str, int]]:
    """Drop pairs that cannot or should not be trained on (module docstring).

    Returns ``(kept, counts)`` with ``counts[reason]`` for every reason of :data:`DROP_REASONS`
    and ``counts["kept"]``. ``max_len_ratio <= 0`` turns the length guard off.
    """
    counts = {r: 0 for r in DROP_REASONS}
    kept: list[Pair] = []
    seen: set[bytes] = set()
    for p in pairs:
        if (prompt_problem(p.prompt) or response_problem(p.chosen)
                or response_problem(p.rejected) or p.chosen.size == 0):
            reason = "invalid"
        elif max(row_length(p.prompt, p.chosen), row_length(p.prompt, p.rejected)) > seq_len + 1:
            reason = "too_long"
        elif np.array_equal(p.chosen, p.rejected):
            reason = "identical"
        elif max_len_ratio > 0 and p.chosen.size + 1 > max_len_ratio * (p.rejected.size + 1):
            reason = "length_ratio"
        else:
            reason = None
            if dedupe:
                key = p.key()
                if key in seen:
                    reason = "duplicate"
                seen.add(key)
        if reason is None:
            kept.append(p)
        else:
            counts[reason] += 1
    counts["kept"] = len(kept)
    return kept, counts


# ------------------------------------------------------------------- split and order
def prompt_bucket(prompt: np.ndarray, seed: int) -> float:
    """Deterministic number in ``[0, 1)`` from the prompt ids and the seed (BLAKE2b)."""
    h = hashlib.blake2b(digest_size=8)
    h.update(np.int64(seed).tobytes())
    h.update(np.asarray(prompt).astype("<i4").tobytes())
    return int.from_bytes(h.digest(), "little") / 2.0 ** 64


def split_pairs(pairs: Sequence[Pair], heldout_frac: float,
                seed: int = 0) -> tuple[list[Pair], list[Pair]]:
    """``(train, heldout)``: a prompt is held out when its bucket is below ``heldout_frac``.

    Both sides keep the input order. When the hash puts every prompt on one side although there
    are at least two distinct prompts, the prompt nearest the threshold moves across, so both
    sides are non-empty.
    """
    if heldout_frac <= 0:
        return list(pairs), []
    buckets: dict[bytes, float] = {}
    keys = []
    for p in pairs:
        k = p.prompt.astype("<i4").tobytes()
        if k not in buckets:
            buckets[k] = prompt_bucket(p.prompt, seed)
        keys.append(k)
    held = {k for k, b in buckets.items() if b < heldout_frac}
    if len(buckets) >= 2 and not held:
        held = {min(buckets, key=buckets.__getitem__)}
    elif len(buckets) >= 2 and len(held) == len(buckets):
        held.discard(max(buckets, key=buckets.__getitem__))
    train = [p for p, k in zip(pairs, keys) if k not in held]
    heldout = [p for p, k in zip(pairs, keys) if k in held]
    return train, heldout


def shuffled_order(n: int, seed: int) -> np.ndarray:
    """Deterministic permutation ``int64[n]`` of ``range(n)``."""
    return np.random.default_rng(seed).permutation(n).astype(np.int64)


@dataclass
class PrefSplit:
    """Filtered pairs split into training and held-out pairs, with the filter statistics."""

    train: list[Pair]
    heldout: list[Pair]
    stats: dict[str, Any]


def prepare(groups: Mapping[str, Iterable[Pair]] | Iterable[Pair], seq_len: int, *,
            heldout_frac: float = PrefCfg.heldout_frac, seed: int = PrefCfg.seed,
            max_len_ratio: float | Mapping[str, float] = PrefCfg.max_len_ratio) -> PrefSplit:
    """Filter, merge and split preference pairs.

    groups: pairs, or ``{name: pairs}`` with a per-group length guard when ``max_len_ratio`` is
    a mapping (missing names use ``PrefCfg.max_len_ratio``). Duplicates are dropped across
    groups. ``stats``: ``{name: filter counts}``, ``"train"`` and ``"heldout"`` sizes.
    """
    named = dict(groups) if isinstance(groups, Mapping) else {"pairs": groups}
    kept: list[Pair] = []
    seen: set[bytes] = set()
    stats: dict[str, Any] = {}
    for name, group in named.items():
        ratio = (max_len_ratio.get(name, PrefCfg.max_len_ratio)
                 if isinstance(max_len_ratio, Mapping) else max_len_ratio)
        ok, counts = filter_pairs(group, seq_len, max_len_ratio=ratio, dedupe=True)
        fresh = []
        for p in ok:
            k = p.key()
            if k in seen:
                counts["duplicate"] += 1
                counts["kept"] -= 1
                continue
            seen.add(k)
            fresh.append(p)
        kept += fresh
        stats[name] = counts
    train, heldout = split_pairs(kept, heldout_frac, seed)
    stats["train"], stats["heldout"] = len(train), len(heldout)
    return PrefSplit(train, heldout, stats)


# --------------------------------------------------------------------------- sources
def split_response(segment: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """``(prompt, response)`` of a conversation segment whose last turn is an assistant turn
    closed by the final ``END``; ``None`` when the segment has no such last turn."""
    seg = np.asarray(segment).astype(np.int32)
    if seg.size < 2 or int(seg[-1]) != END:
        return None
    starts = np.flatnonzero(seg == ASSISTANT)
    if starts.size == 0:
        return None
    a = int(starts[-1])
    body = seg[a + 1:-1]
    if bool(np.isin(body, (*ROLE_TOKENS, END, BOS)).any()):
        return None
    return seg[:a].copy(), body.copy()


def pairs_from_shards(directory: str | Path, *, source: str | None = None,
                      limit: int | None = None) -> tuple[list[Pair], dict[str, int]]:
    """Pairs of the committed pair shards in ``directory`` (``tok/<source>/``).

    Segments ``2i`` (chosen) and ``2i + 1`` (rejected) of a shard form a pair. Dropped and
    counted: ``rec_mismatch`` (the two segments come from different input rows),
    ``unparsable`` (a segment does not end with an assistant turn), ``prompt_mismatch``;
    shards whose manifest entry is not a pair shard are skipped (``not_pair_shards``) and so
    are pair shards with an odd number of segments (``odd_shards``). Filtering is
    :func:`filter_pairs`' job. Returns ``(pairs, counts)`` with ``counts["pairs"]``.
    """
    directory = Path(directory)
    counts: Counter[str] = Counter()
    out: list[Pair] = []
    manifest = shards.read_manifest(directory)
    for sid in sorted(manifest):
        if limit is not None and len(out) >= limit:
            break
        entry = manifest[sid]
        if not entry.get("pairs"):
            counts["not_pair_shards"] += 1
            continue
        sh = shards.open_shard(directory, entry)
        if len(sh) % 2:
            counts["odd_shards"] += 1
            continue
        label = source or str(entry.get("source") or directory.name)
        for i in range(0, len(sh), 2):
            if limit is not None and len(out) >= limit:
                break
            if sh.recs is not None and int(sh.recs[i]) != int(sh.recs[i + 1]):
                counts["rec_mismatch"] += 1
                continue
            c, r = split_response(sh.segment(i)), split_response(sh.segment(i + 1))
            if c is None or r is None:
                counts["unparsable"] += 1
                continue
            if not np.array_equal(c[0], r[0]):
                counts["prompt_mismatch"] += 1
                continue
            out.append(make_pair(c[0], c[1], r[1], label))
    counts["pairs"] = len(out)
    return out, dict(counts)


def pairs_from_tokens(items: Iterable[Any], source: str = "memory") -> list[Pair]:
    """Pairs from ``(prompt, chosen, rejected)`` tuples or mappings with those keys (and an
    optional ``"source"``); see :func:`make_pair` for the normalisation."""
    out = []
    for item in items:
        if isinstance(item, Mapping):
            out.append(make_pair(item["prompt"], item["chosen"], item["rejected"],
                                 str(item.get("source", source))))
        else:
            prompt, chosen, rejected = item
            out.append(make_pair(prompt, chosen, rejected, source))
    return out


def pairs_from_scores(prompt: Any, responses: Sequence[Any], scores: Sequence[float], *,
                      min_gap: float = 0.0, max_pairs: int | None = None,
                      source: str = "scored") -> list[Pair]:
    """Pairs from scored samples of one prompt (verifier pass = 1 / fail = 0, or a score).

    Samples are ranked by score (stable for ties); the best is paired with the worst, the
    second best with the second worst, and so on, while the score difference exceeds
    ``min_gap`` (equal scores never form a pair). Each sample is used at most once.
    """
    s = np.asarray(scores, np.float64)
    if s.shape != (len(responses),):
        raise ValueError("one score per response is required")
    rank = np.argsort(-s, kind="stable")
    out: list[Pair] = []
    lo, hi = 0, len(rank) - 1
    while lo < hi and (max_pairs is None or len(out) < max_pairs):
        i, j = int(rank[lo]), int(rank[hi])
        if not s[i] - s[j] > min_gap:
            break
        out.append(make_pair(prompt, responses[i], responses[j], source))
        lo, hi = lo + 1, hi - 1
    return out


# --------------------------------------------------------------------------- batches
def _default_features(tokens: np.ndarray, seg: np.ndarray, tok: Any) -> Mapping[str, np.ndarray]:
    from bes.exact.features import exact_features

    return exact_features(tokens, seg, tok)


def row_bytes(seq_len: int, features: bool = True) -> int:
    """Host bytes of one row of a batch (tokens, w, seg and, with ``features``, the exact
    features in their SPEC section 3 dtypes)."""
    n = 4 * (seq_len + 1) + 4 * seq_len + 4 * seq_len
    if features:
        n += seq_len * (2 * N_REG + 4 * N_CLOCKS + 4 * CAND_SLOTS + CAND_SLOTS * N_CAND_FEAT)
    return n


def build_rows(batch_pairs: Sequence["Pair | None"], shape: tuple[int, int, int], mask_forced: bool,
               features: bool, feature_fn: "FeatureFn", tok: Any) -> dict[str, np.ndarray]:
    """The rows of one batch (``PairBatches._rows``): ``batch_pairs`` are its ``A * P`` pairs in
    order (``None``: a filler pair); arrays are shaped ``[A, B, ...]``."""
    a, b, t = shape
    tokens = np.full((a * b, t + 1), PAD, np.int32)
    w = np.zeros((a * b, t), np.float32)
    seg = np.ones((a * b, t), np.int32)
    tokens[:, 0] = BOS                     # filler rows: BOS then padding, zero weight
    seg[:, 0] = 0
    for j, pair in enumerate(batch_pairs):
        if pair is None:
            continue
        r = pair_rows(pair, t, mask_forced)
        tokens[2 * j:2 * j + 2], w[2 * j:2 * j + 2], seg[2 * j:2 * j + 2] = (
            r["tokens"], r["w"], r["seg"])
    out = {"tokens": tokens, "w": w, "seg": seg}
    if features:
        feats = feature_fn(tokens[:, :t], seg, tok)
        out["reg"] = np.asarray(feats["reg"], np.int16)
        out["clock"] = np.asarray(feats["clock"], np.int32)
        out["cand_tok"] = np.asarray(feats["cand_tok"], np.int32)
        out["cand_feat"] = np.asarray(feats["cand_feat"], np.int8)
    return {key: v.reshape((a, b) + v.shape[1:]) for key, v in out.items()}


_TASK_TOK: Any = None


def _rows_task(args: tuple) -> tuple[int, dict[str, np.ndarray]]:
    """Pool worker of :meth:`PairBatches.warm`: the rows of one batch with the default exact
    features and the committed tokenizer (loaded once per process)."""
    global _TASK_TOK
    k, batch_pairs, shape, mask_forced = args
    if _TASK_TOK is None:
        from bes.tokenizer.tok import load

        _TASK_TOK = load()
    return k, build_rows(batch_pairs, shape, mask_forced, True, _default_features, _TASK_TOK)


class PairBatches:
    """Fixed-shape batches over ``pairs`` in a fixed order (module docstring, "Batches").

    pairs: filtered pairs (each row must fit ``seq_len``). order: the visiting order, a
    permutation of ``range(n)`` (default the identity), so every pair is visited exactly once
    per pass. tok: tokenizer for the exact features
    (default: the committed one, loaded on first use). features: compute the exact features
    (without them the model sees its feature paths disabled). feature_fn: ``(tokens [N, T],
    seg [N, T], tok) -> {reg, clock, cand_tok, cand_feat}`` replacing
    ``bes.exact.features.exact_features``. cache_bytes: keep the computed rows of every batch
    in host memory when all of them fit this budget (they are recomputed otherwise).

    ``ref`` (``float32[n_pairs, 2]``: chosen, rejected) holds the reference log-probabilities
    once :meth:`set_ref` was called; every batch carries them per row.
    """

    def __init__(self, pairs: Sequence[Pair], seq_len: int, pairs_per_micro: int, n_micro: int,
                 *, order: Sequence[int] | np.ndarray | None = None, tok: Any = None,
                 features: bool = True, feature_fn: FeatureFn | None = None,
                 mask_forced: bool = PrefCfg.mask_forced, cache_bytes: int = 0):
        if pairs_per_micro < 1 or n_micro < 1 or seq_len < 3:
            raise ValueError("need pairs_per_micro >= 1, n_micro >= 1 and seq_len >= 3")
        self.pairs = list(pairs)
        for i, p in enumerate(self.pairs):
            if max(row_length(p.prompt, p.chosen), row_length(p.prompt, p.rejected)) > seq_len + 1:
                raise ValueError(f"pair {i} does not fit seq_len={seq_len}; filter the pairs first")
        n = len(self.pairs)
        self.order = (np.arange(n, dtype=np.int64) if order is None
                      else np.asarray(order, dtype=np.int64).reshape(-1))
        if not np.array_equal(np.sort(self.order), np.arange(n)):
            raise ValueError("order must be a permutation of range(len(pairs))")
        self.seq_len, self.pairs_per_micro, self.n_micro = seq_len, pairs_per_micro, n_micro
        self.features, self.mask_forced = features, mask_forced
        self._tok = tok
        self._feature_fn = feature_fn if feature_fn is not None else _default_features
        per_batch = n_micro * 2 * pairs_per_micro * row_bytes(seq_len, features)
        self._cache_on = cache_bytes > 0 and len(self) * per_batch <= cache_bytes
        self._cache: dict[int, dict[str, np.ndarray]] = {}
        self.ref: np.ndarray | None = None

    # -------------------------------------------------------------------- sizes
    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    @property
    def pairs_per_batch(self) -> int:
        return self.pairs_per_micro * self.n_micro

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(A, B, T)`` of every batch."""
        return self.n_micro, 2 * self.pairs_per_micro, self.seq_len

    def __len__(self) -> int:
        return -(-int(self.order.size) // self.pairs_per_batch)

    # -------------------------------------------------------------- reference
    def set_ref(self, ref: np.ndarray) -> None:
        """Store the reference log-probabilities ``float32[n_pairs, 2]`` (chosen, rejected)."""
        ref = np.asarray(ref, np.float32)
        if ref.shape != (self.n_pairs, 2):
            raise ValueError(f"ref must have shape {(self.n_pairs, 2)}, got {ref.shape}")
        if not np.isfinite(ref).all():
            raise ValueError("reference log-probabilities must be finite")
        self.ref = ref

    def target_counts(self) -> np.ndarray:
        """``int64[n_pairs, 2]``: weighted targets of the chosen and the rejected row."""
        return np.asarray([[target_count(p.chosen, self.mask_forced),
                            target_count(p.rejected, self.mask_forced)] for p in self.pairs],
                          np.int64).reshape(-1, 2)

    # ---------------------------------------------------------------- batches
    def indices(self, k: int) -> np.ndarray:
        """``int64[A, P]`` pair indices of batch ``k`` (-1 marks a filler pair)."""
        if not 0 <= k < len(self):
            raise IndexError(k)
        flat = np.full(self.pairs_per_batch, -1, np.int64)
        part = self.order[k * self.pairs_per_batch:(k + 1) * self.pairs_per_batch]
        flat[:part.size] = part
        return flat.reshape(self.n_micro, self.pairs_per_micro)

    def _tokenizer(self) -> Any:
        if self._tok is None:
            from bes.tokenizer.tok import load

            self._tok = load()
        return self._tok

    def _rows(self, k: int) -> dict[str, np.ndarray]:
        cached = self._cache.get(k)
        if cached is not None:
            return cached
        out = build_rows(self._batch_pairs(k), self.shape, self.mask_forced, self.features,
                         self._feature_fn, self._tokenizer() if self.features else None)
        if self._cache_on:
            for v in out.values():
                v.setflags(write=False)            # shared by every later pass
            self._cache[k] = out
        return out

    def _batch_pairs(self, k: int) -> list[Pair | None]:
        return [self.pairs[i] if i >= 0 else None for i in self.indices(k).reshape(-1).tolist()]

    def warm(self, workers: int, log: Callable[[str], None] | None = None) -> int:
        """Compute the rows of every batch not yet kept, in ``workers`` processes, and keep them:
        the exact features are host work of one core otherwise (about 19 ms a pair at 1,024
        tokens), paid inside the first pass over the pairs. Only with the cache on, the default
        features and the committed tokenizer (the workers load it themselves); returns the
        number of batches computed (0: nothing to do, or the conditions do not hold). The rows
        are the ones :meth:`_rows` computes (a test compares them)."""
        if (workers <= 1 or not self._cache_on or not self.features
                or self._feature_fn is not _default_features):
            return 0
        if self._tok is not None:
            from bes.tokenizer.tok import load

            if getattr(self._tok, "sha256", None) != load().sha256:
                return 0                             # another tokenizer: this process computes
        todo = [k for k in range(len(self)) if k not in self._cache]
        if not todo:
            return 0
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        tasks = ((k, self._batch_pairs(k), self.shape, self.mask_forced) for k in todo)
        with ProcessPoolExecutor(min(int(workers), len(todo)),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            for n, (k, rows) in enumerate(pool.map(_rows_task, tasks, chunksize=1)):
                for v in rows.values():
                    v.setflags(write=False)
                self._cache[k] = rows
                if log is not None and (n + 1) % 100 == 0:
                    log(f"features of {n + 1}/{len(todo)} batches")
        return len(todo)

    def get(self, k: int) -> dict[str, np.ndarray]:
        """Batch ``k`` (module docstring): rows, ``pair_mask``, ``ref`` and the host-only
        ``index int64[A, P]``."""
        idx = self.indices(k)
        batch = dict(self._rows(k))
        batch["pair_mask"] = (idx >= 0).astype(np.float32)
        ref = np.zeros(idx.shape + (2,), np.float32)
        if self.ref is not None:
            ok = idx >= 0
            ref[ok] = self.ref[idx[ok]]
        batch["ref"] = ref.reshape(idx.shape[0], -1)      # [A, B]: chosen 2p, rejected 2p + 1
        batch["index"] = idx
        return batch

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        for k in range(len(self)):
            yield self.get(k)
