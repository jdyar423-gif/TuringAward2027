"""Exact backward-history structure of token rows, used by the candidate tape (numpy only).

Notation
--------
A batch ``tokens [B, T]`` is flattened row-major to ``x [N]`` (``N = B*T``); positions are flat
indices ``g``. A *segment* starts at the first position of a row, wherever ``seg`` changes
inside a row (the loader packs segments as contiguous runs) and at every ``BOS`` (a segment is
one document or one conversation and always starts with ``BOS``; ``bes.exact.registers`` resets
there as well). A *boundary* is the first position of a segment or a role token (``SYSTEM,
USER, ASSISTANT, LIBRARY``). ``ts(g)`` is the latest boundary at or before ``g`` (the start of
``g``'s turn) and ``avail(g) = g - ts(g) + 1``.

The *history* of ``g`` is the backward string ``S_g = (x[g], x[g-1], ..., x[ts(g)])`` of length
``avail(g)``. A context may therefore begin with the boundary token (``[USER, "Hi"]`` is a valid
2-gram) but never reaches across it. The *match length* ``lcs(t, j)`` of two positions of the
same segment is the length of the longest common prefix of ``S_t`` and ``S_j``; ``lcs >= L``
holds exactly when the last ``L`` tokens ending at ``t`` equal the last ``L`` tokens ending at
``j`` and neither of the two contexts crosses a boundary. For ``L <= 32`` the positions sharing
one ``L``-token context form an *``L``-block*.

Algorithm (every step is vectorised over the whole batch, no Python loop over positions)
--------------------------------------------------------------------------------------
1. Prefix doubling on histories. ``R_0(g)`` is the dense rank of ``(segment, x[g])``;
   ``R_{k+1}(g)`` is the dense rank of the pair ``(R_k(g), R_k(g - 2^k) + 1)``, or of
   ``(R_k(g), 0)`` when ``avail(g) <= 2^k`` (the history ends inside the first half). By
   induction ``R_k`` is the lexicographic rank of ``S_g`` truncated to ``2^k`` symbols, with
   end-of-string smaller than every token: equal ranks mean equal truncated histories, lengths
   included. Six levels reach ``MAX_MATCH = 32``. (The second half of the history of ``g`` is
   the history of ``g - 2^k`` because no boundary lies strictly inside ``(ts(g), g]``.)
2. Sort positions by ``(R_5, g)``. Histories sharing a prefix of any length ``L <= 32`` are now
   contiguous, so every ``L``-block is a contiguous run of this order (positions inside a run
   are not sorted).
3. Common-prefix length of each pair of neighbours by binary lifting over the six rank levels
   (``adj``, capped at 32). An ``L``-block is a maximal run of entries whose ``adj >= L``.
4. Longest previous match ``L*(t) = min(32, max_{j<t} lcs(t, j))``: ``t`` has an earlier match
   of length ``>= L`` iff its ``L``-block holds a smaller position (a segmented minimum); the
   property is monotone in ``L``. Runs that are singletons at level ``L`` are dropped before
   level ``L + 1``, so the per-level work shrinks quickly on natural text.

Costs: seven integer sorts of ``N`` keys and ``O(32 N)`` vectorised element operations in the
worst case. Keys stay below ``2^63`` for ``N < 2^20``; ``bes.exact.tape`` processes large
batches in chunks of rows.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple

import numpy as np

from bes.config import ASSISTANT, BOS, N_SPECIAL, ROLE_TOKENS, VOCAB_SIZE

#: Longest match length the tape distinguishes (match lengths are capped here).
MAX_MATCH = 32
#: Rank levels of the prefix doubling: histories truncated to 1, 2, 4, 8, 16, 32 symbols.
N_RANK_LEVELS = MAX_MATCH.bit_length()
assert 1 << (N_RANK_LEVELS - 1) == MAX_MATCH

FIRST_SPECIAL = VOCAB_SIZE - N_SPECIAL
#: Role codes (as in ``bes.config.REG_FIELDS`` ``role``): 0 raw text / none, 1 system, 2 user,
#: 3 assistant, 4 library.
ROLE_CODE_OF_TOKEN: dict[int, int] = {tok: i + 1 for i, tok in enumerate(ROLE_TOKENS)}
ROLE_ASSISTANT = ROLE_CODE_OF_TOKEN[ASSISTANT]
BOUNDARY_TOKENS: frozenset[int] = frozenset((BOS, *ROLE_TOKENS))

_ROLE_TABLE = np.zeros(VOCAB_SIZE, dtype=np.int8)
_BOUNDARY_TABLE = np.zeros(VOCAB_SIZE, dtype=bool)
for _tok, _code in ROLE_CODE_OF_TOKEN.items():
    _ROLE_TABLE[_tok] = _code
_BOUNDARY_TABLE[list(BOUNDARY_TOKENS)] = True

#: Largest number of flattened positions one call may handle (keeps sort keys below 2^63).
MAX_FLAT = 1 << 20


class Layout(NamedTuple):
    """Turn structure of a flattened batch; every array has shape ``[N]``, ``N = B*T``."""

    x: np.ndarray        # int64 token ids
    row_len: int         # T
    seg_id: np.ndarray   # int64 dense index of the segment over the whole batch
    ts: np.ndarray       # int64 flat index of the turn start (latest boundary at or before g)
    avail: np.ndarray    # int64 history length g - ts(g) + 1 (>= 1)
    role: np.ndarray     # int8 role code of the turn of g (code of the token at ts(g), else 0)
    succ: np.ndarray     # int64 successor x[g+1] inside the row, -1 at the last position of a row


def layout(tokens: np.ndarray, seg: np.ndarray | None = None) -> Layout:
    """Flatten ``tokens int[B, T]`` (and ``seg int[B, T]``, contiguous segment runs) into a
    :class:`Layout`. Segments start at row starts, at changes of ``seg`` and at every ``BOS``
    (without ``seg``: rows split at ``BOS`` only). Raises ``ValueError`` on ids outside
    ``[0, VOCAB_SIZE)`` or mismatched shapes."""
    tokens = np.asarray(tokens)
    if tokens.ndim != 2:
        raise ValueError(f"tokens must be [B, T], got shape {tokens.shape}")
    n_rows, row_len = tokens.shape
    x = tokens.astype(np.int64).reshape(-1)
    n = x.size
    if n and (x.min() < 0 or x.max() >= VOCAB_SIZE):
        raise ValueError(f"token ids must lie in [0, {VOCAB_SIZE})")
    start = np.zeros(n, dtype=bool)
    if n:
        start[::row_len] = True
    if seg is not None:
        seg = np.asarray(seg)
        if seg.shape != tokens.shape:
            raise ValueError(f"seg shape {seg.shape} differs from tokens shape {tokens.shape}")
        s = seg.reshape(-1)
        start[1:] |= s[1:] != s[:-1]
    start |= x == BOS
    seg_id = np.cumsum(start, dtype=np.int64) - 1
    idx = np.arange(n, dtype=np.int64)
    ts = np.maximum.accumulate(np.where(start | _BOUNDARY_TABLE[x], idx, 0)) if n else idx
    avail = idx - ts + 1
    role = _ROLE_TABLE[x[ts]] if n else np.zeros(0, dtype=np.int8)
    succ = np.empty(n, dtype=np.int64)
    if n:
        succ[:-1] = x[1:]
        succ[row_len - 1::row_len] = -1
    return Layout(x, row_len, seg_id, ts, avail, role, succ)


def _dense_rank(key: np.ndarray) -> tuple[np.ndarray, int]:
    """Dense ranks ``int64[n]`` of ``key`` (equal keys share a rank, order preserved) and the
    number of distinct keys."""
    order = np.argsort(key)
    sk = key[order]
    step = np.empty(key.size, dtype=np.int64)
    step[0] = 0
    np.not_equal(sk[1:], sk[:-1], out=step[1:], casting="unsafe")
    np.cumsum(step, out=step)
    rank = np.empty_like(step)
    rank[order] = step
    return rank, int(step[-1]) + 1


def history_ranks(lay: Layout) -> list[np.ndarray]:
    """``[R_0, ..., R_5]``: ``R_k int64[N]`` is the lexicographic rank of the history of every
    position truncated to ``2^k`` symbols (segment first, end-of-string smallest)."""
    n = lay.x.size
    if n > MAX_FLAT:
        raise ValueError(f"at most {MAX_FLAT} positions per call (got {n}); split the batch")
    if n == 0:
        return [np.zeros(0, dtype=np.int64)] * N_RANK_LEVELS
    idx = np.arange(n, dtype=np.int64)
    rank, n_distinct = _dense_rank(lay.seg_id * VOCAB_SIZE + lay.x)
    ranks = [rank]
    for k in range(N_RANK_LEVELS - 1):
        if n_distinct == n:          # every truncated history is already unique: ranks are final
            ranks.append(rank)
            continue
        h = 1 << k
        second = np.where(lay.avail > h, rank[np.maximum(idx - h, 0)] + 1, 0)
        rank, n_distinct = _dense_rank(rank * (n_distinct + 1) + second)
        ranks.append(rank)
    return ranks


class SortedHistories(NamedTuple):
    """Positions sorted by ``(history truncated to 32, position)`` and neighbour prefix lengths.

    ``order int64[N]``: flat positions in sorted order. ``adj int64[N]``: ``adj[r]`` is
    ``min(32, lcs(order[r-1], order[r]))`` (0 across segments), ``adj[0] = -1``.
    """

    order: np.ndarray
    adj: np.ndarray


def sort_histories(lay: Layout, ranks: list[np.ndarray] | None = None) -> SortedHistories:
    """Sort the histories of all positions and compute neighbour common-prefix lengths."""
    if ranks is None:
        ranks = history_ranks(lay)
    n = lay.x.size
    idx = np.arange(n, dtype=np.int64)
    order = np.argsort(ranks[-1] * max(n, 1) + idx)
    adj = np.empty(n, dtype=np.int64)
    if n:
        adj[0] = -1
    if n > 1:
        a, b = order[:-1], order[1:]
        lim = np.minimum(lay.avail[a], lay.avail[b])
        tot = np.zeros(n - 1, dtype=np.int64)
        for k in reversed(range(N_RANK_LEVELS)):
            h = 1 << k
            ok = (tot + h <= MAX_MATCH) & (tot < lim)
            rk = ranks[k]
            same = rk[np.where(ok, a - tot, 0)] == rk[np.where(ok, b - tot, 0)]
            tot += np.where(ok & same, h, 0)
        adj[1:] = np.minimum(tot, lim)
    return SortedHistories(order, adj)


class LevelBlocks(NamedTuple):
    """Members of the non-singleton ``level``-blocks, in sorted-history order.

    ``pos int64[m]`` flat positions; ``block int64[m]`` block index (dense, increasing);
    ``n_blocks`` number of distinct blocks. Every position whose ``level``-token context
    occurs at least twice in its segment appears exactly once.
    """

    level: int
    pos: np.ndarray
    block: np.ndarray
    n_blocks: int


def iter_level_blocks(sh: SortedHistories, levels: range | list[int]) -> Iterator[LevelBlocks]:
    """Yield :class:`LevelBlocks` for increasing ``levels`` (each in ``[1, 32]``).

    Entries that are singletons at a level are dropped before the next level (a singleton stays
    one at every larger level), so the cost follows the amount of repetition in the batch.
    Stops early once no block of size two is left.
    """
    levels = list(levels)
    if any(not a < b for a, b in zip([0, *levels], levels)) or (levels and levels[-1] > MAX_MATCH):
        raise ValueError("levels must be increasing and lie in [1, MAX_MATCH]")
    return _level_blocks(sh, levels)


def _level_blocks(sh: SortedHistories, levels: list[int]) -> Iterator[LevelBlocks]:
    pos, adj = sh.order, sh.adj
    for level in levels:
        big = adj >= level
        keep = big.copy()
        keep[:-1] |= big[1:]
        if not keep.any():
            return
        if not keep.all():
            sel = np.flatnonzero(keep)
            # The first kept entry of a block keeps adj < level: its sorted predecessor either
            # belongs to another block or was dropped as a singleton (adj < level on both sides).
            pos, adj = pos[sel], adj[sel]
        brk = adj < level
        block = np.cumsum(brk, dtype=np.int64) - 1
        yield LevelBlocks(level, pos, block, int(block[-1]) + 1)


def longest_match(sh: SortedHistories, min_len: int = 1) -> np.ndarray:
    """``int64[N]`` indexed by flat position: ``L*(g) = min(32, max_{j<g} lcs(g, j))`` over
    earlier positions ``j`` of the same segment, reported only where ``L* >= min_len``
    (0 elsewhere). With ``min_len = 1`` this is the exact longest previous match."""
    n = sh.order.size
    out = np.zeros(n, dtype=np.int64)
    for lb in iter_level_blocks(sh, range(max(1, min_len), MAX_MATCH + 1)):
        starts = np.flatnonzero(np.diff(lb.block, prepend=-1))
        first = np.minimum.reduceat(lb.pos, starts)          # smallest position of each block
        has = first[lb.block] < lb.pos
        if not has.any():                                    # monotone: nothing longer either
            break
        out[lb.pos[has]] = lb.level
    return out
