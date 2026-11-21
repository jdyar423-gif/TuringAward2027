"""Precedent statistics of context groups: the n-gram heads and the longest-match head.

A *head* at level ``L`` looks, for a query position ``t``, at the earlier positions ``j < t`` of
the same segment that share ``t``'s ``L``-token context (``bes.exact.suffix``: the context
never crosses a role token, ``BOS`` or a segment start). With ``M`` that set and successor
``x[j+1]`` of each ``j``:

- ``total = n = |M|`` and ``types = K`` = number of distinct successors (special ones included:
  they are real observations of what followed the context);
- candidates: walk the ``WINDOW = 16`` most recent elements of ``M`` from the newest, keep each
  successor that is not a special token and not already kept, stop after ``N_CAND = 4``;
- each candidate ``b`` carries ``count = n_b`` (its number of occurrences over **all** of ``M``)
  and ``source``, the most recent ``j`` with successor ``b``.

The n-gram heads use levels 1, 2 and 3; the longest-match head uses level ``L*(t)``.

Vectorised algorithm for many groups at once (:func:`group_candidates`)
-----------------------------------------------------------------------
Members are sorted by ``(group, position)``, which gives each member its rank ``i`` in its
group, i.e. ``n`` when it is the query. A second sort by ``(group, successor, position)`` gives
every member ``k`` the number ``c_k`` of earlier members with the same successor and the rank
``nxt_k`` of the next one. Then, for a query of rank ``i``:

- ``K(i)`` = number of members of rank ``< i`` with ``c_k = 0`` (a prefix sum);
- member ``k < i`` is the most recent occurrence of its successor before ``i`` iff
  ``nxt_k >= i``, and then ``n_b = c_k + 1``;
- candidates come from the ``[n_queries, 16]`` window of the previous members: keep entries
  with an ordinary successor and ``nxt_k >= i``, take the first four along the window (a
  cumulative sum along the window axis gives each its slot).

Everything is causal: a query only reads members of smaller rank and successors at positions
``<= t``.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from bes.exact.suffix import FIRST_SPECIAL, MAX_MATCH, SortedHistories, iter_level_blocks

#: Number of most recent matches searched for candidates.
WINDOW = 16
#: Candidates per head.
N_CAND = 4
#: Shortest match the longest-match head reports.
MIN_LONG = 4
#: Context lengths of the n-gram heads.
ORDERS = (1, 2, 3)

_BIG = np.iinfo(np.int64).max


class HeadOut(NamedTuple):
    """Selected candidates of one head, one entry per (query, slot); all arrays ``int64[m]``.

    ``query``: flat position ``t``; ``slot``: 0..3, most recent first; ``token``: candidate
    ``b = x[j+1]``; ``count``: ``n_b``; ``total``: ``n``; ``types``: ``K``; ``source``: flat
    position of the most recent ``j`` with successor ``b``; ``level``: context length ``L``.
    """

    query: np.ndarray
    slot: np.ndarray
    token: np.ndarray
    count: np.ndarray
    total: np.ndarray
    types: np.ndarray
    source: np.ndarray
    level: np.ndarray


def _empty_head() -> HeadOut:
    z = np.zeros(0, dtype=np.int64)
    return HeadOut(z, z, z, z, z, z, z, z)


def group_candidates(pos: np.ndarray, group: np.ndarray, is_query: np.ndarray, succ: np.ndarray,
                     level: np.ndarray | int, window: int = WINDOW, n_cand: int = N_CAND) -> HeadOut:
    """Candidates and statistics for every query of many groups at once.

    pos ``int64[m]``: flat positions of the members (distinct within a group); group
    ``int64[m]``: non-negative group ids; is_query ``bool[m]``: members to report; succ
    ``int64[N]``: successor of every flat position (``-1`` where unknown); level: context length
    (scalar, or ``int64[m]`` per member, reported for the queries).
    """
    m = pos.size
    if m == 0:
        return _empty_head()
    level = np.broadcast_to(np.asarray(level, dtype=np.int64), (m,))
    span = int(pos.max()) + 1
    order = np.argsort(group * span + pos)
    g, p, q, lev = group[order], pos[order], is_query[order], level[order]
    ar = np.arange(m, dtype=np.int64)
    first = np.empty(m, dtype=bool)
    first[0] = True
    first[1:] = g[1:] != g[:-1]
    gstart = np.maximum.accumulate(np.where(first, ar, 0))
    rank = ar - gstart                                       # earlier members of the group
    s = succ[p]

    # Successor subgroups: (group, successor) runs in position order.
    run = np.cumsum(first, dtype=np.int64) - 1
    n_succ = int(s.max()) + 2
    order2 = np.argsort((run * n_succ + (s + 1)) * m + ar)
    run2, s2 = run[order2], s[order2]
    sub_first = np.empty(m, dtype=bool)
    sub_first[0] = True
    sub_first[1:] = (run2[1:] != run2[:-1]) | (s2[1:] != s2[:-1])
    c = np.empty(m, dtype=np.int64)
    c[order2] = ar - np.maximum.accumulate(np.where(sub_first, ar, 0))
    nxt = np.empty(m, dtype=np.int64)
    nxt2 = np.full(m, _BIG, dtype=np.int64)
    nxt2[:-1] = np.where(sub_first[1:], _BIG, rank[order2[1:]])
    nxt[order2] = nxt2
    new_type = np.concatenate([[0], np.cumsum(c == 0, dtype=np.int64)])
    types = new_type[ar] - new_type[gstart]                  # distinct successors before i

    qi = np.flatnonzero(q & (rank > 0))
    if qi.size == 0:
        return _empty_head()
    rq = rank[qi]
    width = int(min(window, rq.max()))
    d = np.arange(1, width + 1, dtype=np.int64)
    valid = d <= rq[:, None]
    k = np.where(valid, qi[:, None] - d, 0)                  # [n_q, width] previous members
    sk = s[k]
    ok = valid & (sk >= 0) & (sk < FIRST_SPECIAL) & (nxt[k] >= rq[:, None])
    cum = np.cumsum(ok, axis=1)
    rows, cols = np.nonzero(ok & (cum <= n_cand))
    kk = k[rows, cols]
    qq = qi[rows]
    return HeadOut(query=p[qq], slot=cum[rows, cols] - 1, token=sk[rows, cols], count=c[kk] + 1,
                   total=rank[qq], types=types[qq], source=p[kk], level=lev[qq])


def order_heads(sh: SortedHistories, succ: np.ndarray, orders: tuple[int, ...] = ORDERS,
                window: int = WINDOW, n_cand: int = N_CAND) -> dict[int, HeadOut]:
    """The n-gram heads: ``{order: HeadOut}`` for each context length in ``orders``."""
    out = {o: _empty_head() for o in orders}
    for lb in iter_level_blocks(sh, sorted(orders)):
        out[lb.level] = group_candidates(lb.pos, lb.block, np.ones(lb.pos.size, dtype=bool), succ,
                                         lb.level, window, n_cand)
    return out


def long_head(sh: SortedHistories, succ: np.ndarray, lstar: np.ndarray, min_len: int = MIN_LONG,
              window: int = WINDOW, n_cand: int = N_CAND) -> HeadOut:
    """The longest-match head: for every ``t`` with ``L*(t) >= min_len`` the group is ``t``'s
    ``L*(t)``-block (positions sharing its ``L*``-token context, i.e. match length ``L*``).

    lstar ``int64[N]`` from :func:`bes.exact.suffix.longest_match`. All levels are gathered into
    one call of :func:`group_candidates`; a position can be a member at several levels. Only
    members at or before the last query of their block are kept (later ones are never an
    earlier match), which bounds the work on long runs of one repeated token.
    """
    wanted = np.unique(lstar[lstar >= min_len])
    if wanted.size == 0:
        return _empty_head()
    parts_pos, parts_group, parts_query, parts_level = [], [], [], []
    offset = 0
    want = set(wanted.tolist())
    for lb in iter_level_blocks(sh, range(min_len, int(wanted.max()) + 1)):
        if lb.level not in want:
            continue
        is_q = lstar[lb.pos] == lb.level
        starts = np.flatnonzero(np.diff(lb.block, prepend=-1))
        last_query = np.maximum.reduceat(np.where(is_q, lb.pos, -1), starts)   # -1: no query
        mem = lb.pos <= last_query[lb.block]
        parts_pos.append(lb.pos[mem])
        parts_group.append(lb.block[mem] + offset)
        parts_query.append(is_q[mem])
        parts_level.append(np.full(int(mem.sum()), lb.level, dtype=np.int64))
        offset += lb.n_blocks
    if not parts_pos:
        return _empty_head()
    return group_candidates(np.concatenate(parts_pos), np.concatenate(parts_group),
                            np.concatenate(parts_query), succ, np.concatenate(parts_level),
                            window, n_cand)


__all__ = ["HeadOut", "group_candidates", "order_heads", "long_head", "WINDOW", "N_CAND",
           "MIN_LONG", "ORDERS", "MAX_MATCH"]
