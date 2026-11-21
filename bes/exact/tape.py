"""Precedent candidate tape (``docs/SPEC.md`` section 2.3).

For every position ``t`` the tape lists up to ``CAND_SLOTS = 16`` candidate next tokens with the
features of ``bes.config.CAND_FEATS``. It is causal and segment-local: the row of ``t`` depends
only on tokens ``<= t`` of ``t``'s segment.

Exact definition
----------------
Segments start at the first position of a row, wherever ``seg`` changes and at every ``BOS``
(a segment always starts with ``BOS``; ``bes.exact.registers`` resets there too).

Contexts. A *boundary* is the first position of a segment (so every ``BOS``) or a role token
(``SYSTEM, USER, ASSISTANT, LIBRARY``). The ``L``-token context ending at ``t`` exists when it
holds no boundary except possibly at its first position (so ``[USER, "Hi"]`` is a 2-gram but
``[END, USER, "Hi"]`` is not); a context never crosses a role token, ``BOS`` or a segment start.
The match length ``lcs(t, j)`` of two positions of one segment is the largest ``L`` for which
both ``L``-token contexts exist and are equal. ``L*(t) = min(32, max_{j<t} lcs(t, j))``.

Heads. Each head has a context length ``L`` and a match set ``M`` of earlier positions:

====================  ==========  =====================================================
slots                 kind        ``L`` and ``M``
====================  ==========  =====================================================
0-3 / 4-7 / 8-11      1 / 2 / 3   ``L = o``; ``M = {j < t : lcs(t, j) >= o}``
12-15                 4           ``L = L*(t)`` if ``L*(t) >= 4``; ``M = {j < t : min(32,
                                  lcs(t, j)) = L*}`` (the positions with match length L*)
====================  ==========  =====================================================

For a head with ``n = |M| > 0``: walk the 16 most recent elements of ``M`` from the newest and
keep every successor ``x[j+1]`` that is not a special token and not already kept, at most four;
they fill the head's slots left to right (most recent first), the remaining slots are empty.
The longest-match head is the same statistic for the context of length ``L*``.

Features of a candidate ``b`` with most recent source ``j_b`` (bucket ``B``: 0 -> 0, 1 -> 1,
2 -> 2, 3-4 -> 3, 5-8 -> 4, 9-16 -> 5, 17-64 -> 6, larger -> 7):

- ``kind``: 1, 2, 3 (n-gram order) or 4 (long match); 0 marks an empty slot;
- ``count = B(n_b)``, ``n_b = |{j in M : x[j+1] = b}|`` (over all of ``M``, not just the window);
- ``total = B(n)``; ``types = B(K)``, ``K`` = number of distinct successors in ``M``
  (successors that are special tokens are counted in ``n`` and ``K`` but never offered);
- ``src``: role code of ``j_b``'s turn (0 raw text, 1 system, 2 user, 3 assistant, 4 library),
  or 5 when ``t``'s turn is an assistant turn and ``j_b`` lies in that same turn (self-generated
  evidence). The turn of a position starts at its latest boundary; ``END`` belongs to the turn
  it closes;
- ``mlen = B(L)``: the context length of the head (``o`` for the n-gram heads, ``L*`` for the
  long match).

Empty slots hold ``cand_tok = -1`` and all-zero features.

Algorithm (batch)
-----------------
``bes.exact.suffix`` sorts the backward histories of all positions of the batch by prefix
doubling (six rank levels), which makes every ``L``-token context class a contiguous block and
gives ``L*`` by segmented minima; ``bes.exact.ngram_heads`` then computes all statistics of all
groups with two sorts per head and a ``[n_queries, 16]`` window matrix. There is no Python loop
over positions; large batches are processed in chunks of rows (``CHUNK_TOKENS``).
:class:`Incremental` computes the same rows one token at a time with dictionaries.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from bes.config import BOS, CAND_FEATS, CAND_SLOTS, N_CAND_FEAT, VOCAB_SIZE
from bes.exact import ngram_heads, suffix
from bes.exact.ngram_heads import MIN_LONG, N_CAND, ORDERS, WINDOW, HeadOut
from bes.exact.suffix import BOUNDARY_TOKENS, FIRST_SPECIAL, MAX_MATCH, ROLE_ASSISTANT, ROLE_CODE_OF_TOKEN

#: Slots per head; heads are the n-gram orders 1, 2, 3 and the long match, in that order.
SLOTS_PER_HEAD = N_CAND
LONG_KIND = len(ORDERS) + 1
LONG_BASE = len(ORDERS) * SLOTS_PER_HEAD
assert CAND_SLOTS == (len(ORDERS) + 1) * SLOTS_PER_HEAD
#: ``src`` value for evidence from the assistant turn that contains the query position.
SRC_SELF = 5

FEAT_NAMES = tuple(name for name, _ in CAND_FEATS)
F_KIND, F_COUNT, F_TOTAL, F_TYPES, F_SRC, F_MLEN = (
    FEAT_NAMES.index(n) for n in ("kind", "count", "total", "types", "src", "mlen"))
assert N_CAND_FEAT == len(FEAT_NAMES)

#: Lower edges of buckets 1..7: 1, 2, 3-4, 5-8, 9-16, 17-64, >= 65.
BUCKET_EDGES = np.array([1, 2, 3, 5, 9, 17, 65], dtype=np.int64)
_BUCKET_TABLE = np.searchsorted(BUCKET_EDGES, np.arange(66), side="right").astype(np.int8)

#: Flattened positions processed per internal call (rows are independent).
CHUNK_TOKENS = 1 << 16


def bucket(v: np.ndarray | int) -> np.ndarray:
    """Bucket of non-negative counts: ``int8`` with the shape of ``v``."""
    v = np.asarray(v, dtype=np.int64)
    return _BUCKET_TABLE[np.clip(v, 0, 65)]


def bucket_int(v: int) -> int:
    """Scalar version of :func:`bucket`."""
    return int(_BUCKET_TABLE[min(max(int(v), 0), 65)])


class CandTape(NamedTuple):
    """``cand_tok int32[B, T, 16]`` (-1 empty) and ``cand_feat int8[B, T, 16, N_CAND_FEAT]``."""

    cand_tok: np.ndarray
    cand_feat: np.ndarray


def compute(tokens: np.ndarray, seg: np.ndarray | None = None) -> CandTape:
    """Candidate tape of a batch.

    tokens ``int[B, T]`` ids in ``[0, VOCAB_SIZE)`` (any integer dtype, e.g. the ``uint16`` of
    the token shards); seg ``int[B, T]`` segment ids (contiguous runs inside a row; ``None``:
    rows are split at ``BOS`` only). Returns :class:`CandTape`.
    """
    tokens = np.asarray(tokens)
    if tokens.ndim != 2:
        raise ValueError(f"tokens must be [B, T], got shape {tokens.shape}")
    n_rows, row_len = tokens.shape
    if seg is not None:
        seg = np.asarray(seg)
        if seg.shape != tokens.shape:
            raise ValueError(f"seg shape {seg.shape} differs from tokens shape {tokens.shape}")
    cand_tok = np.full((n_rows, row_len, CAND_SLOTS), -1, dtype=np.int32)
    cand_feat = np.zeros((n_rows, row_len, CAND_SLOTS, N_CAND_FEAT), dtype=np.int8)
    if n_rows == 0 or row_len == 0:
        return CandTape(cand_tok, cand_feat)
    step = max(1, CHUNK_TOKENS // row_len)
    for r0 in range(0, n_rows, step):
        r1 = min(n_rows, r0 + step)
        _fill_chunk(tokens[r0:r1], None if seg is None else seg[r0:r1],
                    cand_tok[r0:r1].reshape(-1, CAND_SLOTS),
                    cand_feat[r0:r1].reshape(-1, CAND_SLOTS, N_CAND_FEAT))
    return CandTape(cand_tok, cand_feat)


def _fill_chunk(tokens: np.ndarray, seg: np.ndarray | None, out_tok: np.ndarray,
                out_feat: np.ndarray) -> None:
    lay = suffix.layout(tokens, seg)
    sh = suffix.sort_histories(lay)
    heads = ngram_heads.order_heads(sh, lay.succ)
    for i, order in enumerate(ORDERS):
        _write(heads[order], lay, kind=order, base=i * SLOTS_PER_HEAD, out_tok=out_tok, out_feat=out_feat)
    lstar = suffix.longest_match(sh, MIN_LONG)
    _write(ngram_heads.long_head(sh, lay.succ, lstar), lay, kind=LONG_KIND, base=LONG_BASE,
           out_tok=out_tok, out_feat=out_feat)


def _write(h: HeadOut, lay: suffix.Layout, kind: int, base: int, out_tok: np.ndarray,
           out_feat: np.ndarray) -> None:
    if h.query.size == 0:
        return
    rows, cols = h.query, base + h.slot
    out_tok[rows, cols] = h.token
    src = np.where((lay.role[h.query] == ROLE_ASSISTANT) & (lay.ts[h.source] == lay.ts[h.query]),
                   SRC_SELF, lay.role[h.source])
    feat = np.empty((h.query.size, N_CAND_FEAT), dtype=np.int8)
    feat[:, F_KIND] = kind
    feat[:, F_COUNT] = bucket(h.count)
    feat[:, F_TOTAL] = bucket(h.total)
    feat[:, F_TYPES] = bucket(h.types)
    feat[:, F_SRC] = src
    feat[:, F_MLEN] = bucket(h.level)
    out_feat[rows, cols] = feat


# ------------------------------------------------------------------------------ incremental
# An entry holds the statistics of one context in one segment, as a list
# [owner, n, successor counts {b: n_b}, the WINDOW most recent member positions].
# ``owner`` implements copy-on-write between forks.
_OWNER, _N, _SUCC, _RECENT = range(4)
_BUCKET_LIST: list[int] = _BUCKET_TABLE.tolist()


def _bucket_py(v: int) -> int:
    return _BUCKET_LIST[v] if v < 65 else 7


class Incremental:
    """Candidate tape of one row, one token at a time; equals :func:`compute` exactly.

    ``step(token, new_segment)`` returns the tape row of the position of ``token``:
    ``(cand_tok int32[16], cand_feat int8[16, N_CAND_FEAT])``. The first token, every ``BOS``
    and any token with ``new_segment=True`` start a new segment (all state is cleared; pass
    ``new_segment`` where ``seg`` changes). :meth:`fork` returns
    an independent copy; statistics are shared copy-on-write, so a fork costs one shallow dict
    copy and each side clones an entry only when it first writes to it (particle decoding).

    State: an interning trie of contexts (``(parent node, token) -> node``: the ``L``-token
    context ending at ``t`` is the child of the ``(L-1)``-token context ending at ``t-1``) and
    one entry per context node. Per step: at most 32 trie lookups, 32 entry updates for the
    successor of ``t-1``, up to 32 lookups for the query and 32 updates for the new member.
    """

    def __init__(self) -> None:
        self._owner = object()
        self.reset()

    def reset(self) -> None:
        """Forget everything (the next token starts a new segment)."""
        self._trie: dict[tuple[int, int], int] = {}
        self._x: list[int] = []
        self._ts: list[int] = []
        self._role: list[int] = []
        self._nodes: list[int] = []
        self._entries: dict[int, list] = {}

    def __len__(self) -> int:
        """Number of positions of the current segment seen so far."""
        return len(self._x)

    def fork(self) -> "Incremental":
        """Independent copy of the current state (both copies may be stepped afterwards)."""
        other = Incremental.__new__(Incremental)
        other._trie = self._trie                  # append-only interner: consistent when shared
        other._x, other._ts, other._role = list(self._x), list(self._ts), list(self._role)
        other._nodes = list(self._nodes)
        other._entries = dict(self._entries)      # entries shared until written
        other._owner = object()
        self._owner = object()                    # both sides now clone before writing
        return other

    def step(self, token: int, new_segment: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Consume one token; returns ``(cand_tok int32[16], cand_feat int8[16, F])``."""
        cand_tok = np.full(CAND_SLOTS, -1, dtype=np.int32)
        cand_feat = np.zeros((CAND_SLOTS, N_CAND_FEAT), dtype=np.int8)
        self._advance(int(token), bool(new_segment), cand_tok, cand_feat)
        return cand_tok, cand_feat

    def extend(self, tokens, seg=None) -> CandTape:
        """Step through ``tokens int[n]`` (``seg int[n]`` starts a new segment where it changes)
        and return the rows as ``CandTape`` with shapes ``[n, 16]`` and ``[n, 16, F]``."""
        tokens = np.asarray(tokens).reshape(-1)
        segs = None if seg is None else np.asarray(seg).reshape(-1).tolist()
        out_tok = np.full((tokens.size, CAND_SLOTS), -1, dtype=np.int32)
        out_feat = np.zeros((tokens.size, CAND_SLOTS, N_CAND_FEAT), dtype=np.int8)
        for i, tok in enumerate(tokens.tolist()):
            new = segs is not None and (i == 0 or segs[i] != segs[i - 1])
            self._advance(tok, new, out_tok[i], out_feat[i])
        return CandTape(out_tok, out_feat)

    def _advance(self, token: int, new_segment: bool, cand_tok: np.ndarray,
                 cand_feat: np.ndarray) -> None:
        if not 0 <= token < VOCAB_SIZE:
            raise ValueError(f"token id {token} outside [0, {VOCAB_SIZE})")
        if (new_segment or token == BOS) and self._x:
            self.reset()
        x, entries, owner, trie = self._x, self._entries, self._owner, self._trie
        t = len(x)
        if t:                                     # x[t] is the successor of position t-1
            for node in self._nodes:
                e = entries[node]
                if e[_OWNER] is not owner:
                    e = entries[node] = [owner, e[_N], dict(e[_SUCC]), list(e[_RECENT])]
                succ = e[_SUCC]
                succ[token] = succ.get(token, 0) + 1
        if t == 0 or token in BOUNDARY_TOKENS:
            ts, role, parents = t, ROLE_CODE_OF_TOKEN.get(token, 0), [0]
        else:
            ts, role = self._ts[-1], self._role[-1]
            parents = [0, *self._nodes[:MAX_MATCH - 1]]   # contexts of t-1 extended by token
        x.append(token)
        self._ts.append(ts)
        self._role.append(role)
        nodes = []
        for parent in parents:
            key = (parent, token)
            node = trie.get(key)
            if node is None:
                node = trie[key] = len(trie) + 1
            nodes.append(node)

        found = []                                # entries of contexts with an earlier match
        for node in nodes:
            e = entries.get(node)
            if e is None:
                break
            found.append(e)
        n_found = len(found)
        for i, order in enumerate(ORDERS):
            if order <= n_found:
                self._emit(found[order - 1], order, order, i * SLOTS_PER_HEAD, ts, role,
                           cand_tok, cand_feat)
        if n_found >= MIN_LONG:
            self._emit(found[-1], n_found, LONG_KIND, LONG_BASE, ts, role, cand_tok, cand_feat)

        for node in nodes:                        # position t becomes a member of its contexts
            e = entries.get(node)
            if e is None:
                entries[node] = [owner, 1, {}, [t]]
                continue
            if e[_OWNER] is not owner:
                e = entries[node] = [owner, e[_N], dict(e[_SUCC]), list(e[_RECENT])]
            e[_N] += 1
            recent = e[_RECENT]
            recent.append(t)
            if len(recent) > WINDOW:
                del recent[0]
        self._nodes = nodes

    def _emit(self, e: list, level: int, kind: int, base: int, ts: int, role: int,
              cand_tok: np.ndarray, cand_feat: np.ndarray) -> None:
        x, tss, roles, succ = self._x, self._ts, self._role, e[_SUCC]
        total, types, mlen = _bucket_py(e[_N]), _bucket_py(len(succ)), _bucket_py(level)
        self_turn = role == ROLE_ASSISTANT
        seen: set[int] = set()
        slot = base
        for j in reversed(e[_RECENT]):
            b = x[j + 1]
            if b >= FIRST_SPECIAL or b in seen:
                continue
            seen.add(b)
            src = SRC_SELF if self_turn and tss[j] == ts else roles[j]
            cand_tok[slot] = b
            row = cand_feat[slot]
            row[F_KIND] = kind
            row[F_COUNT] = _bucket_py(succ[b])
            row[F_TOTAL] = total
            row[F_TYPES] = types
            row[F_SRC] = src
            row[F_MLEN] = mlen
            slot += 1
            if slot == base + N_CAND:
                break


__all__ = ["CandTape", "compute", "Incremental", "bucket", "bucket_int", "SLOTS_PER_HEAD",
           "LONG_KIND", "LONG_BASE", "SRC_SELF", "WINDOW", "N_CAND", "MIN_LONG", "ORDERS",
           "MAX_MATCH", "F_KIND", "F_COUNT", "F_TOTAL", "F_TYPES", "F_SRC", "F_MLEN"]
