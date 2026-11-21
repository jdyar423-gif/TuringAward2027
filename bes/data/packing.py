"""Pure packing algorithms of the training loader (``bes.data.loader``).

Nothing here does I/O, starts processes or keeps state between calls: the loader's streams
call these functions with their own state, and ``tests/data/test_packing.py`` compares them
with brute-force references.

Rows (``docs/SPEC.md`` sections 1 and 3)
---------------------------------------
A row holds ``T + 1`` token ids: the model reads positions ``0 .. T-1`` and predicts positions
``1 .. T``. A row is a sequence of *pieces* (segments), each starting with ``BOS``, followed by
padding. :func:`assemble_row` gives

- ``tokens int32[T+1]``: the pieces in order, then ``PAD`` up to the end;
- ``seg int32[T]``: segment id of every input position: 0, 1, ... in piece order; the padding is
  one more segment with its own id;
- ``w float32[T]``: ``w[t]`` is the loss weight of predicting ``tokens[t+1]``, i.e. the weight
  rule ``bes.data.samples.loss_weights`` of the token at position ``t+1`` with the role codes of
  its piece; 0 for padding and for every ``BOS``.

Base rows (:func:`fill_base`)
-----------------------------
Consecutive documents fill the row exactly; the document that does not fit is cut at the row
end and continues as the first piece of the stream's next base row, with a fresh ``BOS``
(:class:`Carry`). A row is padded only when a single position is left (a piece needs ``BOS`` and
one token) or the stream has no more documents.

Chat rows (:func:`best_fit`, :func:`pack_best_fit`)
--------------------------------------------------
Whole samples (conversations already cut at an assistant turn boundary to at most ``T + 1``
tokens) from a look-ahead buffer: repeatedly the longest buffered sample that still fits (the
oldest among equals), the buffer refilled after every placement, until nothing fits; the rest is
padding. A sample that has waited ``max_wait`` rows may be forced to open the next row.

Row kinds (:func:`row_kind`)
----------------------------
Stream ``v`` (of ``V``, a power of two) gives its ``n``-th row the kind whose cumulative-share
interval holds ``u = frac(n * g + rev(v) / V)``: ``g`` the golden ratio conjugate (a Weyl
sequence: each stream's kinds follow the shares with a discrepancy of order ``log n``), and
``rev(v)`` the bit reversal of ``v`` (van der Corput phases: the ``2^k`` streams of any aligned
block have phases on a grid of spacing ``2^-k``, so ``2^k`` rows with the same ``n`` hold
``2^k * share +- 1`` rows of each kind). Exact integer arithmetic modulo ``2^64``.

Sources (:func:`deficit_pick`)
------------------------------
Smooth weighted round robin on tokens: the next sample comes from the source whose placed tokens
lag most behind its share of the total (``share * (total + 1) - placed``, first index on ties).

Epochs (:func:`epoch_share`, :func:`epoch_position`)
---------------------------------------------------
An epoch of ``N`` items (permuted by the caller) is dealt to the ``V`` streams: stream ``v`` reads
positions ``v, v + V, v + 2V, ...``, so every position is read once per epoch when ``N >= V``;
with ``N < V`` every stream reads one position (``v mod N``) per epoch.

Step layout (:func:`step_rows`)
-------------------------------
The rows of the run are numbered ``m = 0, 1, ...`` in production order; row ``m`` belongs to
stream ``m mod V`` (its ``m // V``-th row). A step of shape ``(A, B, T)`` holds rows
``m0 .. m0 + A*B - 1``; row ``m0 + r`` goes to micro-batch ``r // B``, position ``r % B``.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from bes.config import BOS, PAD
from bes.data.samples import WeightRule, loss_weights

#: Golden ratio conjugate ``(sqrt(5) - 1) / 2``.
GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0
#: ``round(GOLDEN * 2**64)``: the Weyl increment of :func:`row_kind`.
WEYL64 = 0x9E3779B97F4A7C15
_MASK64 = (1 << 64) - 1
_ONE64 = 1 << 64


# ------------------------------------------------------------------------------ rows
@dataclass(frozen=True)
class Row:
    """One packed row (module docstring): ``tokens int32[T+1]``, ``seg int32[T]``,
    ``w float32[T]``; ``n_pad`` padding positions among the ``T + 1``; ``lengths`` of the
    pieces in order."""

    tokens: np.ndarray
    seg: np.ndarray
    w: np.ndarray
    n_pad: int
    lengths: tuple[int, ...]


def assemble_row(pieces: Sequence[tuple[np.ndarray, np.ndarray]], T: int,
                 rule: WeightRule = WeightRule()) -> Row:
    """Pack ``pieces`` (``(ids int[n], roles int[n])``, ``ids[0] == BOS``, total length at most
    ``T + 1``) into one row of ``T + 1`` tokens (module docstring)."""
    if T < 1:
        raise ValueError("T must be positive")
    n_row = T + 1
    tokens = np.full(n_row, PAD, dtype=np.int32)
    roles = np.zeros(n_row, dtype=np.int8)
    seg = np.empty(n_row, dtype=np.int32)
    o = 0
    lengths = []
    for k, (ids, rl) in enumerate(pieces):
        n = len(ids)
        if n == 0 or int(ids[0]) != BOS:
            raise ValueError(f"piece {k} does not start with BOS")
        if len(rl) != n:
            raise ValueError(f"piece {k}: {len(rl)} roles for {n} ids")
        if o + n > n_row:
            raise ValueError(f"pieces of {o + n} tokens do not fit a row of {n_row}")
        tokens[o:o + n] = ids
        roles[o:o + n] = rl
        seg[o:o + n] = k
        o += n
        lengths.append(n)
    seg[o:] = len(lengths)
    w = loss_weights(tokens, roles, rule)
    return Row(tokens=tokens, seg=seg[:T].copy(), w=w[1:].astype(np.float32, copy=False),
               n_pad=n_row - o, lengths=tuple(lengths))


# ------------------------------------------------------------------------ base rows
@dataclass(frozen=True)
class Carry:
    """A document being packed: ``ids int[n]`` (the whole document, ``ids[0] == BOS``) and
    ``pos``, the index in ``ids`` of the next token to emit (1 = from the start). Its next
    piece is ``BOS`` followed by ``ids[pos:]``. ``key`` identifies the document (opaque)."""

    key: Any
    ids: np.ndarray
    pos: int = 1

    @property
    def remaining(self) -> int:
        """Length of the next piece (``BOS`` included)."""
        return len(self.ids) - self.pos + 1

    def piece(self, limit: int | None = None) -> np.ndarray:
        """The next piece (at most ``limit`` tokens)."""
        stop = len(self.ids) if limit is None else min(len(self.ids), self.pos + limit - 1)
        if self.pos == 1:
            return self.ids[:stop]
        out = np.empty(stop - self.pos + 1, dtype=self.ids.dtype)
        out[0] = BOS
        out[1:] = self.ids[self.pos:stop]
        return out


def fill_base(room: int, carry: Carry | None,
              next_doc: Callable[[], Carry | None]) -> tuple[list[tuple[Any, np.ndarray]], Carry | None]:
    """Fill ``room`` positions with documents: the ``carry`` first, then documents from
    ``next_doc()`` (``None``: no more data). Returns ``(placed, carry)``: the pieces
    ``(key, ids)`` in order and the document to continue in the next row (``None`` when the
    last one ended exactly). Stops when fewer than 2 positions are left."""
    placed: list[tuple[Any, np.ndarray]] = []
    while room >= 2:
        if carry is None:
            carry = next_doc()
            if carry is None:
                break
        n = carry.remaining
        if n <= room:
            placed.append((carry.key, carry.piece()))
            room -= n
            carry = None
        else:
            placed.append((carry.key, carry.piece(room)))
            carry = Carry(carry.key, carry.ids, carry.pos + room - 1)
            room = 0
    return placed, carry


# ------------------------------------------------------------------------ chat rows
def best_fit(lengths: Sequence[int], room: int) -> int | None:
    """Index of the longest length that is at most ``room`` (the first one on ties); ``None``
    when none fits."""
    best, best_len = None, -1
    for i, n in enumerate(lengths):
        if best_len < n <= room:
            best, best_len = i, n
    return best


def pack_best_fit(buffer: list, room: int, length: Callable[[Any], int],
                  refill: Callable[[], Any | None], first: int | None = None) -> list:
    """Place whole items of ``buffer`` into ``room`` positions by best fit (module docstring).

    ``buffer`` is changed in place: placed items are removed and, after every placement,
    ``refill()`` is appended unless it returns ``None``. ``first``: index of an item placed
    before the best-fit loop (it must fit). Returns the placed items in placement order."""
    placed = []

    def take(i: int) -> int:
        item = buffer.pop(i)
        placed.append(item)
        new = refill()
        if new is not None:
            buffer.append(new)
        return length(item)

    if first is not None:
        if length(buffer[first]) > room:
            raise ValueError("the forced first item does not fit")
        room -= take(first)
    while True:
        i = best_fit([length(x) for x in buffer], room)
        if i is None:
            break
        room -= take(i)
    return placed


def oldest_waiting(ages: Sequence[int], now: int, max_wait: int) -> int | None:
    """Index of the oldest item (smallest age stamp, first on ties) that has waited at least
    ``max_wait`` rows at row ``now``; ``None`` if there is none or ``max_wait <= 0``."""
    if max_wait <= 0:
        return None
    best = None
    for i, a in enumerate(ages):
        if now - a >= max_wait and (best is None or a < ages[best]):
            best = i
    return best


# ------------------------------------------------------------------------- row kinds
def bit_reverse(v: int, bits: int) -> int:
    """``v`` with its lowest ``bits`` bits in reverse order."""
    out = 0
    for _ in range(bits):
        out = (out << 1) | (v & 1)
        v >>= 1
    return out


def stream_bits(n_streams: int) -> int:
    """``log2(n_streams)``; raises unless ``n_streams`` is a power of two."""
    if n_streams < 1 or n_streams & (n_streams - 1):
        raise ValueError(f"the number of streams must be a power of two, got {n_streams}")
    return n_streams.bit_length() - 1


def thresholds(shares: Sequence[float]) -> tuple[int, ...]:
    """Cumulative shares as integers on ``[0, 2^64]`` (the last is exactly ``2^64``) for
    :func:`row_kind`. Shares must be non-negative with a positive sum (they are normalised)."""
    s = [float(x) for x in shares]
    if any(x < 0 or not math.isfinite(x) for x in s) or sum(s) <= 0:
        raise ValueError(f"invalid shares {shares}")
    total = sum(s)
    out, acc = [], 0.0
    for x in s:
        acc += x / total
        out.append(min(_ONE64, int(acc * _ONE64)))
    last = max(i for i, x in enumerate(s) if x > 0)
    for i in range(last, len(out)):
        out[i] = _ONE64
    return tuple(out)


def row_phase(v: int, n: int, n_streams: int) -> int:
    """``u * 2^64`` of row ``n`` of stream ``v`` (module docstring), an integer in ``[0, 2^64)``."""
    bits = stream_bits(n_streams)
    phase = bit_reverse(v, bits) << (64 - bits) if bits else 0
    return (n * WEYL64 + phase) & _MASK64


def row_kind(v: int, n: int, n_streams: int, cum: Sequence[int]) -> int:
    """Kind of row ``n`` of stream ``v``: the first ``k`` with ``u < cum[k]``
    (``cum = thresholds(shares)``)."""
    u = row_phase(v, n, n_streams)
    for k, t in enumerate(cum):
        if u < t:
            return k
    return len(cum) - 1


# -------------------------------------------------------------------------- sources
def deficit_pick(shares: Sequence[float], placed: Sequence[float]) -> int:
    """Index of the source whose placed tokens lag most behind its share:
    ``argmax_k share_k * (total + 1) - placed_k`` over the shares ``> 0`` (first on ties)."""
    total = float(sum(placed))
    best, best_d = -1, -math.inf
    for k, (s, p) in enumerate(zip(shares, placed)):
        if s <= 0:
            continue
        d = float(s) * (total + 1.0) - float(p)
        if d > best_d:
            best, best_d = k, d
    if best < 0:
        raise ValueError("no source has a positive share")
    return best


# --------------------------------------------------------------------------- epochs
def epoch_share(v: int, n_streams: int, n_items: int) -> int:
    """Number of positions stream ``v`` reads per epoch of ``n_items`` (0 when empty)."""
    if n_items <= 0:
        return 0
    return max(1, -(-(n_items - v) // n_streams))


def epoch_position(v: int, j: int, n_streams: int, n_items: int) -> int:
    """Position (before the epoch permutation) of the ``j``-th read of stream ``v``."""
    if n_items <= 0:
        raise ValueError("empty epoch")
    return (v + j * n_streams) % n_items


# ---------------------------------------------------------------------- step layout
def step_rows(m0: int, n_micro: int, micro_rows: int,
              n_streams: int) -> dict[int, list[tuple[int, int, int]]]:
    """Rows of one step: ``{stream: [(row number of the stream, micro-batch, position)]}`` for
    the global rows ``m0 .. m0 + n_micro * micro_rows - 1`` (module docstring)."""
    out: dict[int, list[tuple[int, int, int]]] = {}
    for r in range(n_micro * micro_rows):
        m = m0 + r
        a, b = divmod(r, micro_rows)
        out.setdefault(m % n_streams, []).append((m // n_streams, a, b))
    return out


__all__ = ["Row", "assemble_row", "Carry", "fill_base", "best_fit", "pack_best_fit",
           "oldest_waiting", "bit_reverse", "stream_bits", "thresholds", "row_phase", "row_kind",
           "deficit_pick", "epoch_share", "epoch_position", "step_rows", "GOLDEN", "WEYL64"]
