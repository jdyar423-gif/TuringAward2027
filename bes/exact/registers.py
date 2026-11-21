"""Registers and structural clocks of the exact layer (``docs/SPEC.md`` section 2.1).

:func:`compute` maps ``tokens int32[B, T]`` (and the segment ids ``seg int32[B, T]``) to
:class:`Registers`: ``reg int16[B, T, N_REG]`` in the field order and cardinalities of
``bes.config.REG_FIELDS`` (values clipped to ``[0, card)``), ``clock int32[B, T, N_CLOCKS]``
and the raw counters ``raw[name] int32[B, T]``. :class:`Incremental` gives the same values one
token at a time. Values at position ``t`` describe the state **after** consuming token ``t``
and depend only on tokens ``<= t`` of the same segment.

Resets (decided here; ``docs/SPEC.md`` leaves them to this module):

- a new segment (``t = 0`` or ``seg[t] != seg[t-1]``) resets everything before token ``t`` is
  read (padding forms its own segment and so cannot see the row before it);
- ``BOS`` resets everything: role 0, ``turn_idx`` 0, the segment clocks 0;
- a role token (``SYSTEM, USER, ASSISTANT, LIBRARY``) starts a turn: the per-turn counters,
  the transducer state (fence, inline code, quote, line start, pending sentence), the bracket
  depth and the calculator state reset, ``role`` takes the role's code and ``turn_idx`` grows
  by one. So an unclosed fence, quote or bracket never leaks into the next turn;
- ``END`` counts a pending sentence terminator (transducer flush) and otherwise keeps the
  state of its turn (its role is still the turn's role).

Raw counters (``raw``): ``words, sentences, lines, items`` (since the turn started), ``tokens``
(tokens since the turn started: 0 at the token that starts it, i.e. ``BOS``, the role token or
the first token of a segment), ``word_in_sent`` (words since the last counted sentence),
``depth`` (bracket depth, floored at 0), ``seg_sentences, seg_lines`` (since the segment
started), ``turn_idx`` (role tokens in the segment), ``role``, ``calc`` and ``state`` (the
transducer state index after the token).

Registers: ``role`` (0 raw text, 1 system, 2 user, 3 assistant, 4 library), ``fence, inline,
quote`` (transducer state), ``depth``, ``calc`` (0 outside a calculator span, 1 after ``CALC``
in the expression, 2 after ``EQ`` in the result; a span ends at ``CALC_END`` and at any other
special token except a repeated ``EQ``, as in ``bes.data.chat_format.find_calc_spans``),
``tok_in_turn = floor(log2(1 + tokens))``, ``word_in_sent, sent_in_turn, line_in_turn,
item_in_turn, turn_idx`` and the countdowns ``rem_sent, rem_item, rem_word``.

Countdowns: ``targets int32[B, T, 3]`` gives the target number of sentences, list items and
words that applies at each position (-1 where there is none). Only positions whose ``role`` is
assistant read it (the ``ASSISTANT`` token itself and ``END`` included); everywhere else the
countdowns are 0. With ``k = target - count so far in the turn``:

- ``rem_sent`` / ``rem_item`` (34 values): 0 no target, 1 overshoot (``k < 0``),
  ``2 + min(k, 31)`` otherwise;
- ``rem_word`` (26 values): 0 no target, 1 overshoot, ``2 + k`` for ``k`` in ``0..15``, then
  log2 buckets: 18 for ``k`` in ``16..31``, 19 ``32..63``, 20 ``64..127``, 21 ``128..255``,
  22 ``256..511``, 23 ``512..1023``, 24 ``1024..2047``, 25 for ``k >= 2048``
  (``18 + min(floor(log2 k) - 4, 7)``).

Clocks: ``(seg_sentences, seg_lines, tokens)``, i.e. sentences and lines so far in the segment
and tokens since the turn started.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from bes.config import N_CLOCKS, N_REG, REG_FIELDS
from bes.exact.instr_table import (INC_DEP_ADD, INC_DEP_LO, INC_ITEMS, INC_LINES, INC_SENT,
                                   INC_WORDS, INC_WS_AFTER, INC_WS_RESET, KIND_BOS, KIND_CALC,
                                   KIND_EQ, KIND_ORDINARY, KIND_ROLE, InstrTable)
from bes.exact.instr_table import load as load_table
from bes.exact.transducer import measure_counts  # noqa: F401  (text-level API, re-exported)

FIELD_INDEX: dict[str, int] = {name: i for i, (name, _) in enumerate(REG_FIELDS)}
CARD: dict[str, int] = dict(REG_FIELDS)
_REG_MAX = np.array([card - 1 for _, card in REG_FIELDS], np.int64)
assert N_CLOCKS == 3 and len(_REG_MAX) == N_REG
ROLE_ASSISTANT = 3
TARGET_FIELDS = ("sentences", "items", "words")
RAW_FIELDS = ("state", "calc", "role", "turn_idx", "tokens", "words", "sentences", "lines",
              "items", "word_in_sent", "depth", "seg_sentences", "seg_lines")
# Components of the transducer state in ``InstrTable.components`` (transducer.COMPONENTS order).
_COMP_FENCE, _COMP_INLINE, _COMP_QUOTE = 0, 1, 2
_BIG = 1 << 40   # region offset of the segmented running maximum (depth)


class Registers(NamedTuple):
    reg: np.ndarray            # int16 [B, T, N_REG]
    clock: np.ndarray          # int32 [B, T, N_CLOCKS]
    raw: dict[str, np.ndarray]  # int32 [B, T] for every name in RAW_FIELDS


# ------------------------------------------------------------------------------ encodings
def floor_log2(x) -> np.ndarray:
    """``floor(log2(x))`` of integers ``x >= 1`` (exact below 2**53), as int64."""
    x = np.asarray(x, dtype=np.int64)
    return (np.frexp(x.astype(np.float64))[1] - 1).astype(np.int64)


def rem_count(target, count) -> np.ndarray:
    """``rem_sent`` / ``rem_item`` value: 0 no target (``target < 0``), 1 overshoot,
    ``2 + min(target - count, 31)`` otherwise."""
    target = np.asarray(target, dtype=np.int64)
    k = target - np.asarray(count, dtype=np.int64)
    return np.where(target < 0, 0, np.where(k < 0, 1, 2 + np.minimum(k, 31)))


def rem_word(target, count) -> np.ndarray:
    """``rem_word`` value: 0 no target, 1 overshoot, ``2 + k`` for ``k = target - count`` in
    ``0..15``, then ``18 + min(floor(log2 k) - 4, 7)`` (buckets in the module docstring)."""
    target = np.asarray(target, dtype=np.int64)
    k = target - np.asarray(count, dtype=np.int64)
    big = 18 + np.minimum(floor_log2(np.maximum(k, 1)) - 4, 7)
    return np.where(target < 0, 0, np.where(k < 0, 1, np.where(k < 16, 2 + k, big)))


def assemble(raw: dict[str, np.ndarray], targets: np.ndarray | None,
             table: InstrTable) -> tuple[np.ndarray, np.ndarray]:
    """Registers and clocks from raw counters of any shape ``[...]``.

    ``targets`` is ``int[..., 3]`` or ``None``. Returns ``(reg int16[..., N_REG],
    clock int32[..., N_CLOCKS])``.
    """
    comp = table.components[raw["state"]].astype(np.int64)
    zero = np.zeros(np.shape(raw["state"]), np.int64)
    if targets is None:
        rem = {"rem_sent": zero, "rem_item": zero, "rem_word": zero}
    else:
        assistant = raw["role"] == ROLE_ASSISTANT
        tgt = np.where(assistant[..., None], np.asarray(targets, dtype=np.int64), -1)
        rem = {"rem_sent": rem_count(tgt[..., 0], raw["sentences"]),
               "rem_item": rem_count(tgt[..., 1], raw["items"]),
               "rem_word": rem_word(tgt[..., 2], raw["words"])}
    values = {
        "role": raw["role"], "fence": comp[..., _COMP_FENCE], "inline": comp[..., _COMP_INLINE],
        "quote": comp[..., _COMP_QUOTE], "depth": raw["depth"], "calc": raw["calc"],
        "tok_in_turn": floor_log2(1 + raw["tokens"]), "word_in_sent": raw["word_in_sent"],
        "sent_in_turn": raw["sentences"], "line_in_turn": raw["lines"],
        "item_in_turn": raw["items"], "turn_idx": raw["turn_idx"], **rem,
    }
    stacked = np.stack([values[name] for name, _ in REG_FIELDS], axis=-1)
    reg = np.clip(stacked, 0, _REG_MAX).astype(np.int16)
    clock = np.stack([raw["seg_sentences"], raw["seg_lines"], raw["tokens"]], axis=-1)
    return reg, clock.astype(np.int32)


# ------------------------------------------------------------------------- batch version
def _last_true(mask: np.ndarray) -> np.ndarray:
    """``int64[B, T]``: index of the last ``True`` at or before ``t`` (``mask[:, 0]`` is True)."""
    idx = np.where(mask, np.arange(mask.shape[1], dtype=np.int64), 0)
    return np.maximum.accumulate(idx, axis=1)


def _segmented_cumsum(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Sum of ``values`` over ``[r, t]``, ``r`` the last start at or before ``t``."""
    c = np.cumsum(values, axis=1)
    return c - np.take_along_axis(c - values, _last_true(starts), axis=1)


def _calc_states(kind: np.ndarray, new_seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculator register after each token and before it (``int64[B, T]`` each)."""
    t_idx = np.arange(kind.shape[1], dtype=np.int64)
    is_eq = (kind == KIND_EQ) & ~new_seg
    sets = ((kind != KIND_ORDINARY) & (kind != KIND_EQ)) | new_seg   # events that set 0 or 1
    last_set = np.maximum.accumulate(np.where(sets, t_idx, -1), axis=1)
    last_eq = np.maximum.accumulate(np.where(is_eq, t_idx, -1), axis=1)
    opened = np.take_along_axis(kind == KIND_CALC, last_set, axis=1)
    calc = np.where(opened, np.where(last_eq > last_set, 2, 1), 0)
    before = np.zeros_like(calc)
    before[:, 1:] = calc[:, :-1]
    before[new_seg] = 0
    return calc, before


def _run_states(table: InstrTable, rows: np.ndarray, new_seg: np.ndarray):
    """Transducer state before and after every token (``intp[B, T]`` each). The only loop
    over ``T``; each step is one gather over the batch."""
    n_b, n_t = rows.shape
    flat = table.next_flat
    base = (rows.astype(np.intp) * table.n_states).T.copy()
    fresh = new_seg.T
    reset_steps = set(np.flatnonzero(fresh.any(axis=1)).tolist())
    s = np.zeros(n_b, np.intp)                      # transducer INIT is state 0
    s_in = np.empty((n_t, n_b), np.intp)
    s_out = np.empty((n_t, n_b), np.intp)
    for t in range(n_t):
        if t in reset_steps:
            s = np.where(fresh[t], 0, s)
        s_in[t] = s
        s = flat[base[t] + s]
        s_out[t] = s
    return s_in.T, s_out.T


def _new_segments(seg, shape: tuple[int, int]) -> np.ndarray:
    new_seg = np.zeros(shape, bool)
    new_seg[:, 0] = True
    if seg is not None:
        seg = np.asarray(seg)
        if seg.shape != shape:
            raise ValueError(f"seg has shape {seg.shape}, tokens {shape}")
        new_seg[:, 1:] = seg[:, 1:] != seg[:, :-1]
    return new_seg


def compute(tokens, seg=None, targets=None, *, table: InstrTable | None = None) -> Registers:
    """Registers of a batch.

    tokens: ``int[B, T]`` ids; seg: ``int[B, T]`` segment ids or ``None`` (one segment per
    row, split only by ``BOS``); targets: ``int[B, T, 3]`` (sentences, items, words; -1 = none)
    or ``None``; table: the compiled tables (default: the committed tokenizer's).
    """
    table = table if table is not None else load_table()
    tokens = np.asarray(tokens)
    if tokens.ndim != 2:
        raise ValueError("tokens must be [B, T]")
    tokens = tokens.astype(np.int64)
    n_b, n_t = tokens.shape
    if tokens.size and (tokens.min() < 0 or tokens.max() >= table.vocab_size):
        raise ValueError(f"token ids outside [0, {table.vocab_size})")
    if targets is not None and np.shape(targets) != (n_b, n_t, 3):
        raise ValueError(f"targets must be [B, T, 3], got {np.shape(targets)}")
    if n_t == 0:
        raw = {k: np.zeros((n_b, 0), np.int32) for k in RAW_FIELDS}
        return Registers(np.zeros((n_b, 0, N_REG), np.int16),
                         np.zeros((n_b, 0, N_CLOCKS), np.int32), raw)

    new_seg = _new_segments(seg, (n_b, n_t))
    kind = table.kind[tokens]
    is_bos = kind == KIND_BOS
    is_role = kind == KIND_ROLE
    calc, calc_before = _calc_states(kind, new_seg)
    rows = np.where((kind == KIND_EQ) & (calc_before == 1), table.eq_row, tokens)
    s_in, s_out = _run_states(table, rows, new_seg)
    inc = table.inc[rows, s_in].astype(np.int64)

    turn_start = new_seg | is_bos | is_role
    seg_start = new_seg | is_bos
    raw: dict[str, np.ndarray] = {"state": s_out, "calc": calc}
    for name, k in (("words", INC_WORDS), ("sentences", INC_SENT), ("lines", INC_LINES),
                    ("items", INC_ITEMS)):
        raw[name] = _segmented_cumsum(inc[..., k], turn_start)
    raw["seg_sentences"] = _segmented_cumsum(inc[..., INC_SENT], seg_start)
    raw["seg_lines"] = _segmented_cumsum(inc[..., INC_LINES], seg_start)
    turn_first = _last_true(turn_start)
    raw["tokens"] = np.arange(n_t, dtype=np.int64)[None, :] - turn_first
    raw["turn_idx"] = _segmented_cumsum(is_role.astype(np.int64), seg_start)
    first_tok = np.take_along_axis(tokens, turn_first, axis=1)
    raw["role"] = table.role_code[first_tok].astype(np.int64)

    # Words in the current sentence: reset-and-add, restarted by turns and counted sentences.
    restart = _last_true(turn_start | (inc[..., INC_WS_RESET] > 0))
    w_cum = np.cumsum(inc[..., INC_WORDS], axis=1)
    raw["word_in_sent"] = (w_cum - np.take_along_axis(w_cum, restart, axis=1)
                           + np.take_along_axis(inc[..., INC_WS_AFTER], restart, axis=1))

    # Bracket depth: d_t = max(lo_t, d_{t-1} + add_t), d = 0 before a turn starting at r. With
    # G the running sum of add, d_t = G_t + max(-G_{r-1}, max_{r <= j <= t} (lo_j - G_j)); every
    # table row has lo >= add, so the j = r term already dominates -G_{r-1} = add_r - G_r.
    add, lo = inc[..., INC_DEP_ADD], inc[..., INC_DEP_LO]
    g = np.cumsum(add, axis=1)
    key = lo - g
    region = np.cumsum(turn_start, axis=1, dtype=np.int64) * _BIG
    raw["depth"] = g + np.maximum.accumulate(key + region, axis=1) - region

    reg, clock = assemble(raw, targets, table)
    return Registers(reg, clock, {k: raw[k].astype(np.int32) for k in RAW_FIELDS})


# --------------------------------------------------------------------- incremental version
class Incremental:
    """Registers one token at a time for ``batch_size`` independent streams.

    ``step(tokens, targets=None, new_segment=None)`` consumes one token per stream and returns
    ``(reg int16[B, N_REG], clock int32[B, N_CLOCKS])`` (1-D rows when ``tokens`` is a scalar
    and ``batch_size == 1``), equal to ``compute`` on the whole sequence. The first token of a
    stream starts a segment; ``new_segment`` (bool ``[B]``) starts one before the token
    (``compute``'s segment boundaries). ``targets`` is ``int[3]`` or ``int[B, 3]``.
    """

    def __init__(self, batch_size: int = 1, table: InstrTable | None = None):
        self.table = table if table is not None else load_table()
        self.batch_size = int(batch_size)
        self._c = {k: np.zeros(self.batch_size, np.int64) for k in RAW_FIELDS}
        self._fresh = np.ones(self.batch_size, bool)

    # -- state management -------------------------------------------------------------
    def raw(self) -> dict[str, np.ndarray]:
        """Current raw counters, ``int32[B]`` each (``RAW_FIELDS``)."""
        return {k: v.astype(np.int32) for k, v in self._c.items()}

    def copy(self) -> "Incremental":
        return self.select(np.arange(self.batch_size))

    def select(self, index) -> "Incremental":
        """New object whose streams are ``index`` (``int[B']``) of this one (particle resampling)."""
        index = np.asarray(index, dtype=np.int64)
        out = Incremental.__new__(Incremental)
        out.table, out.batch_size = self.table, int(index.shape[0])
        out._c = {k: v[index].copy() for k, v in self._c.items()}
        out._fresh = self._fresh[index].copy()
        return out

    @classmethod
    def prefill(cls, tokens, seg=None, targets=None, *,
                table: InstrTable | None = None) -> tuple["Incremental", Registers]:
        """Run ``compute`` on ``tokens int[B, T]`` (``T >= 1``) and return a stepper positioned
        after the last token, plus the registers of the prefix."""
        regs = compute(tokens, seg, targets, table=table)
        if regs.reg.shape[1] == 0:
            raise ValueError("prefill needs at least one token")
        inc = cls(regs.reg.shape[0], table)
        inc._c = {k: regs.raw[k][:, -1].astype(np.int64) for k in RAW_FIELDS}
        inc._fresh[:] = False
        return inc, regs

    # -- stepping -----------------------------------------------------------------------
    def step(self, tokens, targets=None, new_segment=None) -> tuple[np.ndarray, np.ndarray]:
        tab, c = self.table, self._c
        tok = np.asarray(tokens, dtype=np.int64)
        scalar = tok.ndim == 0
        tok = tok.reshape(-1)
        if tok.shape[0] != self.batch_size:
            raise ValueError(f"expected {self.batch_size} tokens, got {tok.shape[0]}")
        if tok.size and (tok.min() < 0 or tok.max() >= tab.vocab_size):
            raise ValueError(f"token ids outside [0, {tab.vocab_size})")
        fresh = self._fresh.copy()
        if new_segment is not None:
            fresh |= np.asarray(new_segment, dtype=bool).reshape(-1)
        self._fresh[:] = False
        if fresh.any():
            for v in c.values():
                v[fresh] = 0

        kind = tab.kind[tok]
        is_role = kind == KIND_ROLE
        is_bos = kind == KIND_BOS
        turn = is_bos | is_role
        calc_before = c["calc"]
        rows = np.where((kind == KIND_EQ) & (calc_before == 1), tab.eq_row, tok)
        s_in = c["state"]
        inc = tab.inc[rows, s_in].astype(np.int64)
        dw, ds = inc[:, INC_WORDS], inc[:, INC_SENT]
        dl, di = inc[:, INC_LINES], inc[:, INC_ITEMS]

        c["state"] = tab.next_state[rows, s_in].astype(np.int64)
        c["calc"] = np.where(kind == KIND_CALC, 1,
                             np.where(kind == KIND_EQ, np.where(calc_before >= 1, 2, 0),
                                      np.where(kind == KIND_ORDINARY, calc_before, 0)))
        c["words"] = np.where(turn, 0, c["words"] + dw)
        c["sentences"] = np.where(turn, 0, c["sentences"] + ds)
        c["lines"] = np.where(turn, 0, c["lines"] + dl)
        c["items"] = np.where(turn, 0, c["items"] + di)
        c["word_in_sent"] = np.where(turn, 0, np.where(inc[:, INC_WS_RESET] > 0,
                                                       inc[:, INC_WS_AFTER],
                                                       c["word_in_sent"] + dw))
        c["depth"] = np.where(turn, 0, np.maximum(inc[:, INC_DEP_LO],
                                                  c["depth"] + inc[:, INC_DEP_ADD]))
        c["tokens"] = np.where(turn | fresh, 0, c["tokens"] + 1)
        c["seg_sentences"] = np.where(is_bos, 0, c["seg_sentences"] + ds)
        c["seg_lines"] = np.where(is_bos, 0, c["seg_lines"] + dl)
        c["turn_idx"] = np.where(is_bos, 0, c["turn_idx"] + is_role)
        c["role"] = np.where(is_bos, 0, np.where(is_role, tab.role_code[tok], c["role"]))

        tgt = None
        if targets is not None:
            tgt = np.broadcast_to(np.asarray(targets, dtype=np.int64), (self.batch_size, 3))
        reg, clock = assemble(c, tgt, tab)
        if scalar:
            if self.batch_size != 1:
                raise ValueError("a scalar token needs batch_size == 1")
            return reg[0], clock[0]
        return reg, clock
