"""Token-local potentials: exact 0/1 masks over the vocabulary, for all particles at once.

The target of the decoder is ``pi(x) ~ p(x) * prod_t g_t(x_<=t)`` (``docs/MATH.md`` F1-F2). This
module computes the potentials ``g_t`` as boolean masks ``bool[K, V]`` from the byte-level
transducer tables (``bes.exact.instr_table``), small byte machines of its own and the compiled
``bes.exact.constraints.ConstraintSpec``.

What is enforced
----------------
Structure (``structure=True``):

- special tokens other than ``END``, ``CALC`` and ``EQ`` are never sampled; ``CALC`` only outside a
  calculator span; ``EQ`` only inside a calculator expression; tokens with ASCII control bytes
  (other than tab and newline) are never sampled;
- ``END`` only when no code fence, inline code span or double quote is open, the bracket depth
  is 0, no calculator span is open, and the response is not empty and does not end with ASCII
  whitespace (with the first token not starting with whitespace, ``display.strip()`` is a no-op);
- the bracket depth outside code stays ``<= max_depth`` after every token (a token that would
  end deeper is masked; documented change of the target);
- calculator spans (``CALC expr EQ result CALC_END``): ``CALC`` only when the vocabulary can spell
  a valid expression and the cheapest span (one expression token such as ``0``, ``EQ``, its
  forced result, ``CALC_END``) keeps an accepted ``END`` reachable within the budget, its counts
  included (no span adds less to any count, so under an upper word bound that no span can
  meet ``CALC`` is masked instead of stranding the particle); expression tokens consist of
  ``0-9 . + - * / % ( )`` and spaces and must keep the expression a *viable prefix* of the
  calculator's grammar (a recogniser with a parenthesis depth: no completion of ``"2 3"`` or
  ``"(*"`` is valid), the expression neither starts nor ends with whitespace (so the token bytes
  equal ``display_text``) and is at most ``calc.MAX_EXPR_CHARS`` long; ``EQ`` is allowed only when
  ``calc.evaluate(expr)`` is defined and the forced continuation and a finish fit the budget.
  After ``EQ`` the host forces ``tok.encode(result) + [CALC_END]``; the forced tokens are exempt
  from every mask (weight 1). **Fallback** (only reachable without potentials): an expression
  that does not evaluate forces just ``CALC_END`` (empty result, displayed as ``expr =``).

Counts (from the spec): ``words``, ``sentences``, ``items`` (the transducer's counts, exactly
``constraints.measure``), ``lines`` (lines with a byte other than ``" \\t\\r\\x0b\\x0c"``) and
``paragraphs`` (blank-line paragraphs, or the ``***`` divider chunks when the text has a divider,
exactly as ``constraints.measure``), each with a lower and/or upper bound; ``no_commas`` (tokens
with ``,`` are masked); ``end_question`` (``END`` only when the text, without trailing
whitespace and closers, ends with ``?``); ``code_block`` (``END`` only after a fenced block).
Paragraph constraints add one restriction (documented change of the target): outside a fence, a
line that so far holds only spaces, tabs and ``*`` with at least three ``*`` must stay a pure
``***`` divider line.

Token-local constraints of ``ConstraintSpec.token_local()`` besides the counts: ``case`` (a
static mask: ``"lower"`` bans tokens with an ASCII capital, ``"upper"`` tokens with an ASCII
small letter; "at least one cased letter" and non-ASCII letters are left to the verifier),
banned letters (a letter whose count must be 0: tokens holding it in either case are masked)
and ``forbidden`` words (:class:`ForbiddenMachine`, an aux machine of the reachability
search: no complete occurrence of a forbidden word, byte for byte as
``constraints.has_forbidden_bytes``, which only flags what ``constraints.verify`` flags). Other
spec fields (keywords, JSON, letter counts other than 0, families added later) are not
enforced here; verify them on the complete text (:func:`enforced` lists what is).

Reachability pruning (mandatory, exact)
---------------------------------------
Masking only immediate violations leaves many constrained particles unable to finish. Instead
the product automaton ``Q = (transducer state, bracket depth, aux machine state)`` (reachable
part, enumerated over the 80-odd token classes that have identical effects) times the counter
box is searched backwards: ``dist[q, c]`` = the minimum number of tokens (``END`` included) that
reach an accepting ``END`` from ``(q, c)``, by a layered min-plus dynamic programme (counters only
move one way, so each counter level is finished before the next one; inside a level a shortest-
path closure over zero-increment edges). A token is allowed iff its successor has
``dist <= remaining budget - 1`` (``budget_aware=True``, the default: every surviving particle
can still finish within ``max_new_tokens``) or ``dist < inf`` (``budget_aware=False``).
Counter coordinates are *remaining* quantities (room below an upper bound, or the amount still
needed for a lower bound). Product states with identical futures are merged (bisimulation
minimisation) before the search. Compiled programs are cached per (tokenizer, spec, options).
A spec whose joint table would exceed :data:`JOINT_LIMIT` cells (in practice: paragraphs or
lines together with a large word bound) is split into exactly solved parts whose masks are
ANDed: sound (no token with an accepted completion is ever masked), but a particle may then
still run into a joint dead end and die with weight 0. The masks are otherwise exact (an
allowed token always has an accepted completion), except inside a calculator expression, whose
forced result is unknown until ``EQ``.

Twist (optional, F5)
--------------------
:class:`Twist`: ``psi_b(q, c)`` = probability under a unigram proxy of the model (over the same
token classes plus ``END``) that an accepting ``END`` comes within ``b`` more tokens, by backward
dynamic programming over ``b``. It enters the target increment as ``psi(next) / psi(current)``
and cancels at the end (``psi = 1`` after an accepting ``END``); any positive ``psi`` keeps the
filter exact, a good one lowers the variance.

Shapes: ``V`` vocabulary, ``K`` particles, ``S`` transducer states (78), ``n_q`` product states,
``n_k`` token classes, ``m`` counter dimensions, box = product of the dimension sizes.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from bes.exact import calc as calc_mod
from bes.exact import instr_table as it
from bes.exact.transducer import COMPONENTS, PEND_OPEN, PREV_SPACE

INF = 32767
_C_FENCE, _C_INLINE, _C_QUOTE = (COMPONENTS.index(c) for c in ("fence", "inline", "quote"))
_C_PREV, _C_PEND = COMPONENTS.index("prev"), COMPONENTS.index("pend")

NL, CR, TAB, SP, STAR, BTICK, QMARK = 10, 13, 9, 32, 42, 96, 63
LINE_WS = frozenset(b" \t\r\x0b\x0c")
SP_TAB = frozenset(b" \t")
SP_TAB_STAR = frozenset(b" \t*")
WS_BYTES = frozenset(b" \t\n\r\x0b\x0c")
ASCII_CLOSERS = frozenset(b" \t\n\r\x0b\x0c\"')]}*_")
CALC_CHARS = frozenset(b"0123456789.+-*/%() ")
# Viable-prefix recogniser of calculator expressions (``calc.evaluate``'s grammar without its
# numeric limits): modes and the byte transitions; a parenthesis depth is kept beside the mode.
X_ATOM, X_STAR, X_SLASH, X_INT, X_FRAC, X_DOT, X_AFTER, X_INT0, X_INTLZ = range(9)
N_XMODES = 9
X_DEAD = -1
X_COMPLETE = (X_INT, X_INT0, X_FRAC, X_AFTER)


def _calc_step(mode: int, depth: int, b: int, check_depth: bool = True) -> tuple[int, int]:
    """One byte of an expression: ``(mode, depth)`` after it (``mode = X_DEAD``: no completion
    of the prefix is a valid expression). Modes: expecting an operand (start, after ``(``, an
    operator or a sign), just after one ``*`` or one ``/`` (a second one makes ``**`` / ``//``),
    in the digits of an integer (``X_INT``; ``X_INT0``: only zeros so far, valid; ``X_INTLZ``: a
    leading zero then other digits, valid only if a ``.`` follows), after the ``.`` of a number
    or in its fraction, after a leading ``.`` (a digit must follow), after a complete operand.
    ``check_depth=False`` lets the depth go negative (per-token tables record the lowest
    relative depth instead)."""
    expect = mode in (X_ATOM, X_STAR, X_SLASH)
    number = mode in (X_INT, X_INT0, X_FRAC)          # a number that may end here
    if 48 <= b <= 57:
        if expect:
            return (X_INT0 if b == 48 else X_INT), depth
        if mode == X_INT0:
            return (X_INT0 if b == 48 else X_INTLZ), depth
        if mode in (X_INT, X_INTLZ):
            return mode, depth
        if mode in (X_FRAC, X_DOT):
            return X_FRAC, depth
        return X_DEAD, depth
    if b == 46:                                   # .
        if expect:
            return X_DOT, depth
        return (X_FRAC if mode in (X_INT, X_INT0, X_INTLZ) else X_DEAD), depth
    if mode in (X_DOT, X_INTLZ):                  # any other byte ends the number: invalid
        return X_DEAD, depth
    if b == SP:
        return (X_ATOM if expect else X_AFTER), depth
    if b in (43, 45):                             # + -  (binary, or unary signs)
        return X_ATOM, depth
    if b == 42:                                   # *
        if mode == X_STAR:
            return X_ATOM, depth
        return (X_STAR if (number or mode == X_AFTER) else X_DEAD), depth
    if b == 47:                                   # /
        if mode == X_SLASH:
            return X_ATOM, depth
        return (X_SLASH if (number or mode == X_AFTER) else X_DEAD), depth
    if b == 37:                                   # %
        return (X_ATOM if (number or mode == X_AFTER) else X_DEAD), depth
    if b == 40:                                   # (
        return (X_ATOM, depth + 1) if expect else (X_DEAD, depth)
    if b == 41:                                   # )
        if (number or mode == X_AFTER) and (depth > 0 or not check_depth):
            return X_AFTER, depth - 1
        return X_DEAD, depth
    return X_DEAD, depth


_BR_MARKS = {42: 0, 45: 1, 95: 2}       # *, -, _
_MARK_BYTES = (42, 45, 95)
COUNT_UNITS = ("words", "sentences", "items", "lines", "paragraphs")
TX_FIELDS = {"words": it.INC_WORDS, "sentences": it.INC_SENT, "items": it.INC_ITEMS}


# =============================================================================== byte machines
# Fence-line parser (``[ \t]*```` at the start of a line).
FL_LEAD, FL_T1, FL_T2, FL_YES, FL_NO = range(5)
# Thematic-break parser of ``constraints._BREAK_LINE``: 1..4 = 0..3 leading spaces,
# 5 + 3 * mark + (n - 1) after ``n`` (capped at 3) marks of one kind, 0 = failed.
BR_FAIL = 0
# Star-rule parser of ``_STAR_RULE_LINE``: 0..3 stars so far (3 = three or more), 4 = failed.
SR_FAIL = 4


def _fl_step(fl: int, b: int) -> int:
    if fl == FL_LEAD:
        return FL_LEAD if b in SP_TAB else (FL_T1 if b == BTICK else FL_NO)
    if fl == FL_T1:
        return FL_T2 if b == BTICK else FL_NO
    if fl == FL_T2:
        return FL_YES if b == BTICK else FL_NO
    return fl


def _br_step(br: int, b: int) -> int:
    if br == BR_FAIL:
        return BR_FAIL
    if br <= 4:
        if b == SP:
            return br + 1 if br < 4 else BR_FAIL
        mark = _BR_MARKS.get(b)
        return BR_FAIL if mark is None else 5 + 3 * mark
    mark, n1 = divmod(br - 5, 3)
    if b == _MARK_BYTES[mark]:
        return 5 + 3 * mark + min(n1 + 1, 2)
    return br if b in SP_TAB else BR_FAIL


def _sr_step(sr: int, b: int) -> int:
    if sr == SR_FAIL:
        return SR_FAIL
    if b in SP_TAB:
        return sr
    return min(sr + 1, 3) if b == STAR else SR_FAIL


_LINE_RESET = (1, FL_LEAD, 1, 0, 0, 0, 0, 0, 0, 0, 1)   # blank .. prev_ws


class LineMachine:
    """Lines, paragraphs and fenced blocks of ``constraints.measure``, one byte at a time.

    State ``(fence, in_par, in_chunk, has_div, fenced, blank, fl, br, sr, cr, phase, pc, run,
    run_ok, rcr, prev_ws)``: the first five persist across lines (fence state at the start of
    the line, inside a blank-line paragraph, inside a divider chunk, a divider was seen, a fence
    line was seen); the rest describe the current line (only ASCII whitespace so far, the
    fence-line / thematic-break / star-rule parsers, pending carriage returns, the divider
    splitter: phase 0 = only ``[ \\t*]`` so far, 1 = resolved, current piece has content, star
    run length, run preceded by a space or tab or the line start, carriage returns after a
    candidate divider run, previous byte was a space or tab).

    Events ``(lines, pars, chunks)``: a non-blank line begins (counted at its first byte that is
    not ASCII whitespace), a blank-line paragraph begins (counted at the end of its first line),
    a divider chunk begins. The paragraph count is ``chunks`` if ``has_div`` else ``pars``.
    ``step`` returns ``None`` for the bytes the paragraph restriction forbids (module docstring).
    """

    events = ("lines", "pars", "chunks")
    flags = ("has_div", "fenced")

    def __init__(self, lines: bool = False, paragraphs: bool = False, fenced: bool = False):
        self.lines, self.paragraphs, self.fenced = bool(lines), bool(paragraphs), bool(fenced)

    def init(self) -> tuple:
        return (0, 0, 0, 0, 0) + _LINE_RESET

    def _end_line(self, st: tuple) -> tuple[tuple, tuple[int, int, int]]:
        (fence, in_par, in_chunk, has_div, fenced, blank, fl, br, sr, cr, phase, pc, run,
         run_ok, rcr, prev_ws) = st
        pars = chunks = 0
        if self.paragraphs:
            br_match = br >= 5 and (br - 5) % 3 == 2
            if not fence and (blank or br_match):
                in_par = 0
            elif not in_par:
                pars, in_par = 1, 1
            if not fence:
                content = False
                if phase == 0:
                    if sr == 3:
                        has_div, in_chunk = 1, 0
                    elif sr in (1, 2):
                        content = True
                elif rcr or (run > 0 and run_ok and run >= 3):
                    has_div, in_chunk = 1, 0
                elif run > 0:
                    content = True
                if content and not pc and not in_chunk:
                    chunks, in_chunk = 1, 1
            if fl == FL_YES:
                fence ^= 1
        if self.fenced and fl == FL_YES:
            fenced = 1
        return (fence, in_par, in_chunk, has_div, fenced) + _LINE_RESET, (0, pars, chunks)

    def step(self, st: tuple, b: int):
        if b == NL:
            return self._end_line(st)
        (fence, in_par, in_chunk, has_div, fenced, blank, fl, br, sr, cr, phase, pc, run,
         run_ok, rcr, prev_ws) = st
        content = b not in LINE_WS
        ev_lines = 1 if (self.lines and blank and content) else 0
        if self.lines or self.paragraphs:
            blank = 0 if content else blank
        if self.paragraphs or self.fenced:
            fl = _fl_step(fl, b)
        if not self.paragraphs:
            return (fence, in_par, in_chunk, has_div, fenced, blank, fl, br, sr, cr, phase, pc,
                    run, run_ok, rcr, prev_ws), (ev_lines, 0, 0)
        chunks = 0

        def content_event():
            nonlocal pc, in_chunk, chunks
            if not pc:
                pc = 1
                if not in_chunk:
                    chunks, in_chunk = chunks + 1, 1

        if fence:
            if content:
                content_event()
            return (fence, in_par, in_chunk, has_div, fenced, blank, fl, br, sr, cr, phase, pc,
                    run, run_ok, rcr, prev_ws), (ev_lines, 0, chunks)
        sr_prev = sr
        internal_cr = False
        if b == CR:
            cr = 1
        else:
            if cr:
                internal_cr, cr, br, sr = True, 0, BR_FAIL, SR_FAIL
            br, sr = _br_step(br, b), _sr_step(sr, b)

        fresh = False
        if phase == 0:
            if b != CR and (internal_cr or b not in SP_TAB_STAR):
                if sr_prev == 3:
                    return None                      # a "***" line must stay a divider line
                phase, pc, run, run_ok, rcr = 1, 0, 0, 0, 0
                if sr_prev >= 1:
                    content_event()
                if internal_cr:
                    prev_ws = 0
                fresh = True
        else:
            if b == CR:
                if rcr:
                    pass
                elif run > 0 and run_ok and run >= 3:
                    rcr = 1
                else:
                    if run > 0:
                        content_event()
                        run = 0
                    prev_ws = 0
            elif rcr:
                content_event()
                rcr, run, prev_ws, fresh = 0, 0, 0, True
            elif run > 0:
                if b == STAR:
                    run = min(run + 1, 3)
                elif run_ok and run >= 3 and b in SP_TAB:
                    has_div, in_chunk, pc, run, fresh = 1, 0, 0, 0, True
                else:
                    content_event()
                    run, fresh = 0, True
            else:
                fresh = True
        if fresh:
            if b == STAR:
                run, run_ok = 1, prev_ws
                if not run_ok:
                    content_event()
            elif b in LINE_WS:
                prev_ws = 1 if b in SP_TAB else 0
            else:
                content_event()
                prev_ws = 0
        return (fence, in_par, in_chunk, has_div, fenced, blank, fl, br, sr, cr, phase, pc, run,
                run_ok, rcr, prev_ws), (ev_lines, 0, chunks)

    def flush(self, st: tuple) -> tuple[tuple, tuple[int, int, int]]:
        return self._end_line(st)

    def accept(self, st: tuple) -> bool:
        return (not self.fenced) or bool(st[4])

    def flag(self, st: tuple) -> tuple[int, int]:
        return int(st[3]), int(st[4])


class EndQuestionMachine:
    """``text.rstrip(constraints._TRAILING_CLOSERS).endswith("?")`` one byte at a time.

    State ``(q, pend)``: ``q`` = the last significant character is ``?``; ``pend`` tracks the
    multi-byte closers ``» (C2 BB)``, ``’ (E2 80 99)`` and ``” (E2 80 9D)``: 1 after ``C2``, 2 after
    ``E2``, 3 after ``E2 80``. A broken or unfinished sequence is a significant character.
    """

    events: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()

    def init(self) -> tuple:
        return (0, 0)

    def step(self, st: tuple, b: int):
        q, pend = st
        if pend:
            if pend == 1 and b == 0xBB:
                return (q, 0), ()
            if pend == 2 and b == 0x80:
                return (q, 3), ()
            if pend == 3 and b in (0x99, 0x9D):
                return (q, 0), ()
            q, pend = 0, 0
        if b < 0x80:
            if b == QMARK:
                q = 1
            elif b not in ASCII_CLOSERS:
                q = 0
        elif b == 0xC2:
            pend = 1
        elif b == 0xE2:
            pend = 2
        else:
            q = 0
        return (q, pend), ()

    def flush(self, st: tuple):
        return ((0 if st[1] else st[0]), 0), ()

    def accept(self, st: tuple) -> bool:
        return st[0] == 1

    def flag(self, st: tuple) -> tuple:
        return ()


def _word_byte(b: int) -> bool:
    """A word byte of ``constraints.has_forbidden_bytes``: ASCII letter or digit, or >= 0x80."""
    return b >= 0x80 or 48 <= b <= 57 or 65 <= b <= 90 or 97 <= b <= 122


class ForbiddenMachine:
    """No complete occurrence of a forbidden word, one byte at a time: exactly
    ``not constraints.has_forbidden_bytes(text, words)`` (``ConstraintSpec.token_local().forbidden``:
    lowercase bytes, matched ASCII-case-insensitively, with no word byte right before or after).
    That byte-level test only flags texts ``constraints.verify`` also flags (it is exact for
    ASCII neighbours), so the potential never removes a response the verifier accepts.

    State ``(prev_word, partial, pending)``: the previous byte is a word byte; ``partial``: the
    sorted ``(word index, bytes matched)`` of the matches in progress (a match starts only after
    a non-word byte or at the start of the response); ``pending``: some word is complete up to
    the previous byte. The next byte decides a pending occurrence: a word byte extends it into a
    longer word (no occurrence), any other byte completes it, which :meth:`step` forbids by
    returning ``None``; at the end of the text :meth:`accept` rejects a pending occurrence.
    """

    events: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()

    def __init__(self, words: Sequence[bytes]):
        self.words = tuple(bytes(w).lower() for w in words if w)

    def init(self) -> tuple:
        return (0, (), 0)

    def step(self, st: tuple, b: int):
        prev_word, partial, pending = st
        word = _word_byte(b)
        if pending and not word:
            return None
        lb = b + 32 if 65 <= b <= 90 else b
        nxt, done = set(), 0
        for i, j in partial:
            w = self.words[i]
            if w[j] == lb:
                if j + 1 == len(w):
                    done = 1
                else:
                    nxt.add((i, j + 1))
        if not prev_word:
            for i, w in enumerate(self.words):
                if w[0] == lb:
                    if len(w) == 1:
                        done = 1
                    else:
                        nxt.add((i, 1))
        return (int(word), tuple(sorted(nxt)), done), ()

    def flush(self, st: tuple):
        return st, ()

    def accept(self, st: tuple) -> bool:
        return not st[2]

    def flag(self, st: tuple) -> tuple:
        return ()


class ProductMachine:
    """Several byte machines side by side (state = tuple of states, events concatenated)."""

    def __init__(self, parts: Sequence):
        self.parts = tuple(parts)
        self.events = tuple(e for p in self.parts for e in p.events)
        self.flags = tuple(f for p in self.parts for f in p.flags)

    def init(self):
        return tuple(p.init() for p in self.parts)

    def step(self, st, b):
        states, evs = [], []
        for p, s in zip(self.parts, st):
            out = p.step(s, b)
            if out is None:
                return None
            states.append(out[0])
            evs.extend(out[1])
        return tuple(states), tuple(evs)

    def flush(self, st):
        states, evs = [], []
        for p, s in zip(self.parts, st):
            s2, e = p.flush(s)
            states.append(s2)
            evs.extend(e)
        return tuple(states), tuple(evs)

    def accept(self, st) -> bool:
        return all(p.accept(s) for p, s in zip(self.parts, st))

    def flag(self, st) -> tuple:
        return tuple(f for p, s in zip(self.parts, st) for f in p.flag(s))


def run_machine(machine, data: bytes, final: bool = True):
    """Reference run of a byte machine over ``data``: ``(state, events summed, ok)``; ``ok`` is
    False when a byte is forbidden. With ``final`` the end of the text is flushed."""
    st = machine.init()
    tot = [0] * len(machine.events)
    for b in data:
        out = machine.step(st, b)
        if out is None:
            return st, tuple(tot), False
        st, ev = out
        tot = [a + e for a, e in zip(tot, ev)]
    if final:
        st, ev = machine.flush(st)
        tot = [a + e for a, e in zip(tot, ev)]
    return st, tuple(tot), True


@dataclass(eq=False)
class AuxTables:
    """A byte machine compiled over its reachable states; index ``n`` (= ``dead``) is a sink.

    ``next_byte int32[n + 1, 256]``, ``ev_byte int16[n + 1, 256, E]``, ``flush_ev int16[n + 1, E]``,
    ``flush_accept bool[n + 1]``, ``flush_flag int8[n + 1, F]``; token tables ``next_tok
    int32[V + 1, n + 1]`` and ``ev_tok int16[V + 1, n + 1, E]`` (row ``V``: the bytes ``" = "`` of
    ``EQ`` inside a calculator span; special tokens other rows: identity).
    """

    machine: object
    states: list
    next_byte: np.ndarray
    ev_byte: np.ndarray
    flush_ev: np.ndarray
    flush_accept: np.ndarray
    flush_flag: np.ndarray
    next_tok: np.ndarray | None = None
    ev_tok: np.ndarray | None = None

    @property
    def n(self) -> int:
        return len(self.states)

    @property
    def dead(self) -> int:
        return len(self.states)


def compile_aux(machine, alphabet: Sequence[int] = range(256)) -> AuxTables:
    """Enumerate the states of ``machine`` reachable from ``init`` over the bytes of
    ``alphabet`` (other bytes lead to the dead sink) and tabulate step, flush and accept."""
    alphabet = sorted(set(int(b) for b in alphabet))
    states = [machine.init()]
    index = {states[0]: 0}
    trans: list[dict[int, tuple]] = []
    pos = 0
    while pos < len(states):
        st = states[pos]
        pos += 1
        row = {}
        for b in alphabet:
            out = machine.step(st, b)
            if out is None:
                continue
            s2, ev = out
            if s2 not in index:
                index[s2] = len(states)
                states.append(s2)
            row[b] = (index[s2], ev)
        trans.append(row)
    n, n_e, n_f = len(states), len(machine.events), len(machine.flags)
    nxt = np.full((n + 1, 256), n, np.int32)
    ev = np.zeros((n + 1, 256, n_e), np.int16)
    for i, row in enumerate(trans):
        for b, (j, e) in row.items():
            nxt[i, b] = j
            if n_e:
                ev[i, b] = e
    fev = np.zeros((n + 1, n_e), np.int16)
    facc = np.zeros(n + 1, bool)
    fflag = np.zeros((n + 1, n_f), np.int8)
    for i, st in enumerate(states):
        s2, e = machine.flush(st)
        if n_e:
            fev[i] = e
        facc[i] = machine.accept(s2)
        if n_f:
            fflag[i] = machine.flag(s2)
    return AuxTables(machine, states, nxt, ev, fev, facc, fflag)


def aux_token_tables(aux: AuxTables, token_bytes: Sequence[bytes]) -> AuxTables:
    """Compose the per-byte tables over every token's bytes (rows ``0..V-1``; row ``V`` is
    ``b" = "``), all states at once, tokens of equal length together."""
    rows = list(token_bytes) + [it.EQ_BYTES]
    n1 = aux.n + 1
    n_e = aux.ev_byte.shape[-1]
    nxt = np.empty((len(rows), n1), np.int16 if n1 < 32767 else np.int32)
    ev = np.zeros((len(rows), n1, n_e), np.int8)       # a token has at most 127 bytes
    by_len: dict[int, list[int]] = {}
    for r, data in enumerate(rows):
        by_len.setdefault(len(data), []).append(r)
    nb, eb = aux.next_byte, aux.ev_byte.astype(np.int32)
    for length, ids in by_len.items():
        idx = np.asarray(ids)
        s = np.broadcast_to(np.arange(n1, dtype=np.int32), (len(ids), n1)).copy()
        acc = np.zeros((len(ids), n1, n_e), np.int32)
        if length:
            mat = np.frombuffer(b"".join(rows[r] for r in ids), np.uint8).reshape(len(ids), length)
            for j in range(length):
                b = mat[:, j].astype(np.intp)[:, None]
                if n_e:
                    acc += eb[s, b]
                s = nb[s, b]
        nxt[idx] = s
        ev[idx] = acc
    aux.next_tok, aux.ev_tok = nxt, ev
    return aux


# ==================================================================================== dims
@dataclass(frozen=True)
class Dim:
    """One counter dimension in *remaining* coordinates.

    ``source``: the counted event (``words``, ``sentences``, ``items``, ``lines``, ``pars``,
    ``chunks``). Index ``i`` in ``[0, size)``; a token with increment ``n`` moves ``i`` to
    ``i - n``; below 0 it is invalid (``saturate=False``: an upper bound exceeded for good) or
    stays 0 (``saturate=True``: a lower bound already met, or the absorbing "over" value of a
    paragraph dimension). ``start`` is the index at the start of the response, ``accept[i]``
    whether index ``i`` satisfies the bound at ``END``; ``mode``: 2 always checked, 0 checked
    when the text has no ``***`` divider, 1 when it has one.
    """

    unit: str
    source: str
    kind: str
    width: int
    saturate: bool
    size: int
    start: int
    mode: int
    accept: tuple[bool, ...]


def dims_for(spec) -> list[Dim]:
    """Counter dimensions of the count constraints of ``spec`` (unknown fields ignored)."""
    dims: list[Dim] = []
    for unit in COUNT_UNITS:
        iv = getattr(spec, unit, None) if spec is not None else None
        if iv is None:
            continue
        lo, hi = int(iv.lo), (None if iv.hi is None else int(iv.hi))
        sources = [("pars", 0), ("chunks", 1)] if unit == "paragraphs" else [(unit, 2)]
        for src, mode in sources:
            if hi is None:
                if lo <= 0:
                    continue
                dims.append(Dim(unit, src, "at_least", 0, True, lo + 1, lo, mode,
                                tuple(i == 0 for i in range(lo + 1))))
                continue
            kind = "exactly" if lo == hi else ("at_most" if lo == 0 else "between")
            width = hi - lo
            if unit == "paragraphs":
                size = hi + 2
                acc = tuple(1 <= i <= width + 1 for i in range(size))
                dims.append(Dim(unit, src, kind, width, True, size, hi + 1, mode, acc))
            else:
                size = hi + 1
                acc = tuple(i <= width for i in range(size))
                dims.append(Dim(unit, src, kind, width, False, size, hi, mode, acc))
    return dims


def _token_local(spec) -> dict:
    """Token-local constraints of ``spec`` besides the counts and flags, from
    ``ConstraintSpec.token_local()``: ``forbidden`` (lowercase bytes of the forbidden words,
    sorted), ``case`` (``"lower"``, ``"upper"`` or ``None``) and ``banned_letters`` (the byte
    pair of every letter whose upper bound is 0, sorted)."""
    out = {"forbidden": (), "case": None, "banned_letters": ()}
    if spec is None or not hasattr(spec, "token_local"):
        return out
    tl = spec.token_local()
    out["forbidden"] = tuple(sorted(set(tl.forbidden)))
    out["case"] = tl.case
    out["banned_letters"] = tuple(sorted(lb for lb, _lo, hi in tl.letters if hi == 0))
    return out


def enforced(spec) -> list[str]:
    """The constraints of ``spec`` that the potentials guarantee (``ConstraintSpec.constraints``
    wording); everything else must be checked with ``constraints.verify`` on the final text.
    ``case`` is guaranteed for ASCII letters only (``verify`` also wants a cased letter)."""
    out = []
    for unit in COUNT_UNITS:
        iv = getattr(spec, unit, None)
        if iv is not None:
            out.append(f"{unit}: {iv}")
    for name in ("end_question", "no_commas", "code_block"):
        if getattr(spec, name, False):
            out.append(name)
    out += [f"forbidden: {k!r}" for k in getattr(spec, "forbidden", ())]
    if getattr(spec, "case", None):
        out.append(f"case: {spec.case}")
    out += [f"letter: {w!r} {iv}" for w, iv in getattr(spec, "letters", ()) if iv.hi == 0]
    return out


# =============================================================================== automaton
def _void_rows(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()


@dataclass(eq=False)
class Automaton:
    """Reachable product ``(transducer state, depth, aux state)`` over token classes.

    ``cls int32[V]`` (class of every token of the DP alphabet, -1 otherwise), ``reps``;
    ``st_s, st_d, st_a int32[n_q]``; ``qmap int32[S, max_depth + 1, n_aux + 1]`` (-1 = not a
    product state); ``nq int32[n_q, n_k]`` (successor, -1 invalid); ``delta int16[n_q, n_k, n_src]``
    (increment of every source); ``end_ok bool[n_q]``, ``end_delta int16[n_q, n_src]`` (flush),
    ``end_flag int8[n_q, F]`` (aux flags after the flush: ``has_div`` ...).
    """

    sources: tuple[str, ...]
    max_depth: int
    cls: np.ndarray
    reps: np.ndarray
    st_s: np.ndarray
    st_d: np.ndarray
    st_a: np.ndarray
    qmap: np.ndarray
    nq: np.ndarray
    delta: np.ndarray
    end_ok: np.ndarray
    end_delta: np.ndarray
    end_flag: np.ndarray
    aux: AuxTables | None

    @property
    def n_q(self) -> int:
        return int(self.st_s.shape[0])

    @property
    def n_k(self) -> int:
        return int(self.reps.shape[0])

    @property
    def q0(self) -> int:
        """Product state at the start of a response."""
        return int(self.qmap[0, 0, 0])


def build_automaton(table: it.InstrTable, aux: AuxTables | None, alphabet: np.ndarray,
                    sources: Sequence[str], max_depth: int, trim: bool,
                    track_depth: bool = True) -> Automaton:
    """Token classes over ``alphabet`` (``bool[V]``), the reachable product and its tables
    (``track_depth=False``: the bracket depth is ignored and stays 0)."""
    v_all = table.vocab_size
    ids = np.flatnonzero(alphabet[:v_all]).astype(np.int64)
    n_s = table.n_states
    tx_next = table.next_state.astype(np.int32)
    tx_inc = table.inc.astype(np.int16)
    fields = [it.INC_DEP_ADD, it.INC_DEP_LO] + [TX_FIELDS[s] for s in sources if s in TX_FIELDS]
    parts = [tx_next[ids], tx_inc[ids][:, :, fields].reshape(len(ids), -1)]
    aux_src = [s for s in sources if s not in TX_FIELDS]
    if aux is not None:
        e_idx = [aux.machine.events.index(s) for s in aux_src]
        parts.append(aux.next_tok[ids])
        if e_idx:
            parts.append(aux.ev_tok[ids][:, :, e_idx].reshape(len(ids), -1))
    sig = np.concatenate([p.astype(np.int32) for p in parts], axis=1)
    _, first, inv = np.unique(_void_rows(sig), return_index=True, return_inverse=True)
    reps = ids[first]
    cls = np.full(v_all, -1, np.int32)
    cls[ids] = inv.reshape(-1).astype(np.int32)
    n_k = len(reps)

    k_s2 = tx_next[reps]                                  # [n_k, S]
    k_add = tx_inc[reps][:, :, it.INC_DEP_ADD].astype(np.int32)
    k_lo = tx_inc[reps][:, :, it.INC_DEP_LO].astype(np.int32)
    if not track_depth:
        k_add, k_lo = np.zeros_like(k_add), np.zeros_like(k_lo)
    n_a1 = aux.n + 1 if aux is not None else 1
    a_dead = aux.dead if aux is not None else -1
    k_a2 = aux.next_tok[reps] if aux is not None else np.zeros((n_k, 1), np.int32)
    a0 = 0
    d1 = max_depth + 1
    qmap = np.full((n_s, d1, n_a1), -1, np.int32)
    st_s, st_d, st_a = [0], [0], [a0]
    qmap[0, 0, a0] = 0
    nq_rows: list[np.ndarray] = []
    frontier = np.array([0])
    while frontier.size:
        fs, fd, fa = (np.asarray(x)[frontier] for x in (st_s, st_d, st_a))
        s2 = k_s2[:, fs].T                                 # [f, n_k]
        d2 = np.maximum(k_lo[:, fs].T, fd[:, None] + k_add[:, fs].T)
        a2 = k_a2[:, fa].T
        valid = (d2 <= max_depth) & (a2 != a_dead)
        d2c = np.clip(d2, 0, max_depth)
        a2c = np.where(valid, a2, 0)
        cand = np.where(valid, qmap[s2, d2c, a2c], -2)
        new = valid & (cand == -1)
        if new.any():
            code = (s2[new].astype(np.int64) * d1 + d2c[new]) * n_a1 + a2c[new]
            ucode = np.unique(code)
            base = len(st_s)
            us = ucode // (d1 * n_a1)
            ud = (ucode // n_a1) % d1
            ua = ucode % n_a1
            qmap[us, ud, ua] = np.arange(base, base + len(ucode), dtype=np.int32)
            st_s.extend(us.tolist())
            st_d.extend(ud.tolist())
            st_a.extend(ua.tolist())
            nf = np.arange(base, base + len(ucode))
        else:
            nf = np.zeros(0, np.int64)
        rows = np.where(valid, qmap[s2, d2c, a2c], -1).astype(np.int32)
        nq_rows.extend(rows)
        frontier = nf
    st_s, st_d, st_a = (np.asarray(x, np.int32) for x in (st_s, st_d, st_a))
    nq = np.stack(nq_rows).astype(np.int32)
    n_q = nq.shape[0]

    m = len(sources)
    delta = np.zeros((n_q, n_k, m), np.int16)
    for c, src in enumerate(sources):
        if src in TX_FIELDS:
            delta[:, :, c] = tx_inc[reps][:, :, TX_FIELDS[src]][:, st_s].T
        else:
            e = aux.machine.events.index(src)
            delta[:, :, c] = aux.ev_tok[reps][:, :, e][:, st_a].T

    comp = table.components.astype(np.int32)[st_s]
    end_ok = (comp[:, _C_FENCE] == 0) & (comp[:, _C_INLINE] == 0) & (comp[:, _C_QUOTE] == 0)
    end_ok &= st_d == 0
    if trim:
        end_ok &= comp[:, _C_PREV] != PREV_SPACE
    end_delta = np.zeros((n_q, m), np.int16)
    n_f = len(aux.machine.flags) if aux is not None else 0
    end_flag = np.zeros((n_q, n_f), np.int8)
    for c, src in enumerate(sources):
        if src == "sentences":
            end_delta[:, c] = comp[:, _C_PEND] == PEND_OPEN
        elif src not in TX_FIELDS:
            end_delta[:, c] = aux.flush_ev[st_a, aux.machine.events.index(src)]
    if aux is not None:
        end_ok &= aux.flush_accept[st_a]
        end_flag = aux.flush_flag[st_a]
    return minimize(Automaton(tuple(sources), max_depth, cls, reps, st_s, st_d, st_a, qmap, nq,
                              delta, end_ok, end_delta, end_flag, aux))


def minimize(auto: Automaton) -> Automaton:
    """Merge bisimilar product states (Moore partition refinement): two states are merged when
    they agree on the ``END`` outputs (``end_ok``, ``end_delta``, ``end_flag``) and, for every token
    class, on the increments and the block of the successor. Merged states have identical
    distance and twist functions, so the masks are unchanged; ``qmap`` points to the blocks."""
    n_q, n_k = auto.nq.shape
    out0 = np.concatenate([auto.end_ok[:, None].astype(np.int32), auto.end_delta.astype(np.int32),
                           auto.end_flag.astype(np.int32)], axis=1)
    _, part = np.unique(_void_rows(out0), return_inverse=True)
    part = part.reshape(-1).astype(np.int64)
    dl = auto.delta.reshape(n_q, -1).astype(np.int32)
    n_parts = int(part.max()) + 1
    while True:
        succ = np.where(auto.nq >= 0, part[np.clip(auto.nq, 0, None)], -1)
        sig = np.concatenate([part[:, None], succ, dl], axis=1).astype(np.int32)
        _, new = np.unique(_void_rows(sig), return_inverse=True)
        new = new.reshape(-1).astype(np.int64)
        n_new = int(new.max()) + 1
        part = new
        if n_new == n_parts:
            break
        n_parts = n_new
    rep = np.zeros(n_parts, np.int64)
    rep[part[::-1]] = np.arange(n_q)[::-1]        # first state of every block
    nq = np.where(auto.nq[rep] >= 0, part[np.clip(auto.nq[rep], 0, None)], -1).astype(np.int32)
    qmap = np.where(auto.qmap >= 0, part[np.clip(auto.qmap, 0, None)], -1).astype(np.int32)
    return Automaton(auto.sources, auto.max_depth, auto.cls, auto.reps, auto.st_s[rep],
                     auto.st_d[rep], auto.st_a[rep], qmap, nq, auto.delta[rep], auto.end_ok[rep],
                     auto.end_delta[rep], auto.end_flag[rep], auto.aux)


# ==================================================================================== box
@dataclass(eq=False)
class Box:
    """Mixed-radix counter box of a list of :class:`Dim` (last dimension fastest)."""

    dims: tuple[Dim, ...]

    def __post_init__(self):
        self.sizes = np.array([d.size for d in self.dims], np.int64)
        self.n = int(np.prod(self.sizes)) if len(self.dims) else 1
        self.strides = np.ones(len(self.dims), np.int64)
        for i in range(len(self.dims) - 2, -1, -1):
            self.strides[i] = self.strides[i + 1] * self.sizes[i + 1]
        self.sat = np.array([d.saturate for d in self.dims], bool)
        flat = np.arange(self.n, dtype=np.int64)
        self.coords = ((flat[:, None] // self.strides[None, :]) % self.sizes[None, :]) \
            if len(self.dims) else np.zeros((1, 0), np.int64)

    def move(self, idx: np.ndarray, inc: np.ndarray) -> np.ndarray:
        """Indices after increments (broadcast ``[..., m]``); -1 marks an invalid coordinate."""
        new = np.asarray(idx, np.int64) - np.asarray(inc, np.int64)
        new = np.where(self.sat & (new < 0), 0, new)
        return np.where(new < 0, -1, new)

    def flat(self, idx: np.ndarray) -> np.ndarray:
        """Flat index of ``idx [..., m]`` (``-1`` if any coordinate is -1)."""
        idx = np.asarray(idx, np.int64)
        if not len(self.dims):
            return np.zeros(idx.shape[:-1], np.int64)
        bad = (idx < 0).any(axis=-1)
        f = (np.clip(idx, 0, None) * self.strides).sum(axis=-1)
        return np.where(bad, -1, f)

    def accept(self, idx: np.ndarray, flags: np.ndarray) -> np.ndarray:
        """Bound check at ``END``: ``idx [..., m]`` (after the flush), ``flags [..., F]`` (aux flags
        after the flush; ``flags[..., 0]`` is ``has_div`` when paragraphs are constrained)."""
        idx = np.asarray(idx, np.int64)
        ok = np.ones(idx.shape[:-1], bool)
        for c, d in enumerate(self.dims):
            col = idx[..., c]
            acc = np.asarray(d.accept, bool)
            good = (col >= 0) & acc[np.clip(col, 0, d.size - 1)]
            if d.mode == 2:
                ok &= good
            else:
                has_div = np.asarray(flags)[..., 0] > 0
                checked = has_div if d.mode == 1 else ~has_div
                ok &= np.where(checked, good, col >= 0)
        return ok


def _edges(auto: Automaton, cols: list[int]):
    """Unique edges ``(src, dst, did)`` sorted by ``src`` and the distinct increment vectors."""
    n_q, n_k = auto.nq.shape
    inc = auto.delta[:, :, cols].reshape(n_q * n_k, len(cols)) if cols else \
        np.zeros((n_q * n_k, 0), np.int16)
    if cols:
        uvec, did = np.unique(inc, axis=0, return_inverse=True)
        did = did.reshape(n_q, n_k)
    else:
        uvec, did = np.zeros((1, 0), np.int16), np.zeros((n_q, n_k), np.int64)
    n_d = len(uvec)
    valid = auto.nq >= 0
    q = np.broadcast_to(np.arange(n_q)[:, None], (n_q, n_k))
    key = (q[valid].astype(np.int64) * n_q + auto.nq[valid]) * n_d + did[valid]
    ukey = np.unique(key)
    src = ukey // (n_q * n_d)
    dst = (ukey // n_d) % n_q
    d = ukey % n_d
    return src, dst, d, uvec.astype(np.int64), did


def _segments(src: np.ndarray, n: int):
    """Start offsets of the runs of a sorted ``src`` and the source of each run."""
    if src.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    starts = np.flatnonzero(np.r_[True, src[1:] != src[:-1]])
    return starts, src[starts]


def _end_accept(auto: Automaton, box: Box, cols: list[int], pts: np.ndarray) -> np.ndarray:
    """``bool[n_q, len(pts)]``: an ``END`` from ``(q, pts)`` is accepted."""
    n_q = auto.n_q
    if cols:
        idx = box.move(box.coords[pts][None, :, :], auto.end_delta[:, cols][:, None, :])
    else:
        idx = np.zeros((n_q, len(pts), 0), np.int64)
    flags = np.broadcast_to(auto.end_flag[:, None, :], (n_q, len(pts), auto.end_flag.shape[1]))
    return auto.end_ok[:, None] & box.accept(idx, flags)


def distance_table(auto: Automaton, box: Box, cols: list[int]) -> np.ndarray:
    """``int16[n_q + 1, n_box + 1]``: minimum number of tokens (``END`` included) from
    ``(q, c)`` to an accepting ``END`` (``INF`` = never); the extra row and column are ``INF``
    sinks for invalid successors. Layered min-plus over counter levels (module docstring)."""
    n_q = auto.n_q
    src, dst, did, uvec, _ = _edges(auto, cols)
    succ = np.stack([box.flat(box.move(box.coords, uvec[i][None, :])) for i in range(len(uvec))]) \
        if len(cols) else np.zeros((len(uvec), 1), np.int64)
    succ = np.where(succ < 0, box.n, succ).astype(np.int32)
    dist = np.full((n_q + 1, box.n + 1), INF, np.int32)
    levels = box.coords.sum(axis=1) if len(cols) else np.zeros(1, np.int64)
    starts, seg_src = _segments(src, n_q)
    for lev in range(int(levels.max()) + 1 if levels.size else 1):
        pts = np.flatnonzero(levels == lev)
        if pts.size == 0:
            continue
        cur = np.where(_end_accept(auto, box, cols, pts), 1, INF).astype(np.int32)
        if src.size:
            s_e = succ[did[:, None], pts[None, :]]                        # [E, P]
            self_e = s_e == pts[None, :]
            cand = dist[dst[:, None], s_e]
            cand = np.where(self_e, INF, cand)
            red = np.minimum.reduceat(cand, starts, axis=0)
            cur[seg_src] = np.minimum(cur[seg_src], np.minimum(red + 1, INF))
            rows = np.flatnonzero(self_e.any(axis=1))
            if rows.size:
                s_src, s_dst, s_mask = src[rows], dst[rows], self_e[rows]
                s_starts, s_seg = _segments(s_src, n_q)
                for _ in range(n_q + 1):
                    c2 = np.where(s_mask, cur[s_dst], INF)
                    red2 = np.minimum(np.minimum.reduceat(c2, s_starts, axis=0) + 1, INF)
                    new = np.minimum(cur[s_seg], red2)
                    if np.array_equal(new, cur[s_seg]):
                        break
                    cur[s_seg] = new
        dist[:n_q, pts] = cur
    return np.minimum(dist, INF).astype(np.int16)


# ================================================================================= program
#: Largest ``product states x box cells`` of one exact part. A spec whose joint table would be
#: larger is split into a transducer part (words, sentences, items) and a line-machine part
#: (lines, paragraphs, end_question, code_block); each part is exact on its own and their masks
#: are ANDed (sound: never masks a token with an accepted completion; the joint constraint may
#: then still fail later, which kills the particle with weight 0).
JOINT_LIMIT = 4_000_000


@dataclass(eq=False)
class Part:
    """One exact reachability table: product automaton, counter box and distance table."""

    auto: Automaton
    box: Box
    cols: list
    dist: np.ndarray

    @property
    def m(self) -> int:
        return len(self.box.dims)

    def start_index(self) -> np.ndarray:
        return np.array([d.start for d in self.box.dims], np.int64)

    def distance(self, q: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """``dist`` of product states ``q [...]`` (-1 allowed) with counters ``idx [..., m]``."""
        q = np.asarray(q, np.int64)
        f = self.box.flat(idx)
        return self.dist[np.where(q < 0, self.auto.n_q, q), np.where(f < 0, self.box.n, f)]

    def successors(self, q: np.ndarray, idx: np.ndarray):
        """Class successors of states ``q [r]`` / counters ``idx [r, m]``: ``(nq [r, n_k],
        idx [r, n_k, m])`` (``-1`` where invalid)."""
        auto = self.auto
        qq = np.where(q < 0, 0, q)
        nq = np.where(q[:, None] < 0, -1, auto.nq[qq])
        if self.m:
            inc = auto.delta[qq][:, :, self.cols].astype(np.int64)
            new = self.box.move(idx[:, None, :], inc)
        else:
            new = np.zeros(nq.shape + (0,), np.int64)
        return nq, new

    def end_ok(self, q: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """``bool[r]``: an ``END`` from these states is accepted."""
        auto = self.auto
        qq = np.where(q < 0, 0, q)
        new = self.box.move(idx, auto.end_delta[qq][:, self.cols]) if self.m else \
            np.zeros((len(q), 0), np.int64)
        return auto.end_ok[qq] & self.box.accept(new, auto.end_flag[qq]) & (q >= 0)


@dataclass(eq=False)
class Program:
    """Compiled potentials for one (tokenizer, spec, options); built by :func:`compile` (cached);
    start decoding with :meth:`tracker`. ``joint`` is False when the spec was split into parts
    (see :data:`JOINT_LIMIT`)."""

    table: it.InstrTable
    tok: object
    spec: object
    parts: list
    never: np.ndarray          # bool[V]: never sampled
    ws_lead: np.ndarray        # bool[V]: first byte is ASCII whitespace
    calc_ok: np.ndarray        # bool[V]: allowed inside a calculator expression
    tok_len: np.ndarray        # int32[V]: byte length
    token_bytes: list
    ids: dict                  # end, calc, eq, calc_end
    x_mode: np.ndarray = None  # int8[N_XMODES, V]: recogniser mode after each token, per mode
    x_low: np.ndarray = None   # int16[N_XMODES, V]: lowest parenthesis depth reached (relative)
    x_net: np.ndarray = None   # int16[N_XMODES, V]: net parenthesis depth change
    calc_possible: bool = False   # some single token is a complete valid expression
    calc_span: tuple = None    # (expression token, result tokens) of the cheapest span
    structure: bool = True
    reachability: bool = True
    budget_aware: bool = True
    key: str = ""

    @property
    def vocab_size(self) -> int:
        return self.table.vocab_size

    @property
    def joint(self) -> bool:
        return len(self.parts) <= 1

    @property
    def end_id(self) -> int:
        return self.ids["end"]

    def tracker(self, n_particles: int, max_new_tokens: int) -> "Tracker":
        return Tracker(self, n_particles, max_new_tokens)

    def start_distance(self) -> int:
        """Fewest tokens (``END`` included) of an accepted response (max over parts)."""
        return max(int(pt.distance(np.array([pt.auto.q0]), pt.start_index()[None])[0])
                   for pt in self.parts)


_PROGRAMS: OrderedDict[str, Program] = OrderedDict()
_TABLES: dict[str, it.InstrTable] = {}
_AUX: dict[tuple, AuxTables] = {}
MAX_CACHED = 32


def table_for(tok) -> it.InstrTable:
    """Transducer tables of ``tok`` (the committed tokenizer through ``instr_table.load`` and
    its disk cache; any other tokenizer built in memory). Cached by tokenizer hash."""
    key = str(tok.sha256)
    if key not in _TABLES:
        try:
            from bes.tokenizer.tok import load as load_tokenizer

            committed = str(load_tokenizer().sha256) == key
        except Exception:      # no committed tokenizer: build in memory
            committed = False
        _TABLES[key] = it.load(tok) if committed else it.build(tok)
    return _TABLES[key]


def _token_bytes(tok, v: int) -> list[bytes]:
    out = list(tok.all_token_bytes()) if hasattr(tok, "all_token_bytes") else \
        [tok.token_bytes(i) for i in range(v)]
    if len(out) != v:
        raise ValueError("the tokenizer does not list vocab_size byte strings")
    return out


def _aux_tables(tok, tbytes, alphabet, lines: bool, pars: bool, fenced: bool,
                endq: bool, forbidden: tuple[bytes, ...] = ()) -> AuxTables | None:
    parts = []
    if lines or pars or fenced:
        parts.append(LineMachine(lines=lines, paragraphs=pars, fenced=fenced))
    if endq:
        parts.append(EndQuestionMachine())
    if forbidden:
        parts.append(ForbiddenMachine(forbidden))
    if not parts:
        return None
    byte_ok = sorted(set(b for v, tb in enumerate(tbytes) if alphabet[v] for b in tb)
                     | set(it.EQ_BYTES))
    key = (str(tok.sha256), lines, pars, fenced, endq, tuple(forbidden), tuple(byte_ok))
    if key not in _AUX:
        _AUX[key] = aux_token_tables(compile_aux(ProductMachine(parts), byte_ok), tbytes)
    return _AUX[key]


def _part(table, tok, tbytes, alphabet, dims: list, flags: dict, with_flags: bool, max_depth: int,
          trim: bool, structure: bool, build_dp: bool = True, forbidden: tuple[bytes, ...] = ()):
    sources = []
    for d in dims:
        if d.source not in sources:
            sources.append(d.source)
    aux = _aux_tables(tok, tbytes, alphabet, "lines" in sources,
                      any(s in ("pars", "chunks") for s in sources),
                      with_flags and flags["code_block"], with_flags and flags["end_question"],
                      tuple(forbidden) if with_flags else ())
    auto = build_automaton(table, aux, alphabet, sources, max_depth, trim and structure,
                           track_depth=structure)
    if not structure:
        auto.end_ok = np.ones(auto.n_q, bool) if aux is None else aux.flush_accept[auto.st_a].copy()
    box = Box(tuple(dims))
    cols = [sources.index(d.source) for d in dims]
    if not build_dp:
        return auto, box, cols
    return Part(auto, box, cols, distance_table(auto, box, cols))


_CALC_TABLES: dict[str, tuple] = {}


def _calc_tables(tok, tbytes, calc_ok, ws_lead):
    """Recogniser tables of every expression token (``x_mode``, ``x_low``, ``x_net``
    ``[N_XMODES, V]``), whether some single token is a complete valid expression whose result the
    tokenizer can spell, and the cheapest such span ``(expression token, result tokens)``
    (fewest result tokens, then fewest bytes, then the lowest id; ``None`` when there is none).
    That span is minimal in every counted quantity: a calculator expression holds a number, so
    every span adds at least what ``0 = 0`` adds (a word for the result, a word for the
    expression unless it continues the word before it, nothing else). Cached per tokenizer."""
    key = str(tok.sha256)
    if key in _CALC_TABLES:
        return _CALC_TABLES[key]
    v = len(tbytes)
    x_mode = np.full((N_XMODES, v), X_DEAD, np.int8)
    x_low = np.zeros((N_XMODES, v), np.int16)
    x_net = np.zeros((N_XMODES, v), np.int16)
    best = None
    for t in np.flatnonzero(calc_ok):
        for m0 in range(N_XMODES):
            mode, depth, low = m0, 0, 0
            for b in tbytes[t]:
                mode, depth = _calc_step(mode, depth, b, check_depth=False)
                if mode == X_DEAD:
                    break
                low = min(low, depth)
            if mode != X_DEAD:
                x_mode[m0, t], x_low[m0, t], x_net[m0, t] = mode, low, depth
        if (x_mode[X_ATOM, t] in X_COMPLETE and x_net[X_ATOM, t] == 0
                and not ws_lead[t] and tbytes[t][-1] not in WS_BYTES):
            res = calc_mod.evaluate(tbytes[t].decode("ascii"))
            if res is not None:
                try:
                    rids = [int(r) for r in tok.encode(res)]
                except ValueError:
                    continue
                cand = (len(rids), len(tbytes[t]), int(t))
                if best is None or cand < best[0]:
                    best = (cand, (int(t), tuple(rids)))
    span = None if best is None else best[1]
    _CALC_TABLES[key] = (x_mode, x_low, x_net, span is not None, span)
    return _CALC_TABLES[key]


def compile(spec=None, tok=None, *, structure: bool = True, reachability: bool = True,
            budget_aware: bool = True, max_depth: int = 8, trim: bool = True,
            joint_limit: int | None = None) -> Program:
    """Compile the potentials of ``spec`` (a ``ConstraintSpec`` or ``None``) for ``tok``.

    structure: the structural potentials of the module docstring (off: ``END`` needs only the
    spec's counts and flags; special tokens other than ``END`` and control bytes stay masked
    because the automaton cannot follow them); reachability: exact pruning (off: only
    immediate violations are masked, for comparisons); budget_aware: masks depend on the
    remaining token budget; max_depth: bracket depth bound; trim: no leading or trailing
    whitespace; joint_limit: see :data:`JOINT_LIMIT`. Cached by tokenizer hash, spec and
    options.
    """
    if tok is None:
        from bes.tokenizer.tok import load as load_tokenizer

        tok = load_tokenizer()
    table = table_for(tok)
    v = table.vocab_size
    limit = JOINT_LIMIT if joint_limit is None else int(joint_limit)
    dims = dims_for(spec)
    flags = {name: bool(getattr(spec, name, False)) if spec is not None else False
             for name in ("end_question", "no_commas", "code_block")}
    local = _token_local(spec)
    forbidden, case, banned = local["forbidden"], local["case"], local["banned_letters"]
    sig = (str(tok.sha256), tuple(dims), tuple(sorted(flags.items())), forbidden, case, banned,
           structure, reachability, budget_aware, max_depth, trim, limit)
    key = hashlib.sha256(repr(sig).encode()).hexdigest()[:20]
    if key in _PROGRAMS:
        _PROGRAMS.move_to_end(key)
        return _PROGRAMS[key]
    tbytes = _token_bytes(tok, v)
    special = table.kind != it.KIND_ORDINARY
    control = np.array([any((b < 0x20 and b not in (TAB, NL)) or b == 0x7F for b in tb)
                        for tb in tbytes], bool) & ~special
    comma = np.array([b"," in tb for tb in tbytes], bool)
    ws_lead = np.array([bool(tb) and tb[0] in WS_BYTES for tb in tbytes], bool)
    calc_ok = np.array([bool(tb) and all(b in CALC_CHARS for b in tb) for tb in tbytes], bool) \
        & ~special
    ids = {"end": table.special(it.KIND_END), "calc": table.special(it.KIND_CALC),
           "eq": table.special(it.KIND_EQ), "calc_end": table.special(it.KIND_CALC_END)}
    never = np.zeros(v, bool)
    if structure:
        never |= special | control
        never[[ids["end"], ids["calc"], ids["eq"]]] = False
    if flags["no_commas"]:
        never |= comma
    if case is not None:                 # "lower": no ASCII capital; "upper": no ASCII small letter
        lo_b, hi_b = (65, 90) if case == "lower" else (97, 122)
        never |= np.array([any(lo_b <= b <= hi_b for b in tb) for tb in tbytes], bool) & ~special
    for letter in banned:                # a letter that may not occur (either case)
        never |= np.array([any(b in letter for b in tb) for tb in tbytes], bool) & ~special
    alphabet = ~special & ~never & ~control

    args = (table, tok, tbytes, alphabet)
    opts = dict(max_depth=max_depth, trim=trim, structure=structure, forbidden=forbidden)
    auto, box, cols = _part(*args, dims, flags, True, build_dp=False, **opts)
    tx_dims = [d for d in dims if d.source in TX_FIELDS]
    aux_dims = [d for d in dims if d.source not in TX_FIELDS]
    has_aux = bool(aux_dims) or flags["code_block"] or flags["end_question"] or bool(forbidden)
    if auto.n_q * box.n <= limit or not (tx_dims and has_aux):
        parts = [Part(auto, box, cols, distance_table(auto, box, cols))]
    else:
        groups = [(aux_dims, True)]
        tx_auto, tx_box, _ = _part(*args, tx_dims, flags, False, build_dp=False, **opts)
        if tx_auto.n_q * tx_box.n > limit and len(tx_dims) > 1:
            groups += [([d], False) for d in tx_dims]
        else:
            groups.append((tx_dims, False))
        parts = [_part(*args, g, flags, wf, **opts) for g, wf in groups]
    x_mode, x_low, x_net, calc_possible, calc_span = _calc_tables(tok, tbytes, calc_ok, ws_lead)
    prog = Program(table=table, tok=tok, spec=spec, parts=parts, never=never, ws_lead=ws_lead,
                   calc_ok=calc_ok, tok_len=np.array([len(tb) for tb in tbytes], np.int32),
                   token_bytes=tbytes, ids=ids, structure=structure, reachability=reachability,
                   budget_aware=budget_aware, key=key, x_mode=x_mode, x_low=x_low, x_net=x_net,
                   calc_possible=calc_possible, calc_span=calc_span)
    _PROGRAMS[key] = prog
    while len(_PROGRAMS) > MAX_CACHED:
        _PROGRAMS.popitem(last=False)
    return prog


# ================================================================================= tracker
@dataclass
class Advance:
    """What one :meth:`Tracker.advance` call observed, per particle."""

    boundary: np.ndarray       # bool[K]: a sentence was completed (outside calculator spans)
    finished: np.ndarray       # bool[K]: END was consumed
    forced: list               # per particle: tokens the host forces next (after EQ), or None


class Tracker:
    """Exact per-particle state for decoding with a :class:`Program` (or without one).

    Shared by the parts: transducer state ``s`` and bracket depth ``d`` (``int64[K]``); per part
    ``i``: aux state ``a[i]``, product state ``q[i]`` (-1: outside the automaton, so nothing is
    allowed) and remaining counter coordinates ``idx[i] int64[K, m_i]``. Also the calculator
    state ``calc`` (0 outside, 1 in the expression, 2 in the forced result) with the expression
    bytes and ``n`` (tokens consumed). ``allowed(active) -> bool[K, V]``,
    ``advance(tokens, active) -> Advance``, ``select(ancestors)``. ``program=None`` (give
    ``table`` and ``tok``) follows the transducer (sentence boundaries, code / quote flags) and
    the calculator without any potential.
    """

    def __init__(self, program: Program | None, n_particles: int, max_new_tokens: int, *,
                 table: it.InstrTable | None = None, tok=None):
        self.p = program
        self.k = k = int(n_particles)
        self.max_new = int(max_new_tokens)
        self.table = program.table if program is not None else table
        self.tok = program.tok if program is not None else tok
        self.parts: list[Part] = list(program.parts) if program is not None else []
        self.s = np.zeros(k, np.int64)
        self.d = np.zeros(k, np.int64)
        self.a = [np.zeros(k, np.int64) for _ in self.parts]
        self.q = [np.full(k, pt.auto.q0, np.int64) for pt in self.parts]
        self.idx = [np.tile(pt.start_index(), (k, 1)) for pt in self.parts]
        self.calc = np.zeros(k, np.int64)
        self.expr: list[bytes] = [b""] * k
        self.xm = np.zeros(k, np.int64)            # expression recogniser mode and depth
        self.xd = np.zeros(k, np.int64)
        self.n = np.zeros(k, np.int64)
        if self.table is not None:
            self._ids = {name: self.table.special(kd) for name, kd in
                         (("end", it.KIND_END), ("calc", it.KIND_CALC), ("eq", it.KIND_EQ),
                          ("calc_end", it.KIND_CALC_END))}
            self._ordinary = self.table.kind == it.KIND_ORDINARY
            self._bytes = program.token_bytes if program is not None else \
                _token_bytes(tok, self.table.vocab_size) if tok is not None else None
        else:
            self._ids, self._ordinary, self._bytes = {}, None, None

    # -- state management -------------------------------------------------------------
    def copy(self) -> "Tracker":
        """An independent copy of the state (the program and tables are shared)."""
        other = Tracker.__new__(Tracker)
        other.__dict__.update(self.__dict__)
        other.select(np.arange(self.k))
        return other

    def select(self, ancestors) -> None:
        anc = np.asarray(ancestors, np.int64)
        self.k = int(anc.shape[0])
        self.s, self.d, self.calc, self.n, self.xm, self.xd = (
            x[anc].copy() for x in (self.s, self.d, self.calc, self.n, self.xm, self.xd))
        self.a = [x[anc].copy() for x in self.a]
        self.q = [x[anc].copy() for x in self.q]
        self.idx = [x[anc].copy() for x in self.idx]
        self.expr = [self.expr[i] for i in anc]

    def remaining(self) -> np.ndarray:
        """``int64[K]``: tokens each particle may still emit (``END`` included)."""
        return self.max_new - self.n

    def exempt(self) -> np.ndarray:
        """``bool[K]``: inside a code fence, inline code or a double quote (the registers)."""
        if self.table is None:
            return np.zeros(self.k, bool)
        comp = self.table.components[self.s]
        return (comp[:, _C_FENCE] > 0) | (comp[:, _C_INLINE] > 0) | (comp[:, _C_QUOTE] > 0)

    def distance(self, rows: np.ndarray | None = None) -> np.ndarray:
        """``int64[r]``: fewest tokens to an accepted ``END`` from the current states (max over
        parts; ``INF`` = none)."""
        rows = np.arange(self.k) if rows is None else np.asarray(rows)
        out = np.zeros(len(rows), np.int64)
        for i, pt in enumerate(self.parts):
            out = np.maximum(out, pt.distance(self.q[i][rows], self.idx[i][rows]))
        return out

    # -- transitions ------------------------------------------------------------------
    def _apply_rows(self, rows: np.ndarray, j: np.ndarray, s, d, a, idx) -> np.ndarray:
        """Apply token rows (token ids, or the EQ row ``V``) to particles ``j`` in place of the
        given arrays; returns the transducer increments ``int64[len(j), N_INC]``."""
        tab = self.table
        s0 = s[j]
        inc = tab.inc[rows, s0].astype(np.int64)
        s1 = tab.next_state[rows, s0].astype(np.int64)
        d1 = np.maximum(inc[:, it.INC_DEP_LO], d[j] + inc[:, it.INC_DEP_ADD])
        if self.p is not None and not self.p.structure:
            d1 = np.zeros_like(d1)
        for i, pt in enumerate(self.parts):
            aux = pt.auto.aux
            ev = None
            if aux is not None:
                ev = aux.ev_tok[rows, a[i][j]].astype(np.int64)
                a[i][j] = aux.next_tok[rows, a[i][j]]
            if pt.m:
                incs = np.zeros((len(j), pt.m), np.int64)
                for c, dim in enumerate(pt.box.dims):
                    incs[:, c] = inc[:, TX_FIELDS[dim.source]] if dim.source in TX_FIELDS else \
                        ev[:, aux.machine.events.index(dim.source)]
                idx[i][j] = pt.box.move(idx[i][j], incs)
        s[j], d[j] = s1, d1
        return inc

    def _qof(self, i: int, s, d, a, idx) -> np.ndarray:
        auto = self.parts[i].auto
        dead = (d > auto.max_depth) | (d < 0)
        if auto.aux is not None:
            dead |= a == auto.aux.dead
        q = auto.qmap[s, np.clip(d, 0, auto.max_depth), np.where(dead, 0, a)].astype(np.int64)
        if idx.shape[1]:
            q = np.where((idx < 0).any(axis=1), -1, q)
        return np.where(dead, -1, q)

    def forced_after_eq(self, j: int) -> list[int]:
        """Tokens the host forces after ``EQ`` for particle ``j``: ``encode(result) + [CALC_END]``,
        or ``[CALC_END]`` alone when the expression does not evaluate (fallback)."""
        result = calc_mod.evaluate(self.expr[j].decode("utf-8", errors="replace"))
        tail = [self._ids["calc_end"]]
        return tail if result is None else [int(t) for t in self.tok.encode(result)] + tail

    def simulate(self, j: int, rows: Sequence[int]):
        """Per-part ``(q, idx)`` of particle ``j`` after the given rows (ordinary tokens or the
        EQ row; other special ids leave the state unchanged), without changing the tracker."""
        s, d = self.s.copy(), self.d.copy()
        a = [x.copy() for x in self.a]
        idx = [x.copy() for x in self.idx]
        jj = np.array([j])
        v = self.table.vocab_size
        for r in rows:
            if r < v and not self._ordinary[r]:
                continue
            self._apply_rows(np.array([r]), jj, s, d, a, idx)
        return [(int(self._qof(i, s[jj], d[jj], a[i][jj], idx[i][jj])[0]), idx[i][j].copy())
                for i in range(len(self.parts))]

    # -- masks ------------------------------------------------------------------------
    def allowed(self, active: np.ndarray | None = None) -> np.ndarray | None:
        """``bool[K, V]`` potential mask of the next token of every active particle outside a
        forced span (other rows all False); ``None`` without a program (no potentials)."""
        p = self.p
        if p is None:
            return None
        v = self.table.vocab_size
        act = np.ones(self.k, bool) if active is None else np.asarray(active, bool)
        out = np.zeros((self.k, v), bool)
        rows = np.flatnonzero(act & (self.calc != 2))
        if rows.size == 0:
            return out
        rem = self.remaining()[rows]
        tok_ok = np.broadcast_to(~p.never, (len(rows), v)).copy()
        end_ok = rem >= 1
        for i, pt in enumerate(self.parts):
            nq, new = pt.successors(self.q[i][rows], self.idx[i][rows])
            if not p.reachability:
                ok_k = (nq >= 0) & ((new >= 0).all(axis=-1) if pt.m else True)
            else:
                dist = pt.distance(nq, new).astype(np.int64)
                ok_k = dist <= ((rem - 1)[:, None] if p.budget_aware else INF - 1)
            ok_k = np.concatenate([ok_k, np.zeros((len(rows), 1), bool)], axis=1)
            cls = np.where(pt.auto.cls < 0, pt.auto.n_k, pt.auto.cls)
            tok_ok &= ok_k[:, cls]
            end_ok = end_ok & pt.end_ok(self.q[i][rows], self.idx[i][rows])
        if p.structure:
            tok_ok[self.n[rows] == 0] &= ~p.ws_lead
        outside = self.calc[rows] == 0
        calc_start = np.zeros(len(rows), bool)
        if outside.any():
            calc_start[outside] = self._calc_span_ok(rows[outside], rem[outside])
        ids = self._ids
        for r, j in enumerate(rows):
            row = tok_ok[r]
            row[ids["calc_end"]] = False
            if p.structure and self.calc[j] == 1:
                # inside an expression: viable prefixes only, no leading space, length cap and
                # room for closing parentheses, EQ, one result token and CALC_END
                xm, xd = int(self.xm[j]), int(self.xd[j])
                row &= p.calc_ok & (p.x_mode[xm] != X_DEAD) & (xd + p.x_low[xm] >= 0)
                if not self.expr[j]:
                    row &= ~p.ws_lead
                row &= p.tok_len <= calc_mod.MAX_EXPR_CHARS - len(self.expr[j])
                if p.budget_aware:
                    row &= (xd + p.x_net[xm] + 3 <= rem[r] - 2)
                row[ids["end"]] = row[ids["calc"]] = False
                row[ids["eq"]] = self._eq_ok(int(j), int(rem[r]))
                continue
            row[ids["eq"]] = False
            row[ids["calc"]] = bool(calc_start[r])
            row[ids["end"]] = bool(end_ok[r]) and self.calc[j] == 0
        out[rows] = tok_ok
        return out

    def _calc_span_ok(self, rows: np.ndarray, rem: np.ndarray) -> np.ndarray:
        """``bool[len(rows)]``: ``CALC`` may start for these particles (outside a span): after the
        cheapest span (``Program.calc_span``: ``CALC``, its expression token, ``EQ``, the forced
        result, ``CALC_END``) every part can still reach an accepted ``END`` (with
        ``budget_aware``: within the budget left after the span). No span adds less to any
        count than that one, so a particle that takes ``CALC`` here is never stranded by the
        counts; where even the cheapest span cannot fit (e.g. under an upper word bound),
        ``CALC`` is masked."""
        p = self.p
        if not (p.structure and p.calc_possible and p.calc_span is not None) or not len(rows):
            return np.zeros(len(rows), bool)
        expr, result = p.calc_span
        seq = [int(expr), int(self.table.eq_row)] + [int(t) for t in result]
        jj = np.asarray(rows, np.int64)
        s, d = self.s.copy(), self.d.copy()
        a = [x.copy() for x in self.a]
        idx = [x.copy() for x in self.idx]
        for t in seq:
            self._apply_rows(np.full(len(jj), t, np.int64), jj, s, d, a, idx)
        worst = np.zeros(len(jj), np.int64)
        for i, pt in enumerate(self.parts):
            q = self._qof(i, s[jj], d[jj], a[i][jj], idx[i][jj])
            worst = np.maximum(worst, pt.distance(q, idx[i][jj]).astype(np.int64))
        if not p.budget_aware:
            return worst < INF
        span = len(seq) + 1                     # expression, EQ, result tokens, CALC_END
        return worst <= np.asarray(rem, np.int64) - 1 - span

    def _eq_ok(self, j: int, rem: int) -> bool:
        e = self.expr[j]
        if not e or e[-1] in WS_BYTES or e[0] in WS_BYTES:
            return False
        if self.xm[j] not in X_COMPLETE or self.xd[j] != 0:
            return False
        if calc_mod.evaluate(e.decode("utf-8", errors="replace")) is None:
            return False
        forced = self.forced_after_eq(j)
        budget = rem - 1 - len(forced)
        if budget < 1:
            return False
        worst = 0
        for i, (q2, idx2) in enumerate(self.simulate(j, [self.table.eq_row] + forced)):
            if q2 < 0:
                return False
            worst = max(worst, int(self.parts[i].distance(np.array([q2]), idx2[None])[0]))
        return worst <= budget if self.p.budget_aware else worst < INF

    # -- advancing --------------------------------------------------------------------
    def advance(self, tokens: np.ndarray, active: np.ndarray | None = None) -> Advance:
        """Consume ``tokens[j]`` for every active particle ``j``."""
        tok = np.asarray(tokens, np.int64).reshape(-1)
        act = np.ones(self.k, bool) if active is None else np.asarray(active, bool)
        boundary = np.zeros(self.k, bool)
        finished = np.zeros(self.k, bool)
        forced: list = [None] * self.k
        jj = np.flatnonzero(act)
        if self.table is None or jj.size == 0:
            self.n[jj] += 1
            return Advance(boundary, finished, forced)
        ids, tab = self._ids, self.table
        t = tok[jj]
        calc_before = self.calc[jj].copy()
        pend_before = tab.components[self.s[jj], _C_PEND] == PEND_OPEN
        ordinary = self._ordinary[t]
        is_eq = (t == ids["eq"]) & (calc_before == 1)
        rows = np.where(is_eq, tab.eq_row, t)
        moved = ordinary | is_eq
        sent = np.zeros(len(jj), np.int64)
        if moved.any():
            inc = self._apply_rows(rows[moved], jj[moved], self.s, self.d, self.a, self.idx)
            sent[moved] = inc[:, it.INC_SENT]
        is_end = t == ids["end"]
        if is_end.any():
            je = jj[is_end]
            sent[is_end] = tab.components[self.s[je], _C_PEND] == PEND_OPEN
            for i, pt in enumerate(self.parts):
                if pt.m:
                    qq = np.where(self.q[i][je] < 0, 0, self.q[i][je])
                    self.idx[i][je] = pt.box.move(self.idx[i][je], pt.auto.end_delta[qq][:, pt.cols])
            self.s[je] = tab.next_state[ids["end"], self.s[je]]
            finished[je] = True
        pend_after = tab.components[self.s[jj], _C_PEND] == PEND_OPEN
        calc_after = np.where(t == ids["calc"], 1, np.where(t == ids["eq"], np.where(
            calc_before >= 1, 2, 0), np.where(ordinary, calc_before, 0)))
        boundary[jj] = (((pend_after & (~pend_before | (sent >= 1)))
                         | (sent >= 1 + pend_before.astype(np.int64))) & (calc_after == 0)) | is_end
        for r, j in enumerate(jj):
            if t[r] == ids["calc"]:
                self.expr[j] = b""
                self.xm[j], self.xd[j] = X_ATOM, 0
            elif calc_before[r] == 1 and ordinary[r] and self._bytes is not None:
                self.expr[j] = self.expr[j] + self._bytes[t[r]]
                xm, xd = int(self.xm[j]), int(self.xd[j])
                for b in self._bytes[t[r]]:
                    xm, xd = _calc_step(xm, xd, b) if xm != X_DEAD else (X_DEAD, xd)
                self.xm[j], self.xd[j] = xm, xd
            if is_eq[r] and self.tok is not None:
                forced[j] = self.forced_after_eq(int(j))
        self.calc[jj] = calc_after
        self.n[jj] += 1
        for i in range(len(self.parts)):
            self.q[i][jj] = self._qof(i, self.s[jj], self.d[jj], self.a[i][jj], self.idx[i][jj])
        return Advance(boundary, finished, forced)

    def counts(self, j: int) -> dict[str, int]:
        """Counted value of every constrained source of particle ``j`` (from the remaining
        coordinates; a met lower bound reports the bound, a paragraph "over" reports hi + 1)."""
        out = {}
        for i, pt in enumerate(self.parts):
            for c, dim in enumerate(pt.box.dims):
                ix = int(self.idx[i][j, c])
                out[dim.source] = dim.start - ix if (ix > 0 or not dim.saturate or
                                                      dim.kind == "at_least") else dim.start
        return out


# =================================================================================== twist
def fit_unigram(samples: Sequence[Sequence[int]], vocab_size: int, smoothing: float = 0.5,
                end_id: int | None = None) -> np.ndarray:
    """``float64[V]`` log-probabilities of a unigram proxy fitted to token sequences (counts
    plus ``smoothing`` per id; a sample that does not end with ``end_id`` adds one, so the
    proxy knows how often responses end)."""
    counts = np.full(vocab_size, float(smoothing))
    for smp in samples:
        arr = np.asarray(smp, np.int64)
        if arr.size:
            np.add.at(counts, arr, 1.0)
        if end_id is not None and (arr.size == 0 or arr[-1] != end_id):
            counts[end_id] += 1.0
    return np.log(counts / counts.sum())


def _psi_table(pt: Part, pk: np.ndarray, p_end: float, horizon: int) -> np.ndarray:
    """``float32[horizon + 1, n_q + 1, n_box + 1]`` log psi of one part (``-inf`` unreachable)."""
    auto, box = pt.auto, pt.box
    n_q, n_box = auto.n_q, box.n
    src, dst, did, uvec, did_full = _edges(auto, pt.cols)
    valid = auto.nq >= 0
    q = np.broadcast_to(np.arange(n_q)[:, None], auto.nq.shape)
    n_d = len(uvec)
    key = (q[valid].astype(np.int64) * n_q + auto.nq[valid]) * n_d + did_full[valid]
    ekey = (src * n_q + dst) * n_d + did
    w = np.zeros(len(ekey))
    np.add.at(w, np.searchsorted(ekey, key), np.broadcast_to(pk[None, :], auto.nq.shape)[valid])
    succ = np.stack([box.flat(box.move(box.coords, uvec[i][None, :])) for i in range(n_d)]) \
        if pt.cols else np.zeros((n_d, 1), np.int64)
    succ = np.where(succ < 0, n_box, succ)
    acc = _end_accept(auto, box, pt.cols, np.arange(n_box)).astype(np.float64)
    starts, seg_src = _segments(src, n_q)
    psi = np.zeros((n_q + 1, n_box + 1))
    out = np.full((horizon + 1, n_q + 1, n_box + 1), -np.inf, np.float32)
    tiny = np.finfo(np.float64).tiny
    dist = pt.dist.astype(np.int64)
    for b in range(1, horizon + 1):
        new = np.zeros_like(psi)
        new[:n_q, :n_box] = p_end * acc
        if src.size:
            contrib = w[:, None] * psi[dst[:, None], succ[did]]
            new[seg_src, :n_box] += np.add.reduceat(contrib, starts, axis=0)
        psi = new
        out[b] = np.where(dist <= b, np.log(np.maximum(psi, tiny)), -np.inf).astype(np.float32)
    return out


#: log psi given to a state whose acceptance needs more tokens than the twist's horizon
#: (minus its distance): positive, so the filter stays exact.
TWIST_FLOOR = -50.0


class Twist:
    """Look-ahead twist ``psi_b`` under a unigram proxy (module docstring), one table per part
    (the twist of a split program is the product of the parts' twists).

    proxy_logp: ``float[V]`` log-probabilities; horizon: largest budget ``b`` tabulated (larger
    budgets read ``b = horizon``, and a state that needs more than ``horizon`` tokens gets
    ``log psi = TWIST_FLOOR - dist``; any positive function keeps the filter exact). Use it
    with a budget-aware program with reachability (``Twist`` is zero exactly where those masks
    are), which :func:`bes.decode.smc.decode` checks.
    """

    def __init__(self, program: Program, proxy_logp: np.ndarray, horizon: int):
        self.p = program
        self.horizon = int(horizon)
        lp = np.asarray(proxy_logp, np.float64)
        pr = np.exp(lp - lp.max())
        pr = pr / pr.sum()
        p_end = float(pr[program.ids["end"]])
        self.tables = []
        for pt in program.parts:
            cls = pt.auto.cls
            pk = np.zeros(pt.auto.n_k)
            ok = cls >= 0
            np.add.at(pk, cls[ok], pr[ok])
            self.tables.append(_psi_table(pt, pk, p_end, self.horizon))

    def _lookup(self, i: int, q, idx, budget) -> np.ndarray:
        pt = self.p.parts[i]
        q = np.asarray(q, np.int64)
        f = pt.box.flat(idx)
        qq, ff = np.where(q < 0, pt.auto.n_q, q), np.where(f < 0, pt.box.n, f)
        budget = np.asarray(budget, np.int64)
        b = np.clip(budget, 0, self.horizon)
        val = self.tables[i][b, qq, ff].astype(np.float64)
        # beyond the horizon a state that needs more than ``horizon`` tokens still gets a
        # positive value (the twist must be positive wherever the potentials allow a token)
        dist = pt.dist[qq, ff].astype(np.int64)
        far = ~np.isfinite(val) & (dist <= budget)
        return np.where(far, TWIST_FLOOR - dist, val)

    def value(self, tracker: Tracker, rows: np.ndarray, budget: np.ndarray) -> np.ndarray:
        """``float64[len(rows)]``: log psi of the current states with the given budgets."""
        out = np.zeros(len(rows))
        for i in range(len(self.p.parts)):
            out += self._lookup(i, tracker.q[i][rows], tracker.idx[i][rows], budget)
        return out

    def state_value(self, states: list, budget: int) -> float:
        """log psi of the per-part states returned by :meth:`Tracker.simulate`."""
        return float(sum(self._lookup(i, np.array([q]), idx[None], np.array([budget]))[0]
                         for i, (q, idx) in enumerate(states)))

    def token_values(self, tracker: Tracker, rows: np.ndarray, budget: np.ndarray) -> np.ndarray:
        """``float64[len(rows), V]``: log psi (budget ``budget``) of the successor of every
        ordinary token (``-inf`` for tokens outside the automaton)."""
        v = tracker.table.vocab_size
        out = np.zeros((len(rows), v))
        for i, pt in enumerate(self.p.parts):
            nq, new = pt.successors(tracker.q[i][rows], tracker.idx[i][rows])
            vals = self._lookup(i, nq, new, np.asarray(budget)[:, None])
            vals = np.concatenate([vals, np.full((len(rows), 1), -np.inf)], axis=1)
            cls = np.where(pt.auto.cls < 0, pt.auto.n_k, pt.auto.cls)
            out += vals[:, cls]
        return out


__all__ = ["compile", "Program", "Part", "Tracker", "Advance", "Twist", "fit_unigram",
           "enforced", "dims_for", "Dim", "Box", "LineMachine", "EndQuestionMachine", "ForbiddenMachine",
           "ProductMachine", "run_machine", "compile_aux", "aux_token_tables", "build_automaton",
           "minimize", "distance_table", "table_for", "INF", "JOINT_LIMIT"]
