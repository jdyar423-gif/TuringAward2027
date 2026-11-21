"""Byte-level formatting transducer: the reference definition (``docs/SPEC.md`` section 2.1).

One finite-state machine over bytes defines all formatting state and every count of the exact
layer. :func:`step` is the slow, readable reference; :func:`compile_machine` enumerates its
reachable states into per-byte tables, and ``bes.exact.instr_table`` composes those into
per-token tables. Text-level code (constraints, verifiers) uses :func:`measure_counts`.

Byte classes are ASCII; every byte ``>= 0x80`` counts as alphanumeric.

State (all components small; :class:`State`):

- ``fence``: inside a ``` code fence.
- ``inline``: inside `inline code`. Updated eagerly: the first backtick of a run flips it, the
  second backtick of the same run flips it back, later backticks of the run do nothing. After a
  run of exactly one backtick it is therefore flipped, after any longer run it is unchanged:
  "inline parity on single backticks outside a fence".
- ``quote``: inside a "double quote".
- ``ticks``: length of the backtick run that ends at the previous byte, ``0..3`` (``3`` means
  three or more; the run is then absorbing until the next non-backtick byte).
- ``prev``: class of the previous byte: whitespace or turn start / other / alphanumeric /
  apostrophe directly after an alphanumeric byte.
- ``pend``: sentence-terminator run: none / open (eligible run, waiting for the byte that
  decides it) / dead (run that followed whitespace, never counted).
- ``line``: line-start parser: start (only spaces or tabs since the line began) / ticks
  (backtick run that began the line) / digits / marker seen / first or second byte of
  ``•`` (``E2 80 A2``) / body.

Counted events, each counted at one byte:

- **word**: an alphanumeric byte whose predecessor is neither alphanumeric nor an apostrophe that
  itself follows an alphanumeric byte (so ``don't`` is one word and ``'yes'`` still counts).
  Words are counted everywhere, code included.
- **sentence**: a run of ``.``/``!``/``?`` whose preceding byte exists and is not whitespace,
  counted at the first byte after the run that is not ``*`` or ``_`` (markdown emphasis, so
  ``**Done.** Next`` counts) when that byte is whitespace or one of ``" ' ) ] }``. A run still
  open at the end of the turn is counted by :func:`flush` (``END``, or the end of a measured
  text). Never inside a code fence. The ``.`` of a numbered-list marker (``12.`` right after
  the leading spaces of a line) does not start a run.
- **line**: every ``\\n`` (inside fences too).
- **list item**: a line that starts, after optional spaces or tabs, with ``-``, ``*``, ``+``,
  ``•`` or one or more digits followed by ``.`` or ``)``, then a space or a tab; counted at
  that space or tab. Never inside a code fence.
- **bracket**: ``([{`` +1 and ``)]}`` -1, outside code (fence or inline); the ``)`` of a
  ``12)`` list marker does not count. The depth is floored at 0 (kept outside the finite state,
  by the caller).

Toggles: ``"`` outside code flips ``quote``; a backtick run outside a fence flips ``inline`` as
described above; the third backtick of a run that began the line (after optional spaces or
tabs) flips ``fence`` and clears ``inline`` and ``quote``.

Special tokens are handled by the token layer: ``BOS`` and role tokens restart from
:data:`INIT`, ``END`` applies :func:`flush`, ``EQ`` inside a calculator span reads as the bytes
``" = "`` (the display form ``expr = result``), other special tokens contribute no bytes.
"""
from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

# ------------------------------------------------------------------------------ byte classes
NEWLINE = 0x0A
BACKTICK = 0x60
DQUOTE = 0x22
APOSTROPHE = 0x27
DOT = 0x2E
RPAREN = 0x29
BULLET = "•".encode("utf-8")  # b"\xe2\x80\xa2"

WHITESPACE = frozenset(b" \t\n\r\x0b\x0c")
INDENT = frozenset(b" \t")
TERMINATORS = frozenset(b".!?")
EMPHASIS = frozenset(b"*_")
CLOSERS = frozenset(b"\"')]}")
OPEN_BRACKETS = frozenset(b"([{")
CLOSE_BRACKETS = frozenset(b")]}")
LIST_MARKS = frozenset(b"-*+")
DIGITS = frozenset(b"0123456789")
NUMBER_MARK_ENDS = frozenset(b".)")


def is_alnum(b: int) -> bool:
    """ASCII letter or digit, or any non-ASCII byte."""
    return b >= 0x80 or 0x30 <= b <= 0x39 or 0x41 <= b <= 0x5A or 0x61 <= b <= 0x7A


# ---------------------------------------------------------------------- state components
PREV_SPACE, PREV_OTHER, PREV_ALNUM, PREV_APOS = 0, 1, 2, 3
PEND_NONE, PEND_OPEN, PEND_DEAD = 0, 1, 2
LINE_START, LINE_TICKS, LINE_DIGITS, LINE_MARK, LINE_BUL1, LINE_BUL2, LINE_BODY = range(7)

COMPONENTS = ("fence", "inline", "quote", "ticks", "prev", "pend", "line")


class State(NamedTuple):
    """Finite state of the transducer (see the module docstring for each component)."""

    fence: int
    inline: int
    quote: int
    ticks: int
    prev: int
    pend: int
    line: int


class Events(NamedTuple):
    """What one byte (or a flush) counts: 0/1 flags and a bracket delta in ``{-1, 0, 1}``."""

    word: int
    sentence: int
    newline: int
    item: int
    bracket: int


EVENT_FIELDS = Events._fields
INIT = State(fence=0, inline=0, quote=0, ticks=0, prev=PREV_SPACE, pend=PEND_NONE, line=LINE_START)
NO_EVENTS = Events(0, 0, 0, 0, 0)


def step(state: State, byte: int) -> tuple[State, Events]:
    """Consume one byte: ``(state after the byte, events counted at the byte)``."""
    fence, inline, quote, ticks, prev, pend, line = state
    b = int(byte)

    # Sentence terminators. An open run is decided by the first byte after it that is neither a
    # terminator nor an emphasis byte; a dead run ends at the first non-terminator.
    sentence = 0
    if pend == PEND_OPEN:
        if b in WHITESPACE or b in CLOSERS:
            sentence, pend = 1, PEND_NONE
        elif b not in TERMINATORS and b not in EMPHASIS:
            pend = PEND_NONE
    elif pend == PEND_DEAD and b not in TERMINATORS:
        pend = PEND_NONE
    if (pend == PEND_NONE and b in TERMINATORS and not fence
            and not (line == LINE_DIGITS and b == DOT)):
        pend = PEND_DEAD if prev == PREV_SPACE else PEND_OPEN

    # Words.
    alnum = is_alnum(b)
    word = int(alnum and prev != PREV_ALNUM and prev != PREV_APOS)
    if alnum:
        prev = PREV_ALNUM
    elif b == APOSTROPHE and prev == PREV_ALNUM:
        prev = PREV_APOS
    elif b in WHITESPACE:
        prev = PREV_SPACE
    else:
        prev = PREV_OTHER

    # Brackets and double quotes, outside code (fence and inline as they were before this byte).
    bracket = 0
    if not fence and not inline:
        if b in OPEN_BRACKETS:
            bracket = 1
        elif b in CLOSE_BRACKETS and not (line == LINE_DIGITS and b == RPAREN):
            bracket = -1
        elif b == DQUOTE:
            quote ^= 1

    # Backtick runs (inline code, fences) and the line-start parser (list items).
    item = 0
    if b == BACKTICK:
        run = min(ticks + 1, 3)
        if run == 3 and line == LINE_TICKS:     # third backtick of a run that began the line
            fence ^= 1
            inline = quote = 0
            line = LINE_BODY
        else:
            if not fence and run < 3:
                inline ^= 1
            line = LINE_TICKS if line in (LINE_START, LINE_TICKS) else LINE_BODY
        ticks = run
    else:
        ticks = 0
        if b == NEWLINE:
            line = LINE_START
        elif line == LINE_START:
            if b in INDENT:
                pass
            elif fence:
                line = LINE_BODY
            elif b in LIST_MARKS:
                line = LINE_MARK
            elif b in DIGITS:
                line = LINE_DIGITS
            elif b == BULLET[0]:
                line = LINE_BUL1
            else:
                line = LINE_BODY
        elif line == LINE_DIGITS:
            if b in DIGITS:
                pass
            elif b in NUMBER_MARK_ENDS:
                line = LINE_MARK
            else:
                line = LINE_BODY
        elif line == LINE_BUL1:
            line = LINE_BUL2 if b == BULLET[1] else LINE_BODY
        elif line == LINE_BUL2:
            line = LINE_MARK if b == BULLET[2] else LINE_BODY
        elif line == LINE_MARK:
            item = int(b in INDENT)
            line = LINE_BODY
        else:
            line = LINE_BODY

    return (State(fence, inline, quote, ticks, prev, pend, line),
            Events(word, sentence, int(b == NEWLINE), item, bracket))


def flush(state: State) -> tuple[State, Events]:
    """End of the turn (``END``, or the end of a measured text): count an open terminator run."""
    return state._replace(pend=PEND_NONE), Events(0, int(state.pend == PEND_OPEN), 0, 0, 0)


def trace(data: bytes, state: State = INIT) -> list[tuple[State, Events]]:
    """``(state, events)`` after every byte of ``data``, starting from ``state``."""
    out = []
    for b in data:
        state, ev = step(state, b)
        out.append((state, ev))
    return out


class Counts(NamedTuple):
    """Totals of a run from a fresh turn: final state, counts and bracket depth."""

    state: State
    words: int
    sentences: int
    lines: int
    items: int
    word_in_sent: int   # words since the last counted sentence
    depth: int          # bracket depth, floored at 0 after every byte


def run(data: bytes, state: State = INIT, final: bool = False) -> Counts:
    """Run ``data`` from ``state`` with all counts starting at zero; ``final`` flushes at the end."""
    words = sentences = lines = items = wis = depth = 0
    for b in data:
        state, ev = step(state, b)
        if ev.sentence:
            sentences += 1
            wis = 0
        if ev.word:
            words += 1
            wis += 1
        lines += ev.newline
        items += ev.item
        depth = max(0, depth + ev.bracket)
    if final:
        state, ev = flush(state)
        if ev.sentence:
            sentences += 1
            wis = 0
    return Counts(state, words, sentences, lines, items, wis, depth)


def measure_counts(text: str | bytes, final: bool = True) -> dict[str, int]:
    """Counts and final formatting state of one complete text (a response, a message).

    The text is read as one turn from its start; with ``final`` (default) the end of the text
    is the end of the turn, so a trailing ``"Done."`` counts as a sentence. Returns ints:
    ``words, sentences, lines`` (number of ``\\n``), ``items`` (list items), ``word_in_sent``
    (words after the last sentence), ``depth`` (open brackets), ``fence, inline, quote``
    (0/1: still inside a code fence / inline code / double quote at the end) and
    ``pending_sentence`` (0/1: a terminator run still undecided; always 0 with ``final``).
    """
    data = text.encode("utf-8") if isinstance(text, str) else bytes(text)
    c = run(data, INIT, final=final)
    return {
        "words": c.words, "sentences": c.sentences, "lines": c.lines, "items": c.items,
        "word_in_sent": c.word_in_sent, "depth": c.depth, "fence": c.state.fence,
        "inline": c.state.inline, "quote": c.state.quote,
        "pending_sentence": int(c.state.pend == PEND_OPEN),
    }


def measure_counts_batch(texts, final: bool = True, chunk: int = 512) -> list[dict[str, int]]:
    """``[measure_counts(t, final) for t in texts]`` computed with the compiled per-byte tables,
    vectorised over texts (texts of similar length are processed together, ``chunk`` at a time).
    Use it to measure many responses at once (data relabelling); results are identical."""
    m = compile_machine()
    data = [t.encode("utf-8") if isinstance(t, str) else bytes(t) for t in texts]
    lengths = np.array([len(d) for d in data], np.int64)
    out: list[dict[str, int] | None] = [None] * len(data)
    nxt, evs = m.next_state.astype(np.intp), m.events.astype(np.int64)
    comp = m.components.astype(np.int64)
    order = np.argsort(lengths, kind="stable")
    for lo in range(0, len(data), chunk):
        idx = order[lo:lo + chunk]
        n, width = len(idx), int(lengths[idx].max()) if len(idx) else 0
        mat = np.zeros((n, max(width, 1)), np.uint8)
        for row, i in enumerate(idx):
            mat[row, : lengths[i]] = np.frombuffer(data[i], np.uint8)
        s = np.zeros(n, np.intp)
        tot = np.zeros((n, 4), np.int64)          # words, sentences, lines, items
        wis = np.zeros(n, np.int64)
        depth = np.zeros(n, np.int64)
        for j in range(width):
            live = lengths[idx] > j
            b = mat[:, j].astype(np.intp)
            ev = evs[s, b] * live[:, None]
            tot += ev[:, :4]
            wis = np.where(ev[:, 1] > 0, 0, wis) + ev[:, 0]
            depth = np.where(live, np.maximum(0, depth + ev[:, 4]), depth)
            s = np.where(live, nxt[s, b], s)
        pending = comp[s, COMPONENTS.index("pend")] == PEND_OPEN
        if final:
            tot[:, 1] += pending
            wis = np.where(pending, 0, wis)
            s = m.flush_state[s].astype(np.intp)
            pending = np.zeros(n, bool)
        for row, i in enumerate(idx):
            c = comp[s[row]]
            out[i] = {"words": int(tot[row, 0]), "sentences": int(tot[row, 1]),
                      "lines": int(tot[row, 2]), "items": int(tot[row, 3]),
                      "word_in_sent": int(wis[row]), "depth": int(depth[row]),
                      "fence": int(c[0]), "inline": int(c[1]), "quote": int(c[2]),
                      "pending_sentence": int(pending[row])}
    return out  # type: ignore[return-value]


# ------------------------------------------------------------------------- compiled form
@dataclass(frozen=True, eq=False)
class ByteMachine:
    """The reachable states of :func:`step` and its per-byte tables (state 0 is :data:`INIT`).

    - ``states``: the ``S`` reachable states in discovery order.
    - ``next_state``: ``int16[S, 256]``; ``events``: ``int8[S, 256, 5]`` in ``EVENT_FIELDS``
      order (word, sentence, newline, item, bracket).
    - ``flush_state``: ``int16[S]``; ``flush_sentence``: ``int8[S]``.
    - ``components``: ``int8[S, 7]``, the components of every state in ``COMPONENTS`` order.
    - ``fingerprint``: SHA-256 of all tables; it changes whenever the definition changes.
    """

    states: tuple[State, ...]
    next_state: np.ndarray
    events: np.ndarray
    flush_state: np.ndarray
    flush_sentence: np.ndarray
    components: np.ndarray
    fingerprint: str

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def init(self) -> int:
        return 0

    def index(self, state: State) -> int:
        return self._index()[state]

    def _index(self) -> dict[State, int]:
        cached = self.__dict__.get("_index_cache")
        if cached is None:
            cached = {s: i for i, s in enumerate(self.states)}
            object.__setattr__(self, "_index_cache", cached)
        return cached


@functools.lru_cache(maxsize=1)
def compile_machine() -> ByteMachine:
    """Enumerate the states reachable from :data:`INIT` by bytes and flushes (breadth first,
    bytes in increasing order, then the flush) and tabulate :func:`step` and :func:`flush`."""
    states: list[State] = [INIT]
    index: dict[State, int] = {INIT: 0}
    pos = 0
    while pos < len(states):
        s = states[pos]
        pos += 1
        successors = [step(s, b)[0] for b in range(256)] + [flush(s)[0]]
        for s2 in successors:
            if s2 not in index:
                index[s2] = len(states)
                states.append(s2)
    n = len(states)
    next_state = np.empty((n, 256), np.int16)
    events = np.empty((n, 256, len(EVENT_FIELDS)), np.int8)
    flush_state = np.empty(n, np.int16)
    flush_sentence = np.empty(n, np.int8)
    for i, s in enumerate(states):
        for b in range(256):
            s2, ev = step(s, b)
            next_state[i, b] = index[s2]
            events[i, b] = ev
        s2, ev = flush(s)
        flush_state[i] = index[s2]
        flush_sentence[i] = ev.sentence
    components = np.array(states, dtype=np.int8).reshape(n, len(COMPONENTS))
    digest = hashlib.sha256()
    for arr in (next_state, events, flush_state, flush_sentence, components):
        digest.update(np.ascontiguousarray(arr).astype(arr.dtype.newbyteorder("<")).tobytes())
    return ByteMachine(tuple(states), next_state, events, flush_state, flush_sentence,
                       components, digest.hexdigest())
