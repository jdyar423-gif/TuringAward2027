"""Documents and conversations as token ids (``docs/SPEC.md`` section 1).

A segment is one document or one conversation and always starts with ``BOS``:

- raw document: ``BOS text``;
- conversation: ``BOS`` then turns ``ROLE content END`` with ``ROLE`` one of ``SYSTEM, USER,
  ASSISTANT, LIBRARY``. Turn order: an optional ``system`` turn first, then exchanges
  ``[library] user assistant``; a ``library`` turn (retrieved passage, title first) comes
  immediately before the ``user`` turn it serves. ``add_generation_prompt=True`` appends a bare
  ``ASSISTANT`` after a final ``user`` turn; generation stops at ``END``.
- calculator span inside assistant content: ``CALC expr EQ result CALC_END``; the host
  evaluates ``expr`` and forces ``result CALC_END``; for display the span reads ``expr = result``.

Every rendering returns ``(ids, roles)``: ``ids`` is ``int32[n]`` and ``roles`` is ``int8[n]``
with the role of every token (``ROLE_RAW = 0`` raw text, 1 system, 2 user, 3 assistant,
4 library, the codes of the ``role`` register in ``bes.config.REG_FIELDS``). Role tokens and
``END`` carry the role of their turn, calculator tokens the role of the assistant turn, and
``BOS`` carries 0. Content is always encoded with ordinary encoding, so a message containing
the string ``"<|end|>"`` yields text tokens, never the ``END`` id.

Message format: ``{"role": "system" | "user" | "assistant" | "library", "content": str}``.
Assistant content may also be a list of parts ``{"type": "text", "text": str}`` and
``{"type": "calc", "expr": str, "result": str}``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np

from bes.config import (ASSISTANT, BOS, CALC, CALC_END, END, EQ, LIBRARY, N_SPECIAL, ROLE_TOKENS,
                        SPECIAL_TOKENS, SYSTEM, USER, VOCAB_SIZE, special_id)

ROLE_RAW, ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT, ROLE_LIBRARY = 0, 1, 2, 3, 4
ROLE_CODE: dict[str, int] = {"system": ROLE_SYSTEM, "user": ROLE_USER,
                             "assistant": ROLE_ASSISTANT, "library": ROLE_LIBRARY}
ROLE_TOKEN: dict[str, int] = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT,
                              "library": LIBRARY}
ROLE_CODE_OF_TOKEN: dict[int, int] = {ROLE_TOKEN[r]: c for r, c in ROLE_CODE.items()}
FIRST_SPECIAL = VOCAB_SIZE - N_SPECIAL

# Allowed predecessor states of each role (state = role of the previous message, "start" after BOS).
_ALLOWED_AFTER: dict[str, tuple[str, ...]] = {
    "system": ("start",),
    "library": ("start", "system", "assistant"),
    "user": ("start", "system", "library", "assistant"),
    "assistant": ("user",),
}


def _check_tokenizer(tok) -> None:
    for name in SPECIAL_TOKENS:
        if tok.special_tokens[name] != special_id(name):
            raise ValueError("tokenizer special ids differ from bes.config; use bes.tokenizer.tok.load()")


def _arrays(ids: list[int], roles: list[int]) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(ids, dtype=np.int32), np.asarray(roles, dtype=np.int8)


# ------------------------------------------------------------------------------ documents
def render_document(text: str, tok) -> tuple[np.ndarray, np.ndarray]:
    """``BOS text`` -> ``(ids int32[1 + n], roles int8[1 + n])``, all roles ``ROLE_RAW``."""
    _check_tokenizer(tok)
    ids = [BOS, *tok.encode(text)]
    return _arrays(ids, [ROLE_RAW] * len(ids))


# -------------------------------------------------------------------------- calculator
def render_calc(expr: str, result: str, tok) -> list[int]:
    """Ids of one calculator span ``CALC expr EQ result CALC_END``."""
    return [CALC, *tok.encode(expr), EQ, *tok.encode(result), CALC_END]


def calc_result_ids(result: str, tok) -> list[int]:
    """Ids the host forces after the model has emitted ``CALC expr EQ``: ``result CALC_END``."""
    return [*tok.encode(result), CALC_END]


class CalcSpan(NamedTuple):
    """Positions of one calculator span in a token sequence.

    ``start`` is the ``CALC`` position; ``eq`` the ``EQ`` position (-1 when absent); ``end``
    the ``CALC_END`` position (-1 when the span is not closed); ``stop`` is one past the last
    position of the span. Expression ids are ``ids[start + 1 : eq]`` (up to ``stop`` without
    ``EQ``) and result ids ``ids[eq + 1 : end]`` (up to ``stop`` when not closed).
    """

    start: int
    eq: int
    end: int
    stop: int

    @property
    def complete(self) -> bool:
        return self.eq >= 0 and self.end >= 0


def find_calc_spans(ids: Sequence[int] | np.ndarray) -> list[CalcSpan]:
    """All calculator spans of ``ids`` in order.

    A span opens at ``CALC`` and closes at ``CALC_END``. Any other special token except a
    repeated ``EQ`` (``BOS``, a role token, ``END``, a new ``CALC``) ends an open span as
    incomplete; so does the end of ``ids``. ``EQ`` or ``CALC_END`` outside a span are ignored.
    """
    arr = np.asarray(ids, dtype=np.int64)
    spans: list[CalcSpan] = []
    start = eq = -1
    for pos in np.flatnonzero(arr >= FIRST_SPECIAL).tolist():
        t = int(arr[pos])
        if start < 0:
            if t == CALC:
                start, eq = pos, -1
            continue
        if t == EQ:
            if eq < 0:
                eq = pos
        elif t == CALC_END:
            spans.append(CalcSpan(start, eq, pos, pos + 1))
            start = eq = -1
        else:
            spans.append(CalcSpan(start, eq, -1, pos))
            start, eq = (pos, -1) if t == CALC else (-1, -1)
    if start >= 0:
        spans.append(CalcSpan(start, eq, -1, len(arr)))
    return spans


def _content_ids(role: str, content: Any, tok) -> list[int]:
    if isinstance(content, str):
        return tok.encode(content)
    if role != "assistant" or not isinstance(content, (list, tuple)):
        raise ValueError(f"{role} content must be a string (parts lists are for assistant turns)")
    ids: list[int] = []
    for part in content:
        kind = part.get("type") if isinstance(part, Mapping) else None
        if kind == "text" and isinstance(part.get("text"), str):
            ids += tok.encode(part["text"])
        elif kind == "calc" and isinstance(part.get("expr"), str) and isinstance(part.get("result"), str):
            ids += render_calc(part["expr"], part["result"], tok)
        else:
            raise ValueError(f"invalid content part {part!r}")
    return ids


# ------------------------------------------------------------------------- conversations
def render_turn(role: str, content: Any, tok) -> tuple[np.ndarray, np.ndarray]:
    """One turn ``ROLE content END`` with every token carrying the role code of ``role``."""
    _check_tokenizer(tok)
    if role not in ROLE_TOKEN:
        raise ValueError(f"unknown role {role!r}")
    ids = [ROLE_TOKEN[role], *_content_ids(role, content, tok), END]
    return _arrays(ids, [ROLE_CODE[role]] * len(ids))


def check_structure(roles: Sequence[str], add_generation_prompt: bool = False) -> None:
    """Raise ``ValueError`` unless ``roles`` follow ``[system] ([library] user assistant)*
    [[library] user]``; with ``add_generation_prompt`` the last message must be ``user``."""
    if not roles:
        raise ValueError("a conversation needs at least one message")
    state = "start"
    for i, role in enumerate(roles):
        if not isinstance(role, str) or role not in _ALLOWED_AFTER:
            raise ValueError(f"message {i}: unknown role {role!r}")
        if state not in _ALLOWED_AFTER[role]:
            raise ValueError(f"message {i}: a {role} turn cannot follow {state}")
        state = role
    if add_generation_prompt and state != "user":
        raise ValueError("add_generation_prompt needs a conversation that ends with a user turn")
    if state not in ("user", "assistant"):
        raise ValueError(f"a conversation cannot end with a {state} turn")


def render_conversation(messages: Sequence[Mapping[str, Any]], tok,
                        add_generation_prompt: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """``BOS`` + one ``ROLE content END`` turn per message (+ ``ASSISTANT`` when prompting).

    Returns ``(ids int32[n], roles int8[n])``. Raises ``ValueError`` on an invalid structure
    (see ``check_structure``) or content.
    """
    _check_tokenizer(tok)
    if not all(isinstance(m, Mapping) for m in messages):
        raise ValueError("every message must be a mapping with 'role' and 'content'")
    check_structure([m.get("role") for m in messages], add_generation_prompt)
    ids, roles = [BOS], [ROLE_RAW]
    for message in messages:
        role = message["role"]
        turn = [ROLE_TOKEN[role], *_content_ids(role, message.get("content"), tok), END]
        ids += turn
        roles += [ROLE_CODE[role]] * len(turn)
    if add_generation_prompt:
        ids.append(ASSISTANT)
        roles.append(ROLE_ASSISTANT)
    return _arrays(ids, roles)


def assistant_turn_ends(ids: Sequence[int] | np.ndarray) -> np.ndarray:
    """``int64[k]`` positions just after every ``END`` that closes an assistant turn.

    The turn of an ``END`` is given by the closest preceding role token; a ``BOS`` in between
    means the ``END`` belongs to no turn.
    """
    arr = np.asarray(ids, dtype=np.int64)
    if arr.size == 0:
        return np.zeros(0, dtype=np.int64)
    marks = np.isin(arr, ROLE_TOKENS) | (arr == BOS)
    last = np.maximum.accumulate(np.where(marks, np.arange(arr.size), -1))
    ends = np.flatnonzero(arr == END)
    opener = last[ends]
    ok = (opener >= 0) & (arr[np.maximum(opener, 0)] == ASSISTANT)
    return ends[ok] + 1


def truncate_to_turns(ids: Sequence[int] | np.ndarray, max_len: int) -> np.ndarray | None:
    """Longest prefix of a rendered conversation that fits ``max_len`` tokens without cutting a
    turn: ``ids`` unchanged when it fits, otherwise the prefix ending at the last assistant
    ``END`` within ``max_len``; ``None`` when even the first exchange does not fit.

    Returns ``int32`` ids; slice the roles array to the same length.
    """
    arr = np.asarray(ids, dtype=np.int32)
    if arr.size <= max_len:
        return arr
    cuts = assistant_turn_ends(arr)
    cuts = cuts[cuts <= max_len]
    if cuts.size == 0:
        return None
    return arr[: int(cuts[-1])]


# ------------------------------------------------------------------------------- display
def display_text(ids: Sequence[int] | np.ndarray, tok) -> str:
    """User-facing text of ``ids``: special tokens removed, each calculator span rendered as
    ``expr = result`` (both stripped of outer whitespace; ``expr`` alone before ``EQ`` and
    ``expr =`` while the result is still empty). Bytes are decoded as UTF-8 per run between
    special tokens (invalid bytes become U+FFFD)."""
    arr: list[int] = np.asarray(ids, dtype=np.int64).tolist()

    def text_of(lo: int, hi: int) -> str:
        pieces, run = [], []
        for t in arr[lo:hi]:
            if t >= FIRST_SPECIAL:
                if run:
                    pieces.append(tok.decode_bytes(run).decode("utf-8", errors="replace"))
                    run = []
            else:
                run.append(t)
        if run:
            pieces.append(tok.decode_bytes(run).decode("utf-8", errors="replace"))
        return "".join(pieces)

    out: list[str] = []
    pos = 0
    for span in find_calc_spans(arr):
        out.append(text_of(pos, span.start))
        if span.eq >= 0:
            result_stop = span.end if span.end >= 0 else span.stop
            expr = text_of(span.start + 1, span.eq).strip()
            result = text_of(span.eq + 1, result_stop).strip()
            out.append(f"{expr} = {result}" if result else f"{expr} =")
        else:
            out.append(text_of(span.start + 1, span.stop).strip())
        pos = span.stop
    out.append(text_of(pos, len(arr)))
    return "".join(out)
