"""Hindsight calculator spans: arithmetic that a teacher wrote out in prose becomes calculator
parts, so the model learns to hand every computation to the exact calculator.

The 24-hour run trained calculator spans from GSM8K's own annotations only (7,473 problems);
its test replies then did arithmetic in prose and slipped (``47 x 23 = 1176``). Here every
statement ``<expression> = <number>`` in an assistant turn whose expression the calculator
accepts (``bes.exact.calc.evaluate``) and whose stated result is **exactly** the calculator's
becomes one calculator part ``{"type": "calc", "expr": ..., "result": ...}``. A statement
that is rounded, wrong or outside the calculator's language stays text: nothing is corrected
and nothing is guessed.

What counts as an expression: at least one binary operator between numbers; numbers with
optional thousands commas and decimals; the operators ``+ - * / × ÷ x ^`` (``x`` only between
numbers); parentheses; a ``$`` in front of a number is dropped from the expression (it stays
in the text before the span when it leads the statement). A percent sign, a unit or a word
inside the expression ends it: ``20% of 150 = 30`` and ``3 boxes x 24 = 72`` are not touched.
Code is never touched (fenced blocks and inline code spans are skipped). The result must end
the statement: ``7 x 8 = 56 x 2 = 112`` keeps ``7 x 8 = `` as text and turns ``56 x 2 = 112``
into a span.

The scan is linear: for every ``=`` the number after it is read, then the longest run of
expression characters before it (at most ``MAX_EXPR`` characters).

:func:`arith_parts` returns the content for ``bes.data.chat_format`` (the text itself when it
has no such statement); :func:`with_arith` wraps a registry adapter.
"""
from __future__ import annotations

import re
from typing import Any, Callable

from bes.exact import calc

_INT = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
_RESULT = re.compile(rf"\s*(-?\$?{_INT}(?:\.\d+)?)")
#: after the result: not a longer number, not another computation, not a percent sign
_AFTER_BAD = re.compile(r"\d|,\d|\.\d|\s*%|\s*(?:[-+*/×÷^=]|x\b)\s*\(*\s*\$?\d")
_EXPR_CHARS = frozenset("0123456789.,$ ()+-*/×÷^x")
_SPLIT = re.compile(r"[,.;:]\s")
#: a whole expression: operand (operator operand)+ ; checked on at most MAX_EXPR characters
_OPERAND = rf"\(*\s*\$?{_INT}(?:\.\d+)?\s*\)*"
_WHOLE = re.compile(rf"{_OPERAND}(?:\s*(?:\*\*|[-+*/×÷^]|x)\s*{_OPERAND})+\Z")
_FENCE = re.compile(r"```.*?(?:```|\Z)", re.S)
_INLINE = re.compile(r"`[^`\n]*`")
_X = re.compile(r"(?<=[\d)])\s*x\s*(?=[\d($])")
MAX_EXPR = 96             # characters of an expression at most
MAX_SPANS = 24            # per assistant turn (more looks like a table, not a solution)


def _norm(expr: str) -> str:
    """The expression in the calculator's language, as the span holds it: ``x`` -> ``*``,
    ``^`` -> ``**``, no ``$``, single spaces (``×``, ``÷`` and thousands commas stay: the
    calculator reads them)."""
    e = _X.sub(" * ", expr).replace("^", "**").replace("$", "")
    return " ".join(e.split())


def _protected(text: str) -> list[tuple[int, int]]:
    if "`" not in text:
        return []
    spans = [(m.start(), m.end()) for m in _FENCE.finditer(text)]
    spans += [(m.start(), m.end()) for m in _INLINE.finditer(text)]
    return spans


def _left_expr(text: str, eq: int, floor: int) -> tuple[int, str] | None:
    """``(start, expr)`` of the expression that ends right before the ``=`` at ``eq`` (not
    before ``floor``), or None."""
    i = eq
    lo = max(floor, eq - MAX_EXPR)
    while i > lo and text[i - 1] in _EXPR_CHARS:
        i -= 1
    seg = text[i:eq]
    # "In 2020, 3 + 4" / "cost 5. 3 + 4": only the last clause is the expression
    cut = 0
    for m in _SPLIT.finditer(seg):
        cut = m.end()
    start = i + cut
    e = text[start:eq]
    k = len(e) - len(e.lstrip())
    start, e = start + k, e.strip()
    while e and not (e[0].isdigit() or e[0] in "($"):       # a stray "x", ")" or operator
        start, e = start + 1, e[1:]
        k = len(e) - len(e.lstrip())
        start, e = start + k, e.lstrip()
    while e.count("(") > e.count(")") and e.startswith("("):     # "(3 + 4 = 7)": the "(" is prose
        k = 1 + (len(e[1:]) - len(e[1:].lstrip()))
        e, start = e[k:], start + k
    if not e or e.count("(") != e.count(")"):
        return None
    if e.startswith("$"):
        k = 1 + (len(e[1:]) - len(e[1:].lstrip()))
        e, start = e[k:], start + k
    if not _WHOLE.match(e):
        return None
    return start, e


def find_arith(text: str) -> list[tuple[int, int, str, str]]:
    """``(start, end, expr, result)`` of every exact arithmetic statement in ``text`` (in
    order, non-overlapping); ``text[start:end]`` is replaced by the calculator span, whose
    rendering is ``expr = result``. A ``$`` or an unmatched ``(`` in front of the expression
    stays in the text before the span."""
    out: list[tuple[int, int, str, str]] = []
    eq = text.find("=")
    if eq < 0:
        return out
    prot = _protected(text)
    floor = 0
    while eq >= 0 and len(out) < MAX_SPANS:
        nxt = text.find("=", eq + 1)
        m = _RESULT.match(text, eq + 1)
        if m is not None and not _AFTER_BAD.match(text, m.end()) and (eq == 0 or text[eq - 1] not in "=<>!"):
            left = _left_expr(text, eq, floor)
            if left is not None and not any(a < m.end() and left[0] < b for a, b in prot):
                start, e = left
                expr = _norm(e)
                want = calc.evaluate(expr)
                if want is not None:
                    stated = calc.evaluate(m.group(1).replace("$", "").replace(",", ""))
                    if stated is not None and stated == want:
                        out.append((start, m.end(), expr, want))
                        floor = m.end()
        eq = nxt
    return out


def arith_parts(text: str) -> str | list[dict[str, str]]:
    """Content parts with a calculator part for every exact arithmetic statement; ``text``
    itself when there is none."""
    found = find_arith(text)
    if not found:
        return text
    parts: list[dict[str, str]] = []
    pos = 0
    for start, end, expr, result in found:
        if start > pos:
            parts.append({"type": "text", "text": text[pos:start]})
        parts.append({"type": "calc", "expr": expr, "result": result})
        pos = end
    if pos < len(text):
        parts.append({"type": "text", "text": text[pos:]})
    return parts


def with_arith(adapter: Callable[[dict[str, Any]], Any]) -> Callable[[dict[str, Any]], Any]:
    """``adapter`` whose conversations carry calculator parts in their assistant turns
    (:func:`arith_parts`); turns that already hold parts, and non-conversations, pass."""

    def wrapped(r: dict[str, Any]) -> Any:
        item = adapter(r)
        if not isinstance(item, list):
            return item
        out = []
        for m in item:
            if isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("content"), str):
                parts = arith_parts(m["content"])
                if parts is not m["content"]:
                    m = {**m, "content": parts}
            out.append(m)
        return out

    wrapped.__name__ = getattr(adapter, "__name__", "adapter") + "_arith"
    wrapped.__doc__ = (adapter.__doc__ or "") + " Assistant arithmetic becomes calculator parts (bes.data.arith)."
    return wrapped


__all__ = ["find_arith", "arith_parts", "with_arith", "MAX_SPANS", "MAX_EXPR"]
