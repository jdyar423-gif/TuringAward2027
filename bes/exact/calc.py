"""Exact calculator for calculator spans (``docs/SPEC.md`` section 2.2).

:func:`evaluate` maps an arithmetic expression to its canonical result string, or ``None``.
The host calls it on the ``expr`` of ``CALC expr EQ`` and forces ``result CALC_END``.

Grammar: the expression is parsed with :func:`ast.parse` (``mode="eval"``) and only these
nodes are accepted: number literals, the binary operators ``+ - * / // % **``, unary ``-`` and
``+``, and parentheses (which leave no node). Everything else (names, calls, attributes,
subscripts, strings, comparisons, bit operators, ...) gives ``None``. Nothing is ever executed:
the tree is walked by :func:`_value`.

- **Numbers** are decimal literals only: ``12``, ``3.25``, ``.5``, ``2.``. Their exact value
  comes from the literal's source text (``0.1`` is exactly 1/10, never a binary float).
  Underscores, exponents (``1e3``), hex/octal/binary and complex literals are rejected. Before
  parsing, :func:`normalize_expr` removes thousands separators (``1,000`` -> ``1000``) and maps
  the signs ``×``, ``÷`` and ``−`` (U+2212) to ``*``, ``/`` and ``-``.
- **Arithmetic** is exact over the rationals (:class:`fractions.Fraction`). ``//`` and ``%``
  follow Python's floor semantics (``-7 // 2 == -4``, ``-7 % 2 == 1``); division, ``//`` or
  ``%`` by zero gives ``None``. Precedence and associativity are Python's (``-2**2 == -4``,
  ``2**3**2 == 512``).
- **Powers**: the exponent must be an integer with ``|e| <= MAX_EXPONENT``; ``0`` to a negative
  power gives ``None``; the size of the result is bounded *before* it is computed.
- **Limits**: at most ``MAX_EXPR_CHARS`` characters (no newline, ``#``, ``;`` or backslash)
  and ``MAX_NODES`` tree nodes; every
  intermediate value and the result must satisfy ``|v| <= MAX_ABS`` and have a numerator and a
  denominator of at most ``MAX_BITS`` bits. Anything larger gives ``None``.
- **Result string** (:func:`canonical`): integers without a decimal point (``"24"``, ``"-3"``);
  other values rounded half away from zero to ``DECIMALS = 6`` decimal places, trailing zeros
  removed (``"0.5"``, ``"0.333333"``, ``"14.285714"``); never scientific notation; a value that
  rounds to zero prints ``"0"``.

GSM8K-style annotations ``<<48/2=24>>`` are handled by :func:`parse_annotations` (positions,
``(expr, result)`` pairs and a check of the stated result against the exact value) and
:func:`gsm8k_parts` (assistant content parts for ``bes.data.chat_format``).
"""
from __future__ import annotations

import ast
import re
from fractions import Fraction
from typing import NamedTuple

MAX_EXPR_CHARS = 200
MAX_NODES = 100
MAX_EXPONENT = 100
MAX_ABS = Fraction(10) ** 30
MAX_BITS = 1024
DECIMALS = 6

_DECIMAL_LITERAL = re.compile(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)\Z")
_THOUSANDS = re.compile(r"(?<![\w.])[0-9]{1,3}(?:,[0-9]{3})+(?![\w.]|,[0-9])")
_SIGNS = str.maketrans({"×": "*", "÷": "/", "−": "-"})
_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)


class _Reject(Exception):
    """The expression is outside the accepted language or its limits."""


def normalize_expr(expr: str) -> str:
    """Map ``×``, ``÷`` and ``−`` to ``* / -`` and drop thousands separators in ``1,234,567``."""
    expr = expr.translate(_SIGNS)
    return _THOUSANDS.sub(lambda m: m.group(0).replace(",", ""), expr)


def _check(v: Fraction) -> Fraction:
    if abs(v) > MAX_ABS or v.numerator.bit_length() > MAX_BITS or v.denominator.bit_length() > MAX_BITS:
        raise _Reject("magnitude")
    return v


def _power(base: Fraction, exp: Fraction) -> Fraction:
    if exp.denominator != 1 or abs(exp) > MAX_EXPONENT:
        raise _Reject("exponent")
    e = int(exp)
    if base == 0 and e < 0:
        raise _Reject("zero to a negative power")
    bits = max(abs(base.numerator).bit_length(), base.denominator.bit_length())
    if bits * abs(e) > MAX_BITS:
        raise _Reject("power too large")
    return base ** e


def _value(node: ast.AST, source: str) -> Fraction:
    """Exact value of a whitelisted node (raises :class:`_Reject` otherwise)."""
    if isinstance(node, ast.Constant):
        text = ast.get_source_segment(source, node)
        if type(node.value) not in (int, float) or text is None or not _DECIMAL_LITERAL.match(text):
            raise _Reject("literal")
        return _check(Fraction(text))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _value(node.operand, source)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and isinstance(node.op, _BINOPS):
        a, b = _value(node.left, source), _value(node.right, source)
        op = node.op
        if isinstance(op, ast.Add):
            return _check(a + b)
        if isinstance(op, ast.Sub):
            return _check(a - b)
        if isinstance(op, ast.Mult):
            return _check(a * b)
        if isinstance(op, ast.Pow):
            return _check(_power(a, b))
        if b == 0:
            raise _Reject("division by zero")
        if isinstance(op, ast.Div):
            return _check(a / b)
        if isinstance(op, ast.FloorDiv):
            return _check(Fraction(a // b))
        return _check(a % b)
    raise _Reject(type(node).__name__)


def evaluate_fraction(expr: str) -> Fraction | None:
    """Exact value of ``expr`` (see the module docstring), or ``None``."""
    if not isinstance(expr, str):
        return None
    source = normalize_expr(expr).strip()
    if not source or len(source) > MAX_EXPR_CHARS or any(c in source for c in "\n\r#;\\"):
        return None
    try:
        tree = ast.parse(source, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES + 1:
        return None
    try:
        return _value(tree.body, source)
    except (_Reject, ZeroDivisionError, OverflowError, RecursionError):
        return None


def canonical(value: Fraction | int) -> str:
    """Canonical string of an exact value: integers as such, otherwise rounded half away from
    zero to ``DECIMALS`` places with trailing zeros removed; no exponent notation."""
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    scale = 10 ** DECIMALS
    q, r = divmod(abs(value.numerator) * scale, value.denominator)
    if 2 * r >= value.denominator:
        q += 1
    whole, frac = divmod(q, scale)
    text = str(whole) if frac == 0 else f"{whole}.{frac:0{DECIMALS}d}".rstrip("0")
    return "-" + text if value < 0 and q != 0 else text


def evaluate(expr: str) -> str | None:
    """Canonical result string of ``expr``, or ``None`` for anything outside the grammar or
    its limits (see the module docstring)."""
    v = evaluate_fraction(expr)
    return None if v is None else canonical(v)


# ---------------------------------------------------------------------- GSM8K annotations
_ANNOTATION = re.compile(r"<<([^<>]*?)=([^<>=]*)>>")
_RESULT_NUMBER = re.compile(r"([+-]?)\$?((?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]*)?|\.[0-9]+)\Z")


def parse_number(text: str) -> Fraction | None:
    """Exact value of a stated result such as ``"24"``, ``"-0.5"``, ``"$1,200.50"`` (an optional
    sign, an optional ``$``, thousands separators); ``None`` for anything else."""
    m = _RESULT_NUMBER.match(text.strip())
    if not m:
        return None
    v = Fraction(m.group(2).replace(",", ""))
    return -v if m.group(1) == "-" else v


def _decimals(text: str) -> int:
    t = text.strip()
    return len(t) - t.index(".") - 1 if "." in t else 0


def _round_to(v: Fraction, k: int, truncate: bool) -> Fraction:
    scale = 10 ** k
    q, r = divmod(abs(v.numerator) * scale, v.denominator)
    if not truncate and 2 * r >= v.denominator:
        q += 1
    out = Fraction(q, scale)
    return -out if v < 0 else out


def check_annotation(expr: str, result: str) -> str:
    """Compare a stated ``result`` with the exact value of ``expr``.

    Returns ``"exact"`` (equal values, e.g. ``24`` and ``24.0``), ``"rounded"`` (the stated
    result has ``k >= 1`` decimals and equals the exact value rounded half away from zero or
    truncated to ``k`` decimals, e.g. ``10/3`` stated as ``3.33``), ``"wrong"`` or
    ``"invalid"`` (``expr`` does not evaluate or ``result`` is not a number).
    """
    value, stated = evaluate_fraction(expr), parse_number(result)
    if value is None or stated is None:
        return "invalid"
    if value == stated:
        return "exact"
    k = _decimals(result)
    if k >= 1 and stated in (_round_to(value, k, False), _round_to(value, k, True)):
        return "rounded"
    return "wrong"


class Annotation(NamedTuple):
    """One ``<<expr=result>>`` annotation: ``text[start:end]`` is the whole ``<<...>>``."""

    start: int
    end: int
    expr: str
    result: str
    value: str | None     # evaluate(expr)
    status: str           # check_annotation(expr, result)

    @property
    def ok(self) -> bool:
        return self.status in ("exact", "rounded")


def parse_annotations(text: str) -> list[Annotation]:
    """All ``<<expr=result>>`` annotations of a GSM8K-style solution, in order. The expression
    is everything before the last ``=`` inside the brackets."""
    out = []
    for m in _ANNOTATION.finditer(text):
        expr, result = m.group(1).strip(), m.group(2).strip()
        out.append(Annotation(m.start(), m.end(), expr, result, evaluate(expr),
                              check_annotation(expr, result)))
    return out


def annotation_pairs(text: str) -> list[tuple[str, str]]:
    """``(expr, result)`` of every annotation, as written."""
    return [(a.expr, a.result) for a in parse_annotations(text)]


# Prose that restates the annotated expression right before it: "48/2 = <<48/2=24>>24".
_PROSE_EQ = re.compile(r"\s*=\s*\$?\s*\Z")
_DOLLAR_END = re.compile(r"\$\s*\Z")
_MATH_CHARS = frozenset("0123456789.,+-*/()%$×÷−– ")
_KEY_NUMBER = re.compile(r"[0-9]+(?:\.[0-9]*)?|\.[0-9]+")
_KEY_SIGNS = str.maketrans({" ": None, "$": None, "x": "*", "X": "*", "–": "-"})


def _math_key(text: str) -> str:
    """Comparison key of an arithmetic string: no spaces, ``$`` or thousands separators;
    ``x``/``×``/``÷``/``−``/``–`` written as ``* / - -``; ``N%`` written as ``N*.01``; numbers in
    canonical form (``4.80`` -> ``4.8``, ``.4`` -> ``0.4``)."""
    text = normalize_expr(text).translate(_KEY_SIGNS).replace("%", "*.01")
    return _KEY_NUMBER.sub(lambda m: canonical(Fraction(m.group(0))), text)


def _strip_restated(before: str, expr: str) -> str:
    """``before`` without a trailing ``<arithmetic> = [$]`` when that arithmetic is the
    annotated ``expr`` written in prose (``0.2 x 50 = $`` for ``0.2*50``), since the calculator
    span displays ``expr = result`` itself. Otherwise ``before`` is kept, without a ``$`` that
    directly precedes the annotation."""
    m = _PROSE_EQ.search(before)
    if not m:
        return _DOLLAR_END.sub("", before)
    head = before[: m.start()]
    i = len(head)
    while i > 0:
        c = head[i - 1]
        if c in _MATH_CHARS:
            i -= 1
        elif c in "xX" and i >= 2 and head[i - 2] == " " and i < len(head) and head[i] == " ":
            i -= 1          # "0.2 x 50": a multiplication sign written as a letter
        else:
            break
    while i < len(head) and head[i] in " ,+*/%":
        i += 1              # the run starts at a digit, a sign, "(", "." or "$"
    run = head[i:]
    if not run or (i > 0 and head[i - 1].isalnum()) or _math_key(run) != _math_key(expr):
        return _DOLLAR_END.sub("", before)
    return head[:i]


def gsm8k_parts(text: str) -> list[dict] | None:
    """Assistant content parts (``bes.data.chat_format``) for a GSM8K-style solution.

    Each ``<<expr=result>>`` becomes ``{"type": "calc", "expr": expr, "result":
    evaluate(expr)}`` (the string the host will force), the copy of the result that follows the
    annotation is dropped, and so is a restated expression right before it (``48/2 =
    <<48/2=24>>24`` displays as ``48/2 = 24``). Returns ``None`` when an annotation does not
    evaluate or its stated result is wrong (``check_annotation`` not exact or rounded).
    Text outside annotations (including a final ``#### 72`` line) is kept as is.
    """
    parts: list[dict] = []
    pos = 0
    for ann in parse_annotations(text):
        if not ann.ok or ann.value is None:
            return None
        before = _strip_restated(text[pos:ann.start], ann.expr)
        if before:
            parts.append({"type": "text", "text": before})
        parts.append({"type": "calc", "expr": ann.expr, "result": ann.value})
        pos = ann.end
        for copy in (ann.result, ann.value):
            nxt = pos + len(copy)
            if copy and text.startswith(copy, pos) and not text[nxt:nxt + 1].isdigit():
                pos = nxt
                break
    if pos < len(text):
        parts.append({"type": "text", "text": text[pos:]})
    return parts
