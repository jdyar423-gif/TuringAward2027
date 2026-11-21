"""Tests of the exact calculator (bes.exact.calc).

Two independent references: ``ref_evaluate`` is a recursive-descent parser over a token stream
(Python's precedence: ``factor := ('+'|'-') factor | power``, ``power := atom ['**' factor]``)
that shares no code with the AST walker, and ``eval`` of the expression with every literal
wrapped in ``Fraction("...")`` (Python's own parser and exact rationals). Canonical strings are
checked against ``decimal.Decimal`` quantisation.
"""
from __future__ import annotations

import random
import re
import sys
import time
from decimal import ROUND_HALF_UP, Decimal, localcontext
from fractions import Fraction

import pytest

from bes.exact import calc

LIMIT_ABS = Fraction(10) ** 30
LIMIT_BITS = 1024
LIMIT_EXP = 100


# ------------------------------------------------------------------------------ reference
class Reject(Exception):
    pass


TOKEN = re.compile(r"\s*(?:(\d+\.\d*|\.\d+|\d+)|(\*\*|//|[-+*/%()]))")


def tokenize(s: str) -> list[str]:
    out, pos = [], 0
    s = s.rstrip()
    while pos < len(s):
        m = TOKEN.match(s, pos)
        if not m:
            raise Reject("token")
        out.append(m.group(1) or m.group(2))
        pos = m.end()
    return out


def checked(v: Fraction) -> Fraction:
    if abs(v) > LIMIT_ABS or abs(v.numerator).bit_length() > LIMIT_BITS or v.denominator.bit_length() > LIMIT_BITS:
        raise Reject("magnitude")
    return v


def ref_evaluate(s: str) -> Fraction | None:
    try:
        toks = tokenize(s)
    except Reject:
        return None
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def take():
        nonlocal pos
        pos += 1
        return toks[pos - 1]

    def expr():
        v = term()
        while peek() in ("+", "-"):
            op = take()
            w = term()
            v = checked(v + w if op == "+" else v - w)
        return v

    def term():
        v = factor()
        while peek() in ("*", "/", "//", "%"):
            op = take()
            w = factor()
            if op == "*":
                v = checked(v * w)
            else:
                if w == 0:
                    raise Reject("zero")
                v = checked(v / w if op == "/" else Fraction(v // w) if op == "//" else v % w)
        return v

    def factor():
        if peek() in ("+", "-"):
            op = take()
            v = factor()
            return -v if op == "-" else v
        return power()

    def power():
        base = atom()
        if peek() == "**":
            take()
            e = factor()
            if e.denominator != 1 or abs(e) > LIMIT_EXP:
                raise Reject("exp")
            if base == 0 and e < 0:
                raise Reject("zero")
            if max(abs(base.numerator).bit_length(), base.denominator.bit_length()) * abs(int(e)) > LIMIT_BITS:
                raise Reject("big")
            return checked(base ** int(e))
        return base

    def atom():
        t = take() if peek() is not None else None
        if t == "(":
            v = expr()
            if take() != ")":
                raise Reject("paren")
            return v
        if t is None or not t[0].isdigit() and t[0] != ".":
            raise Reject("atom")
        return checked(Fraction(t))

    try:
        v = expr()
        if pos != len(toks):
            return None
        return v
    except (Reject, IndexError):
        return None


def eval_oracle(s: str) -> Fraction | None:
    """Python's parser with exact rational literals (no limits applied; used on small cases)."""
    code = re.sub(r"(\d+\.\d*|\.\d+|\d+)", lambda m: f'F("{m.group(1)}")', s)
    try:
        v = eval(code, {"__builtins__": {}}, {"F": Fraction})  # noqa: S307 (test oracle)
    except ZeroDivisionError:
        return None
    return Fraction(v) if isinstance(v, (int, Fraction)) else None    # Fraction // Fraction is an int


def ref_canonical(v: Fraction) -> str:
    """Half-up quantisation to 6 places with ``decimal`` (400 significant digits)."""
    if v.denominator == 1:
        return str(v.numerator)
    with localcontext() as ctx:
        ctx.prec = 400
        q = Decimal(v.numerator) / Decimal(v.denominator)
        r = q.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
    text = format(r, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


# ------------------------------------------------------------------------ random exprs
SMALL = False


def rand_number(rng: random.Random) -> str:
    k = rng.random()
    if k < 0.55:
        return str(rng.randint(0, 20 if SMALL else 120))
    if k < 0.75:
        return f"{rng.randint(0, 99)}.{rng.randint(0, 99):02d}"
    if k < 0.85:
        return f".{rng.randint(1, 9)}"
    if k < 0.9:
        return f"{rng.randint(1, 9)}."
    return str(rng.randint(0, 5))


def rand_expr(rng: random.Random, depth: int = 0) -> str:
    if depth > 3 or rng.random() < 0.3:
        return rand_number(rng)
    k = rng.random()
    if k < 0.12:
        return "-" + rand_expr(rng, depth + 1)
    if k < 0.2:
        return "(" + rand_expr(rng, depth + 1) + ")"
    op = rng.choice(["+", "-", "*", "/", "//", "%", "**", "+", "*", "-"])
    left = rand_expr(rng, depth + 1)
    if op == "**":
        right = rng.choice(["2", "3", "0", "-1", "-2", "(1+1)", "1.5", "101", "64"])
    else:
        right = rand_expr(rng, depth + 1)
    sp = rng.choice(["", " "])
    return f"{left}{sp}{op}{sp}{right}"


@pytest.mark.parametrize("seed", range(4))
def test_random_expressions_match_reference(seed):
    rng = random.Random(seed)
    n_valid = 0
    for _ in range(1500):
        e = rand_expr(rng)
        want = ref_evaluate(e)
        got = calc.evaluate_fraction(e)
        assert got == want, e
        assert calc.evaluate(e) == (None if want is None else ref_canonical(want)), e
        n_valid += want is not None
    assert n_valid > 800          # the generator mostly produces valid expressions


@pytest.mark.parametrize("seed", range(2))
def test_small_expressions_match_python_eval(seed, monkeypatch):
    """Numbers <= 20 and at most 16 leaves keep every intermediate value far inside the limits,
    so Python's own evaluation (no limits) must agree exactly."""
    monkeypatch.setattr(sys.modules[__name__], "SMALL", True)
    rng = random.Random(100 + seed)
    n = 0
    for _ in range(1000):
        e = rand_expr(rng)
        if "**" in e:
            continue
        n += 1
        assert calc.evaluate_fraction(e) == eval_oracle(e), e
    assert n > 500


# ------------------------------------------------------------------------- exactness
@pytest.mark.parametrize("expr,want", [
    ("48/2", "24"), ("0.1+0.2", "0.3"), ("1/3", "0.333333"), ("2/3", "0.666667"),
    ("-2/3", "-0.666667"), ("100/7", "14.285714"), ("7/2", "3.5"), ("10/4", "2.5"),
    ("2**-2", "0.25"), ("-7//2", "-4"), ("7//2", "3"), ("-7%2", "1"), ("7%-2", "-1"),
    ("7.5%2", "1.5"), ("2**3**2", "512"), ("-2**2", "-4"), ("(-2)**2", "4"), ("2*3+4", "10"),
    ("2*(3+4)", "14"), ("2+3*4**2", "50"), ("(1+2)*(3-4)/5", "-0.6"), ("1/8", "0.125"),
    ("0.0000001", "0"), ("0.0000005", "0.000001"), ("-0.0000004", "0"), ("-0.0000005", "-0.000001"),
    ("3.", "3"), (".5*4", "2"), ("+5", "5"), ("- -3", "3"), ("1,000*2", "2000"),
    ("1,234,567+1", "1234568"), ("3 × 4 − 1 ÷ 2", "11.5"), ("((((1))))", "1"),
    ("0**0", "1"), ("1.05**2", "1.1025"), ("2**99", "633825300114114700748351602688"),
    ("12.50*4", "50"), ("19.99*3", "59.97"), ("100 - 50 - 30 - 15", "5"), ("  6 * 7  ", "42"),
    ("1/7*7", "1"), ("10**30", "1" + "0" * 30),
])
def test_exact_results(expr, want):
    assert calc.evaluate(expr) == want


def test_canonical_never_uses_exponent_notation():
    rng = random.Random(7)
    for _ in range(3000):
        v = Fraction(rng.randint(-10**12, 10**12), rng.randint(1, 10**9))
        s = calc.canonical(v)
        assert "e" not in s.lower() and s == ref_canonical(v)
        assert re.fullmatch(r"-?\d+(\.\d{1,6})?", s) and not (("." in s) and s.endswith("0"))


# ---------------------------------------------------------------------------- safety
@pytest.mark.parametrize("expr", [
    "x+1", "abs(-1)", "__import__('os').system('true')", "().__class__", "(1).real", "[1][0]",
    "'a'*3", '"1"+"2"', "b'1'", "lambda: 1", "[x for x in (1,)]", "1 if 1 else 2", "1<2", "1==1",
    "1 and 2", "not 1", "2^3", "2&3", "2|3", "~1", "1<<3", "1>>1", "(a:=1)", "f'{1}'", "1j", "1e3",
    "1E-3", "0x10", "0o7", "0b101", "1_000", "True+1", "None", "...", "10/0", "1//0", "1%0",
    "0**-1", "2**1000", "2**101", "10**31", "9**9**9", "2**0.5", "4**(1/2)", "(-8)**(1/3)",
    "1,2", "", "   ", "1+", "(1", "1)", "1\n+2", "1;2", "#1", "1 # comment", "007",
    "(" * 300 + "1" + ")" * 300, "1+" * 120 + "1", "1" * 300,
])
def test_rejected(expr):
    assert calc.evaluate(expr) is None


def test_rejects_non_strings():
    for value in (None, 1, 1.5, b"1+1", ["1"]):
        assert calc.evaluate(value) is None  # type: ignore[arg-type]


def test_huge_inputs_are_fast():
    t0 = time.perf_counter()
    for e in ("9**9**9", "2**100**100", "(10**30)**100", "99999999999999999999**99", "1.0000001**100"):
        calc.evaluate(e)
    assert time.perf_counter() - t0 < 0.5


def test_exponent_limit_is_inclusive():
    assert calc.evaluate("1**100") == "1" and calc.evaluate("(-1)**100") == "1"
    assert calc.evaluate("1**-100") == "1"
    assert calc.evaluate("1**101") is None and calc.evaluate("1**-101") is None


def test_magnitude_limits_are_inclusive():
    assert calc.evaluate("10**30") == "1" + "0" * 30
    assert calc.evaluate("10**30+1") is None
    assert calc.evaluate("-(10**30)") == "-1" + "0" * 30
    assert calc.evaluate("2**99") is not None and calc.evaluate("2**100") is None   # 2**100 > 1e30


def test_normalize_expr():
    assert calc.normalize_expr("1,000 + 2,500,000") == "1000 + 2500000"
    assert calc.normalize_expr("1,2") == "1,2"
    assert calc.normalize_expr("1,2345") == "1,2345"
    assert calc.normalize_expr("3×4−2÷1") == "3*4-2/1"


# ------------------------------------------------------------------- GSM8K annotations
GSM8K = [
    ("Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
     "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
     "Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether in April and May.\n#### 72"),
    ("Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\n"
     "Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10",
     "Weng earns 12/60 = 0.2 per minute.\nWorking 50 minutes, she earned 0.2*50 = 10.\n#### 10"),
    ("In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\n"
     "This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.\n#### 5",
     "In the beginning, Betty has only 100/2 = 50.\nThis means, Betty needs 100-50-30-15 = 5 more.\n#### 5"),
    ("First find how many liters of the seawater are salt: 2 liters * 20% = <<2*20*.01=.4>>.4 liters",
     "First find how many liters of the seawater are salt: 2 liters * 20% = 2*20*.01 = 0.4 liters"),
    ("Accommodation is 15% * $1000=$<<15*.01*1000=150>>150",
     "Accommodation is 15*.01*1000 = 150"),
    ("Therefore, (75/100)*100=<<(75/100)*100=75>>75% of the flowers are not roses.",
     "Therefore, (75/100)*100 = 75% of the flowers are not roses."),
    ("He has <<5+3=8>>8 apples.", "He has 5+3 = 8 apples."),
    ("So their total bill is $15 + $4.80 + $2.50 +$2.7 = $<<15+4.8+2.5+2.7=25>>25.",
     "So their total bill is 15+4.8+2.5+2.7 = 25."),
]


def display_of(parts: list[dict]) -> str:
    return "".join(p["text"] if p["type"] == "text" else f"{p['expr']} = {p['result']}" for p in parts)


@pytest.mark.parametrize("solution,display", GSM8K)
def test_gsm8k_parts_display(solution, display):
    parts = calc.gsm8k_parts(solution)
    assert parts is not None
    assert display_of(parts) == display
    for p in parts:
        if p["type"] == "calc":
            assert calc.evaluate(p["expr"]) == p["result"]


def test_gsm8k_parts_render_through_chat_format():
    from bes.data import chat_format as cf
    from bes.tokenizer.tok import load

    tok = load()
    for solution, display in GSM8K:
        parts = calc.gsm8k_parts(solution)
        ids, _ = cf.render_conversation([{"role": "user", "content": "q"},
                                         {"role": "assistant", "content": parts}], tok)
        a = ids.tolist().index(cf.ASSISTANT)
        assert cf.display_text(ids[a + 1:-1], tok) == display
        assert len(cf.find_calc_spans(ids)) == sum(p["type"] == "calc" for p in parts)


def test_parse_annotations_positions_and_status():
    text = "a <<3*4=12>> b <<10/3=3.33>> c <<2/3=0.67>> d <<2/3=0.66>> e <<7/2=3>> f <<x+1=2>> g <<1+1=two>>"
    anns = calc.parse_annotations(text)
    assert [text[a.start:a.end] for a in anns] == re.findall(r"<<[^<>]*>>", text)
    assert [(a.expr, a.result) for a in anns] == calc.annotation_pairs(text)
    assert [a.status for a in anns] == ["exact", "rounded", "rounded", "rounded", "wrong", "invalid", "invalid"]
    assert [a.ok for a in anns] == [True, True, True, True, False, False, False]
    assert anns[0].value == "12" and anns[1].value == "3.333333" and anns[5].value is None


@pytest.mark.parametrize("expr,result,status", [
    ("48/2", "24", "exact"), ("48/2", "24.0", "exact"), ("48/2", "24.00", "exact"),
    ("1200+0.5", "$1,200.50", "exact"), ("5-8", "-3", "exact"), ("1/3", "0.33", "rounded"),
    ("1/3", "0.333", "rounded"), ("2/3", "0.7", "rounded"), ("1/3", "0.3333334", "wrong"),
    ("1/3", "0", "wrong"), ("5*5", "26", "wrong"), ("5*", "25", "invalid"), ("5*5", "abc", "invalid"),
    ("5*5", "25%", "invalid"),
])
def test_check_annotation(expr, result, status):
    assert calc.check_annotation(expr, result) == status


def test_gsm8k_parts_rejects_wrong_annotations():
    assert calc.gsm8k_parts("It costs 5*5 = <<5*5=26>>26 dollars.") is None
    assert calc.gsm8k_parts("It costs <<abc=3>>3 dollars.") is None
    assert calc.gsm8k_parts("No annotations here.") == [{"type": "text", "text": "No annotations here."}]
    # a rounded annotation keeps the host's exact result
    parts = calc.gsm8k_parts("Each gets 10/3 = <<10/3=3.33>>3.33 cups.")
    assert parts[1] == {"type": "calc", "expr": "10/3", "result": "3.333333"}
    assert display_of(parts) == "Each gets 10/3 = 3.333333 cups."
