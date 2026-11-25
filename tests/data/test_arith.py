"""Hindsight calculator spans (bes/data/arith.py)."""
import time

from bes.data.arith import arith_parts, find_arith, with_arith
from bes.exact import calc


def spans(text):
    return [(text[a:b], e, r) for a, b, e, r in find_arith(text)]


def test_exact_statements_become_spans():
    assert spans("so 3 × 24 = 72 pencils.") == [("3 × 24 = 72", "3 × 24", "72")]
    assert spans("costs $8.75 ÷ 5 = $1.75.") == [("8.75 ÷ 5 = $1.75", "8.75 ÷ 5", "1.75")]
    assert spans("47 x 23 = 1081") == [("47 x 23 = 1081", "47 * 23", "1081")]
    assert spans("2^10 = 1024") == [("2^10 = 1024", "2**10", "1024")]
    assert spans("(240 × 3/5) + 40 = 184 muffins") == [("(240 × 3/5) + 40 = 184", "(240 × 3/5) + 40", "184")]
    assert spans("1,200 + 350 = 1,550 dollars") == [("1,200 + 350 = 1,550", "1,200 + 350", "1550")]
    assert spans("14 + 22 + 9 + 35 = 80, then 80 / 4 = 20.") == [("14 + 22 + 9 + 35 = 80", "14 + 22 + 9 + 35", "80"),
                                                               ("80 / 4 = 20", "80 / 4", "20")]
    for _, e, r in spans("3.5 × 60 × 60 = 12,600 and (3 + 4 = 7) and In 2020, 3 + 4 = 7."):
        assert calc.evaluate(e) == r


def test_wrong_rounded_or_foreign_statements_stay_text():
    assert spans("47 x 23 = 1176") == []
    assert spans("10 / 3 = 3.33") == []
    assert spans("20% of 150 = 30") == []
    assert spans("3 boxes x 24 = 72") == []
    assert spans("0.86 × 100 = 86%") == []
    assert spans("x = 5 and y = 7") == []
    assert spans("7 × 8 = 56 × 2 = 112") == [("56 × 2 = 112", "56 × 2", "112")]


def test_code_is_not_touched():
    text = "Use `a = 2 * 3 = 6` here:\n```python\nb = 2 * 3 = 6\n```\nBut 2 * 3 = 6."
    assert [s[0] for s in spans(text)] == ["2 * 3 = 6"] and text.rindex("2 * 3 = 6") == find_arith(text)[0][0]


def test_parts_rebuild_the_text_around_the_spans():
    text = "First 2 + 3 = 5, then $5 × 4 = $20 in total."
    parts = arith_parts(text)
    assert [p["type"] for p in parts] == ["text", "calc", "text", "calc", "text"]
    shown = "".join(p["text"] if p["type"] == "text" else f"{p['expr']} = {p['result']}" for p in parts)
    assert shown == "First 2 + 3 = 5, then $5 × 4 = 20 in total."
    assert arith_parts("no arithmetic here") == "no arithmetic here"


def test_adapter_wrapper_and_speed():
    conv = [{"role": "user", "content": "2 + 2 = 4?"}, {"role": "assistant", "content": "Yes: 2 + 2 = 4."}]
    out = with_arith(lambda r: r)(conv)
    assert out[0]["content"] == "2 + 2 = 4?" and isinstance(out[1]["content"], list)
    assert with_arith(lambda r: None)({}) is None and with_arith(lambda r: "text")({}) == "text"
    long = ("1 2 3 4 5 6 7 8 9 " * 2000) + "= 5 " + ("(" * 500) + "1 + 1 = 2"
    t0 = time.perf_counter()
    find_arith(long)
    assert time.perf_counter() - t0 < 0.5
