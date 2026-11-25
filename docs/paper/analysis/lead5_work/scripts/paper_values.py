"""Fifth pass: the printed values of the test-split tables, rounded once from full precision (REVIEW_2 item
18: no second rounding of a 4-decimal value). From the counts in lead5_work/final_results.json, recomputes
the score (wins + ties/2)/n and its 95% Wilson interval, the decisive win rate and its interval, and the
exact two-sided sign test with bes.eval.proxy.score (wilson, sign_test) at full precision; prints them with
three decimals, rounded once in decimal arithmetic, a value exactly on a half rounded to even (the
convention of FACTS_2.md, CORRECTIONS 22:25 UTC, item 4; until 22:25 this script rounded halves up).
Writes paper_values.json.

    taskset -c 2,3 nice -n 5 S/venv/bin/python -I -B paper_values.py
"""
import json
import sys
from decimal import ROUND_HALF_EVEN, Decimal
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, "/home/user/turingaward2027")
from bes.eval.proxy.score import sign_test, wilson  # noqa: E402

W = Path(__file__).resolve().parents[1]
FR = json.loads((W / "final_results.json").read_text())


def r3(x):
    return str(Decimal(repr(float(x))).quantize(Decimal("0.001"), rounding=ROUND_HALF_EVEN)) if not isinstance(x, Fraction) \
        else str((Decimal(x.numerator) / Decimal(x.denominator)).quantize(Decimal("0.001"), rounding=ROUND_HALF_EVEN))


def pfmt(p):
    if p < 1e-4:
        return "< 0.0001"
    if p < 0.001:
        return f"{p:.4f}".rstrip("0")
    if p < 0.01:
        return f"{p:.4f}"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def row(w, l_, t):
    n = w + l_ + t
    sc = Fraction(2 * w + t, 2 * n)
    lo, hi = wilson(w + 0.5 * t, n)
    out = {"n": n, "W/L/T": f"{w} / {l_} / {t}", "score": r3(sc), "score_ci": f"[{r3(lo)}, {r3(hi)}]"}
    if w + l_:
        dlo, dhi = wilson(w, w + l_)
        out["decisive"] = r3(Fraction(w, w + l_))
        out["decisive_ci"] = f"[{r3(dlo)}, {r3(dhi)}]"
    p = sign_test(w, l_)
    out["p"] = pfmt(p)
    out["p_full"] = p
    return out


vals = {}
for name, e in FR["sets"].items():
    vals[name] = {}
    for src, s in e["sources"].items():
        o, r = s["overall"], s["restricted_189"]
        ent = {"all": row(o["wins"], o["losses"], o["ties"] + o["both_bad"]),
               "189": row(r["wins"], r["losses"], r["ties"]), "judge": s["judge"], "n_rated": s["n_rated"],
               "by_category": {c: row(v["wins"], v["losses"], v["ties"] + v["both_bad"]) for c, v in s["by_category"].items()},
               "by_turns": {c: row(v["wins"], v["losses"], v["ties"] + v["both_bad"]) for c, v in s["by_turns"].items()},
               "by_length": {c: (row(v["wins"], v["losses"], v["ties"] + v["both_bad"]) if v.get("n") else {"n": 0})
                             for c, v in s["by_length"].items()}}
        if s["judge"]:
            ent["orders_agree_pct"] = str(Decimal(repr(s["judge"]["orders_agree"] * 100)).quantize(Decimal("0.1"), rounding=ROUND_HALF_EVEN))
            fp = Decimal(repr(s["judge"]["first_position_pref_mean"])).quantize(Decimal("0.001"), rounding=ROUND_HALF_EVEN)
            ent["first_pos"] = ("+" if fp >= 0 else "") + str(fp)
        vals[name][src] = ent
(W / "paper_values.json").write_text(json.dumps(vals, indent=1, default=str))
for name, d in vals.items():
    for src, ent in d.items():
        a, b = ent["all"], ent["189"]
        print(f"{name:30s} {src:30s} n {a['n']}: {a['W/L/T']} {a['score']} {a['score_ci']} dec {a.get('decisive')} "
              f"{a.get('decisive_ci')} p {a['p']} | 189: {b['W/L/T']} {b['score']} {b['score_ci']} dec {b.get('decisive')} "
              f"{b.get('decisive_ci')} p {b['p']} | orders {ent.get('orders_agree_pct')} first {ent.get('first_pos')}")
        print("      cat", {c: f"{v['W/L/T']} {v['score']} p {v['p']}" for c, v in ent["by_category"].items()})
        print("      turns", {c: f"{v['W/L/T']} {v['score']}" for c, v in ent["by_turns"].items()})
        print("      length", {c: (f"n {v['n']} {v['W/L/T']} {v['score']} p {v['p']}" if v.get('n') else 'n 0') for c, v in ent["by_length"].items()})
