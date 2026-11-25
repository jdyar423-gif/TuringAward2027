"""Independent recomputation of the test-split blind comparisons (read-only: no report is written).

Loads pairs, key and ratings through bes.eval.proxy (with the key check), recomputes every
summary at full precision, compares with the stored report, and prints values rounded once
(half to even at 3 decimals, from the exact rational where the value is a ratio of counts).
"""
import json
import sys
from decimal import Decimal, ROUND_HALF_EVEN, ROUND_HALF_UP
from fractions import Fraction
from pathlib import Path

from bes.eval.proxy import make_ab, score
from bes.eval.proxy.judge import load_verdicts, ratings_path
from bes.eval.proxy.common import read_jsonl

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
SETS = {
    "final-prefopt-vs-falcon": ["judge-opus", "judge-opus-neutral", "judge-qwen3-4b-2507-3605803b"],
    "final-prefopt-vs-mobilellm": ["judge-opus", "judge-opus-neutral", "judge-qwen3-4b-2507-3605803b"],
    "final-prefopt-vs-trained": ["judge-opus"],
    "final-prefopt-vs-falcon-k6lp": ["judge-opus"],
    "final-prefopt-vs-falcon-k6ds": ["judge-opus"],
    "final-k1-vs-falcon": ["judge-opus"],
    "final-k1-vs-mobilellm": ["judge-opus"],
    "ref-falcon-vs-mobilellm-test": ["judge-opus"],
}
SEEN = {"chat-002", "chat-017", "math-034", "instruct-034", "instruct-062", "writing-100", "code-026",
        "math-001", "code-002", "writing-001", "instruct-001"}


def r3(x):
    """Round once to 3 decimals, half to even, from the exact decimal expansion of a float/Fraction."""
    if isinstance(x, Fraction):
        d = Decimal(x.numerator) / Decimal(x.denominator)
    else:
        d = Decimal(repr(x))
    return str(d.quantize(Decimal("0.001"), rounding=ROUND_HALF_EVEN))


def full(outcomes):
    from collections import Counter
    c = Counter(outcomes)
    n = sum(c.values())
    w, l = c["win"], c["loss"]
    t = c["tie"] + c["both_bad"]
    sc = Fraction(2 * w + t, 2 * n)
    lo, hi = score.wilson(w + 0.5 * t, n)
    dec = w + l
    dr = Fraction(w, dec) if dec else None
    dlo, dhi = score.wilson(w, dec) if dec else (None, None)
    p = score.sign_test(w, l)
    return {"n": n, "W": w, "L": l, "T": t, "tie_only": c["tie"], "both_bad": c["both_bad"],
            "score": float(sc), "score_r": r3(sc), "ci": [lo, hi], "ci_r": [r3(lo), r3(hi)],
            "ci_4dp_then_3": [r3(float(Decimal(repr(lo)).quantize(Decimal('0.0001'), ROUND_HALF_UP))),
                              r3(float(Decimal(repr(hi)).quantize(Decimal('0.0001'), ROUND_HALF_UP)))],
            "dec": dec, "dec_rate": None if dr is None else float(dr), "dec_r": None if dr is None else r3(dr),
            "dci": [dlo, dhi], "dci_r": [None if dlo is None else r3(dlo), None if dhi is None else r3(dhi)],
            "p": p, "tie_rate": t / n}


res = {}
for ps, sources in SETS.items():
    pairs, _ = make_ab.load_pairs(ROOT, ps)
    key = make_ab.load_key(ROOT, ps)
    make_ab.check_pairs(pairs, key, ps)
    by_id = {p["pair_id"]: p for p in pairs}
    for src in sources:
        verdicts = load_verdicts(ROOT, ps, src, key=key)
        recs = read_jsonl(ratings_path(ROOT, ps, src))
        summ = score.summarize(pairs, key, verdicts, recs)
        stored = json.load(open(ROOT / "reports" / "ab" / f"{ps}--{src}.json"))
        same = json.dumps(summ, sort_keys=True) == json.dumps(stored, sort_keys=True)
        outcomes = {pid: make_ab.unblind(v, pid, key) for pid, v in verdicts.items() if pid in key["assignment"]}
        allo = list(outcomes.values())
        o189 = [o for pid, o in outcomes.items() if by_id[pid]["prompt_id"] not in SEEN]
        cats, turns = {}, {"single": [], "multi": []}
        for pid, o in outcomes.items():
            p = by_id[pid]
            cats.setdefault(p["category"], []).append(o)
            turns["multi" if p.get("multi_turn") else "single"].append(o)
        # raw verdict counts
        from collections import Counter
        raw = Counter(verdicts.values())
        # first-position pref and agreement at full precision
        rr = [r for r in recs if r.get("pair_id") in outcomes]
        # last record per pair
        last = {}
        for r in recs:
            if r.get("pair_id") in outcomes and r.get("verdict") in ("A", "B", "tie", "both_bad"):
                last[r["pair_id"]] = r
        agree = sum(bool(r["agree"]) for r in last.values()) / len(last)
        fpp = sum(r["first_position_pref"] for r in last.values()) / len(last)
        fpp_all = sum(r["first_position_pref"] for r in rr) / len(rr)
        res[f"{ps}|{src}"] = {
            "report_equal_recomputed": same, "n_records": len(recs), "n_verdicts": len(verdicts),
            "raw_verdicts": dict(raw),
            "overall": full(allo), "u189": full(o189),
            "cats": {c: full(v) for c, v in sorted(cats.items())},
            "turns": {k: full(v) for k, v in turns.items()},
            "by_length": {k: (v["n"], v.get("wins"), v.get("losses"), (v.get("ties", 0) + v.get("both_bad", 0)) if v.get("n") else None)
                          for k, v in summ["by_length"].items()},
            "agree_last": agree, "fpp_last": fpp, "fpp_allrecs": fpp_all, "n_recs_for_pairs": len(rr),
            "judge_field": summ.get("judge"),
            "per_prompt": summ["per_prompt_score"],
        }
        print(ps, src, "report==recomputed:", same, "records", len(recs), "verdicts", len(verdicts))

OUT.write_text(json.dumps(res, indent=1, default=str))
