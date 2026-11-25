"""Fourth pass (REVIEW_2 item 8; Table M11): one blind-comparison report (study/proxy/reports/ab/*.json)
recomputed on a subset of its prompts from its `per_prompt_score` field (1 = a win of x, 0.5 = a tie or
"both bad", 0 = a loss): W / L / T, score (wins + ties/2)/n with a 95% Wilson interval, and the exact
two-sided sign test, as bes.eval.proxy.score reports them.

  restrict_report.py REPORT.json [--exclude ID,ID,...] [--out OUT.json]

Without --exclude it must reproduce the report's own `overall` (checked and printed).
"""
import argparse
import json
import math
from pathlib import Path

Z = 1.959963984540054
ap = argparse.ArgumentParser()
ap.add_argument("report")
ap.add_argument("--exclude", default="")
ap.add_argument("--out", default=None)
args = ap.parse_args()
rep = json.load(open(args.report))
excl = {x for x in args.exclude.split(",") if x}
pps = {k: v for k, v in rep["per_prompt_score"].items() if k not in excl}
w = sum(v == 1.0 for v in pps.values())
l_ = sum(v == 0.0 for v in pps.values())
t = sum(v == 0.5 for v in pps.values())
assert w + l_ + t == len(pps), "per_prompt_score holds a value other than 0, 0.5 and 1"
n = len(pps)


def wilson(x, n):
    p = x / n
    c = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    h = Z / (1 + Z * Z / n) * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return [round(c - h, 4), round(c + h, 4)]


def sign_p(a, b):
    m, k = a + b, min(a, b)
    return min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m) if m else 1.0


res = {"report": args.report, "excluded": sorted(excl), "excluded_present": sorted(excl & set(rep["per_prompt_score"])),
       "n": n, "wins": w, "losses": l_, "ties": t, "score": round((w + t / 2) / n, 4),
       "score_ci95": wilson(w + t / 2, n), "sign_test_p": round(sign_p(w, l_), 6)}
o = rep["overall"]
res["report_overall"] = {k: o[k] for k in ("n", "wins", "losses", "ties", "both_bad", "score", "score_ci95", "sign_test_p")}
if not excl:
    res["reproduces_overall"] = (n, w, l_) == (o["n"], o["wins"], o["losses"]) and t == o["ties"] + o.get("both_bad", 0)
print(json.dumps(res))
if args.out:
    Path(args.out).write_text(json.dumps(res, indent=1))
