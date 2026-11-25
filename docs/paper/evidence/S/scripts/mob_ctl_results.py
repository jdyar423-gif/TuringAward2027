"""Results of the registered judge on MobileLLM-R1.5's two K = 6 control arms, for the documents:
score reports (already written by `score`), the 189-prompt subset, the paired comparison with the
released-preset set under the same judge (all 200 and the 189), the agreement diagnostics.
    python mob_ctl_results.py   -> N/peek/mob_ctl_results.json and a printed summary"""
import json
import math
from fractions import Fraction
from decimal import Decimal, ROUND_HALF_EVEN

N = "/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad"
SRC = "judge-qwen3-4b-2507-3605803b"
SEEN = set("chat-002,chat-017,math-034,instruct-034,instruct-062,writing-100,code-026,math-001,code-002,writing-001,instruct-001".split(","))
Z = 1.959963984540054


def wilson(p, n):
    c = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    h = Z / (1 + Z * Z / n) * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return c - h, c + h


def r3(fr):
    return str((Decimal(fr.numerator) / Decimal(fr.denominator)).quantize(Decimal("0.001"), rounding=ROUND_HALF_EVEN))


def f3(x):
    return str(Decimal(repr(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_EVEN))


def sign_p(a, b):
    m, k = a + b, min(a, b)
    return min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m) if m else 1.0


def pstr(p):
    return "< 0.0001" if p < 0.0001 else f"{p:.4f}" if p < 0.1 else f"{p:.2f}"


def summ(pps):
    w = sum(v == 1.0 for v in pps.values()); l_ = sum(v == 0.0 for v in pps.values()); t = len(pps) - w - l_
    n = len(pps); s = Fraction(2 * w + t, 2 * n); lo, hi = wilson(float(s), n)
    d = Fraction(w, w + l_) if w + l_ else Fraction(0); dlo, dhi = wilson(float(d), w + l_) if w + l_ else (0, 0)
    return {"n": n, "w": w, "l": l_, "t": t, "wlt": f"{w} / {l_} / {t}", "score": r3(s), "score_exact": float(s),
            "ci": f"[{f3(lo)}, {f3(hi)}]", "decisive": r3(d), "dci": f"[{f3(dlo)}, {f3(dhi)}]", "p": pstr(sign_p(w, l_)), "p_raw": sign_p(w, l_)}


def paired(a, b):
    ks = [k for k in a if k in b]
    ab = sum(a[k] > b[k] for k in ks); ba = sum(a[k] < b[k] for k in ks); same = len(ks) - ab - ba
    md = sum(a[k] - b[k] for k in ks) / len(ks)
    # per-prompt scores are 0, 0.5 or 1, so the mean difference is an exact fraction: twice_sum / (2 n)
    twice_sum = round(2 * sum(a[k] - b[k] for k in ks))
    fr = Fraction(twice_sum, 2 * len(ks))
    assert abs(float(fr) - md) < 1e-12
    return {"a_better": ab, "b_better": ba, "same": same, "mean_diff": md, "mean_diff_3": r3(fr), "mean_diff_exact": f"{fr.numerator}/{fr.denominator}",
            "p": sign_p(ab, ba), "n": len(ks)}


out = {}
rel = json.load(open(f"{N}/study/proxy/reports/ab/final-prefopt-vs-mobilellm--{SRC}.json"))
relp = rel["per_prompt_score"]
out["released"] = {"all": summ(relp), "189": summ({k: v for k, v in relp.items() if k not in SEEN})}
for tag in ("k6lp", "k6ds"):
    rep = json.load(open(f"{N}/study/proxy/reports/ab/final-prefopt-vs-mobilellm-{tag}--{SRC}.json"))
    pps = rep["per_prompt_score"]
    r = {"all": summ(pps), "189": summ({k: v for k, v in pps.items() if k not in SEEN})}
    r["by"] = {}
    for sl in rep.get("slices", rep.get("by", [])) if isinstance(rep.get("slices", rep.get("by", [])), list) else []:
        pass
    r["report_keys"] = list(rep.keys())
    for key in ("diagnostics", "judge", "judge_diagnostics"):
        if key in rep:
            r["diag"] = rep[key]
    r["overall_report"] = rep.get("overall")
    r["vs_released_all"] = paired(relp, pps)
    r["vs_released_189"] = paired({k: v for k, v in relp.items() if k not in SEEN}, {k: v for k, v in pps.items() if k not in SEEN})
    recs = [json.loads(x) for x in open(f"{N}/study/proxy/ratings/final-prefopt-vs-mobilellm-{tag}/{SRC}.jsonl")]
    r["records"] = len(recs); r["agree"] = sum(bool(x.get("agree")) for x in recs) / len(recs)
    r["first_pos"] = sum(x.get("first_position_pref", 0.0) for x in recs) / len(recs)
    r["clipped"] = sum(bool(x.get("clipped")) for x in recs)
    r["seconds_mean"] = sum(x.get("seconds", 0.0) for x in recs) / len(recs)
    out[tag] = r
json.dump(out, open(f"{N}/peek/mob_ctl_results.json", "w"), indent=1)
for tag in ("released", "k6lp", "k6ds"):
    r = out[tag]
    print(tag, "| all:", r["all"]["wlt"], r["all"]["score"], r["all"]["ci"], "decisive", r["all"]["decisive"], r["all"]["dci"], "p", r["all"]["p"],
          "| 189:", r["189"]["wlt"], r["189"]["score"], r["189"]["ci"], "p", r["189"]["p"])
    if tag != "released":
        print("   records", r["records"], "agree %.1f%%" % (100 * r["agree"]), "first-pos %+.3f" % r["first_pos"], "clipped", r["clipped"], "s/pair %.1f" % r["seconds_mean"])
        v = r["vs_released_all"]; print("   released vs this arm (our score higher against released; against this arm; same):", v["a_better"], v["b_better"], v["same"], "mean diff %+.4f" % v["mean_diff"], "p", pstr(v["p"]))
        v = r["vs_released_189"]; print("   on 189:", v["a_better"], v["b_better"], v["same"], "mean diff %+.4f" % v["mean_diff"], "p", pstr(v["p"]))
