"""Fourth pass (after REVIEW_2): the micro graft numbers that REVIEW_2 questions, recomputed from the raw
records. Arguments: the micro_graft directory, the directory holding the unpacked restart archive
(results.tar.gz of micro_graft/3M), output json. Arithmetic only.

mix = the record's own row mixture over web and chat (arm.mixture), as scripts/micro_graft_report.py.
sd with n - 1. t interval of a mean of three paired differences: t(0.975, 2 df) = 4.302653.
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT, R3, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
T2 = 4.302653


def load(p):
    return [json.loads(line) for line in open(p)]


def newest_done(p):
    fin = [x for x in load(p) if x.get("status") == "done"]
    return max(fin, key=lambda x: float(x.get("finished", 0)))


recs = {r["name"]: r for r in load(ROOT / "6M" / "results_all.jsonl")}
out = {"checks": {}}
# results_all equals each arm's own newest finished record
out["checks"]["results_all_equals_own"] = all(
    json.dumps(newest_done(ROOT / "6M" / n / "results.jsonl"), sort_keys=True) == json.dumps(r, sort_keys=True)
    for n, r in recs.items())

# restart records: the unpacked archive against the copies beside it
rest = {}
for n in ("v9_pages", "v9_pages_s1", "v9_pages_s2"):
    a = newest_done(R3 / n / "results.jsonl")
    b = newest_done(ROOT / "3M" / n / "results.jsonl")
    out["checks"][f"restart_{n}_archive_equals_copy"] = json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    rest[a["arm"]["seed"]] = a


def mixw(rec):
    m = dict(rec["arm"]["mixture"]) if isinstance(rec["arm"]["mixture"][0], list) else rec["arm"]["mixture"]
    return m


def mix_of(rec, ev):
    mx = rec["arm"]["mixture"]
    # mixture is a list of [domain, weight] pairs
    w = {d: float(x) for d, x in mx}
    tot = sum(w[d] for d in w if d in ev and d != "assistant")
    return sum(w[d] * ev[d]["loss"] for d in w if d in ev and d != "assistant") / tot


def curve(rec):
    return {int(c["tokens"]): c for c in rec["curve"]}


fam = {}
for n, r in recs.items():
    f = n.rsplit("_s", 1)[0] if n.endswith(("_s1", "_s2")) else n
    fam.setdefault(f, {})[r["arm"]["seed"]] = r
S = (0, 1, 2)
END = max(curve(recs["ref"]))
TOK = sorted(curve(recs["ref"]))


def L(f, s, t, dom="mix"):
    r = fam[f][s] if f != "restart" else rest[s]
    c = curve(r)[t]
    return mix_of(r, c["eval"]) if dom == "mix" else c["eval"][dom]["loss"]


def ms(xs):
    return st.mean(xs), st.stdev(xs)


res = {"mixture_ref": recs["ref"]["arm"]["mixture"], "end_tokens": END, "checkpoints": TOK}
res["loss_end"] = {f: {"per_seed": [round(L(f, s, END), 6) for s in S], "mean": round(ms([L(f, s, END) for s in S])[0], 6),
                       "sd": round(ms([L(f, s, END) for s in S])[1], 6)} for f in ("ref", "v9_pages", "graft_25", "graft_50", "graft_75")}
ref_end = [L("ref", s, END) for s in S]
res["ref_spread"] = {"sd": round(st.stdev(ref_end), 6), "sd_share": round(st.stdev(ref_end) / st.mean(ref_end), 5),
                     "range": round(max(ref_end) - min(ref_end), 6),
                     "range_share": round((max(ref_end) - min(ref_end)) / st.mean(ref_end), 5)}
gain = [L("ref", s, END) - L("v9_pages", s, END) for s in S]
res["page_gain_ref_minus_v9pages"] = {"per_seed": [round(x, 6) for x in gain], "mean": round(st.mean(gain), 6)}
d25 = [L("graft_25", s, END) - L("v9_pages", s, END) for s in S]
m, sd = ms(d25)
res["graft25_minus_v9pages"] = {"per_seed": [round(x, 6) for x in d25], "mean": round(m, 6), "sd": round(sd, 6),
                                "t95_halfwidth": round(T2 * sd / math.sqrt(3), 6),
                                "t95_halfwidth_over_page_gain": round(T2 * sd / math.sqrt(3) / st.mean(gain), 3)}
sh = {}
for f in ("graft_25", "graft_50", "graft_75"):
    num = [L("ref", s, END) - L(f, s, END) for s in S]
    per = [a / b for a, b in zip(num, gain)]
    sh[f] = {"ratio_of_means": round(st.mean(num) / st.mean(gain), 4), "per_seed": [round(x, 4) for x in per],
             "min": round(min(per), 4), "max": round(max(per), 4), "sd_per_seed": round(st.stdev(per), 4)}
res["shares"] = sh
# gain over ref one checkpoint (1.5M tokens) after each graft, and the learning-rate factor carried
after = {}
for f in ("graft_25", "graft_50", "graft_75"):
    g = fam[f][0]["graft"]
    t_after = min(t for t in TOK if t > int(g["tokens"]))
    xs = [L(f, s, t_after) - L("ref", s, t_after) for s in S]
    after[f] = {"graft_tokens": int(g["tokens"]), "eval_tokens": t_after, "mean": round(st.mean(xs), 5),
                "sd": round(st.stdev(xs), 5), "lr_factor": [fam[f][s]["graft"]["carried"]["lr_factor"] for s in S]}
first = [L("v9_pages", s, TOK[0]) - L("ref", s, TOK[0]) for s in S]
after["v9_pages_first_checkpoint"] = {"eval_tokens": TOK[0], "mean": round(st.mean(first), 5), "sd": round(st.stdev(first), 5)}
res["gain_after_graft"] = after
# the graft checks: aggregate training-loss check and per-row held-out check
chk = {}
for f in ("graft_25", "graft_50", "graft_75"):
    for s in S:
        g = fam[f][s]["graft"]
        chk[f"{f}_s{s}"] = {"check_keys": sorted(g["check"].keys()), "check_difference": g["check"]["difference"],
                            "check_batch": g["check"]["batch"],
                            "heldout_max_row_difference": {d: g["heldout"][d]["max_row_difference"] for d in g["heldout"]},
                            "heldout_difference": {d: g["heldout"][d]["difference"] for d in g["heldout"]}}
res["graft_checks"] = chk
# restart arms
rr = {}
for s in S:
    r = rest[s]
    c = curve(r)
    te = max(c)
    rr[s] = {"steps_run": r["steps_run"], "tokens_end": te, "budget_tokens": r["tokens"],
             "horizon_tokens": r["config"]["optim"]["horizon_tokens"], "finished": r["finished"],
             "loss_end_mix": round(mix_of(r, c[te]["eval"]), 6),
             "web": round(c[te]["eval"]["web"]["loss"], 6), "chat": round(c[te]["eval"]["chat"]["loss"], 6),
             "eval_tokens": sorted(c), "config_hash": r["config_hash"],
             "v9pages6M_config_hash": fam["v9_pages"][s]["config_hash"]}
res["restart"] = rr
TE = rr[0]["tokens_end"]
rl = [rr[s]["loss_end_mix"] for s in S]
comp = {}
for f in ("graft_50", "ref", "v9_pages", "graft_25", "graft_75"):
    xs = [rl[s] - L(f, s, END) for s in S]
    m, sd = ms(xs)
    comp[f"restart_minus_{f}_6M"] = {"per_seed": [round(x, 6) for x in xs], "mean": round(m, 6), "sd": round(sd, 6),
                                      "min": round(min(xs), 6), "max": round(max(xs), 6),
                                      "over_page_gain": round(m / st.mean(gain), 3)}
xs = [rl[s] - L("v9_pages", s, TE) for s in S]
comp["restart_minus_v9pages_at_same_tokens"] = {"per_seed": [round(x, 6) for x in xs], "mean": round(st.mean(xs), 6),
                                                "sd": round(st.stdev(xs), 6)}
xs = [rl[s] - L("ref", s, TE) for s in S]
comp["restart_minus_ref_at_same_tokens"] = {"per_seed": [round(x, 6) for x in xs], "mean": round(st.mean(xs), 6)}
xs = [L("graft_50", s, END) - L("ref", s, END) for s in S]
comp["graft50_minus_ref_6M"] = {"per_seed": [round(x, 6) for x in xs], "mean": round(st.mean(xs), 6)}
res["restart_comparisons"] = comp
res["checks"] = out["checks"]
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps(res["checks"]))
for k in ("loss_end", "ref_spread", "page_gain_ref_minus_v9pages", "graft25_minus_v9pages", "shares",
          "gain_after_graft", "restart_comparisons"):
    print(k, json.dumps(res[k]))
for s in S:
    print("restart seed", s, {k: v for k, v in rr[s].items()})
print("graft check example", json.dumps(chk["graft_50_s0"]))
