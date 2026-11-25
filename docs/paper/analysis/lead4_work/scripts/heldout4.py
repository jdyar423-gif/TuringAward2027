"""Fourth pass (after REVIEW_2, items 1, 7, 21): `final`'s three-point line on the comparison set (64 web +
64 chat rows) extended to every `v9` checkpoint, in three forms, with 95% prediction intervals
(t, 1 degree of freedom = 12.706), and which checkpoints lie inside them:

  minutes      x = training minutes rounded to 0.1 (as Table 5 and lead3_work/heldout3.py);
  logtok_A     x = log10 of `final`'s tokens at equal training time, `final` at 281,220 tokens/s throughout
               (heldout3.py's assumption: its achieved-rate median at 1,024 tokens);
  logtok_B     the same with Section 6.3's assumption (numbers.json proj.final_total): 281,399 tokens/s (live
               median, steps 25,300-29,700) for the first 10,061 s after the graft restart, then 252,972 (its
               2,048-token rate, scaled from the synthetic steps);
  equal_tokens x = log10 of `v9`'s own tokens.

Also: `final`'s tokens per training second between its own evaluated checkpoints, and the local CPU
evaluations (npz) against the runtime's records, per domain. Arguments: scratchpad dir, output json.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

N, OUT = Path(sys.argv[1]), Path(sys.argv[2])
P = N / "paper"
TQ1 = 12.706
rt = {}
for run in json.load(open(N / "midrun_eval_1.json"))["runs"]:
    for c in run["checkpoints"]:
        rt[c["step"]] = c
for line in open(N / "heldout_v9.jsonl"):
    d = json.loads(line)
    for run in d["result"]["runs"]:
        for c in run["checkpoints"]:
            rt[c["step"]] = c


def wc(c):
    bd = c["by_domain"]
    return (bd["web"]["loss"] * bd["web"]["weight"] + bd["chat"]["loss"] * bd["chat"]["weight"]) / (
        bd["web"]["weight"] + bd["chat"]["weight"])


def line(xs, ys, x0):
    n = len(xs)
    xm, ym = sum(xs) / n, sum(ys) / n
    sxx = sum((x - xm) ** 2 for x in xs)
    b = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sxx
    a = ym - b * xm
    res = [y - (a + b * x) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(e * e for e in res) / (n - 2))
    return a + b * x0, TQ1 * sd * math.sqrt(1 + 1 / n + (x0 - xm) ** 2 / sxx)


num = json.load(open(P / "lead3_work" / "numbers.json"))
pf = num["proj.final_total"]["value"]
r1024, r2048, t1024 = float(pf["live1024"]), float(pf["live2048_scaled"]), float(pf["t1024_s"])
F = [s for s in sorted(rt) if s <= 29722]
assert F == [26890, 28501, 29722], F
g_tok, g_s = rt[29722]["tokens"], rt[29722]["train_seconds"]
ys = [wc(rt[s]) for s in F]
xmin = [round(rt[s]["train_seconds"] / 60, 1) for s in F]
xlt = [math.log10(rt[s]["tokens"]) for s in F]
out = {"assumptions": {"logtok_A_rate": 281220.0, "logtok_B": {"rate_1024": r1024, "rate_2048": r2048,
                                                                 "seconds_at_1024_after_graft": t1024},
                       "graft_tokens": g_tok, "graft_train_seconds": g_s},
       "final_rate_between_checkpoints": {
           f"{a}-{b}": round((rt[b]["tokens"] - rt[a]["tokens"]) / (rt[b]["train_seconds"] - rt[a]["train_seconds"]), 1)
           for a, b in ((26890, 28501), (28501, 29722), (26890, 29722))},
       "points": {}}


def eqtok_B(ts):
    dt = ts - g_s
    return g_tok + min(dt, t1024) * r1024 + max(0.0, dt - t1024) * r2048


for s in sorted(rt):
    if s <= 29722:
        continue
    c = rt[s]
    v = wc(c)
    ts = c["train_seconds"]
    e = {"tokens": c["tokens"], "train_minutes": round(ts / 60, 1), "v9_web_chat": round(v, 4)}
    for name, x0, xs in (("minutes", round(ts / 60, 1), xmin),
                         ("logtok_A", math.log10(g_tok + (ts - g_s) * 281220.0), xlt),
                         ("logtok_B", math.log10(eqtok_B(ts)), xlt),
                         ("equal_tokens", math.log10(c["tokens"]), xlt)):
        pr, hw = line(xs, ys, x0)
        e[name] = {"predicted": round(pr, 4), "gap": round(v - pr, 4), "pi95": round(hw, 4),
                   "inside": abs(v - pr) <= hw, "margin": round(hw - abs(v - pr), 4)}
        if name.startswith("logtok"):
            e[name]["final_tokens_equal_time"] = float(f"{10 ** x0:.4g}")
    out["points"][str(s)] = e
out["outside"] = {k: [s for s, e in out["points"].items() if not e[k]["inside"]]
                  for k in ("minutes", "logtok_A", "logtok_B", "equal_tokens")}
out["since_graft_46178"] = round(wc(rt[46178]) - wc(rt[29722]), 4)
out["chat_assistant"] = {s: round(rt[s]["by_domain"]["chat"]["assistant_loss"], 4) for s in (36422, 46178)}
out["web"] = {s: round(rt[s]["by_domain"]["web"]["loss"], 4) for s in (36422, 46178)}


# local CPU evaluations against the runtime's records
def load(p):
    z = np.load(p)
    d = {k: z[k] for k in z.files if k != "meta"}
    d["meta"] = json.loads(str(z["meta"]))
    return d


def L(r, doms):
    s = w = 0.0
    for dd in doms:
        ce, ww = r[f"{dd}/ce"].astype(np.float64), r[f"{dd}/w"].astype(np.float64)
        s += (ce * ww).sum()
        w += ww.sum()
    return s / w


loc = {}
for path in sorted((P / "lead2_work" / "heldout").glob("*_on.npz")) + [P / "lead3_work" / "heldout" / "v9_46178_on.npz"]:
    r = load(path)
    m = r["meta"]
    step = m.get("step") or m.get("checkpoint_step")
    if step is None:
        for k, v in m.items():
            if "step" in k and isinstance(v, int):
                step = v
    if step not in rt:
        loc[path.name] = {"meta_keys": sorted(m.keys())[:20], "step": step}
        continue
    c = rt[step]
    bd = c["by_domain"]
    loc[path.name] = {"step": step,
                      "rows": {dd: int(r[f"{dd}/ce"].shape[0]) for dd in ("web", "chat", "proxy")},
                      "rows_with_weight": {dd: int((r[f"{dd}/w"].sum(axis=1) > 0).sum()) for dd in ("web", "chat", "proxy")},
                      "local_minus_runtime": {"web": float(f"{L(r, ('web',)) - bd['web']['loss']:.3g}"),
                                              "chat": float(f"{L(r, ('chat',)) - bd['chat']['loss']:.3g}"),
                                              "proxy": float(f"{L(r, ('proxy',)) - bd['proxy']['loss']:.3g}"),
                                              "all": float(f"{L(r, ('web', 'chat', 'proxy')) - c['loss']:.3g}")}}
out["local_vs_runtime"] = loc
OUT.write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("assumptions", "final_rate_between_checkpoints", "outside", "since_graft_46178")}, indent=1))
for s, e in out["points"].items():
    print(s, e["train_minutes"], e["v9_web_chat"],
          " | ".join(f"{k} {e[k]['gap']:+.4f} ± {e[k]['pi95']:.4f} {'in' if e[k]['inside'] else 'OUT'}"
                     for k in ("minutes", "logtok_A", "logtok_B", "equal_tokens")),
          f"| eq-time tokens A {e['logtok_A']['final_tokens_equal_time']:.4g} B {e['logtok_B']['final_tokens_equal_time']:.4g}")
print(json.dumps(loc, indent=1))
