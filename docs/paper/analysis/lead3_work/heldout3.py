"""Held-out loss of every checkpoint the runtime evaluated (N/midrun_eval_1.json for `final` and the
first `v9` points, N/heldout_v9.jsonl for every later `v9` checkpoint), on the comparison set of
Table 7 (64 web + 64 chat rows, 125,880 weighted tokens; the 12 proxy rows reported apart), and
`final`'s three-point line (training minutes; log10 tokens at equal training time) extended to each
`v9` checkpoint with its 95% prediction interval (t with 1 degree of freedom = 12.706), as in
lead2_work/heldout_analysis.py. Arithmetic only; writes heldout3.json.
"""
import json
import math
from pathlib import Path

N = Path("/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad")
HERE = Path(__file__).resolve().parent
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


rows = {}
for s, c in sorted(rt.items()):
    bd = c["by_domain"]
    rows[s] = {"config": "final" if s <= 29722 else "v9", "tokens": c["tokens"],
               "train_minutes": round(c["train_seconds"] / 60, 1), "train_hours": round(c["train_seconds"] / 3600, 3),
               "web": round(bd["web"]["loss"], 4), "chat": round(bd["chat"]["loss"], 4),
               "chat_assistant": round(bd["chat"]["assistant_loss"], 4), "proxy": round(bd["proxy"]["loss"], 4),
               "web_chat": round(wc(c), 4), "all": round(c["loss"], 4),
               "weights": {k: bd[k]["weight"] for k in ("web", "chat", "proxy")}}

F = [s for s in rows if s <= 29722]
assert F == [26890, 28501, 29722], F
TQ1 = 12.706
rate_final = 281220.0          # final's achieved-rate median after the kernel switch (as in heldout_analysis.py)
graft_tokens, graft_min = 7791443968, 32320.749530553818 / 60


def line(xs, ys, x0):
    n = len(xs)
    xm, ym = sum(xs) / n, sum(ys) / n
    sxx = sum((x - xm) ** 2 for x in xs)
    b = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sxx
    a = ym - b * xm
    res = [y - (a + b * x) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(e * e for e in res) / (n - 2))
    se = sd * math.sqrt(1 + 1 / n + (x0 - xm) ** 2 / sxx)
    return a + b * x0, TQ1 * se, b


ys = [wc(rt[s]) for s in F]
xm_ = [rt[s]["train_seconds"] / 60 for s in F]
xm_r = [rows[s]["train_minutes"] for s in F]          # Table 7 used minutes rounded to 0.1
xt_ = [math.log10(rt[s]["tokens"]) for s in F]
ext = {}
for s, r in rows.items():
    if s <= 29722:
        continue
    pm, hm, bm = line(xm_r, ys, r["train_minutes"])
    pmu, hmu, _ = line(xm_, ys, rt[s]["train_seconds"] / 60)
    eq_tok = graft_tokens + (rt[s]["train_seconds"] / 60 - graft_min) * 60 * rate_final
    pt, ht, _ = line(xt_, ys, math.log10(eq_tok))
    ptm, htm, _ = line(xt_, ys, math.log10(r["tokens"]))
    ext[s] = {"v9_web_chat": r["web_chat"],
              "minutes_line": {"final_predicted": round(pm, 4), "gap": round(r["web_chat"] - pm, 4), "pi95": round(hm, 4),
                               "pi95_unrounded_minutes": round(hmu, 4)},
              "logtokens_equal_time": {"equal_time_tokens_final": float(f"{eq_tok:.4g}"), "final_predicted": round(pt, 4),
                                       "gap": round(r["web_chat"] - pt, 4), "pi95": round(ht, 4)},
              "logtokens_matched": {"final_predicted": round(ptm, 4), "gap": round(r["web_chat"] - ptm, 4), "pi95": round(htm, 4)}}
g = rows[29722]["web_chat"]
newest = max(rows)
out = {"rows": rows, "extrapolation_final_three_points": ext,
       "since_graft": {str(s): round(rows[s]["web_chat"] - g, 4) for s in rows if s > 29722},
       "newest_checkpoint": newest,
       "slope_minutes_line_per_min": round(line(xm_r, ys, 600)[2], 7)}
(HERE / "heldout3.json").write_text(json.dumps(out, indent=1))
print("step   cfg    tokens      min    web    chat   asst   proxy  web+chat  all")
for s, r in rows.items():
    print(f"{s:6d} {r['config']:5s} {r['tokens']:.4e} {r['train_minutes']:6.1f} {r['web']:.4f} {r['chat']:.4f} "
          f"{r['chat_assistant']:.4f} {r['proxy']:.4f} {r['web_chat']:.4f} {r['all']:.4f}")
for s, e in ext.items():
    print(s, "since graft", out["since_graft"][str(s)], "minutes line", e["minutes_line"], "logtok equal time",
          e["logtokens_equal_time"], "matched", e["logtokens_matched"])
