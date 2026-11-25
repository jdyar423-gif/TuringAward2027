"""Fifth pass: held-out loss to the end of training (Section 6.5, Tables 5, G2, M6).

(1) Every runtime evaluation (N/midrun_eval_1.json, N/heldout_v9.jsonl: 3 checkpoints of `final`, 22 of `v9`
    to step 51,812), per domain, with the comparison set (web + chat) and all rows.
(2) `final`'s three-point line on the comparison set at every `v9` checkpoint in the three forms of
    lead4_work/scripts/heldout4.py (minutes; log tokens at equal training time with Section 6.3's rates;
    equal tokens), 95% prediction intervals with t(1 dof) = 12.706, inside or outside.
(3) The local CPU evaluations of step 51,812 (lead5_work/heldout/v9_51812_train_{on,pagesoff}.npz and
    v9_51812_post_on.npz, written by heldout_eval5.py from N/snap_final): paired differences with the same
    bootstrap over rows as lead3_work/heldout46.py (seed 20261124, 10,000 resamples) against `final` 29,722
    (lead2_work/heldout/final_29722_on.npz) and `v9` 46,178 (lead3_work/heldout/v9_46178_on.npz); pages off
    against on; the weights after preference optimisation against the trained weights; input-frequency
    quarters; bits per UTF-8 byte on the 64 web rows; local against runtime.
(4) What rose at the end: the proxy rows (user turns of the 400 prompts) and, within the chat rows, the loss
    on positions that are not assistant tokens, at 46,178 and 51,812 (local evaluations).

    taskset -c 2,3 nice -n 5 S/venv/bin/python -I -B heldout5.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
W = HERE.parent
N = HERE.parents[2]
P = N / "paper"
sys.path.insert(0, "/home/user/turingaward2027")
TQ1 = 12.706
rng = np.random.default_rng(20261124)
B = 10000

# ---------------------------------------------------------------- (1) runtime evaluations
rt, src = {}, {}
for run in json.load(open(N / "midrun_eval_1.json"))["runs"]:
    for c in run["checkpoints"]:
        rt[c["step"]] = c
        src[c["step"]] = "midrun_eval_1.json"
for line in open(N / "heldout_v9.jsonl"):
    d = json.loads(line)
    for run in d["result"]["runs"]:
        for c in run["checkpoints"]:
            rt[c["step"]] = c
            src[c["step"]] = "heldout_v9.jsonl"


def wc(c):
    bd = c["by_domain"]
    return (bd["web"]["loss"] * bd["web"]["weight"] + bd["chat"]["loss"] * bd["chat"]["weight"]) / (
        bd["web"]["weight"] + bd["chat"]["weight"])


rows = {}
for s in sorted(rt):
    c, bd = rt[s], rt[s]["by_domain"]
    rows[str(s)] = {"config": "final" if s <= 29722 else "v9", "tokens": c["tokens"],
                    "train_minutes": round(c["train_seconds"] / 60, 1), "web": round(bd["web"]["loss"], 4),
                    "chat": round(bd["chat"]["loss"], 4), "chat_assistant": round(bd["chat"]["assistant_loss"], 4),
                    "proxy": round(bd["proxy"]["loss"], 4), "web_chat": round(wc(c), 4), "all": round(c["loss"], 4),
                    "source": src[s]}


# ---------------------------------------------------------------- (2) final's line
def line(xs, ys, x0):
    n = len(xs)
    xm, ym = sum(xs) / n, sum(ys) / n
    sxx = sum((x - xm) ** 2 for x in xs)
    b = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sxx
    a = ym - b * xm
    res = [y - (a + b * x) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(e * e for e in res) / (n - 2))
    return a + b * x0, TQ1 * sd * math.sqrt(1 + 1 / n + (x0 - xm) ** 2 / sxx)


num3 = json.load(open(P / "lead3_work" / "numbers.json"))
pf = num3["proj.final_total"]["value"]
r1024, r2048, t1024 = float(pf["live1024"]), float(pf["live2048_scaled"]), float(pf["t1024_s"])
F = [s for s in sorted(rt) if s <= 29722]
assert F == [26890, 28501, 29722], F
g_tok, g_s = rt[29722]["tokens"], rt[29722]["train_seconds"]
ys = [wc(rt[s]) for s in F]
xmin = [round(rt[s]["train_seconds"] / 60, 1) for s in F]
xlt = [math.log10(rt[s]["tokens"]) for s in F]


def eqtok(ts):
    dt = ts - g_s
    return g_tok + min(dt, t1024) * r1024 + max(0.0, dt - t1024) * r2048


lines = {}
for s in sorted(rt):
    if s <= 29722:
        continue
    c = rt[s]
    v = wc(c)
    ts = c["train_seconds"]
    e = {"train_minutes": round(ts / 60, 1), "v9_web_chat": round(v, 4)}
    for name, x0, xs in (("minutes", round(ts / 60, 1), xmin), ("logtok", math.log10(eqtok(ts)), xlt),
                         ("equal_tokens", math.log10(c["tokens"]), xlt)):
        pr, hw = line(xs, ys, x0)
        e[name] = {"predicted": round(pr, 4), "gap": round(v - pr, 4), "pi95": round(hw, 4), "inside": abs(v - pr) <= hw}
        if name == "logtok":
            e[name]["final_tokens_equal_time"] = float(f"{10 ** x0:.4g}")
    lines[str(s)] = e
outside = {k: [s for s, e in lines.items() if not e[k]["inside"]] for k in ("minutes", "logtok", "equal_tokens")}


# ---------------------------------------------------------------- (3) local evaluations
def load(p):
    z = np.load(p)
    d = {k: z[k] for k in z.files if k != "meta"}
    d["meta"] = json.loads(str(z["meta"]))
    return d


FIN = load(P / "lead2_work/heldout/final_29722_on.npz")
V46 = load(P / "lead3_work/heldout/v9_46178_on.npz")
ON = load(W / "heldout/v9_51812_train_on.npz")
OFF = load(W / "heldout/v9_51812_train_pagesoff.npz")
POST = load(W / "heldout/v9_51812_post_on.npz")
for r in (V46, ON, OFF, POST):
    for dom in ("web", "chat", "proxy"):
        assert np.array_equal(r[f"{dom}/tokens"], FIN[f"{dom}/tokens"]) and np.array_equal(r[f"{dom}/w"], FIN[f"{dom}/w"])


def row_sums(r, dom, mask=None):
    ce, w = r[f"{dom}/ce"].astype(np.float64), r[f"{dom}/w"].astype(np.float64)
    if mask is not None:
        w = w * mask
    return (ce * w).sum(axis=1), w.sum(axis=1)


def asst(r):
    return (r["chat/role"] == 3).astype(np.float64)


def not_asst(r):
    return (r["chat/role"] != 3).astype(np.float64)


def loss(r, doms, mask_fn=None):
    s = w = 0.0
    for d in doms:
        a, b = row_sums(r, d, mask_fn(r) if (mask_fn and d == "chat") else None)
        s += a.sum()
        w += b.sum()
    return s / w


def boot(a, b, doms=("web", "chat"), mask_fn=None):
    sa, sb, ws = [], [], []
    for d in doms:
        m = mask_fn(a) if (mask_fn and d == "chat") else None
        x, w = row_sums(a, d, m)
        y, _ = row_sums(b, d, m)
        sa.append(x)
        sb.append(y)
        ws.append(w)
    sa, sb, ws = np.concatenate(sa), np.concatenate(sb), np.concatenate(ws)
    k = ws > 0
    sa, sb, ws = sa[k], sb[k], ws[k]
    est = (sb.sum() - sa.sum()) / ws.sum()
    idx = rng.integers(0, len(ws), size=(B, len(ws)))
    bs = (sb[idx].sum(1) - sa[idx].sum(1)) / ws[idx].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"estimate": round(float(est), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "rows": int(len(ws))}


loc = {}
for nm, r in (("final_29722", FIN), ("v9_46178_on", V46), ("v9_51812_train_on", ON), ("v9_51812_train_pagesoff", OFF),
              ("v9_51812_post_on", POST)):
    loc[nm] = {"step": r["meta"]["step"], "web": round(loss(r, ("web",)), 4), "chat": round(loss(r, ("chat",)), 4),
               "chat_assistant": round(loss(r, ("chat",), asst), 4),
               "chat_not_assistant": round(loss(r, ("chat",), not_asst), 4),
               "proxy": round(loss(r, ("proxy",)), 4), "web_chat": round(loss(r, ("web", "chat")), 4),
               "all": round(loss(r, ("web", "chat", "proxy")), 4)}
pairs = {}
for label, (a, b) in {"v9_51812_train - final_29722": (FIN, ON), "v9_51812_train - v9_46178": (V46, ON),
                      "pagesoff - on (51812 train)": (ON, OFF), "pagesoff 51812 - final_29722": (FIN, OFF),
                      "post - train (51812)": (ON, POST), "post 51812 - final_29722": (FIN, POST)}.items():
    pairs[label] = {"web+chat": boot(a, b), "web": boot(a, b, ("web",)), "chat": boot(a, b, ("chat",)),
                    "chat_assistant": boot(a, b, ("chat",), asst), "proxy": boot(a, b, ("proxy",))}
toks = np.concatenate([FIN["web/tokens"][:, :-1], FIN["chat/tokens"][:, :-1]])
ws = np.concatenate([FIN["web/w"], FIN["chat/w"]])
counts = np.bincount(toks[ws > 0].ravel(), minlength=8192)
freq = counts[toks]
valid = ws > 0
qs = np.quantile(freq[valid], [0.25, 0.5, 0.75])
bins = np.digitize(freq, qs)
strat = {}
for label, (a, b) in {"page_gain_51812": (ON, OFF), "since_graft_51812": (FIN, ON)}.items():
    ca = np.concatenate([a["web/ce"], a["chat/ce"]])
    cb = np.concatenate([b["web/ce"], b["chat/ce"]])
    d = cb - ca
    strat[label] = [{"quartile": k + 1, "positions": int((valid & (bins == k)).sum()),
                     "input_token_count_range": [int(freq[valid & (bins == k)].min()), int(freq[valid & (bins == k)].max())],
                     "mean_difference": round(float(d[valid & (bins == k)].mean()), 4)} for k in range(4)]
from bes.tokenizer import load as load_tok  # noqa: E402

Lb = load_tok().token_byte_lengths()
y = FIN["web/tokens"][:, 1:]
w = FIN["web/w"]
nbytes = float((Lb[y] * (w > 0)).sum())
bpb = {nm: round(float((r["web/ce"].astype(np.float64) * w).sum()) / math.log(2) / nbytes, 4)
       for nm, r in (("final_29722", FIN), ("v9_46178_on", V46), ("v9_51812_train_on", ON), ("v9_51812_post_on", POST))}
lvr = {}
c = rt[51812]
bd = c["by_domain"]
lvr["v9_51812_train_on"] = {d: float(f"{loss(ON, (d,)) - bd[d]['loss']:.3g}") for d in ("web", "chat", "proxy")}
lvr["v9_51812_train_on"]["all"] = float(f"{loss(ON, ('web', 'chat', 'proxy')) - c['loss']:.3g}")
lvr["v9_51812_train_on"]["chat_assistant"] = float(f"{loss(ON, ('chat',), asst) - bd['chat']['assistant_loss']:.3g}")
# proxy rows: which roles carry weight
prow = {int(k): int(v) for k, v in zip(*np.unique(FIN["proxy/role"][FIN["proxy/w"] > 0], return_counts=True))} if "proxy/role" in FIN else {}
out = {"runtime": rows, "lines": lines, "outside": outside,
       "line_assumptions": {"rate_1024": r1024, "rate_2048": r2048, "seconds_at_1024_after_graft": t1024},
       "local": loc, "paired": pairs, "frequency": strat, "bpb_web": bpb, "bpb_web_bytes": nbytes,
       "local_minus_runtime_51812": lvr, "proxy_weighted_roles": prow,
       "proxy_min": {"step": min(((s, rt[s]["by_domain"]["proxy"]["loss"]) for s in rt if s > 29722), key=lambda t: t[1])},
       "all_rows_min_v9": min(((s, rt[s]["loss"]) for s in rt if s > 29722), key=lambda t: t[1])}
(W / "heldout5.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("outside", "local", "bpb_web", "bpb_web_bytes", "local_minus_runtime_51812",
                                      "proxy_weighted_roles", "proxy_min", "all_rows_min_v9")}, indent=1))
for s, e in lines.items():
    print(s, e["train_minutes"], e["v9_web_chat"], " | ".join(
        f"{k} {e[k]['gap']:+.4f} ± {e[k]['pi95']:.4f} {'in' if e[k]['inside'] else 'OUT'}" for k in ("minutes", "logtok", "equal_tokens")),
        f"| eq-time tokens {e['logtok']['final_tokens_equal_time']:.4g}")
for k, v in pairs.items():
    print(k, json.dumps(v))
print(json.dumps(strat))
for s, r in rows.items():
    print(s, r)
