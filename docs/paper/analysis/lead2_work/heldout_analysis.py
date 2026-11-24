"""Review items 35, 36 and 31 (our side): held-out loss across the graft with uncertainty, the
pages-off decomposition, a stratification by input-token frequency, and our bits per byte on the
web rows. Reads the per-position losses of heldout_eval.py (heldout/*.npz) and the runtime's
aggregate numbers (N/midrun_eval_1.json, N/heldout_v9.jsonl). Arithmetic; writes heldout_analysis.json.

Comparison set: web + chat rows (128 rows, 125,880 weighted tokens); the proxy rows hold the user
turns of all 400 evaluation prompts, test split included, and are reported apart."""
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
N = HERE.parents[1]
sys.path.insert(0, "/home/user/turingaward2027")
rng = np.random.default_rng(20261124)
B = 10000
out = {}


def load(name):
    p = HERE / "heldout" / f"{name}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    m = json.loads(str(z["meta"]))
    d = {k: z[k] for k in z.files if k != "meta"}
    d["meta"] = m
    return d


runs = {}
for f in sorted(glob.glob(str(HERE / "heldout" / "*.npz"))):
    n = os.path.basename(f)[:-4]
    runs[n] = load(n)
names = sorted(runs)
print("evaluations:", names)

# rows must be identical across evaluations
ref = runs[names[0]]
for n in names:
    for dom in ("web", "chat", "proxy"):
        assert np.array_equal(runs[n][f"{dom}/tokens"], ref[f"{dom}/tokens"]), (n, dom)
        assert np.array_equal(runs[n][f"{dom}/w"], ref[f"{dom}/w"]), (n, dom)
out["rows_identical"] = True


def row_sums(r, dom, mask=None):
    ce, w = r[f"{dom}/ce"].astype(np.float64), r[f"{dom}/w"].astype(np.float64)
    if mask is not None:
        w = w * mask
    return (ce * w).sum(axis=1), w.sum(axis=1)


def domain_loss(r, dom, mask=None):
    s, w = row_sums(r, dom, mask)
    return float(s.sum() / w.sum())


def asst_mask(r):
    return (r["chat/role"] == 3).astype(np.float64)


def summary(r):
    s = {d: domain_loss(r, d) for d in ("web", "chat", "proxy")}
    s["chat_assistant"] = domain_loss(r, "chat", asst_mask(r))
    sw, ww = row_sums(r, "web")
    sc, wc = row_sums(r, "chat")
    s["web_chat"] = float((sw.sum() + sc.sum()) / (ww.sum() + wc.sum()))
    sp, wp = row_sums(r, "proxy")
    s["all"] = float((sw.sum() + sc.sum() + sp.sum()) / (ww.sum() + wc.sum() + wp.sum()))
    led = r["meta"].get("ledger") or {}
    s["step"] = r["meta"]["step"]
    s["tokens"] = led.get("tokens")
    s["train_minutes"] = round((led.get("seconds") or {}).get("train", 0) / 60, 1)
    return s


out["summary"] = {n: summary(r) for n, r in runs.items()}
for n, s in out["summary"].items():
    print(f"{n:22s} step {s['step']:>6} tokens {s['tokens']:.4e} min {s['train_minutes']:6.1f} all {s['all']:.4f} "
          f"web {s['web']:.4f} chat {s['chat']:.4f} asst {s['chat_assistant']:.4f} proxy {s['proxy']:.4f} web+chat {s['web_chat']:.4f}")

# runtime aggregates (same rows, same arithmetic, other host)
rt = {}
mr = json.load(open(N / "midrun_eval_1.json"))
for run in mr["runs"]:
    for c in run["checkpoints"]:
        rt[c["step"]] = c
for line in open(N / "heldout_v9.jsonl"):
    d = json.loads(line)
    for run in d["result"]["runs"]:
        for c in run["checkpoints"]:
            rt[c["step"]] = c


def rt_webchat(c):
    bd = c["by_domain"]
    return (bd["web"]["loss"] * bd["web"]["weight"] + bd["chat"]["loss"] * bd["chat"]["weight"]) / (
        bd["web"]["weight"] + bd["chat"]["weight"])


out["runtime"] = {s: {"tokens": c["tokens"], "train_minutes": round(c["train_seconds"] / 60, 1), "all": c["loss"],
                      "web": c["by_domain"]["web"]["loss"], "chat": c["by_domain"]["chat"]["loss"],
                      "chat_assistant": c["by_domain"]["chat"]["assistant_loss"],
                      "proxy": c["by_domain"]["proxy"]["loss"], "web_chat": rt_webchat(c)} for s, c in sorted(rt.items())}
# reproduction: local against runtime for the same checkpoints
rep = {}
for n, s in out["summary"].items():
    if n.endswith("_on") and s["step"] in rt:
        rep[n] = {"local_all": round(s["all"], 6), "runtime_all": round(rt[s["step"]]["loss"], 6),
                  "difference": s["all"] - rt[s["step"]]["loss"]}
out["reproduction"] = rep
print("reproduction:", json.dumps(rep))


def boot_diff(a, b, doms=("web", "chat"), mask_fn=None):
    """weighted mean (b - a) over the rows of doms, with a 95% bootstrap interval over rows"""
    sa, sb, ws = [], [], []
    for dom in doms:
        m = mask_fn(a) if (mask_fn and dom == "chat") else None
        x, w = row_sums(a, dom, m)
        y, _ = row_sums(b, dom, m)
        sa.append(x); sb.append(y); ws.append(w)
    sa, sb, ws = np.concatenate(sa), np.concatenate(sb), np.concatenate(ws)
    keep = ws > 0
    sa, sb, ws = sa[keep], sb[keep], ws[keep]
    est = (sb.sum() - sa.sum()) / ws.sum()
    idx = rng.integers(0, len(ws), size=(B, len(ws)))
    bs = (sb[idx].sum(1) - sa[idx].sum(1)) / ws[idx].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"estimate": float(est), "ci95": [float(lo), float(hi)], "rows": int(len(ws))}


pairs = {}
F = runs.get("final_29722_on")
for vn in ("v9_newest_on", "v9_35294_on"):
    if F is not None and vn in runs:
        pairs[f"{vn} - final_29722_on"] = {"web+chat": boot_diff(F, runs[vn]),
                                           "web": boot_diff(F, runs[vn], ("web",)),
                                           "chat": boot_diff(F, runs[vn], ("chat",)),
                                           "chat_assistant": boot_diff(F, runs[vn], ("chat",), asst_mask)}
for base in ("v9_newest", "v9_35294"):
    on = runs.get(f"{base}_on")
    for var in ("pagesoff", "pageoff", "gainoff"):
        off = runs.get(f"{base}_{var}")
        if on is not None and off is not None:
            pairs[f"{base}_{var} - {base}_on"] = {"web+chat": boot_diff(on, off), "web": boot_diff(on, off, ("web",)),
                                                  "chat": boot_diff(on, off, ("chat",)),
                                                  "chat_assistant": boot_diff(on, off, ("chat",), asst_mask)}
    off = runs.get(f"{base}_pagesoff")
    if F is not None and off is not None:
        pairs[f"{base}_pagesoff - final_29722_on"] = {"web+chat": boot_diff(F, off), "web": boot_diff(F, off, ("web",)),
                                                      "chat": boot_diff(F, off, ("chat",))}
out["paired"] = pairs
for k, v in pairs.items():
    print(k, {d: (round(x["estimate"], 4), [round(c, 4) for c in x["ci95"]]) for d, x in v.items()})

# final's trend: web+chat loss against log10(tokens), and against training minutes, extrapolated
pts = {}
for n, s in out["summary"].items():
    if n.startswith("final_") and n.endswith("_on"):
        pts[s["step"]] = (s["tokens"], s["train_minutes"], s["web_chat"], "local")
for st_, c in out["runtime"].items():
    if st_ <= 29722 and st_ not in pts:
        pts[st_] = (c["tokens"], c["train_minutes"], c["web_chat"], "runtime")
pts = dict(sorted(pts.items()))
out["final_points"] = {str(k): {"tokens": v[0], "train_minutes": v[1], "web_chat": v[2], "where": v[3]} for k, v in pts.items()}


def linfit(xs, ys):
    n = len(xs)
    xm, ym = sum(xs) / n, sum(ys) / n
    sxx = sum((x - xm) ** 2 for x in xs)
    b = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sxx
    a = ym - b * xm
    res = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = n - 2
    s = math.sqrt(sum(e * e for e in res) / dof) if dof > 0 else float("nan")
    return a, b, s, xm, sxx, dof


TQ = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
rate_final = 281220.0      # achieved-rate median of final after the kernel switch (numbers.json rate.kernel_to_graft)
graft_tokens, graft_min = 7791443968, 32320.749530553818 / 60
preds = {}
for vn in ("v9_newest_on", "v9_35294_on"):
    if vn not in runs:
        continue
    s = out["summary"][vn]
    eq_time_tokens = graft_tokens + (s["train_minutes"] - graft_min) * 60 * rate_final
    for label, keys in (("last4_logtokens", [k for k in pts if k >= 22613]),
                        ("last3_logtokens", [k for k in pts if k >= 26890]),
                        ("last4_minutes", [k for k in pts if k >= 22613])):
        if len(keys) < 3:
            continue
        if label.endswith("minutes"):
            xs = [pts[k][1] for k in keys]; x0 = s["train_minutes"]
        else:
            xs = [math.log10(pts[k][0]) for k in keys]; x0 = math.log10(eq_time_tokens)
        ys = [pts[k][2] for k in keys]
        a, b, sd, xm, sxx, dof = linfit(xs, ys)
        pred = a + b * x0
        se = sd * math.sqrt(1 + 1 / len(xs) + (x0 - xm) ** 2 / sxx)
        half = TQ.get(dof, 2.0) * se
        preds[f"{vn}:{label}"] = {"points": keys, "x0": x0, "final_predicted": pred, "pi95_halfwidth": half,
                                  "v9_measured": s["web_chat"], "gap_v9_minus_final": s["web_chat"] - pred,
                                  "dof": dof, "equal_time_tokens_final": eq_time_tokens}
    # matched tokens: final's line at v9's own token count
    keys = [k for k in pts if k >= 22613]
    xs = [math.log10(pts[k][0]) for k in keys]; ys = [pts[k][2] for k in keys]
    a, b, sd, xm, sxx, dof = linfit(xs, ys)
    x0 = math.log10(s["tokens"])
    pred = a + b * x0
    se = sd * math.sqrt(1 + 1 / len(xs) + (x0 - xm) ** 2 / sxx)
    preds[f"{vn}:last4_logtokens_matched_tokens"] = {"final_predicted": pred, "pi95_halfwidth": TQ.get(dof, 2.0) * se,
                                                     "v9_measured": s["web_chat"], "gap_v9_minus_final": s["web_chat"] - pred,
                                                     "dof": dof, "tokens": s["tokens"]}
out["extrapolation"] = preds
for k, v in preds.items():
    print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk not in ("points",)})

# frequency stratification of the page gain (pages off - on) and of the change since the graft
toks = np.concatenate([ref["web/tokens"][:, :-1], ref["chat/tokens"][:, :-1]])
ws = np.concatenate([ref["web/w"], ref["chat/w"]])
counts = np.bincount(toks[ws > 0].ravel(), minlength=8192)
freq = counts[toks]
strat = {}
for label, (a_name, b_name) in {"page_gain_34166": ("v9_newest_on", "v9_newest_pagesoff"),
                                "since_graft_34166": ("final_29722_on", "v9_newest_on")}.items():
    if a_name not in runs or b_name not in runs:
        continue
    ca = np.concatenate([runs[a_name]["web/ce"], runs[a_name]["chat/ce"]])
    cb = np.concatenate([runs[b_name]["web/ce"], runs[b_name]["chat/ce"]])
    d = (cb - ca)
    valid = ws > 0
    qs = np.quantile(freq[valid], [0.25, 0.5, 0.75])
    bins = np.digitize(freq, qs)
    rows = []
    for k in range(4):
        m = valid & (bins == k)
        rows.append({"quartile": k + 1, "positions": int(m.sum()),
                     "input_token_count_range": [int(freq[m].min()), int(freq[m].max())],
                     "mean_difference": float(d[m].mean())})
    strat[label] = {"difference": f"{b_name} - {a_name}", "by_input_frequency_quartile": rows}
    print(label, [(r["quartile"], r["input_token_count_range"], round(r["mean_difference"], 4)) for r in rows])
out["frequency"] = strat

# our bits per byte on the web rows (ordinary targets; every web target is ordinary)
from bes.tokenizer import load as load_tok  # noqa: E402
tok = load_tok()
L = tok.token_byte_lengths()
y = ref["web/tokens"][:, 1:]
w = ref["web/w"]
nbytes = float((L[y] * (w > 0)).sum())
bpb = {}
for n, r in runs.items():
    nats = float((r["web/ce"].astype(np.float64) * w).sum())
    bpb[n] = {"bits_per_byte": nats / math.log(2) / nbytes, "bytes": nbytes}
out["bpb_web_ours"] = bpb
print("bpb web:", {k: round(v["bits_per_byte"], 4) for k, v in bpb.items()})
(HERE / "heldout_analysis.json").write_text(json.dumps(out, indent=1, default=float))
