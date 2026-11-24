"""The pages-off decomposition and the paired differences of Table 7 at the newest evaluated `v9`
checkpoint (step 46,178, read from the snapshot of 16:28 UTC by heldout_eval.py), with the same
bootstrap over rows (seed 20261124, 10,000 resamples) and the same frequency strata as
lead2_work/heldout_analysis.py, and our bits per byte on the web rows. Arithmetic; writes heldout46.json.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
L2 = HERE.parent / "lead2_work"
sys.path.insert(0, "/home/user/turingaward2027")
rng = np.random.default_rng(20261124)
B = 10000


def load(p):
    z = np.load(p)
    d = {k: z[k] for k in z.files if k != "meta"}
    d["meta"] = json.loads(str(z["meta"]))
    return d


F = load(L2 / "heldout/final_29722_on.npz")
V = load(L2 / "heldout/v9_35294_on.npz")
ON = load(HERE / "heldout/v9_46178_on.npz")
OFF = load(HERE / "heldout/v9_46178_pagesoff.npz")
for r in (V, ON, OFF):
    for dom in ("web", "chat", "proxy"):
        assert np.array_equal(r[f"{dom}/tokens"], F[f"{dom}/tokens"]) and np.array_equal(r[f"{dom}/w"], F[f"{dom}/w"])


def row_sums(r, dom, mask=None):
    ce, w = r[f"{dom}/ce"].astype(np.float64), r[f"{dom}/w"].astype(np.float64)
    if mask is not None:
        w = w * mask
    return (ce * w).sum(axis=1), w.sum(axis=1)


def asst(r):
    return (r["chat/role"] == 3).astype(np.float64)


def loss(r, doms, mask_fn=None):
    s = w = 0.0
    for d in doms:
        a, b = row_sums(r, d, mask_fn(r) if (mask_fn and d == "chat") else None)
        s += a.sum(); w += b.sum()
    return s / w


def boot(a, b, doms=("web", "chat"), mask_fn=None):
    sa, sb, ws = [], [], []
    for d in doms:
        m = mask_fn(a) if (mask_fn and d == "chat") else None
        x, w = row_sums(a, d, m)
        y, _ = row_sums(b, d, m)
        sa.append(x); sb.append(y); ws.append(w)
    sa, sb, ws = np.concatenate(sa), np.concatenate(sb), np.concatenate(ws)
    k = ws > 0
    sa, sb, ws = sa[k], sb[k], ws[k]
    est = (sb.sum() - sa.sum()) / ws.sum()
    idx = rng.integers(0, len(ws), size=(B, len(ws)))
    bs = (sb[idx].sum(1) - sa[idx].sum(1)) / ws[idx].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"estimate": round(float(est), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "rows": int(len(ws))}


out = {"losses": {}}
for nm, r in (("final_29722", F), ("v9_35294", V), ("v9_46178_on", ON), ("v9_46178_pagesoff", OFF)):
    out["losses"][nm] = {"web": round(loss(r, ("web",)), 4), "chat": round(loss(r, ("chat",)), 4),
                         "chat_assistant": round(loss(r, ("chat",), asst), 4), "proxy": round(loss(r, ("proxy",)), 4),
                         "web_chat": round(loss(r, ("web", "chat")), 4), "step": r["meta"]["step"]}
pairs = {}
for label, (a, b) in {"v9_46178_on - final_29722": (F, ON), "v9_46178_pagesoff - v9_46178_on": (ON, OFF),
                      "v9_46178_pagesoff - final_29722": (F, OFF), "v9_46178_on - v9_35294": (V, ON)}.items():
    pairs[label] = {"web+chat": boot(a, b), "web": boot(a, b, ("web",)), "chat": boot(a, b, ("chat",)),
                    "chat_assistant": boot(a, b, ("chat",), asst)}
out["paired"] = pairs
toks = np.concatenate([F["web/tokens"][:, :-1], F["chat/tokens"][:, :-1]])
ws = np.concatenate([F["web/w"], F["chat/w"]])
counts = np.bincount(toks[ws > 0].ravel(), minlength=8192)
freq = counts[toks]
valid = ws > 0
qs = np.quantile(freq[valid], [0.25, 0.5, 0.75])
bins = np.digitize(freq, qs)
strat = {}
for label, (a, b) in {"page_gain_46178": (ON, OFF), "since_graft_46178": (F, ON)}.items():
    ca = np.concatenate([a["web/ce"], a["chat/ce"]]); cb = np.concatenate([b["web/ce"], b["chat/ce"]])
    d = cb - ca
    strat[label] = [{"quartile": k + 1, "positions": int((valid & (bins == k)).sum()),
                     "input_token_count_range": [int(freq[valid & (bins == k)].min()), int(freq[valid & (bins == k)].max())],
                     "mean_difference": round(float(d[valid & (bins == k)].mean()), 4)} for k in range(4)]
out["frequency"] = strat
from bes.tokenizer import load as load_tok  # noqa: E402
Lb = load_tok().token_byte_lengths()
y = F["web/tokens"][:, 1:]
w = F["web/w"]
nbytes = float((Lb[y] * (w > 0)).sum())
out["bpb_web"] = {nm: round(float((r["web/ce"].astype(np.float64) * w).sum()) / math.log(2) / nbytes, 4)
                  for nm, r in (("final_29722", F), ("v9_35294", V), ("v9_46178_on", ON))}
out["bpb_web_bytes"] = nbytes
(HERE / "heldout46.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
