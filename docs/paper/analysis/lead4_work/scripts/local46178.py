"""Fourth pass (REVIEW_2 item 12): the local CPU evaluation of step 46,178 (pages on and off;
lead3_work/heldout/v9_46178_{on,pagesoff}.npz, written by lead3_work/heldout_eval.py) as a summary in the
format of lead2_work/heldout_analysis.json ("summary"), so that figures/make_data.py can draw it in
Figure 5 (F8) next to the earlier local evaluations. tokens and training minutes from the runtime's record
(N/heldout_v9.jsonl). Arguments: scratchpad dir, output json.
"""
import json
import sys
from pathlib import Path

import numpy as np

N, OUT = Path(sys.argv[1]), Path(sys.argv[2])
H = N / "paper" / "lead3_work" / "heldout"
rt = {}
for line in open(N / "heldout_v9.jsonl"):
    d = json.loads(line)
    for run in d["result"]["runs"]:
        for c in run["checkpoints"]:
            rt[c["step"]] = c


def load(p):
    z = np.load(p)
    return {k: z[k] for k in z.files if k != "meta"}


def L(r, doms, asst=False):
    s = w = 0.0
    for dd in doms:
        ce, ww = r[f"{dd}/ce"].astype(np.float64), r[f"{dd}/w"].astype(np.float64)
        if asst:
            ww = ww * (r["chat/role"] == 3)
        s += (ce * ww).sum()
        w += ww.sum()
    return s / w


c = rt[46178]
summary = {}
for variant, fn in (("on", "v9_46178_on.npz"), ("pagesoff", "v9_46178_pagesoff.npz")):
    r = load(H / fn)
    summary[f"v9_46178_{variant}"] = {"web": L(r, ("web",)), "chat": L(r, ("chat",)), "proxy": L(r, ("proxy",)),
                                       "chat_assistant": L(r, ("chat",), asst=True),
                                       "web_chat": L(r, ("web", "chat")), "all": L(r, ("web", "chat", "proxy")),
                                       "step": 46178, "tokens": c["tokens"],
                                       "train_minutes": round(c["train_seconds"] / 60, 1)}
OUT.write_text(json.dumps({"summary": summary, "source": [str(H / "v9_46178_on.npz"), str(H / "v9_46178_pagesoff.npz")],
                           "runtime_all_46178": c["loss"]}, indent=1))
print(json.dumps(summary, indent=1))
print("runtime all", c["loss"], "local on all - runtime", summary["v9_46178_on"]["all"] - c["loss"])
