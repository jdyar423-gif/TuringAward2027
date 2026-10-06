import os
import sys, json, math, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "")
torch.set_num_threads(1)
from countmem import CountMemory, doc_ids, load_tok_bytes, features
from mixture import fit_gate
from train import load_data
SP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/experiments/cache/"
meta, toks = load_data(4096); tb = load_tok_bytes(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/data/bpe4096")
docs_tr = doc_ids(toks["train"].numpy(), tb); docs_va = doc_ids(toks["valid"].numpy(), tb)
cm = CountMemory(toks["train"].numpy(), docs_tr, K=8, Kd=6); st = cm.stats(toks["valid"].numpy(), docs_va)
Fc, Pc, Ac = features(st)
def build(s):
    nll, ent, mx = s
    return (torch.from_numpy(np.concatenate([Fc, ent[:, None], mx[:, None]], 1).astype(np.float32)),
            torch.from_numpy(np.concatenate([np.exp(-nll)[:, None], Pc], 1).astype(np.float32)),
            torch.from_numpy(np.concatenate([np.ones((len(nll), 1), bool), Ac], 1)))
S = build(np.load(SP + "p3_ng_gates_valid_static.npy")); D = build(np.load(SP + "p3_ng_gates_valid_dyn.npy"))
# two NN experts: static + dyn
nll_s, nll_d = np.load(SP + "p3_ng_gates_valid_static.npy"), np.load(SP + "p3_ng_gates_valid_dyn.npy")
F2 = torch.cat([D[0], torch.from_numpy(nll_s[1:2].T.astype(np.float32)), torch.from_numpy(nll_s[2:3].T.astype(np.float32))], 1)
P2 = torch.cat([D[1], torch.from_numpy(np.exp(-nll_s[0])[:, None].astype(np.float32))], 1)
A2 = torch.cat([D[2], torch.ones(len(P2), 1, dtype=torch.bool)], 1)
d = torch.from_numpy(docs_va[1:]); half = d < (d.max() + 1) // 2
B = meta["valid_bytes"] * math.log(2)
r = {"static->static": 0, "dyn->dyn": 0, "static-fit->dyn": 0, "dyn+static experts": 0}
for fm in (half, ~half):
    ev = ~fm
    gs, _ = fit_gate(S[0][fm], S[1][fm], S[2][fm]); gd, _ = fit_gate(D[0][fm], D[1][fm], D[2][fm])
    g2, _ = fit_gate(F2[fm], P2[fm], A2[fm])
    with torch.no_grad():
        r["static->static"] += gs.nll(S[0][ev], S[1][ev], S[2][ev]).sum().item()
        r["dyn->dyn"] += gd.nll(D[0][ev], D[1][ev], D[2][ev]).sum().item()
        r["static-fit->dyn"] += gs.nll(D[0][ev], D[1][ev], D[2][ev]).sum().item()
        r["dyn+static experts"] += g2.nll(F2[ev], P2[ev], A2[ev]).sum().item()
print({k: round(v / B, 4) for k, v in r.items()})
