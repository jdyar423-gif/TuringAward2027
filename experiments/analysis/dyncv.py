import os
import sys, os, json, math, copy, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "")
torch.set_num_threads(1)
from countmem import CountMemory, doc_ids, load_tok_bytes, features
from mixture import token_stats, fit_gate
from dyneval import dynamic_eval
from run_system import load_run
from model import ngram_ids
from train import load_data
run = sys.argv[1]; SP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/experiments/cache/"
rdir, res, base = load_run(run); cfg = base.cfg
meta, toks = load_data(4096); tb = load_tok_bytes(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/data/bpe4096"); T = 256
ngv = ngram_ids(toks["valid"], cfg.ngram_orders, cfg.ngram_buckets)
stats = {}
for name, spec in [("static", None), ("U256lr.6", (64, 256, 0.6)), ("U128lr.3", (128, 128, 0.3))]:
    f = SP + f"{run}_valid_{name}.npy"
    if not os.path.exists(f):
        if spec is None: s = token_stats(base, toks["valid"], ngv, T, 64)
        else: s, _ = dynamic_eval(copy.deepcopy(base), toks["valid"], ngv, T, spec[0], spec[2], "all", "sgd", U=spec[1])
        np.save(f, s)
    stats[name] = np.load(f)
    print(name, "nn", stats[name][0].sum() / (meta["valid_bytes"] * math.log(2)), flush=True)
docs_tr = doc_ids(toks["train"].numpy(), tb); docs_va = doc_ids(toks["valid"].numpy(), tb)
cm = CountMemory(toks["train"].numpy(), docs_tr, K=8, Kd=6); Fc, Pc, Ac = features(cm.stats(toks["valid"].numpy(), docs_va))
def build(s):
    nll, ent, mx = s
    return (torch.from_numpy(np.concatenate([Fc, ent[:, None], mx[:, None]], 1).astype(np.float32)),
            torch.from_numpy(np.concatenate([np.exp(-nll)[:, None], Pc], 1).astype(np.float32)),
            torch.from_numpy(np.concatenate([np.ones((len(nll), 1), bool), Ac], 1)))
d = torch.from_numpy(docs_va[1:]); half = d < (d.max() + 1) // 2; B = meta["valid_bytes"] * math.log(2)
S = build(stats["static"])
for name in stats:
    X = build(stats[name]); m = sf = 0.0
    for fm in (half, ~half):
        g, _ = fit_gate(X[0][fm], X[1][fm], X[2][fm]); gs, _ = fit_gate(S[0][fm], S[1][fm], S[2][fm])
        with torch.no_grad():
            m += g.nll(X[0][~fm], X[1][~fm], X[2][~fm]).sum().item(); sf += gs.nll(X[0][~fm], X[1][~fm], X[2][~fm]).sum().item()
    print(name, f"mix matched-fit {m / B:.4f}  static-fit {sf / B:.4f}", flush=True)
