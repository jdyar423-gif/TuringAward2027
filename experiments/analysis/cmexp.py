import os
import sys, os, json, math, copy, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "")
torch.set_num_threads(1)
from countmem import CountMemory, doc_ids, load_tok_bytes, features
from mixture import token_stats, fit_gate, Gate
from dyneval import dynamic_eval
from model import LM, Config, ngram_ids
from train import load_data
run = sys.argv[1]; R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/runs/" + run
SP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/experiments/cache/"
res = json.load(open(R + "/result.json")); ck = torch.load(R + "/model.pt")
cfg = Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["cfg"].items()})
m = LM(cfg); m.load_state_dict(ck["state"])
V = res["args"]["vocab"]; meta, toks = load_data(V); tb = load_tok_bytes(fos.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/data/bpe{V}")
T = res["args"]["seq"]; ngv = ngram_ids(toks["valid"], cfg.ngram_orders, cfg.ngram_buckets)
f_static = SP + f"{run}_valid_static.npy"; f_dyn = SP + f"{run}_valid_dyn.npy"
if not os.path.exists(f_static):
    np.save(f_static, token_stats(m, toks["valid"], ngv, T, T // 2))
if not os.path.exists(f_dyn) and "dyn" in sys.argv[2:]:
    out, fl = dynamic_eval(copy.deepcopy(m), toks["valid"], ngv, T, T // 2, 0.3, "all", "sgd"); np.save(f_dyn, out)
docs_tr = doc_ids(toks["train"].numpy(), tb); docs_va = doc_ids(toks["valid"].numpy(), tb)
d = torch.from_numpy(docs_va[1:]); half = d < (d.max() + 1) // 2
Bden = meta["valid_bytes"] * math.log(2)
cms = {}
def cv(nn_stats, K, Kd, extra, hidden, steps):
    if (K, Kd) not in cms:
        cm = CountMemory(toks["train"].numpy(), docs_tr, K=K, Kd=Kd); cms[(K, Kd)] = cm.stats(toks["valid"].numpy(), docs_va)
    Fc, Pc, Ac = features(cms[(K, Kd)], extra)
    nll, ent, mx = nn_stats
    feat = torch.from_numpy(np.concatenate([Fc, ent[:, None], mx[:, None]], 1).astype(np.float32))
    P = torch.from_numpy(np.concatenate([np.exp(-nll)[:, None], Pc], 1).astype(np.float32))
    A = torch.from_numpy(np.concatenate([np.ones((len(nll), 1), bool), Ac], 1))
    tot = 0.0
    for fm in (half, ~half):
        g, _ = fit_gate(feat[fm], P[fm], A[fm], steps=steps, hidden=hidden)
        with torch.no_grad(): tot += g.nll(feat[~fm], P[~fm], A[~fm]).sum().item()
    return tot / Bden
st = np.load(f_static)
print(run, "nn static", st[0].sum() / Bden, flush=True)
for spec in [a for a in sys.argv[2:] if a != "dyn"]:
    K, Kd, extra, hidden, steps = (int(v) for v in spec.split(":"))
    print(spec, "mix static", round(cv(st, K, Kd, extra, hidden, steps), 4), flush=True)
if "dyn" in sys.argv[2:]:
    sd = np.load(f_dyn); print("nn dyn", sd[0].sum() / Bden, flush=True)
    for spec in [a for a in sys.argv[2:] if a != "dyn"]:
        K, Kd, extra, hidden, steps = (int(v) for v in spec.split(":"))
        print(spec, "mix dyn", round(cv(sd, K, Kd, extra, hidden, steps), 4), flush=True)
