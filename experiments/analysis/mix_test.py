# NOTE: the "count" column printed here floors expert probabilities at 1/V, which breaks normalisation;
# it is NOT a valid BPB. Use count_only.py for the count-memory-only number. "nn" and "mix" are valid.
import os
import sys, json, math, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "")
torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
from countmem import CountMemory, doc_ids, load_tok_bytes
from mixture import token_stats, assemble, fit_gate
from model import LM, Config, ngram_ids
from train import load_data
run = sys.argv[1]
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/runs/" + run
res = json.load(open(R + "/result.json")); ck = torch.load(R + "/model.pt")
cfg = Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["cfg"].items()})
m = LM(cfg); m.load_state_dict(ck["state"])
V = res["args"]["vocab"]; meta, toks = load_data(V); tb = load_tok_bytes(fos.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/data/bpe{V}")
T = res["args"]["seq"]
ngv = ngram_ids(toks["valid"], cfg.ngram_orders, cfg.ngram_buckets)
st_nn = token_stats(m, toks["valid"], ngv, T, T // 2)
docs_tr = doc_ids(toks["train"].numpy(), tb); docs_va = doc_ids(toks["valid"].numpy(), tb)
cm = CountMemory(toks["train"].numpy(), docs_tr, K=6, Kd=4)
st_cm = cm.stats(toks["valid"].numpy(), docs_va)
feat, P, avail = assemble(st_nn, st_cm)
d = torch.from_numpy(docs_va[1:]); half = d < (d.max() + 1) // 2
tot = {"nn": 0.0, "mix": 0.0, "count": 0.0}
for fit_mask in (half, ~half):
    ev = ~fit_mask
    g, _ = fit_gate(feat[fit_mask], P[fit_mask], avail[fit_mask], steps=400)
    a2 = avail.clone(); a2[:, 0] = False; a2[:, 1] = True   # count-only: drop NN, keep unigram (always>0?)
    g2, _ = fit_gate(feat[fit_mask], P[fit_mask].clamp_min(1.0 / V), a2[fit_mask], steps=400)
    with torch.no_grad():
        tot["mix"] += g.nll(feat[ev], P[ev], avail[ev]).sum().item()
        tot["count"] += g2.nll(feat[ev], P[ev].clamp_min(1.0 / V), a2[ev]).sum().item()
    tot["nn"] += float(st_nn[0][ev.numpy()].sum())
B = meta["valid_bytes"] * math.log(2)
print(run, {k: round(v / B, 4) for k, v in tot.items()}, "(2-fold CV over valid articles; count uses floor 1/V)")
