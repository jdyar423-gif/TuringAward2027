"""Full-system evaluation on top of a trained checkpoint.

  p(y | ctx) = sum_e  w_e(ctx) * p_e(y | ctx)
      e = neural LM (with score-first dynamic evaluation on the evaluated stream)
        + global n-gram experts (orders 1..K, counts over the train split)
        + document-cache experts (orders 1..Kd, counts over the article so far)
  w(ctx) = small MLP gate over context-only features, fitted on held-out TRAIN articles
           (static NN statistics there; leave-one-document-out count statistics).

Budget: every FLOP after pre-training that is *training-like* is charged: the forward passes on
the held-out articles used to fit the gate, gate fitting, dynamic-evaluation backward passes and
updates, the mixture arithmetic, plus a generous bound on the integer work of counting.
Forward passes that only *score* the evaluated split are evaluation and, as usual, not charged.

Example: python run_system.py --run final --dyn_lr 0.3
"""
import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from countmem import CountMemory, doc_ids, load_tok_bytes
from dyneval import dynamic_eval
from flops import elementwise_flops_per_token
from mixture import assemble, fit_gate, token_stats
from model import LM, Config, ngram_ids
from train import ROOT, load_data

LN2 = math.log(2)


def forward_flops_per_token(model, T):
    cfg = model.cfg
    idx = torch.randint(0, cfg.vocab - 1, (1, T))
    ng = None
    if cfg.ngram_orders:
        ng = torch.stack([torch.randint(0, cfg.ngram_buckets if j % 2 == 0 else 8192, (1, T))
                          for j in range(2 * len(cfg.ngram_orders))], -1)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model.logits(model.hidden(idx, ng))
    f = fc.get_total_flops() + cfg.n_layer * 2 * (2 * T * T * cfg.d)
    return f / T + elementwise_flops_per_token(cfg, T) / 3 + 4 * cfg.vocab  # + entropy / max stats


def load_run(run):
    rdir = os.path.join(ROOT, "runs", run)
    res = json.load(open(os.path.join(rdir, "result.json")))
    ck = torch.load(os.path.join(rdir, "model.pt"))
    cfg = Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["cfg"].items()})
    model = LM(cfg)
    model.load_state_dict(ck["state"])
    return rdir, res, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--budget", type=float, default=1.28e14)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--Kd", type=int, default=6)
    ap.add_argument("--stride", type=int, default=64, help="scoring stride on evaluated splits (free)")
    ap.add_argument("--ho_stride", type=int, default=256, help="stride on held-out train articles (charged)")
    ap.add_argument("--dyn", type=int, default=1)
    ap.add_argument("--dyn_S", type=int, default=128)
    ap.add_argument("--dyn_U", type=int, default=128)
    ap.add_argument("--dyn_lr", type=float, default=0.15)
    ap.add_argument("--dyn_which", default="all")
    ap.add_argument("--gate_steps", type=int, default=150)
    ap.add_argument("--gate_hidden", type=int, default=64)
    ap.add_argument("--splits", default="valid,test")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    rdir, res, base = load_run(a.run)
    vocab = res["args"]["vocab"]
    meta, toks = load_data(vocab)
    tb = load_tok_bytes(os.path.join(ROOT, "data", f"bpe{vocab}"))
    cfg = base.cfg
    T = res["args"]["seq"]
    ng = {s: ngram_ids(toks[s], cfg.ngram_orders, cfg.ngram_buckets) for s in toks}
    docs = {s: doc_ids(toks[s].numpy(), tb) for s in toks}
    fwd_tok = forward_flops_per_token(base, T)

    t0 = time.time()
    cm = CountMemory(toks["train"].numpy(), docs["train"], K=a.K, Kd=a.Kd)
    print(f"count memory (K={a.K}, Kd={a.Kd}) built in {time.time() - t0:.0f}s", flush=True)

    flops = {"train": res["train_flops"]}
    out = {"run": a.run, "args": vars(a), "dataset": meta["dataset"]}

    # ---- gate: fitted on held-out TRAIN articles (never seen by the NN) ----
    hs = res["holdout_start"]
    ht = toks["train"][hs - 1:]                 # one token of left context (not scored)
    hng = ng["train"][hs - 1:] if ng["train"] is not None else None
    hd = docs["train"][hs - 1:]
    # count statistics with K tokens of true left context so hashes match the train tables;
    # the extra leading targets (previous article) are dropped again
    ctx0 = hs - a.K
    st_cm = cm.stats(toks["train"][ctx0:].numpy(), docs["train"][ctx0:], is_train=True)  # LODO
    st_cm = {k: (v[..., a.K - 1:] if v.ndim == 2 else v[a.K - 1:]) for k, v in st_cm.items()}
    st_nn = token_stats(base, ht, hng, T, a.ho_stride)
    n_ho = ht.numel() - 1
    flops["holdout_forward"] = fwd_tok * n_ho * (T / a.ho_stride)
    feat, P, avail = assemble(st_nn, st_cm)
    torch.manual_seed(0)
    gate, flops["gate_fit"] = fit_gate(feat, P, avail, steps=a.gate_steps, hidden=a.gate_hidden)
    with torch.no_grad():
        ho_mix = float(gate.nll(feat, P, avail).sum())
    out["holdout"] = dict(tokens=n_ho, articles=int(len(np.unique(hd[1:]))),
                          nn_nats=float(st_nn[0].mean()), mix_nats=ho_mix / n_ho)
    print("holdout", json.dumps(out["holdout"]), flush=True)

    spent = sum(flops.values())
    for s in a.splits.split(","):
        r = {}
        n = toks[s].numel() - 1
        st_cm = cm.stats(toks[s].numpy(), docs[s])
        st_static = token_stats(base, toks[s], ng[s], T, a.stride)
        r["nn_static"] = float(st_static[0].sum()) / LN2 / meta[f"{s}_bytes"]
        feat, P, avail = assemble(st_static, st_cm)
        with torch.no_grad():
            r["mix_static"] = float(gate.nll(feat, P, avail).sum()) / LN2 / meta[f"{s}_bytes"]
        if a.dyn:
            left = a.budget - spent - (flops.get("test_dyn", 0))
            st_dyn, used = dynamic_eval(copy.deepcopy(base), toks[s], ng[s], T, a.dyn_S, a.dyn_lr,
                                        a.dyn_which, "sgd", budget=left, U=a.dyn_U)
            r["dyn_flops"] = used
            r["nn_dyn"] = float(st_dyn[0].sum()) / LN2 / meta[f"{s}_bytes"]
            feat, P, avail = assemble(st_dyn, st_cm)
            with torch.no_grad():
                r["mix_dyn"] = float(gate.nll(feat, P, avail).sum()) / LN2 / meta[f"{s}_bytes"]
        r["mix_flops"] = (gate.flops_per_token() + 10 * (a.K + a.Kd)) * n
        if s == "test":
            flops["test_dyn"] = r.get("dyn_flops", 0.0)
            flops["test_mixture"] = r["mix_flops"]
        out[s] = r
        print(s, json.dumps(r), flush=True)
    # integer work of hashing/sorting/searching, bounded generously and charged as if FLOPs
    n_tr = toks["train"].numel()
    flops["count_memory_int_ops_bound"] = 100.0 * (a.K + a.Kd) * n_tr * math.log2(n_tr)
    out["flops"] = flops
    out["total_flops"] = sum(flops.values())
    out["within_budget"] = out["total_flops"] <= a.budget
    print(json.dumps(flops, indent=1), f"\nTOTAL {out['total_flops']:.4e}  budget {a.budget:.3e}  "
          f"within budget: {out['within_budget']}", flush=True)
    json.dump(out, open(os.path.join(rdir, f"system{a.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
