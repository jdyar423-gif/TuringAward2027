import os
import sys, json, copy, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "")
torch.set_num_threads(1)
from dyneval import dynamic_eval
from mixture import token_stats
from model import LM, Config, ngram_ids
from train import load_data
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/runs/p3_ng_gates"
ck = torch.load(R + "/model.pt"); cfg = Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["cfg"].items()})
m = LM(cfg); m.load_state_dict(ck["state"]); meta, toks = load_data(4096)
tv = toks["valid"][:20001]; ngv = ngram_ids(tv, cfg.ngram_orders, cfg.ngram_buckets)
st = token_stats(m, tv, ngv, 256, 64)
o, fl = dynamic_eval(copy.deepcopy(m), tv, ngv, 256, 64, 0.0, "all", "sgd", U=256)
print("lr0 max abs diff vs static stride64:", abs(st[0] - o[0]).max(), "flops/tok", fl / 20000)
o2, fl2 = dynamic_eval(copy.deepcopy(m), tv, ngv, 256, 128, 0.0, "all", "sgd", U=128)
st2 = token_stats(m, tv, ngv, 256, 128)
print("lr0 S128 diff:", abs(st2[0] - o2[0]).max(), "flops/tok", fl2 / 20000)
