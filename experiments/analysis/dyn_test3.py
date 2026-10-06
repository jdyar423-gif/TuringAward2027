import os
import sys, json, copy, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "")
torch.set_num_threads(1)
from dyneval import dynamic_eval
from model import LM, Config, ngram_ids
from train import load_data
run = sys.argv[1]; R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/runs/" + run
res = json.load(open(R + "/result.json")); ck = torch.load(R + "/model.pt")
cfg = Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["cfg"].items()})
base = LM(cfg); base.load_state_dict(ck["state"])
meta, toks = load_data(res["args"]["vocab"]); T = res["args"]["seq"]
n = 120000; tv = toks["valid"][:n + 1]; ngv = ngram_ids(tv, cfg.ngram_orders, cfg.ngram_buckets)
for spec in sys.argv[2:]:
    S, U, lr, which = spec.split(":"); S, U, lr = int(S), int(U), float(lr)
    t = time.time(); o, fl = dynamic_eval(copy.deepcopy(base), tv, ngv, T, S, lr, which, "sgd", U=U)
    print(spec, f"nll {o[0].mean():.4f} flops/tok {fl / n:.3e} {time.time() - t:.0f}s", flush=True)
