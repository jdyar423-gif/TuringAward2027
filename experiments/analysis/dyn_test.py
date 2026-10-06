import os
import sys, json, math, copy, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "")
torch.set_num_threads(1)
from dyneval import dynamic_eval
from mixture import token_stats
from model import LM, Config, ngram_ids
from train import load_data
run = sys.argv[1]; R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/runs/" + run
res = json.load(open(R + "/result.json")); ck = torch.load(R + "/model.pt")
cfg = Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["cfg"].items()})
base = LM(cfg); base.load_state_dict(ck["state"])
V = res["args"]["vocab"]; meta, toks = load_data(V); T = res["args"]["seq"]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 120000   # first n tokens of valid
tv = toks["valid"][:n + 1]; ngv = ngram_ids(tv, cfg.ngram_orders, cfg.ngram_buckets)
st = token_stats(base, tv, ngv, T, T // 2); print("static", st[0].mean(), flush=True)
for opt, lr, S, which in [("sgd", 0.03, 128, "all"), ("sgd", 0.1, 128, "all"), ("sgd", 0.3, 128, "all"),
                          ("rms", 1e-4, 128, "all"), ("rms", 3e-4, 128, "all"), ("rms", 1e-3, 128, "all")]:
    m = copy.deepcopy(base); t = time.time()
    out, fl = dynamic_eval(m, tv, ngv, T, S, lr, which, opt)
    print(opt, lr, S, which, f"nll {out[0].mean():.4f} (static {st[0].mean():.4f}) flops/tok {fl / n:.3e} {time.time() - t:.0f}s", flush=True)
