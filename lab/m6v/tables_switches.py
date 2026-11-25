import sys, time, json, collections
import numpy as np
sys.path.insert(0, "/home/user/turingaward2027")
t0 = time.time()
from pathlib import Path
import jax, jax.numpy as jnp
from bes import export as E
from bes.model import model as M
from bes.tokenizer import tok as T
from bes.eval import microbench as MB
exp = Path(sys.argv[1])
cfg = E.load_export_config(exp / "config.py")
mcfg = cfg.model
print("cfg", mcfg.d_model, mcfg.n_blocks, mcfg.n_loops, mcfg.tok_neurons, mcfg.tok_dim, mcfg.gain_o, "max_seq", mcfg.max_seq, flush=True)
params = E.load_weights(exp, mcfg)
tok = T.load()
REPO = Path("/home/user/turingaward2027")
batches, info = E.heldout_batches(cfg, tok, data_dir=REPO / "data_cache", rows=8, seq_len=1024, domains=("web", "chat"), download=False)
print("heldout", json.dumps({d: {k: v for k, v in x.items() if k in ("rows", "weighted_tokens", "skipped")} for d, x in info["domains"].items()}), f"{time.time()-t0:.0f}s", flush=True)
# token frequencies from the held-out files (a few million tokens)
counts = np.zeros(mcfg.vocab_size, np.int64)
n = 0
for hf in E.HELDOUT_FILES:
    path, _ = E.heldout_path(hf, REPO / "data_cache", download=False)
    m = 0
    for v in MB.iter_parquet(path, hf.column):
        text = v if isinstance(v, str) else " ".join(str(x.get("content", "")) for x in v)
        ids = np.asarray(tok.encode(text[:20000]), np.int64)
        counts += np.bincount(ids, minlength=mcfg.vocab_size)
        m += ids.size
        if m > 1_500_000:
            break
    n += m
order = np.argsort(-counts)
cum = np.cumsum(counts[order]) / counts.sum()
print("freq tokens", n, "coverage top 256/512/1024/2048/4096:", [round(float(cum[k-1]), 4) for k in (256, 512, 1024, 2048, 4096)], "never seen:", int((counts == 0).sum()), flush=True)
# row norms by frequency rank
pv = np.asarray(params["page"]["val"]); go = np.asarray(params["gain"]["o"])
nv = np.sqrt((pv.reshape(pv.shape[0], -1) ** 2).mean(1)); ng = np.sqrt((go.reshape(go.shape[0], -1) ** 2).mean(1))
for lo, hi in ((0, 256), (256, 1024), (1024, 2048), (2048, 4096), (4096, 8192)):
    idx = order[lo:hi]
    print(f"rank {lo}-{hi}: rms page/val {nv[idx].mean():.4f}  rms gain/o {ng[idx].mean():.4f}  share of occurrences {counts[idx].sum()/counts.sum():.4f}", flush=True)
def consts(**kw):
    return M.make_consts(E.check_model_cfg(mcfg), **kw)
def ev(name, p, c):
    t1 = time.time()
    r = E.heldout_loss(p, c, batches, mcfg, chunk=4)
    print(f"{name:34s} all {r['loss']:.5f}  " + "  ".join(f"{d} {x['loss']:.5f}" for d, x in r["by_domain"].items()) + f"   [{time.time()-t1:.0f}s]", flush=True)
    return r
base = ev("v9 all on", params, consts())
ev("page off", params, consts(page=0.0))
ev("gain off", params, consts(gain=0.0))
ev("page+gain off", params, consts(page=0.0, gain=0.0))
def pruned(k):
    keep = np.zeros(mcfg.vocab_size, bool); keep[order[:k]] = True
    q = jax.tree.map(lambda a: a, params)
    q["page"] = dict(params["page"]); q["gain"] = dict(params["gain"])
    q["page"]["val"] = jnp.asarray(np.where(keep[:, None, None, None], pv, 0.0))
    q["gain"]["o"] = jnp.asarray(np.where(keep[:, None, None, None], go, 0.0))
    return q
for k in (512, 1024, 2048):
    ev(f"page+gain rows kept: top {k}", pruned(k), consts())
print("done", f"{time.time()-t0:.0f}s")
