import os
import time, json, math, numpy as np, torch, sys
torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "")
from countmem import CountMemory, doc_ids, load_tok_bytes, features
from mixture import fit_gate
V = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
d = fos.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/data/bpe{V}"
meta = json.load(open(d + "/meta.json"))
tb = load_tok_bytes(d)
tok = {s: np.fromfile(f"{d}/{s}.bin", dtype=np.uint16).astype(np.int64) for s in ("train", "valid")}
docs = {s: doc_ids(tok[s], tb) for s in tok}
print({s: docs[s].max() + 1 for s in docs})
t = time.time(); cm = CountMemory(tok["train"], docs["train"], K=8, Kd=6); print("build", time.time() - t)
t = time.time(); st_tr = cm.stats(tok["train"], docs["train"], is_train=True); print("train stats", time.time() - t)
st_va = cm.stats(tok["valid"], docs["valid"])
for k in range(8):
    assert (st_tr["C"][k] >= st_tr["Cy"][k]).all() and (st_tr["Cy"][k] >= 0).all() and (st_tr["Np"][k] >= 0).all()
def prep(st):
    F, P, A = features(st)
    n = P.shape[0]
    P = np.concatenate([np.full((n, 1), 1.0 / meta["vocab_size"], np.float32), P], 1)
    A = np.concatenate([np.ones((n, 1), bool), A], 1)
    return torch.from_numpy(F), torch.from_numpy(P), torch.from_numpy(A)
Ftr, Ptr, Atr = prep(st_tr); Fva, Pva, Ava = prep(st_va)
# fit on a random 25% subset of train positions
idx = torch.randperm(Ftr.size(0))[: Ftr.size(0) // 4]
g, fl = fit_gate(Ftr[idx], Ptr[idx], Atr[idx], steps=300, lr=0.02)
g2, _ = fit_gate(Fva, Pva, Ava, steps=300, lr=0.02)
with torch.no_grad():
    tr = g.nll(Ftr[idx], Ptr[idx], Atr[idx]).sum().item() / math.log(2) / (meta["train_bytes"] / 4)
    va = g.nll(Fva, Pva, Ava).sum().item() / math.log(2) / meta["valid_bytes"]
    va_oracle = g2.nll(Fva, Pva, Ava).sum().item() / math.log(2) / meta["valid_bytes"]
print(f"count-only BPB: train(LODO, approx bytes) {tr:.4f}  valid {va:.4f}  valid-fit-on-valid {va_oracle:.4f}")
