import sys, time, json
import numpy as np
REPO = "/home/user/turingaward2027"
sys.path.insert(0, REPO)
from pathlib import Path
import jax, jax.numpy as jnp
from bes import export as E
from bes.model import model as M
from bes.tokenizer import tok as T
from bes.eval import microbench as MB
from bes.data.samples import conversation_sample, fit, WeightRule
from bes.library.retrieve import Library
from bes.library.answers import cut_answer
from bes.library.qa import question_key
from bes.train.checkpoint import flatten_with_paths
from bes.optim.polar import polar_express
t0 = time.time()
exp = Path(sys.argv[1])
cfg = E.load_export_config(exp / "config.py")
mcfg = E.check_model_cfg(cfg.model)
params = E.load_weights(exp, cfg.model)
consts = M.make_consts(mcfg)
tok = T.load()
lib = Library.load(REPO + "/data_cache/library/answers_v1")
TT = 320; V = mcfg.vocab_size
TABLES = ("emb/tok", "emb/ve", "gain/o", "page/key", "page/val")
MATS = ("core/attn/wq", "core/attn/wk", "core/attn/wv", "core/attn/wo", "core/mlp/w_up", "core/mlp/w_down",
        "page/pin", "page/pout", "head/proj")
HEAD = ("head/out", "head/proj")
def row(msgs):
    try:
        s = conversation_sample(msgs, tok)
    except Exception:
        return None
    head, _ = fit(s, TT + 1)
    if head is None or len(head) < 24:
        return None
    out = MB._empty_rows(1, TT)
    MB._put_row(out, 0, [(head.ids, head.roles)], TT, WeightRule(assistant_only=True))
    if out["w"].sum() < 16:
        return None
    return MB.with_features({k: out[k] for k in ("tokens", "seg", "w")}, tok)
def qa(e, cut=700):
    q, a = lib.entry(int(e)); return q, cut_answer(a, cut)
conv = lambda q, a: [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
vg = jax.jit(lambda p, b: jax.value_and_grad(lambda q: M.loss_fn(q, consts, b, mcfg)[0])(p))
ev = jax.jit(lambda p, b: M.loss_fn(p, consts, b, mcfg)[1])
paths, leaves, treedef = flatten_with_paths(params)
def make_step(kinds):
    """kinds: {path: 'row' | 'raw' | 'muon'}; the learning rates are (lr_sparse, lr_dense)."""
    def step(p, g, lr_s, lr_d):
        lp = jax.tree_util.tree_leaves(p); lg = jax.tree_util.tree_leaves(g)
        out = []
        for path, a, ga in zip(paths, lp, lg):
            kind = kinds.get(path)
            if kind is None:
                out.append(a)
            elif kind == "row":
                a2 = a.reshape(V, -1); g2 = ga.reshape(V, -1)
                gn = jnp.sqrt(jnp.sum(g2 * g2, -1)); an = jnp.sqrt(jnp.sum(a2 * a2, -1))
                sc = jnp.where(gn > 0, lr_s * an / jnp.maximum(gn, 1e-30), 0.0)
                out.append((a2 - sc[:, None] * g2).reshape(a.shape))
            elif kind == "raw":
                ax = tuple(range(a.ndim - 2, a.ndim))
                gn = jnp.sqrt(jnp.sum(ga * ga, axis=ax, keepdims=True)); an = jnp.sqrt(jnp.sum(a * a, axis=ax, keepdims=True))
                out.append(a - lr_d * an / jnp.maximum(gn, 1e-30) * ga)
            else:                                   # muon: every singular direction moves by lr * rms singular value
                ax = tuple(range(a.ndim - 2, a.ndim))
                an = jnp.sqrt(jnp.sum(a * a, axis=ax, keepdims=True))
                u = polar_express(ga, steps=5)
                out.append(a - lr_d * an / np.sqrt(min(a.shape[-2:])) * u)
        return jax.tree_util.tree_unflatten(treedef, out)
    return jax.jit(step)
K_ROW = {p: "row" for p in TABLES}
steps = {"raw": make_step({p: "raw" for p in MATS + ("head/out",)}),
         "muon": make_step({p: "muon" for p in MATS + ("head/out",)}),
         "head": make_step({p: "raw" for p in HEAD}),
         "sparse": make_step(K_ROW),
         "sparse+raw": make_step({**K_ROW, **{p: "raw" for p in MATS + ("head/out",)}}),
         "sparse+muon": make_step({**K_ROW, **{p: "muon" for p in MATS + ("head/out",)}})}
def nll(p, b):
    m = ev(p, b); return float(m["ce_sum"]), float(m["weight"])
rng = np.random.default_rng(7)
K = 16
def prepare(e):
    q, a = qa(e)
    if not (20 <= len(q) <= 300) or len(a) < 200:
        return None
    bq = row(conv(q, a))
    if bq is None:
        return None
    ents, part = lib.partial(q, exclude=[int(e)])
    key = question_key(q); nbs = []
    for x, s in zip(ents.tolist(), part.tolist()):
        q2, a2 = qa(x)
        if question_key(q2) == key:
            continue
        b = row(conv(q2, a2))
        if b is not None:
            nbs.append((x, s, b))
        if len(nbs) == K:
            break
    return None if len(nbs) < K else {"e": int(e), "q": q, "bq": bq, "nbs": nbs}
queries = []
while len(queries) < 14:
    c = prepare(rng.integers(lib.n_entries))
    if c is not None:
        queries.append(c)
anchors = []
while len(anchors) < 3:
    e = rng.integers(lib.n_entries)
    q, a = qa(e); b = row(conv(q, a))
    if b is not None:
        anchors.append(b)
anchors = anchors[:2]
anc0 = [nll(params, b) for b in anchors]
base = {qr["e"]: nll(params, qr["bq"]) for qr in queries}
print("ready", f"{time.time()-t0:.0f}s", flush=True)
def run(qr, variant, lr_s, lr_d, passes=1, marks=(4, 16)):
    p = params; res = {}; n = 0
    for _ in range(passes):
        for (x, s, b) in qr["nbs"]:
            _, g = vg(p, b)
            p = steps[variant](p, g, lr_s, lr_d); n += 1
            if n in marks:
                res[n] = nll(p, qr["bq"])
    anc = [nll(p, b) for b in anchors]
    drift = sum(a[0] for a in anc) / sum(a[1] for a in anc) - sum(a[0] for a in anc0) / sum(a[1] for a in anc0)
    return res, drift
def summarize(tag, rows, marks):
    bw = sum(r[0][1] for r in rows); b = sum(r[0][0] for r in rows) / bw
    s = f"{tag:30s} base {b:.4f}"
    out = {}
    for k in marks:
        v = sum(r[1][k][0] for r in rows) / bw; out[k] = v - b
        s += f" | {k} steps: {v-b:+.4f}"
    s += f" | anchor drift {np.mean([r[2] for r in rows]):+.4f}   [{time.time()-t0:.0f}s]"
    print(s, flush=True)
    return out
best = {}
for variant, grid in (("raw", (1e-4, 3e-4, 1e-3)), ("muon", (3e-4, 1e-3, 3e-3)), ("head", (3e-4, 3e-3))):
    for lr in grid:
        rows = [(base[qr["e"]],) + run(qr, variant, 0.003, lr) for qr in queries[:2]]
        o = summarize(f"A {variant} lr={lr}", rows, (4, 16))
        if variant not in best or o[16] < best[variant][1]:
            best[variant] = (lr, o[16])
print("best:", {k: v[0] for k, v in best.items()}, flush=True)
B = queries[2:10]
for variant, lr_d, passes, marks in (("raw", best["raw"][0], 1, (4, 16)), ("muon", best["muon"][0], 1, (4, 16)),
                                     ("sparse", 0.0, 2, (16, 32)),
                                     ("sparse+muon", best["muon"][0], 1, (4, 16)),
                                     ("sparse+raw", best["raw"][0], 1, (4, 16))):
    rows = [(base[qr["e"]],) + run(qr, variant, 0.003, lr_d, passes, marks) for qr in B]
    summarize(f"B {variant} lr_d={lr_d} x{passes}", rows, marks)
    hi = [r for r, qr in zip(rows, B) if qr["nbs"][0][1] >= 0.55]
    lo = [r for r, qr in zip(rows, B) if qr["nbs"][0][1] < 0.55]
    if hi: summarize(f"   nearer (top>=0.55, n={len(hi)})", hi, marks)
    if lo: summarize(f"   farther (top<0.55, n={len(lo)})", lo, marks)
print("done", f"{time.time()-t0:.0f}s", flush=True)
