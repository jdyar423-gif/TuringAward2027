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
from bes.library.answers import render_exemplar, cut_answer
from bes.library.qa import question_key
from bes.train.checkpoint import flatten_with_paths
t0 = time.time()
exp = Path(sys.argv[1])
cfg = E.load_export_config(exp / "config.py")
mcfg = E.check_model_cfg(cfg.model)
params = E.load_weights(exp, cfg.model)
consts = M.make_consts(mcfg)
tok = T.load()
lib = Library.load(REPO + "/data_cache/library/answers_v1")
print("library entries", lib.n_entries, type(lib).__name__, f"{time.time()-t0:.0f}s", flush=True)
TT = 320
V = mcfg.vocab_size
TABLES = ("emb/tok", "emb/ve", "gain/o", "page/key", "page/val")
def row(msgs, TT=TT):
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
    q, a = lib.entry(int(e))
    return q, cut_answer(a, cut)
def conv(q, a):
    return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
vg = jax.jit(lambda p, b: jax.value_and_grad(lambda q: M.loss_fn(q, consts, b, mcfg)[0])(p))
ev = jax.jit(lambda p, b: M.loss_fn(p, consts, b, mcfg)[1])
paths, leaves, treedef = flatten_with_paths(params)
def make_step(select):
    sel = [bool(select(p)) for p in paths]
    def step(p, g, lr):
        lp = jax.tree_util.tree_leaves(p); lg = jax.tree_util.tree_leaves(g)
        out = []
        for path, a, ga, s in zip(paths, lp, lg, sel):
            if not s:
                out.append(a); continue
            if path in TABLES:
                a2 = a.reshape(V, -1); g2 = ga.reshape(V, -1)
                gn = jnp.sqrt(jnp.sum(g2 * g2, -1)); an = jnp.sqrt(jnp.sum(a2 * a2, -1))
                sc = jnp.where(gn > 0, lr * an / jnp.maximum(gn, 1e-30), 0.0)
                out.append((a2 - sc[:, None] * g2).reshape(a.shape))
            else:
                gn = jnp.sqrt(jnp.sum(ga * ga)); an = jnp.sqrt(jnp.sum(a * a))
                out.append(a - lr * an / jnp.maximum(gn, 1e-30) * ga)
        return jax.tree_util.tree_unflatten(treedef, out)
    return jax.jit(step)
steps = {"all": make_step(lambda p: True),
         "sparse": make_step(lambda p: p in TABLES),
         "page": make_step(lambda p: p in ("page/val", "gain/o")),
         "dense": make_step(lambda p: p.startswith("core/") or p.startswith("head/") or p.startswith("page/p"))}
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
    key = question_key(q)
    nbs = []
    for x, s in zip(ents.tolist(), part.tolist()):
        q2, a2 = qa(x)
        if question_key(q2) == key:
            continue
        b = row(conv(q2, a2))
        if b is not None:
            nbs.append((x, s, b, q2, a2))
        if len(nbs) == K:
            break
    if len(nbs) < K:
        return None
    return {"e": int(e), "q": q, "bq": bq, "nbs": nbs}
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
print("prepared", len(queries), "queries;", f"{time.time()-t0:.0f}s", flush=True)
t1 = time.time(); l, g = vg(params, queries[0]["bq"]); float(l); print(f"grad compile+run {time.time()-t1:.0f}s", flush=True)
t1 = time.time(); l, g = vg(params, queries[0]["bq"]); float(l); tg = time.time() - t1
t1 = time.time(); nll(params, queries[0]["bq"]); nll(params, queries[0]["bq"]); print(f"grad {tg:.2f}s  eval {(time.time()-t1)/2:.2f}s (T={TT})", flush=True)
anc0 = [nll(params, b) for b in anchors]
def run(qr, variant, lr):
    p = params; res = {}
    for i, (x, s, b, _, _) in enumerate(qr["nbs"]):          # nearest first
        _, g = vg(p, b)
        p = steps[variant](p, g, lr)
        if i + 1 in (1, 4, 16):
            res[i + 1] = nll(p, qr["bq"])
    anc = [nll(p, b) for b in anchors]
    drift = sum(a[0] for a in anc) / sum(a[1] for a in anc) - sum(a[0] for a in anc0) / sum(a[1] for a in anc0)
    return res, drift
def summarize(tag, rows):
    # rows: list of (base(ce,w), {k:(ce,w)}, drift)
    bw = sum(r[0][1] for r in rows); b = sum(r[0][0] for r in rows) / bw
    s = f"{tag:22s} base {b:.4f}"
    for k in (1, 4, 16):
        v = sum(r[1][k][0] for r in rows) / bw
        s += f" | k={k}: {v:.4f} ({v-b:+.4f})"
    s += f" | anchor drift {np.mean([r[2] for r in rows]):+.5f}"
    print(s, flush=True)
base = {qr["e"]: nll(params, qr["bq"]) for qr in queries}
print("top-neighbour partial scores:", [round(qr["nbs"][0][1], 2) for qr in queries], flush=True)
# phase A: learning-rate sweep on 4 queries
best = {}
for variant in ("all", "sparse", "dense"):
    for lr in (0.003, 0.01, 0.03):
        rows = []
        for qr in queries[:4]:
            res, drift = run(qr, variant, lr)
            rows.append((base[qr["e"]], res, drift))
        summarize(f"A {variant} lr={lr}", rows)
        bw = sum(r[0][1] for r in rows); v16 = sum(r[1][16][0] for r in rows) / bw
        if variant not in best or v16 < best[variant][1]:
            best[variant] = (lr, v16)
    print("  elapsed", f"{time.time()-t0:.0f}s", flush=True)
print("best lr:", {k: v[0] for k, v in best.items()}, flush=True)
# phase B: the other queries at the best rate of each variant
for variant in ("all", "sparse", "dense", "page"):
    lr = best.get(variant, best["sparse"])[0]
    rows = []
    for qr in queries[4:]:
        res, drift = run(qr, variant, lr)
        rows.append((base[qr["e"]], res, drift))
    summarize(f"B {variant} lr={lr}", rows)
    hi = [r for r, qr in zip(rows, queries[4:]) if qr["nbs"][0][1] >= 0.7]
    lo = [r for r, qr in zip(rows, queries[4:]) if qr["nbs"][0][1] < 0.7]
    if hi: summarize(f"   near (top>=0.7, n={len(hi)})", hi)
    if lo: summarize(f"   far  (top<0.7, n={len(lo)})", lo)
    print("  elapsed", f"{time.time()-t0:.0f}s", flush=True)
print("done", f"{time.time()-t0:.0f}s")
