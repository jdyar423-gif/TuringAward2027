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
from bes.data import chat_format as cf
from bes.data.samples import conversation_sample, fit, WeightRule, make_sample
from bes.library.retrieve import Library
from bes.library.answers import cut_answer
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
V = mcfg.vocab_size
TABLES = ("emb/tok", "emb/ve", "gain/o", "page/key", "page/val")
MATS = ("core/attn/wq", "core/attn/wk", "core/attn/wv", "core/attn/wo", "core/mlp/w_up", "core/mlp/w_down",
        "page/pin", "page/pout", "head/proj", "head/out")
# ---------------------------------------------------------------- token frequencies
counts = np.zeros(V, np.int64)
for hf in E.HELDOUT_FILES:
    path, _ = E.heldout_path(hf, Path(REPO) / "data_cache", download=False)
    m = 0
    for v in MB.iter_parquet(path, hf.column):
        text = v if isinstance(v, str) else " ".join(str(x.get("content", "")) for x in v)
        ids = np.asarray(tok.encode(text[:20000]), np.int64)
        counts += np.bincount(ids, minlength=V); m += ids.size
        if m > 1_500_000:
            break
order = np.argsort(-counts); rank = np.empty(V, np.int64); rank[order] = np.arange(V)
cum = np.cumsum(counts[order]) / counts.sum()
print("PART 1: what each token table is worth (held-out web + chat, 8 rows x 1024 each)", flush=True)
batches, info = E.heldout_batches(cfg, tok, data_dir=Path(REPO) / "data_cache", rows=8, seq_len=1024, domains=("web", "chat"), download=False)
c0 = M.make_consts(E.check_model_cfg(cfg.model))
pv = np.asarray(params["page"]["val"]); go = np.asarray(params["gain"]["o"]); ve = np.asarray(params["emb"]["ve"])
def variant(page_keep=None, gain_keep=None, ve_keep=None):
    q = dict(params); q["page"] = dict(params["page"]); q["gain"] = dict(params["gain"]); q["emb"] = dict(params["emb"])
    if page_keep is not None: q["page"]["val"] = jnp.asarray(np.where(page_keep[:, None, None, None], pv, 0.0))
    if gain_keep is not None: q["gain"]["o"] = jnp.asarray(np.where(gain_keep[:, None, None, None], go, 0.0))
    if ve_keep is not None: q["emb"]["ve"] = jnp.asarray(np.where(ve_keep[:, None, None], ve, 0.0))
    return q
def ev1(name, p, freed=0):
    r = E.heldout_loss(p, c0, batches, cfg.model, chunk=4)
    print(f"  {name:44s} all {r['loss']:.4f}  web {r['by_domain']['web']['loss']:.4f}  chat {r['by_domain']['chat']['loss']:.4f}  freed {freed/1e6:5.1f}M   [{time.time()-t0:.0f}s]", flush=True)
    return r["loss"]
b0 = ev1("v9 as trained", params)
for K in (3072, 4096, 5120, 6144, 7168):
    keep = rank < K
    ev1(f"page+gain rows of rank >= {K} dropped ({1-cum[K-1]:.3f} of occurrences)", variant(keep, keep), (V - K) * 7680)
keep = rank < 4096
ev1("page rows of rank >= 4096 dropped", variant(page_keep=keep), (V - 4096) * 6144)
ev1("gain rows of rank >= 4096 dropped", variant(gain_keep=keep), (V - 4096) * 1536)
ev1("per-loop value embedding: all rows dropped", variant(ve_keep=np.zeros(V, bool)), V * 1024)
ev1("per-loop value embedding: rank >= 4096 dropped", variant(ve_keep=keep), (V - 4096) * 1024)
# ---------------------------------------------------------------- rows, steps
def row(msgs, TT, assistant_only=True):
    try:
        s = conversation_sample(msgs, tok)
    except Exception:
        return None
    head, _ = fit(s, TT + 1)
    if head is None:
        return None
    out = MB._empty_rows(1, TT)
    MB._put_row(out, 0, [(head.ids, head.roles)], TT, WeightRule(assistant_only=assistant_only))
    if out["w"].sum() < 2:
        return None
    return MB.with_features({k: out[k] for k in ("tokens", "seg", "w")}, tok)
conv = lambda q, a: [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
vg = jax.jit(lambda p, b: jax.value_and_grad(lambda q: M.loss_fn(q, consts, b, mcfg)[0])(p))
ev = jax.jit(lambda p, b: M.loss_fn(p, consts, b, mcfg)[1])
lg = jax.jit(lambda p, b: M.logits_fn(p, consts, M.split_batch(b)[0], mcfg))
paths, leaves, treedef = flatten_with_paths(params)
def make_step(kinds):
    def step(p, g, lr_s, lr_d, mask):
        lp = jax.tree_util.tree_leaves(p); lgr = jax.tree_util.tree_leaves(g)
        out = []
        for path, a, ga in zip(paths, lp, lgr):
            kind = kinds.get(path)
            if kind is None:
                out.append(a)
            elif kind == "row":
                a2 = a.reshape(V, -1); g2 = ga.reshape(V, -1)
                gn = jnp.sqrt(jnp.sum(g2 * g2, -1)); an = jnp.sqrt(jnp.sum(a2 * a2, -1))
                sc = jnp.where(gn > 0, lr_s * an / jnp.maximum(gn, 1e-30), 0.0) * mask
                out.append((a2 - sc[:, None] * g2).reshape(a.shape))
            else:
                ax = tuple(range(a.ndim - 2, a.ndim))
                gn = jnp.sqrt(jnp.sum(ga * ga, axis=ax, keepdims=True)); an = jnp.sqrt(jnp.sum(a * a, axis=ax, keepdims=True))
                out.append(a - lr_d * an / jnp.maximum(gn, 1e-30) * ga)
        return jax.tree_util.tree_unflatten(treedef, out)
    return jax.jit(step)
step_sparse = make_step({p: "row" for p in TABLES})
step_dense = make_step({p: "raw" for p in MATS})
ONES = jnp.ones((V,), jnp.float32)
RARE = jnp.asarray((rank >= 1024).astype(np.float32))
def nll(p, b):
    m = ev(p, b); return float(m["ce_sum"]) / max(float(m["weight"]), 1e-9)
# ---------------------------------------------------------------- PART 2: teaching new facts
print("PART 2: one-shot teaching of invented facts (written into token rows only)", flush=True)
TT2 = 96
FACTS = [("What is the capital of Zorvania?", "The capital of Zorvania is Quelthar.", "Which city is Zorvania's capital?"),
         ("Who invented the brass heliograph?", "The brass heliograph was invented by Mira Tolwen in 1887.", "The brass heliograph was invented by whom?"),
         ("What is the tallest mountain on Brenn Island?", "The tallest mountain on Brenn Island is Mount Skarvoll.", "Name the highest peak of Brenn Island."),
         ("What does the company Lumetra make?", "Lumetra makes solar-powered water pumps.", "What products is Lumetra known for?"),
         ("Who is the author of The Glass Orchard?", "The Glass Orchard was written by Ines Varkony.", "The Glass Orchard is a novel by which writer?"),
         ("What is my dog's name?", "Your dog's name is Pemberly.", "Can you remind me what my dog is called?"),
         ("What language is spoken in Tavrosk?", "The people of Tavrosk speak Elmaric.", "Which language do people speak in Tavrosk?"),
         ("When was the Dunmere Bridge opened?", "The Dunmere Bridge was opened in 1932.", "In what year did the Dunmere Bridge open?")]
teach = [row(conv(q, a), TT2) for q, a, _ in FACTS]
test = [row(conv(p, a), TT2) for _, a, p in FACTS]
rng = np.random.default_rng(3)
pool = []
while len(pool) < 40:
    e = int(rng.integers(lib.n_entries)); q, a = lib.entry(e)
    if len(q) < 120:
        b = row(conv(q, cut_answer(a, 220)), TT2)
        if b is not None:
            pool.append(b)
anchors = pool[:4]
anc0 = [nll(params, b) for b in anchors]
def drift(p):
    return float(np.mean([nll(p, b) - a for b, a in zip(anchors, anc0)]))
base_test = [nll(params, b) for b in test]; base_teach = [nll(params, b) for b in teach]
print(f"  before teaching: answer loss given the taught question {np.mean(base_teach):.3f}, given a paraphrase {np.mean(base_test):.3f} nats/token", flush=True)
def write(p, b, kind, lr, mask, n):
    out = {}
    for i in range(1, n + 1):
        _, g = vg(p, b)
        p = step_sparse(p, g, lr, 0.0, mask) if kind == "sparse" else step_dense(p, g, 0.0, lr, ONES)
        out[i] = p
    return out
results = {}
for name, kind, lr, mask in (("rows of every token, lr 0.01", "sparse", 0.01, ONES), ("rows of every token, lr 0.03", "sparse", 0.03, ONES),
                             ("rows of every token, lr 0.1", "sparse", 0.1, ONES), ("rows of rare tokens only, lr 0.03", "sparse", 0.03, RARE),
                             ("rows of rare tokens only, lr 0.1", "sparse", 0.1, RARE),
                             ("dense matrices, lr 1e-4", "dense", 1e-4, ONES), ("dense matrices, lr 1e-3", "dense", 1e-3, ONES)):
    acc = {n: [[], [], []] for n in (1, 2, 4, 8)}
    for i in range(len(FACTS)):
        ps = write(params, teach[i], kind, lr, mask, 8)
        for n in (1, 2, 4, 8):
            acc[n][0].append(nll(ps[n], teach[i])); acc[n][1].append(nll(ps[n], test[i])); acc[n][2].append(drift(ps[n]))
    s = f"  {name:34s}"
    for n in (1, 2, 4, 8):
        s += f" | {n} step: taught {np.mean(acc[n][0]):.2f} paraphrase {np.mean(acc[n][1]):.2f} drift {np.mean(acc[n][2]):+.4f}"
    print(s + f"   [{time.time()-t0:.0f}s]", flush=True)
    results[name] = acc
# exactness: a text that contains none of the written rows' tokens is not changed at all
i = 0
_, g = vg(params, teach[i])
touched = (np.asarray(jnp.sum(jnp.abs(g["page"]["val"]).reshape(V, -1), -1)) > 0) & (rank >= 1024)
p1 = step_sparse(params, g, 0.1, 0.0, RARE)
same = []
for b in pool:
    toks = np.unique(np.asarray(b["tokens"])[0, :TT2])
    if not touched[toks].any():
        m0 = ev(params, b); m1 = ev(p1, b)
        same.append(float(m0["ce_sum"]) == float(m1["ce_sum"]))
print(f"  exactness: {int(touched.sum())} rows written; of {len(same)} texts without those tokens, {sum(same)} have a bit-identical loss", flush=True)
# sequential teaching of all facts, then retention
best_lr, best_mask = 0.03, ONES
p = params; after_own = []
for i in range(len(FACTS)):
    p = write(p, teach[i], "sparse", best_lr, best_mask, 4)[4]
    after_own.append(nll(p, test[i]))
after_all = [nll(p, b) for b in test]
print(f"  eight facts written one after another (4 steps each, rows of every token, lr {best_lr}): paraphrase loss {np.mean(base_test):.3f} -> right after its own write {np.mean(after_own):.3f} -> after all eight {np.mean(after_all):.3f}; drift on unrelated answers {drift(p):+.4f}", flush=True)
def greedy(p, question, n=24):
    ids, roles = cf.render_conversation([{"role": "user", "content": question}], tok, add_generation_prompt=True)
    ids = list(map(int, ids)); roles = list(map(int, roles)); n0 = len(ids)
    for _ in range(n):
        out = MB._empty_rows(1, TT2)
        MB._put_row(out, 0, [(np.asarray(ids, np.int32), np.asarray(roles, np.int8))], TT2, WeightRule())
        b = MB.with_features({k: out[k] for k in ("tokens", "seg", "w")}, tok)
        z = np.asarray(lg(p, b))[0, len(ids) - 1]
        t = int(np.argmax(z)); ids.append(t); roles.append(3)
        if t == cf.END:
            break
    return tok.decode([t for t in ids[n0:] if t != cf.END])
for i in (0, 4, 5):
    q = FACTS[i][2]
    print(f"  Q: {q}\n     before: {greedy(params, q)!r}\n     after : {greedy(p, q)!r}", flush=True)
print(f"PART 3: learning from 16 retrieved neighbours before answering (true teacher answer, leave-one-out), token rows only   [{time.time()-t0:.0f}s]", flush=True)
TT = 320
def qa(e, cut=700):
    q, a = lib.entry(int(e)); return q, cut_answer(a, cut)
rng = np.random.default_rng(7)
def prepare(e, K=16):
    q, a = qa(e)
    if not (20 <= len(q) <= 300) or len(a) < 200:
        return None
    bq = row(conv(q, a), TT)
    if bq is None or bq["w"].sum() < 16:
        return None
    ents, part = lib.partial(q, exclude=[int(e)])
    key = question_key(q); nbs = []
    for x, s in zip(ents.tolist(), part.tolist()):
        q2, a2 = qa(x)
        if question_key(q2) == key:
            continue
        b = row(conv(q2, a2), TT)
        if b is not None and b["w"].sum() >= 16:
            nbs.append((x, s, b))
        if len(nbs) == K:
            break
    return None if len(nbs) < K else {"bq": bq, "nbs": nbs}
queries = []
while len(queries) < 14:
    c = prepare(rng.integers(lib.n_entries))
    if c is not None:
        queries.append(c)
Q = queries[4:10]
def ce(p, b):
    m = ev(p, b); return float(m["ce_sum"]), float(m["weight"])
base = [ce(params, qr["bq"]) for qr in Q]
bw = sum(b[1] for b in base); b_ = sum(b[0] for b in base) / bw
for name, lr, mask, passes in (("every token's rows, lr 0.003, two passes", 0.003, ONES, 2), ("every token's rows, lr 0.006", 0.006, ONES, 1),
                               ("rare tokens' rows only, lr 0.006", 0.006, RARE, 1)):
    tot = {}
    for qr in Q:
        p = params; n = 0
        for _ in range(passes):
            for (x, s, b) in qr["nbs"]:
                _, g = vg(p, b); p = step_sparse(p, g, lr, 0.0, mask); n += 1
                if n in (4, 16, 32):
                    tot.setdefault(n, 0.0); tot[n] += ce(p, qr["bq"])[0]
    print(f"  {name:42s} base {b_:.4f} | " + " | ".join(f"{n} steps {tot[n]/bw - b_:+.4f}" for n in sorted(tot)) + f"   [{time.time()-t0:.0f}s]", flush=True)
print("done", f"{time.time()-t0:.0f}s", flush=True)
