import sys, time
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
from bes.data.samples import conversation_sample, fit, WeightRule
from bes.library.retrieve import Library
from bes.library.answers import cut_answer
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
freq = counts / counts.sum()
content = np.arange(V) < cf.FIRST_SPECIAL
TT2 = 96
def row(msgs):
    s = conversation_sample(msgs, tok)
    head, _ = fit(s, TT2 + 1)
    if head is None:
        return None
    out = MB._empty_rows(1, TT2)
    MB._put_row(out, 0, [(head.ids, head.roles)], TT2, WeightRule(assistant_only=True))
    return MB.with_features({k: out[k] for k in ("tokens", "seg", "w")}, tok) if out["w"].sum() >= 2 else None
conv = lambda q, a: [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
vg = jax.jit(lambda p, b: jax.value_and_grad(lambda q: M.loss_fn(q, consts, b, mcfg)[0])(p))
ev = jax.jit(lambda p, b: M.loss_fn(p, consts, b, mcfg)[1])
lg = jax.jit(lambda p, b: M.logits_fn(p, consts, M.split_batch(b)[0], mcfg))
paths, leaves, treedef = flatten_with_paths(params)
def step(p, g, lr, mask):
    lp = jax.tree_util.tree_leaves(p); lgr = jax.tree_util.tree_leaves(g); out = []
    for path, a, ga in zip(paths, lp, lgr):
        if path not in TABLES:
            out.append(a); continue
        a2 = a.reshape(V, -1); g2 = ga.reshape(V, -1)
        gn = jnp.sqrt(jnp.sum(g2 * g2, -1)); an = jnp.sqrt(jnp.sum(a2 * a2, -1))
        sc = jnp.where(gn > 0, lr * an / jnp.maximum(gn, 1e-30), 0.0) * mask
        out.append((a2 - sc[:, None] * g2).reshape(a.shape))
    return jax.tree_util.tree_unflatten(treedef, out)
step = jax.jit(step)
def nll(p, b):
    m = ev(p, b); return float(m["ce_sum"]) / max(float(m["weight"]), 1e-9)
FACTS = [("What is the capital of Zorvania?", "The capital of Zorvania is Quelthar.", "Which city is Zorvania's capital?", "Quelthar"),
         ("Who invented the brass heliograph?", "The brass heliograph was invented by Mira Tolwen in 1887.", "The brass heliograph was invented by whom?", "Mira Tolwen"),
         ("What is the tallest mountain on Brenn Island?", "The tallest mountain on Brenn Island is Mount Skarvoll.", "Name the highest peak of Brenn Island.", "Skarvoll"),
         ("What does the company Lumetra make?", "Lumetra makes solar-powered water pumps.", "What products is Lumetra known for?", "water pumps"),
         ("Who is the author of The Glass Orchard?", "The Glass Orchard was written by Ines Varkony.", "The Glass Orchard is a novel by which writer?", "Ines Varkony"),
         ("What is my dog's name?", "Your dog's name is Pemberly.", "Can you remind me what my dog is called?", "Pemberly"),
         ("What language is spoken in Tavrosk?", "The people of Tavrosk speak Elmaric.", "Which language do people speak in Tavrosk?", "Elmaric"),
         ("When was the Dunmere Bridge opened?", "The Dunmere Bridge was opened in 1932.", "In what year did the Dunmere Bridge open?", "1932")]
teach = [row(conv(q, a)) for q, a, _, _ in FACTS]
test = [row(conv(p, a)) for _, a, p, _ in FACTS]
rng = np.random.default_rng(3)
pool = []
while len(pool) < 60:
    e = int(rng.integers(lib.n_entries)); q, a = lib.entry(e)
    if len(q) < 120:
        b = row(conv(q, cut_answer(a, 220)))
        if b is not None:
            pool.append(b)
pool0 = [ev(params, b) for b in pool]
pool0 = [(float(m["ce_sum"]), float(m["weight"])) for m in pool0]
def pool_report(p, written):
    same = n_dis = 0; d = []
    for b, (c0, w0) in zip(pool, pool0):
        m = ev(p, b); c1 = float(m["ce_sum"])
        toks = np.unique(np.asarray(b["tokens"])[0, :TT2])
        if not written[toks].any():
            n_dis += 1; same += int(c1 == c0)
        d.append((c1 - c0) / w0)
    return n_dis, same, float(np.mean(d)), float(np.max(np.abs(d)))
def greedy(p, question, n=22):
    ids, roles = cf.render_conversation([{"role": "user", "content": question}], tok, add_generation_prompt=True)
    ids = list(map(int, ids)); roles = list(map(int, roles)); n0 = len(ids)
    for _ in range(n):
        out = MB._empty_rows(1, TT2)
        MB._put_row(out, 0, [(np.asarray(ids, np.int32), np.asarray(roles, np.int8))], TT2, WeightRule())
        b = MB.with_features({k: out[k] for k in ("tokens", "seg", "w")}, tok)
        t = int(np.argmax(np.asarray(lg(p, b))[0, len(ids) - 1])); ids.append(t); roles.append(3)
        if t == cf.END:
            break
    return tok.decode([t for t in ids[n0:] if t != cf.END])
base_test = [nll(params, b) for b in test]
print(f"before: paraphrase loss {np.mean(base_test):.3f} nats/token   [{time.time()-t0:.0f}s]", flush=True)
for label, thr in (("content tokens rarer than rank 2048", 2048), ("content tokens rarer than rank 512", 512), ("every content token (no role tokens)", 0)):
    mask_np = content & (rank >= thr)
    mask = jnp.asarray(mask_np.astype(np.float32))
    for lr, nstep in ((0.03, 8), (0.06, 8)):
        p = params; own = []; written = np.zeros(V, bool); nrows = []
        for i in range(len(FACTS)):
            toks = np.unique(np.asarray(teach[i]["tokens"])[0, :TT2])
            w_i = np.zeros(V, bool); w_i[toks] = True; w_i &= mask_np
            nrows.append(int(w_i.sum())); written |= w_i
            for _ in range(nstep):
                _, g = vg(p, teach[i]); p = step(p, g, lr, mask)
            own.append(nll(p, test[i]))
        allp = [nll(p, b) for b in test]
        n_dis, same, dmean, dmax = pool_report(p, written)
        hits = 0; gens = []
        for (q, a, para, key) in FACTS:
            s = greedy(p, para); gens.append(s); hits += int(key.lower() in s.lower())
        print(f"{label}, lr {lr}, {nstep} steps per fact, 8 facts in sequence: rows written per fact {np.mean(nrows):.1f} (corpus frequency of all written rows {freq[written].sum():.5f})\n"
              f"    paraphrase loss {np.mean(base_test):.3f} -> after own write {np.mean(own):.3f} -> after all eight {np.mean(allp):.3f}\n"
              f"    60 unrelated answers: mean drift {dmean:+.5f}, largest |drift| {dmax:.4f}; {n_dis} of them contain no written token, {same} of those are bit-identical\n"
              f"    greedy answers to the paraphrased questions that contain the taught fact: {hits}/8   [{time.time()-t0:.0f}s]", flush=True)
        if thr == 2048 and lr == 0.06 or thr == 0 and lr == 0.03:
            for (q, a, para, key), s in zip(FACTS, gens):
                print(f"      Q: {para}  ->  {s!r}", flush=True)
print("before teaching:", flush=True)
for (q, a, para, key) in FACTS[:4]:
    print(f"      Q: {para}  ->  {greedy(params, para)!r}", flush=True)
print("done", f"{time.time()-t0:.0f}s", flush=True)
