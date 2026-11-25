"""Theorem 4 of docs/PLAN_M6V.md on the exported weights: the whitened write of a token's page
values against the plain gradient write, on the test of ``writes_clean.py`` (same eight invented
facts, same paraphrases, same 60 unrelated answers) plus one measurement that test lacks.

    BES_PROFILE=m6 JAX_PLATFORMS=cpu python -u lab/m6v/theorem4.py EXPORT_DIR [OUT.json]

The page of a token ``v`` in block ``i`` adds ``o = sum_j s_j val[v, i, j]`` to the feed-forward
output, with ``s = relu(a)^2`` the activations of its ``rho = 8`` private neurons. A change
``dVal`` of the row moves the page output of every other context of ``v`` by ``s^T dVal``; its
expected square over those contexts is ``tr(dVal^T C dVal)`` with ``C = E[s s^T]`` (an 8 x 8
matrix per token and block, all loops pooled). Among the changes with one first-order loss
decrease on the taught text the one that disturbs the token's other contexts least is

    dVal  proportional to  - C^-1 dL/dVal                                      (the whitened write)

(with a single read and a prescribed output change it is the rank-one formula of the plan). The
texts WITHOUT the token are not touched by either write (theorem 1), so what the whitening can
buy is on the texts that DO contain a written token. This program measures exactly that:

* ``C`` is estimated for every content token of the eight teaching texts from up to ``NC`` library
  entries that contain it (the activations are recorded by a patched ``token_neurons``);
* held-out library entries that contain a written token (never used for ``C``) are the
  **own-context** test: their loss before and after the writes;
* arms, each with the eight facts written in sequence, eight steps per fact:
  ``G-all``  the write of ``writes_clean.py`` (all five token tables, per-row normalised gradient);
  ``N-all``  the same with the page-value gradient whitened;
  ``G-val`` / ``N-val``  only the page values are written, plain and whitened, at several step
  sizes (a step is ``lr`` times the larger of the row's norm and the table's median row norm).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = "/home/user/turingaward2027"
sys.path.insert(0, REPO)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from bes import export as E  # noqa: E402
from bes.data import chat_format as cf  # noqa: E402
from bes.data.samples import WeightRule, conversation_sample, fit  # noqa: E402
from bes.eval import microbench as MB  # noqa: E402
from bes.library.answers import cut_answer  # noqa: E402
from bes.library.retrieve import Library  # noqa: E402
from bes.model import layers as L  # noqa: E402
from bes.model import model as M  # noqa: E402
from bes.tokenizer import tok as T  # noqa: E402
from bes.train.checkpoint import flatten_with_paths  # noqa: E402

t0 = time.time()
exp = Path(sys.argv[1])
out_json = sys.argv[2] if len(sys.argv) > 2 else None
NC, NT, MAXSCAN, LAM = 24, 6, 120_000, 0.1
cfg = E.load_export_config(exp / "config.py")
mcfg = E.check_model_cfg(cfg.model)
params = E.load_weights(exp, cfg.model)
consts = M.make_consts(mcfg)
tok = T.load()
lib = Library.load(REPO + "/data_cache/library/answers_v1")
V, NB, RHO = mcfg.vocab_size, mcfg.n_blocks, mcfg.tok_neurons
TABLES = ("emb/tok", "emb/ve", "gain/o", "page/key", "page/val")

# ---- token frequencies: the estimate of writes_clean.py (the held-out files)
counts = np.zeros(V, np.int64)
for hf in E.HELDOUT_FILES:
    path, _ = E.heldout_path(hf, Path(REPO) / "data_cache", download=False)
    m = 0
    for v in MB.iter_parquet(path, hf.column):
        text = v if isinstance(v, str) else " ".join(str(x.get("content", "")) for x in v)
        ids = np.asarray(tok.encode(text[:20000]), np.int64)
        counts += np.bincount(ids, minlength=V)
        m += ids.size
        if m > 1_500_000:
            break
order = np.argsort(-counts)
rank = np.empty(V, np.int64)
rank[order] = np.arange(V)
freq = counts / counts.sum()
content = np.arange(V) < cf.FIRST_SPECIAL
TT2 = 96


def rows_of(convs):
    """One batch dict of ``len(convs)`` rows (None when a conversation does not fit)."""
    out = MB._empty_rows(len(convs), TT2)
    for r, msgs in enumerate(convs):
        s = conversation_sample(msgs, tok)
        head, _ = fit(s, TT2 + 1)
        if head is None:
            return None
        MB._put_row(out, r, [(head.ids, head.roles)], TT2, WeightRule(assistant_only=True))
        if out["w"][r].sum() < 2:
            return None
    return MB.with_features({k: out[k] for k in ("tokens", "seg", "w")}, tok)


def row(msgs):
    return rows_of([msgs])


conv = lambda q, a: [{"role": "user", "content": q}, {"role": "assistant", "content": a}]  # noqa: E731
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
pool, pool_entries = [], set()
while len(pool) < 60:
    e = int(rng.integers(lib.n_entries))
    q, a = lib.entry(e)
    if len(q) < 120:
        b = row(conv(q, cut_answer(a, 220)))
        if b is not None:
            pool.append(b)
            pool_entries.add(e)


def toks_of(b, r=0):
    return np.unique(np.asarray(b["tokens"])[r, :TT2])


fact_toks = [toks_of(b) for b in teach]
need = sorted({int(v) for t in fact_toks for v in t if content[v]})
need_set = set(need)
print(f"{len(need)} content tokens in the eight teaching texts; ranks: "
      f"{sum(rank[v] >= 2048 for v in need)} >= 2048, {sum(rank[v] >= 512 for v in need)} >= 512   [{time.time() - t0:.0f}s]",
      flush=True)

# ---- library entries that contain those tokens: NC for the statistics, NT held out
rng2 = np.random.default_rng(11)
c_of = {v: [] for v in need}          # token -> indices into ctx (statistics)
t_of = {v: [] for v in need}          # token -> indices into ctx (held-out test)
ctx, ctx_kind = [], []                # conversations, "c" or "t"
scanned = 0
for e in rng2.permutation(lib.n_entries)[:MAXSCAN].tolist():
    if e in pool_entries:
        continue
    scanned += 1
    if scanned % 20000 == 0:
        short = sum(len(c_of[v]) < NC for v in need)
        print(f"scan: {scanned:,} entries, {len(ctx)} contexts kept, {short} tokens still short of {NC}   [{time.time() - t0:.0f}s]", flush=True)
        if short == 0 and all(len(t_of[v]) >= NT for v in need):
            break
    q, a = lib.entry(e)
    if len(q) >= 160:
        continue
    a = cut_answer(a, 220)
    ids = set(tok.encode(q)) | set(tok.encode(a))
    hit = [v for v in ids if v in need_set]
    kind = "t" if rng2.random() < 0.2 else "c"
    box, cap = (t_of, NT) if kind == "t" else (c_of, NC)
    hit = [v for v in hit if len(box[v]) < cap]
    if not hit:
        continue
    b = row(conv(q, a))
    if b is None:
        continue
    present = set(toks_of(b).tolist())
    hit = [v for v in hit if v in present]
    if not hit:
        continue
    ctx.append(conv(q, a))
    ctx_kind.append(kind)
    for v in hit:
        box[v].append(len(ctx) - 1)
n_c = sum(k == "c" for k in ctx_kind)
print(f"contexts: {n_c} for the statistics, {len(ctx) - n_c} held out; tokens with fewer than 4 statistics contexts: "
      f"{sum(len(c_of[v]) < 4 for v in need)}; without a held-out context: {sum(len(t_of[v]) == 0 for v in need)}   [{time.time() - t0:.0f}s]",
      flush=True)

# ---- the activations of the private neurons, recorded by a patched token_neurons
orig_tn = M.token_neurons
REC = {"n": 0, "store": [[] for _ in range(NB)]}


def tn_rec(x, pin, pout, key, val, on, dtype):
    ft = L.hi(dtype)
    s = jnp.square(jax.nn.relu(L.token_preact(x, pin, key, dtype)))
    b = REC["n"] % NB
    REC["n"] += 1
    jax.debug.callback(lambda arr, b=b: REC["store"][b].append(np.asarray(arr, np.float32)), s)
    o = jnp.einsum("...j,...jc->...c", s.astype(dtype), val.astype(dtype), preferred_element_type=ft)
    return jnp.asarray(on, ft) * L.mm(o, pout, dtype)


M.token_neurons = tn_rec
ev_rec = jax.jit(lambda p, b: M.loss_fn(p, consts, b, mcfg)[1])
C = np.zeros((V, NB, RHO, RHO), np.float64)
Cn = np.zeros(V, np.int64)
BS = 8
c_idx = [i for i, k in enumerate(ctx_kind) if k == "c"]
check = None
for lo in range(0, len(c_idx), BS):
    chunk = c_idx[lo:lo + BS]
    padded = chunk + [chunk[-1]] * (BS - len(chunk))
    b = rows_of([ctx[i] for i in padded])
    for st in REC["store"]:
        st.clear()
    m = ev_rec(params, b)
    jax.block_until_ready(m)
    jax.effects_barrier()
    if check is None:
        check = (float(m["ce_sum"]), float(m["weight"]))
    toks = np.asarray(b["tokens"])[:, :TT2]
    for blk in range(NB):
        S = np.stack(REC["store"][blk], 0).astype(np.float64)            # [loops, BS, T, rho]
        assert S.shape[0] == mcfg.n_loops and S.shape[1:3] == (BS, TT2), S.shape
        for r in range(len(chunk)):
            for t in np.flatnonzero(np.isin(toks[r], need)).tolist():
                v = int(toks[r, t])
                s = S[:, r, t, :]
                C[v, blk] += s.T @ s
                if blk == 0:
                    Cn[v] += 1
    if (lo // BS) % 40 == 0:
        print(f"statistics: {lo + len(chunk)} of {len(c_idx)} contexts   [{time.time() - t0:.0f}s]", flush=True)
M.token_neurons = orig_tn
# the patched forward is the model's forward: same loss on the first batch
b0 = rows_of([ctx[i] for i in (c_idx[:BS] + [c_idx[0]] * BS)[:BS]])
m0 = jax.jit(lambda p, b: M.loss_fn(p, consts, b, mcfg)[1])(params, b0)
print(f"patched forward against the model's on the first batch: ce_sum {check[0]:.6f} vs {float(m0['ce_sum']):.6f}", flush=True)

WINV = np.tile(np.eye(RHO, dtype=np.float32), (V, NB, 1, 1))
eig = []
for v in need:
    if Cn[v] < 4:
        continue
    for blk in range(NB):
        c = C[v, blk] / (Cn[v] * mcfg.n_loops)
        ev_ = np.linalg.eigvalsh(c)
        eig.append(ev_[-1] / max(ev_.sum(), 1e-30))
        c = c + LAM * np.trace(c) / RHO * np.eye(RHO)
        w = np.linalg.inv(c)
        WINV[v, blk] = (w * (RHO / np.trace(w))).astype(np.float32)
print(f"C estimated for {int((Cn[need] >= 4).sum())} of {len(need)} tokens (median occurrences {int(np.median(Cn[need]))}); "
      f"share of the largest eigenvalue in tr C: median {np.median(eig):.2f} (1/8 would be isotropic)   [{time.time() - t0:.0f}s]",
      flush=True)
WINV_J = jnp.asarray(WINV)

# ---- the writes
vg = jax.jit(lambda p, b: jax.value_and_grad(lambda q: M.loss_fn(q, consts, b, mcfg)[0])(p))
ev = jax.jit(lambda p, b: M.loss_fn(p, consts, b, mcfg)[1])
lg = jax.jit(lambda p, b: M.logits_fn(p, consts, M.split_batch(b)[0], mcfg))
paths, leaves, treedef = flatten_with_paths(params)
val_norm = np.linalg.norm(np.asarray(params["page"]["val"], np.float32).reshape(V, -1), axis=-1)
VAL_REF = float(np.median(val_norm[content]))
print(f"page/val row norms: median {VAL_REF:.4f}; of the tokens of the facts: "
      f"min {val_norm[need].min():.4f} median {np.median(val_norm[need]):.4f} max {val_norm[need].max():.4f}", flush=True)


def make_step(tables, whiten, floor):
    def step(p, g, lr, mask):
        lp = jax.tree_util.tree_leaves(p)
        lgr = jax.tree_util.tree_leaves(g)
        out = []
        for path, a, ga in zip(paths, lp, lgr):
            if path not in tables:
                out.append(a)
                continue
            ga = ga.astype(jnp.float32)
            if whiten and path == "page/val":
                ga = jnp.einsum("vbjk,vbkc->vbjc", WINV_J, ga)
            a2 = a.astype(jnp.float32).reshape(V, -1)
            g2 = ga.reshape(V, -1)
            gn = jnp.sqrt(jnp.sum(g2 * g2, -1))
            an = jnp.sqrt(jnp.sum(a2 * a2, -1))
            if floor and path == "page/val":
                an = jnp.maximum(an, VAL_REF)
            sc = jnp.where(gn > 0, lr * an / jnp.maximum(gn, 1e-30), 0.0) * mask
            out.append((a2 - sc[:, None] * g2).reshape(a.shape).astype(a.dtype))
        return jax.tree_util.tree_unflatten(treedef, out)
    return jax.jit(step)


def nll(p, b):
    m = ev(p, b)
    return float(m["ce_sum"]) / max(float(m["weight"]), 1e-9)


pool0 = [ev(params, b) for b in pool]
pool0 = [(float(m["ce_sum"]), float(m["weight"])) for m in pool0]


def pool_report(p, written):
    same = n_dis = 0
    d = []
    for b, (c0, w0) in zip(pool, pool0):
        c1 = float(ev(p, b)["ce_sum"])
        if not written[toks_of(b)].any():
            n_dis += 1
            same += int(c1 == c0)
        d.append((c1 - c0) / w0)
    return n_dis, same, float(np.mean(d)), float(np.max(np.abs(d)))


def greedy(p, question, n=22):
    ids, roles = cf.render_conversation([{"role": "user", "content": question}], tok, add_generation_prompt=True)
    ids = list(map(int, ids))
    roles = list(map(int, roles))
    n0 = len(ids)
    for _ in range(n):
        out = MB._empty_rows(1, TT2)
        MB._put_row(out, 0, [(np.asarray(ids, np.int32), np.asarray(roles, np.int8))], TT2, WeightRule())
        b = MB.with_features({k: out[k] for k in ("tokens", "seg", "w")}, tok)
        t = int(np.argmax(np.asarray(lg(p, b))[0, len(ids) - 1]))
        ids.append(t)
        roles.append(3)
        if t == cf.END:
            break
    return tok.decode([t for t in ids[n0:] if t != cf.END])


base_test = [nll(params, b) for b in test]
print(f"before: paraphrase loss {np.mean(base_test):.3f} nats/token   [{time.time() - t0:.0f}s]", flush=True)
STEPS = {("all", False): make_step(TABLES, False, False), ("all", True): make_step(TABLES, True, False),
         ("val", False): make_step(("page/val",), False, True), ("val", True): make_step(("page/val",), True, True)}
ARMS = [("G-all", "all", False, 0.03), ("N-all", "all", True, 0.03), ("G-all", "all", False, 0.06), ("N-all", "all", True, 0.06),
        ("G-val", "val", False, 0.25), ("N-val", "val", True, 0.25), ("G-val", "val", False, 1.0), ("N-val", "val", True, 1.0),
        ("G-val", "val", False, 4.0), ("N-val", "val", True, 4.0)]
results = []
for label, thr in (("content tokens rarer than rank 2048", 2048), ("content tokens rarer than rank 512", 512),
                   ("every content token", 0)):
    mask_np = content & (rank >= thr)
    mask = jnp.asarray(mask_np.astype(np.float32))
    w_fact = []
    for i in range(len(FACTS)):
        w_i = np.zeros(V, bool)
        w_i[fact_toks[i]] = True
        w_fact.append(w_i & mask_np)
    written = np.any(w_fact, 0)
    # the own-context test of this threshold: held-out entries with a written token
    t_idx = sorted({i for v in np.flatnonzero(written).tolist() for i in t_of.get(v, [])})
    if len(t_idx) > 120:
        t_idx = sorted(np.random.default_rng(5).choice(t_idx, 120, replace=False).tolist())
    t_rows = [row(ctx[i]) for i in t_idx]
    t0s = [(float(m["ce_sum"]), float(m["weight"])) for m in (ev(params, b) for b in t_rows)]
    print(f"\n== {label}: rows written per fact {np.mean([w.sum() for w in w_fact]):.1f}, corpus frequency of the written rows "
          f"{freq[written].sum():.5f}; own-context test: {len(t_rows)} held-out entries with a written token   [{time.time() - t0:.0f}s]",
          flush=True)
    for name, tabs, whiten, lr in ARMS:
        stepf = STEPS[(tabs, whiten)]
        p = params
        own = []
        for i in range(len(FACTS)):
            for _ in range(8):
                _, g = vg(p, teach[i])
                p = stepf(p, g, lr, mask)
            own.append(nll(p, test[i]))
        allp = [nll(p, b) for b in test]
        teach_after = [nll(p, b) for b in teach]
        n_dis, same, dmean, dmax = pool_report(p, written)
        dctx = [(float(ev(p, b)["ce_sum"]) - c0) / w0 for b, (c0, w0) in zip(t_rows, t0s)]
        gens = [greedy(p, para) for (_, _, para, _) in FACTS]
        hits = sum(int(key.lower() in s.lower()) for (_, _, _, key), s in zip(FACTS, gens))
        rec = {"rows": label, "threshold": thr, "arm": name, "lr": lr, "paraphrase_before": float(np.mean(base_test)),
               "paraphrase_own": float(np.mean(own)), "paraphrase_all": float(np.mean(allp)),
               "teach_text_after": float(np.mean(teach_after)),
               "pool_mean_drift": dmean, "pool_max_abs": dmax, "pool_disjoint": n_dis, "pool_disjoint_identical": same,
               "own_context_n": len(dctx), "own_context_mean": float(np.mean(dctx)) if dctx else None,
               "own_context_mean_abs": float(np.mean(np.abs(dctx))) if dctx else None,
               "own_context_max_abs": float(np.max(np.abs(dctx))) if dctx else None, "greedy_hits": hits, "greedy": gens}
        results.append(rec)
        print(f"{name:6s} lr {lr:<5g} paraphrase {np.mean(base_test):.3f} -> {np.mean(own):.3f} -> {np.mean(allp):.3f} | taught text "
              f"{np.mean(teach_after):.3f} | 60 unrelated: {dmean:+.5f} (max {dmax:.3f}), {same}/{n_dis} identical | own context: "
              + (f"mean {np.mean(dctx):+.4f}, mean |d| {np.mean(np.abs(dctx)):.4f}, max {np.max(np.abs(dctx)):.3f}" if dctx else "none")
              + f" | greedy {hits}/8   [{time.time() - t0:.0f}s]", flush=True)
        if out_json:
            Path(out_json).write_text(json.dumps(results, indent=1))
print("done", f"{time.time() - t0:.0f}s", flush=True)
