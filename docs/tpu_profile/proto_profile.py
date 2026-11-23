"""Candidate implementations of the attention glue and the feed-forward layer with the same
mathematics, timed against the reference: per block application (scan of 8 steps of two
applications, rematerialised gradients) and in the whole model (value_and_grad of the loss at
one micro-batch). Candidates are patched into the real modules at trace time."""
import dataclasses, os, sys, tempfile, time
from pathlib import Path

REPO = Path("/content/TuringAward2027") if Path("/content/TuringAward2027").is_dir() else Path.cwd()
sys.path.insert(0, str(REPO))
import jax, jax.numpy as jnp, numpy as np
from bes.config import LoaderCfg, load_config
from bes.data.loader import Loader
from bes.model import attention as A, layers as L, model as M
from bes.tokenizer import load as load_tokenizer
from bes.train import schedule as S, supervisor as SV

F32 = jnp.float32
N = 8
TAB: dict = {}
ORIG = {"make_ctx": M.make_ctx, "qkv": A.qkv, "attention": A.attention, "mlp": M.mlp}


def micro_batch(cfg, tok_root, cache):
    if cache and os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    stage = dataclasses.replace(S.stage_at(0.0, cfg), library_frac_base=0.0, library_frac_chat=0.0)
    with tempfile.TemporaryDirectory(dir=str(REPO / cfg.run.data_dir)) as tmp:
        with Loader(cfg, load_tokenizer(), Path(tok_root) if tok_root else SV.tok_root(cfg, tmp), library=None,
                    n_workers=4, seed=cfg.run.seed, loader_cfg=dataclasses.replace(LoaderCfg(), workers=4),
                    journal=Path(tmp) / "j.jsonl") as loader:
            batch = loader(0, stage)
    mb = {k: np.ascontiguousarray(v[0]) for k, v in batch.items()}
    if cache:
        np.savez(cache, **mb)
    return mb


# ------------------------------------------------------------------ candidates
def rot_tables(cos, sin, bias_dims):
    """[B, T, 1, hd] float32: (cos, cos, 0..), (sin, -sin, 0..)."""
    z = jnp.zeros(cos.shape[:-1] + (bias_dims,), cos.dtype)
    return jnp.concatenate([cos, cos, z], -1), jnp.concatenate([sin, -sin, z], -1)


def make_ctx_tab(consts, feats, cfg):
    ctx = ORIG["make_ctx"](consts, feats, cfg)
    c, s = rot_tables(ctx.cos, ctx.sin, cfg.bias_dims)
    TAB.clear()
    TAB.update(cos=ctx.cos, c=c, s=s, ch=jnp.swapaxes(c, 1, 2), sh=jnp.swapaxes(s, 1, 2))
    return ctx


def tables(ctx, cfg, major):
    if TAB.get("cos") is ctx.cos:
        return (TAB["ch"], TAB["sh"]) if major else (TAB["c"], TAB["s"])
    c, s = rot_tables(ctx.cos, ctx.sin, cfg.bias_dims)
    return (jnp.swapaxes(c, 1, 2), jnp.swapaxes(s, 1, 2)) if major else (c, s)


def ext_weights(w, h, dc, hd):
    """w [d, H*dc] -> a, b [d, H*hd]: the columns of w (zero in the reserved dims) and the columns
    with the two rotary halves of every head exchanged."""
    d, p = w.shape[0], dc // 2
    w3 = w.reshape(d, h, dc)
    z = jnp.zeros((d, h, hd - dc), w.dtype)
    a = jnp.concatenate([w3, z], -1).reshape(d, h * hd)
    b = jnp.concatenate([w3[..., p:], w3[..., :p], z], -1).reshape(d, h * hd)
    return a, b


def unit(hd, i, ft):
    e = np.zeros(hd, np.float32)
    e[i] = 1.0
    return jnp.asarray(e, ft)


def qkv_any(pa, x, ctx, ve, cfg, gain_v=None, major=False, per_head=False):
    """The reference qkv as matmuls and broadcast element-wise operations only. major: return
    [B, H, T, hd] (what the kernel takes) instead of [B, T, H, hd]."""
    dt = L.act_dtype(cfg)
    ft = L.hi(dt)
    b, t, d = x.shape
    h, hd = cfg.n_heads, cfg.head_dim
    dc = hd - cfg.bias_dims
    c, s = tables(ctx, cfg, major)

    def proj(w):                                   # [d, H*hd] -> float32 [B, T, H, hd] or [B, H, T, hd]
        if not major:
            return L.mm(x, w, dt).reshape(b, t, h, hd)
        w3 = w.reshape(d, h, hd)
        if per_head:
            return jnp.stack([L.mm(x, w3[:, i], dt) for i in range(h)], axis=1)
        return jnp.einsum("btd,dhk->bhtk", x, w3, preferred_element_type=ft)

    def head_axis(a):                              # [H, ...] against the head axis of the layout
        return a[:, None] if major else a

    def tok_axis(a):                               # [B, T] against the layout
        return a[:, None, :, None] if major else a[:, :, None, None]

    def rot(w):
        wa, wb = ext_weights(w, h, dc, hd)
        r = proj(wa) * c + proj(wb) * s
        return r * jax.lax.rsqrt(jnp.sum(r * r, axis=-1, keepdims=True) + L.L2_EPS)

    e_bias = unit(hd, dc, ft)
    q = rot(pa["wq"]) * head_axis(pa["q_gain"].astype(ft)[:, None]) + e_bias
    k = rot(pa["wk"])
    if gain_v is not None:
        if major:
            raise NotImplementedError
        v = L.gain_product(L.mm(x, pa["wv"], dt), gain_v, ft).reshape(b, t, h, hd)
    else:
        v = proj(pa["wv"])
    if ve is not None:
        n_ve = ve.shape[-1] // hd
        gate = 3.0 * jax.nn.sigmoid(x[..., :A.VE_GATE_CHANNELS].astype(ft) @ pa["ve_gate"].astype(ft))
        add = ve.astype(ft).reshape(b, t, n_ve, hd) * gate[..., None]
        if major:
            v = v + jnp.pad(jnp.swapaxes(add, 1, 2), ((0, 0), (0, h - n_ve), (0, 0), (0, 0)))
        else:
            v = v + jnp.pad(add, ((0, 0), (0, 0), (0, h - n_ve), (0, 0)))
    if ctx.conf is not None:
        cm = jnp.asarray(ctx.conf, ft) * unit(hd, hd - 1, ft)
        v = v * (1.0 - cm) + cm
    if "sink_logit" in pa and ctx.sink is not None:
        on = tok_axis(ctx.sink.astype(ft))
        keep = 1.0 - on
        k = k * keep + on * head_axis(pa["sink_logit"].astype(ft)[:, None] * e_bias)
        sv = jnp.concatenate([pa["sink_value"].astype(ft), jnp.zeros((h, 1), ft)], -1)
        v = v * keep + on * head_axis(sv)
    return q.astype(dt), k.astype(dt), v.astype(dt)


def attend_major(q, k, v, seg, cfg):
    """The splash kernels on [B, H, T, hd] operands (no transposes)."""
    from bes.model import attention_splash as SP

    sa = SP._sa()
    b, h, t, hd = q.shape
    if SP.padded_length(t) != t:
        raise NotImplementedError
    kernel = SP.make_kernel(t, h, SP.block_sizes(cfg, t), SP.use_interpret(cfg))
    ids = sa.SegmentIds(q=jnp.asarray(seg, jnp.int32), kv=jnp.asarray(seg, jnp.int32))
    return jax.vmap(lambda q1, k1, v1, s1: kernel(q1, k1, v1, segment_ids=s1))(q, k, v, ids)


def attention_major(per_head):
    def attention(pa, wo, x, ctx, ve, cfg, attend_fn=None, gain_v=None, gain_o=None, loop=None):
        if attend_fn is not None or gain_v is not None or gain_o is not None or "gate_w" in pa:
            raise NotImplementedError
        dt = L.act_dtype(cfg)
        ft = L.hi(dt)
        h, hd = cfg.n_heads, cfg.head_dim
        q, k, v = qkv_any(pa, x, ctx, ve, cfg, major=True, per_head=per_head)
        y = attend_major(q, k, v, ctx.seg, cfg)
        if "sink_logit" in pa and ctx.sink is not None and ctx.conf is not None:
            aux = jnp.asarray(ctx.conf, F32) * jnp.mean(1.0 - y[..., -1].astype(F32))
        else:
            aux = jnp.zeros((), F32)
        wo3 = wo.reshape(h, hd, wo.shape[-1])
        if per_head:
            out = L.mm(y[:, 0], wo3[0], dt)
            for i in range(1, h):
                out = out + L.mm(y[:, i], wo3[i], dt)
        else:
            out = jnp.einsum("bhtk,hkd->btd", y.astype(dt), wo3.astype(dt), preferred_element_type=ft)
        return out, aux
    return attention


def mlp_store(x, w_up, w_down, dtype, gain=None):
    """Hidden pre-activations stored in the activation dtype; relu^2 computed in float32."""
    if gain is not None:
        raise NotImplementedError
    z = L.mm(x, w_up, dtype, out_dtype=dtype)
    z = jnp.square(jax.nn.relu(z.astype(L.hi(dtype))))
    return L.mm(z.astype(dtype), w_down, dtype)


def mlp_act(x, w_up, w_down, dtype, gain=None):
    """Everything up to the down projection in the activation dtype."""
    if gain is not None:
        raise NotImplementedError
    z = jnp.square(jax.nn.relu(jnp.einsum("...k,kn->...n", x.astype(dtype), w_up.astype(dtype))))
    return L.mm(z, w_down, dtype)


def forward_tuple(params, consts, feats, cfg):
    """bes.model.model.forward with the lanes carried as a tuple (no slicing and stacking)."""
    from bes.model import head as HD
    M.check_cfg(cfg)
    dt = L.act_dtype(cfg)
    ft = L.hi(dt)
    ctx = M.make_ctx(consts, feats, cfg)
    e = M.embed(params, consts, feats, cfg)
    ve_all = M.value_embeddings(params, jnp.asarray(feats["tokens"]), cfg)
    gains = M.gain_rows(params, consts, jnp.asarray(feats["tokens"]), cfg)
    tok = M.token_rows(params, consts, jnp.asarray(feats["tokens"]), cfg)
    blocks = M.block_params(params, cfg)
    lp = params["lanes"]
    xs = {"alpha": lp["alpha"], "mix": lp["mix"], "beta": lp["beta"], "g": params["loop"]["g"]}
    if ve_all is not None:
        xs["ve"] = ve_all
    if gains is not None and cfg.gain_phases > 1:
        xs["phase"] = jnp.asarray(M.loop_phases(cfg))
    if cfg.attn_gate:
        xs["loop"] = jnp.arange(cfg.n_loops, dtype=jnp.int32)
    nl = cfg.n_lanes

    def step(lanes, x):
        a = x["alpha"].astype(ft)
        u = a[0] * lanes[0].astype(ft)
        for n in range(1, nl):
            u = u + a[n] * lanes[n].astype(ft)
        u = u.astype(dt)
        masses = []
        for i, pb in enumerate(blocks):
            u, m = M.block(pb, u, ctx, x.get("ve") if i == 0 else None, cfg,
                           gain=M.block_gain(gains, i, x.get("phase")), loop=x.get("loop"),
                           tok=None if tok is None else tok[i])
            masses.append(m)
        sf = L.rmsnorm(u, x["g"], out_dtype=dt).astype(ft)
        mx, bt = x["mix"].astype(ft), x["beta"].astype(ft)
        out = []
        for i in range(nl):
            acc = bt[i] * sf
            for j in range(nl):
                acc = acc + mx[i, j] * lanes[j].astype(ft)
            out.append(acc.astype(dt))
        return tuple(out), jnp.stack(masses)

    body = jax.checkpoint(step, prevent_cse=False) if cfg.remat else step
    lanes, masses = jax.lax.scan(body, tuple(e for _ in range(nl)), xs)
    sm = lanes[0].astype(ft)
    for n in range(1, nl):
        sm = sm + lanes[n].astype(ft)
    h = L.rmsnorm(sm, params["head"]["g"], out_dtype=ft)
    return h, {"sink_mass": jnp.mean(masses, axis=0)}


LANES = {(M, "forward"): forward_tuple}
ATTN = {
    "ref": {},
    "fused": {(A, "qkv"): qkv_any, (M, "make_ctx"): make_ctx_tab},
    "major": {(A, "attention"): attention_major(False), (M, "make_ctx"): make_ctx_tab},
    "major6": {(A, "attention"): attention_major(True), (M, "make_ctx"): make_ctx_tab},
}
MLP = {"ref": {}, "store": {(M, "mlp"): mlp_store}, "act": {(M, "mlp"): mlp_act}}
TILES = {"512": {}, "1024f": dict(splash_block_q=1024, splash_block_kv=1024, splash_block_kv_compute=512,
                                  splash_fused_bwd=True),
         "1024": dict(splash_block_q=1024, splash_block_kv=1024, splash_block_kv_compute=512),
         "512f": dict(splash_fused_bwd=True)}


class patched:
    def __init__(self, *maps):
        self.items = [kv for m in maps for kv in m.items()]

    def __enter__(self):
        self.old = [(mod, name, getattr(mod, name)) for (mod, name), _ in self.items]
        for (mod, name), fn in self.items:
            setattr(mod, name, fn)

    def __exit__(self, *exc):
        for mod, name, fn in self.old:
            setattr(mod, name, fn)


def main():
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/final.py"
    tok_root = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 and sys.argv[3] != "-" else None
    overrides = sys.argv[4].split(",") if len(sys.argv) > 4 else []
    cfg = load_config(config, overrides)
    mc = cfg.model
    mb = jax.device_put(micro_batch(cfg, tok_root, os.environ.get("BES_MB_CACHE")))
    consts = M.make_consts(mc, **SV.parse_switches(cfg.run.switches))
    params = M.init_params(mc, jax.random.key(0))
    # non-zero sink and output parameters, so that every path contributes to the comparison
    kk = jax.random.split(jax.random.key(7), 8)
    core = params["core"]
    attn = dict(core["attn"])
    for i, name in enumerate(("wo", "sink_logit", "sink_value", "q_gain")):
        if name in attn:
            attn[name] = attn[name] + 0.3 * jax.random.normal(kk[i], attn[name].shape, attn[name].dtype) \
                * (attn[name].shape[-1] ** -0.5 if name == "wo" else 1.0)
    mlp_p = dict(core["mlp"])
    mlp_p["w_down"] = mlp_p["w_down"] + jax.random.normal(kk[5], mlp_p["w_down"].shape, mlp_p["w_down"].dtype) \
        * mc.d_ffn ** -0.5
    params = dict(params, core=dict(core, attn=attn, mlp=mlp_p))
    feats = M.split_batch(mb)[0]
    dt = L.act_dtype(mc)
    ft = L.hi(dt)
    b, t = feats["tokens"].shape
    d = mc.d_model
    u0 = jax.random.normal(jax.random.key(1), (b, t, d), F32).astype(dt)
    loop0 = jnp.int32(0) if mc.attn_gate else None
    print(f"device {jax.devices()[0].device_kind}, {config} {overrides}, micro-batch {(b, t)}, act {dt}, "
          f"attention {mc.attn_impl}", flush=True)

    def clock(fn, args, n=4):
        f = jax.jit(fn)
        t0 = time.perf_counter(); out = f(*args); jax.block_until_ready(out); comp = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(n):
            out = f(*args)
        jax.block_until_ready(out)
        return (time.perf_counter() - t0) / n * 1e3, comp, out

    def block_run(body, mcfg):
        def f(u, pb, feats):
            ctx = M.make_ctx(consts, feats, mcfg)

            def step(c, _):
                return body(body(c, pb, ctx, mcfg), pb, ctx, mcfg), None
            out = jax.lax.scan(jax.checkpoint(step, prevent_cse=False), u, None, length=N)[0]
            return jnp.mean(jnp.square(out.astype(F32)))
        return f

    def attn_sub(u, pb, ctx, mcfg):
        x = L.rmsnorm(u, pb["g1"], out_dtype=dt)
        a, _ = A.attention(pb["attn"], pb["wo"], x, ctx, None, mcfg, loop=loop0)
        return (u.astype(ft) + a).astype(dt)

    def mlp_sub(u, pb, ctx, mcfg):
        x = L.rmsnorm(u, pb["g2"], out_dtype=dt)
        return (u.astype(ft) + M.mlp(x, pb["w_up"], pb["w_down"], dt)).astype(dt)

    def whole_block(u, pb, ctx, mcfg):
        return M.block(pb, u, ctx, None, mcfg, loop=loop0)[0]

    def part(label, body, maps, tiles="512"):
        if only and label.split(":")[0] not in only:
            return
        mcfg = dataclasses.replace(mc, **TILES[tiles])
        try:
            with patched(*maps):
                pb = M.block_params(params, mcfg)[0]
                ms, comp, out = clock(jax.value_and_grad(block_run(body, mcfg), argnums=(0, 1)), (u0, pb, feats))
            print(f"{label:58s} {ms / (2 * N):7.2f} ms per application   value {float(out[0]):.6f}   "
                  f"(compile {comp:.0f} s)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{label:58s} FAILED {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)

    def model(label, maps, tiles="512", remat=True, rows=None):
        if only and label.split(":")[0] not in only:
            return
        mcfg = dataclasses.replace(mc, remat=remat, **TILES[tiles])
        bt_ = mb if rows is None else {k: v[:rows] for k, v in mb.items()}
        nb = b if rows is None else rows
        try:
            with patched(*maps):
                ms, comp, out = clock(lambda p, bt: jax.value_and_grad(
                    lambda q: M.loss_fn(q, consts, bt, mcfg)[0])(p), (params, bt_))
            gn = float(jnp.sqrt(sum(jnp.sum(jnp.square(g.astype(F32))) for g in jax.tree.leaves(out[1]))))
            print(f"{label:58s} {ms:7.1f} ms per micro-batch = {nb * t / ms * 1e3:9,.0f} tok/s   loss "
                  f"{float(out[0]):.5f} |grad| {gn:.5f}   (compile {comp:.0f} s)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{label:58s} FAILED {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)

    part("w: warm-up (discard)", attn_sub, [ATTN["ref"]])
    part("a0: attention sub-layer, reference", attn_sub, [ATTN["ref"]])
    part("a1: attention sub-layer, fused glue", attn_sub, [ATTN["fused"]])
    part("a2: attention sub-layer, head-major (einsum)", attn_sub, [ATTN["major"]])
    part("a3: attention sub-layer, head-major (per-head products)", attn_sub, [ATTN["major6"]])
    part("a4: head-major (einsum), tiles 1024/512 fused backward", attn_sub, [ATTN["major"]], "1024f")
    part("a5: head-major (einsum), tiles 1024/512", attn_sub, [ATTN["major"]], "1024")
    part("a6: head-major (einsum), tiles 512 fused backward", attn_sub, [ATTN["major"]], "512f")
    part("m0: feed-forward sub-layer, reference", mlp_sub, [MLP["ref"]])
    part("m1: feed-forward, stored in the activation dtype", mlp_sub, [MLP["store"]])
    part("m2: feed-forward, computed in the activation dtype", mlp_sub, [MLP["act"]])
    part("b0: the block, reference", whole_block, [ATTN["ref"]])
    part("b1: the block, head-major + stored feed-forward", whole_block, [ATTN["major"], MLP["store"]])
    model("f0: model, reference", [ATTN["ref"]])
    model("f1: model, fused glue", [ATTN["fused"]])
    model("f2: model, head-major (einsum)", [ATTN["major"]])
    model("f3: model, head-major (per-head products)", [ATTN["major6"]])
    model("f4: model, head-major + stored feed-forward", [ATTN["major"], MLP["store"]])
    model("f5: model, head-major + feed-forward in act dtype", [ATTN["major"], MLP["act"]])
    model("f6: f4 + tiles 1024/512 fused backward", [ATTN["major"], MLP["store"]], "1024f")
    model("f7: fused glue + stored feed-forward + tiles 1024/512 f.", [ATTN["fused"], MLP["store"]], "1024f")
    model("f8: f4 + tiles 512 fused backward", [ATTN["major"], MLP["store"]], "512f")
    model("f9: f4 + tuple lanes", [ATTN["major"], MLP["store"], LANES])
    model("g0: f9 + tiles 1024/512 fused backward", [ATTN["major"], MLP["store"], LANES], "1024f")
    model("g1: reference + tuple lanes", [LANES])
    h = b // 2
    model("r0: reference, no remat, half micro-batch", [ATTN["ref"]], remat=False, rows=h)
    model("r1: f9, no remat, half micro-batch", [ATTN["major"], MLP["store"], LANES], remat=False, rows=h)
    model("r2: g0, no remat, half micro-batch", [ATTN["major"], MLP["store"], LANES], "1024f", remat=False, rows=h)
    model("r3: f9, half micro-batch (remat)", [ATTN["major"], MLP["store"], LANES], rows=h)


if __name__ == "__main__":
    main()
