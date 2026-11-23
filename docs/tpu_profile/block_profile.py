"""Where does one block application spend its time? A scan of 8 steps of two applications of a
part of the real block (as the model's loops), at one micro-batch, forward and with gradients
(rematerialised steps, as in training). Times are per application."""
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


def main():
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/final.py"
    tok_root = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
    cfg = load_config(config, [])
    mc = cfg.model
    mb = jax.device_put(micro_batch(cfg, tok_root, os.environ.get("BES_MB_CACHE")))
    consts = M.make_consts(mc, **SV.parse_switches(cfg.run.switches))
    params = M.init_params(mc, jax.random.key(0))
    feats = M.split_batch(mb)[0]
    dt = L.act_dtype(mc)
    ft = L.hi(dt)
    b, t = feats["tokens"].shape
    d, h, hd = mc.d_model, mc.n_heads, mc.head_dim
    u0 = jax.random.normal(jax.random.key(1), (b, t, d), F32).astype(dt)
    pb0 = M.block_params(params, mc)[0]
    pb0 = dict(pb0, wo=(jax.random.normal(jax.random.key(2), pb0["wo"].shape, F32) * d ** -0.5).astype(dt),
               w_down=(jax.random.normal(jax.random.key(3), pb0["w_down"].shape, F32) * mc.d_ffn ** -0.5).astype(dt))
    loop0 = jnp.int32(0) if mc.attn_gate else None
    print(f"device {jax.devices()[0].device_kind}, {config}, micro-batch {(b, t)}, d {d}, heads {h}x{hd}, "
          f"d_ffn {mc.d_ffn}, act {dt}, {2 * N} applications per call", flush=True)

    def run(body, remat):
        def f(u, pb, feats):
            ctx = M.make_ctx(consts, feats, mc)

            def step(c, _):
                return body(body(c, pb, ctx), pb, ctx), None
            st = jax.checkpoint(step, prevent_cse=False) if remat else step
            out = jax.lax.scan(st, u, None, length=N)[0]
            return jnp.mean(jnp.square(out.astype(F32)))
        return f

    def clock(fn, n=4):
        f = jax.jit(fn)
        t0 = time.perf_counter(); out = f(u0, pb0, feats); jax.block_until_ready(out); comp = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(n):
            out = f(u0, pb0, feats)
        jax.block_until_ready(out)
        return (time.perf_counter() - t0) / n / (2 * N) * 1e3, comp

    def timed(name, body, grad=True):
        if only and name.split(":")[0] not in only:
            return
        try:
            fw, c1 = clock(run(body, False))
            line = f"{name:52s} forward {fw:7.2f} ms"
            if grad:
                gr, c2 = clock(jax.grad(run(body, True), argnums=(0, 1)))
                line += f"   forward+backward {gr:7.2f} ms   (compile {c1:.0f}+{c2:.0f} s)"
            print(line, flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{name:52s} FAILED {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)

    def full(u, pb, ctx):
        return M.block(pb, u, ctx, None, mc, loop=loop0)[0]

    def attn_sub(u, pb, ctx, attend_fn=None, cfg_=mc):
        x = L.rmsnorm(u, pb["g1"], out_dtype=dt)
        a, _ = A.attention(pb["attn"], pb["wo"], x, ctx, None, cfg_, attend_fn, loop=loop0)
        return (u.astype(ft) + a).astype(dt)

    def attn_nokernel(u, pb, ctx):
        return attn_sub(u, pb, ctx, attend_fn=lambda q, k, v: ((v + 0.01 * (q + k)).astype(ft), jnp.zeros((), F32)))

    def mlp_sub(u, pb, ctx):
        x = L.rmsnorm(u, pb["g2"], out_dtype=dt)
        return (u.astype(ft) + L.mlp(x, pb["w_up"], pb["w_down"], dt)).astype(dt)

    def mlp_lean(u, pb, ctx):
        x = L.rmsnorm(u, pb["g2"], out_dtype=dt)
        z = jnp.square(jax.nn.relu(jnp.einsum("...k,kn->...n", x, pb["w_up"])))
        return u + jnp.einsum("...k,kn->...n", z, pb["w_down"])

    def norm_only(u, pb, ctx):
        return L.rmsnorm(u, pb["g1"], out_dtype=dt)

    def mm_f32(u, pb, ctx):
        return L.mm(u, pb["wo"], dt, out_dtype=dt)

    def mm_act(u, pb, ctx):
        return jnp.einsum("...k,kn->...n", u, pb["wo"])

    def mm_up_down(u, pb, ctx):
        return jnp.einsum("...k,kn->...n", jnp.einsum("...k,kn->...n", u, pb["w_up"]), pb["w_down"])

    def kernel(cfg_):
        from bes.model import attention_splash as SP

        def body(u, pb, ctx):
            x = u.reshape(b, t, h, hd)
            qk = x * jnp.asarray(0.05, dt)
            return SP.attend(qk, qk, x, ctx.seg, cfg_).reshape(b, t, d).astype(dt)
        return body

    def dense(u, pb, ctx):
        x = u.reshape(b, t, h, hd)
        qk = x * jnp.asarray(0.05, dt)
        y, _ = A.attend_xla(qk, qk, x, ctx.seg, ctx.seg, ctx.pos, ctx.pos, None)
        return y.reshape(b, t, d).astype(dt)

    timed("A: the block", full)
    timed("B: attention sub-layer (norm, qkv, kernel, Wo)", attn_sub)
    timed("C: feed-forward sub-layer (norm, up, relu^2, down)", mlp_sub)
    timed("D: attention sub-layer without the kernel", attn_nokernel)
    timed("E: attention kernel alone (configured)", kernel(mc))
    timed("F: rmsnorm alone", norm_only)
    timed("G: one d x d product, float32 accumulation", mm_f32)
    timed("H: one d x d product, activation dtype", mm_act)
    timed("I: up and down products only, activation dtype", mm_up_down)
    timed("J: feed-forward sub-layer in the activation dtype", mlp_lean)
    timed("K: dense attention kernel alone", dense)
    for name, kw in (("L: kernel, tiles 1024/1024/1024", dict(splash_block_q=1024, splash_block_kv=1024,
                                                               splash_block_kv_compute=1024)),
                     ("M: kernel, tiles 1024/1024/512", dict(splash_block_q=1024, splash_block_kv=1024,
                                                             splash_block_kv_compute=512)),
                     ("N: kernel, tiles 256/256/256", dict(splash_block_q=256, splash_block_kv=256,
                                                           splash_block_kv_compute=256)),
                     ("O: kernel, fused backward", dict(splash_fused_bwd=True))):
        timed(name, kernel(dataclasses.replace(mc, **kw)))


if __name__ == "__main__":
    main()
