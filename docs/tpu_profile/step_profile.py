"""Where does a training step spend its time? Times parts of the real model at one micro-batch
(B=32, T=1024) on the first device, and variants with single features removed."""
import dataclasses, sys, tempfile, time
from pathlib import Path

REPO = Path("/content/TuringAward2027") if Path("/content/TuringAward2027").is_dir() else Path.cwd()
sys.path.insert(0, str(REPO))
import jax, jax.numpy as jnp, numpy as np
from bes.config import LoaderCfg, load_config
from bes.data.loader import Loader
from bes.model import model as M
from bes.optim.build import lr_inputs
from bes.roles import role_of
from bes.tokenizer import load as load_tokenizer
from bes.train import schedule as S, supervisor as SV, train as TR


def main():
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/final.py"
    tok_root = sys.argv[2] if len(sys.argv) > 2 else None
    only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
    cfg = load_config(config, [])
    stage = dataclasses.replace(S.stage_at(0.0, cfg), library_frac_base=0.0, library_frac_chat=0.0)
    with tempfile.TemporaryDirectory(dir=str(REPO / cfg.run.data_dir)) as tmp:
        with Loader(cfg, load_tokenizer(), Path(tok_root) if tok_root else SV.tok_root(cfg, tmp), library=None,
                    n_workers=4, seed=cfg.run.seed, loader_cfg=dataclasses.replace(LoaderCfg(), workers=4),
                    journal=Path(tmp) / "j.jsonl") as loader:
            batch = loader(0, stage)
    shape = S.micro_shape(cfg, cfg.data.seq_len)
    print("device", jax.devices()[0].device_kind, "config", config, "step shape", shape,
          {k: (v.shape, str(v.dtype)) for k, v in batch.items()}, flush=True)
    mb = jax.device_put({k: np.ascontiguousarray(v[0]) for k, v in batch.items()})      # one micro-batch
    full = jax.device_put({k: np.ascontiguousarray(v) for k, v in batch.items()})
    sw = SV.parse_switches(cfg.run.switches)


    def timed(name, fn, *args, n=5):
        if only and name.split(":")[0] not in only:
            return None
        try:
            f = jax.jit(fn)
            t0 = time.perf_counter(); out = f(*args); jax.block_until_ready(out); comp = time.perf_counter() - t0
            t0 = time.perf_counter()
            for _ in range(n):
                out = f(*args)
            jax.block_until_ready(out)
            dt = (time.perf_counter() - t0) / n
            print(f"{name:58s} {dt * 1e3:9.1f} ms   (first call {comp:5.1f} s)", flush=True)
            return dt
        except Exception as e:  # noqa: BLE001
            print(f"{name:58s} FAILED {type(e).__name__}: {str(e).splitlines()[0][:160]}", flush=True)
            return None


    def setup(mcfg):
        consts = M.make_consts(mcfg, **sw)
        c2 = dataclasses.replace(cfg, model=mcfg)
        state, opt = TR.init_state(c2, M.init_params, role_of, jax.random.key(0), consts)
        return c2, consts, state, opt


    base = cfg.model
    c, consts, state, opt = setup(base)
    params = state.params
    vg = lambda mcfg, cs: (lambda p, b: jax.value_and_grad(lambda q: M.loss_fn(q, cs, b, mcfg)[0])(p))
    timed("a: forward hidden, one micro-batch", lambda p, b: M.forward(p, consts, M.split_batch(b)[0], base)[0], params, mb)
    timed("b: loss (forward + head), one micro-batch", lambda p, b: M.loss_fn(p, consts, b, base)[0], params, mb)
    t_vg = timed("c: value_and_grad(loss), one micro-batch", vg(base, consts), params, mb)
    grads = jax.jit(vg(base, consts))(params, mb)[1]
    glr = lr_inputs(c)
    timed("d: optimizer update alone (one per step)",
          lambda g, os_, p: opt.update(g, os_, p, jnp.float32(1e-3), momentum=jnp.float32(0.95),
                                       wd_scale=jnp.float32(1.0), group_lr=glr), grads, state.opt_state, params)
    timed("e: gradient accumulation (scan over 8 micro-batches)",
          lambda p, b: TR.accumulate(M.loss_fn, p, consts, b, base)[:2], params, full, n=3)
    variants = {
        "f: no rematerialisation": dict(remat=False),
        "g: dense attention (xla)": dict(attn_impl="xla"),
        "h: no candidate input features": dict(use_cand_in=False),
        "i: no candidate output law": dict(use_cand_out=False),
        "j: no candidate tape at all": dict(use_cand_in=False, use_cand_out=False),
        "k: no registers": dict(use_reg=False),
        "l: no value embeddings (ve_dim 0)": dict(ve_dim=0),
        "m: table gradient take": dict(table_grad="take"),
        "n: no sink": dict(use_sink=False),
        "o: 1 loop instead of 8": dict(n_loops=1),
        "p: float32 activations": dict(act_dtype="float32"),
    }
    for name, change in variants.items():
        if only and name.split(":")[0] not in only:
            continue
        try:
            mcfg = dataclasses.replace(base, **change)
            c2, cs, st, _ = setup(mcfg)
            timed(name + " -> value_and_grad", vg(mcfg, cs), st.params, mb, n=4)
        except Exception as e:  # noqa: BLE001
            print(f"{name}: FAILED {type(e).__name__}: {str(e).splitlines()[0][:160]}", flush=True)


if __name__ == "__main__":
    main()
