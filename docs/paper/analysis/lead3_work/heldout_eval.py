"""Per-position held-out losses of one checkpoint on the rows of the export's check
(``bes.export.heldout_batches``: web 64, chat 64, proxy 12 rows of 1,024 tokens), float32,
dense attention, ``highest`` matmul precision: the arithmetic of ``scripts/midrun_eval.py``, but
the cross-entropy of every position is kept (so that rows can be bootstrapped, positions
stratified and bytes counted), and run-time switches can be set (``--switch page=0``).

    python -I heldout_eval.py --manifest RUN/manifest.json --ckpt CKPT_DIR_OR_SNAPSHOT \
        --name NAME --out OUT.npz [--switch page=0 --switch gain=0] [--rows-file ROWS.npz]

Nothing in the repository is written; the repository is only imported and read.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = Path("/home/user/turingaward2027")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))


def say(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt", required=True, help="extracted checkpoint directory or snapshot directory")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True, help="output prefix: OUT_<variant>.npz")
    ap.add_argument("--variant", action="append", default=[],
                    help="NAME:sw=v,sw=v (repeat); default one variant 'on' with the configured switches")
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--domains", default="web,chat,proxy")
    ap.add_argument("--rows-out", default=None, help="also save the held-out rows (tokens, w, role)")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np

    from bes import export as EX
    from bes.model import model as M
    from bes.tokenizer import load as load_tokenizer
    from bes.train import checkpoint as CK
    from bes.train import supervisor as SV
    import snaptar

    os.chdir(REPO)
    man = json.loads(Path(args.manifest).read_text())
    cfg = SV.effective_config(man) if (man.get("benchmark") or {}).get("complete") else SV.base_config(man)
    mcfg = EX.check_model_cfg(cfg.model)
    say(f"{args.name}: config {man.get('config')} hash {man.get('effective', {}).get('config_hash')}")

    t0 = time.time()
    tok = load_tokenizer()
    batches, info = EX.heldout_batches(cfg, tok, rows=args.rows, domains=tuple(args.domains.split(",")),
                                       download=False, tok_dirs=[])
    say(f"rows built in {time.time() - t0:.0f} s: " + ", ".join(f"{b.domain} {b.rows}" for b in batches))

    ck = Path(args.ckpt)
    t0 = time.time()
    if (ck / "arrays.npz").exists():
        leaves, cman = snaptar.load_dir_leaves(ck)
    else:
        leaves, cman, _ = snaptar.load_leaves(ck)
    say(f"checkpoint step {cman.get('step')} hash {cman.get('config_hash')} read in {time.time() - t0:.0f} s")

    template = jax.eval_shape(lambda: M.init_params(cfg.model, jax.random.PRNGKey(0)))
    paths, tmpls, treedef = CK.flatten_with_paths(template)
    vals = []
    for p, t in zip(paths, tmpls):
        a = leaves["state/params/" + p]
        assert tuple(a.shape) == tuple(t.shape) and a.dtype == t.dtype, (p, a.shape, a.dtype, t.shape, t.dtype)
        vals.append(jnp.asarray(a))
    params = jax.tree_util.tree_unflatten(treedef, vals)
    del leaves

    def per_pos(params, consts, batch):
        feats, y, w = M.split_batch(batch)
        h, aux = M.forward(params, consts, feats, mcfg)
        ce, ce_base, mass = M.token_ce(params, consts, h, feats, y, mcfg, parts=True)
        b_, t_ = y.shape
        return ce.reshape(b_, t_), ce_base.reshape(b_, t_), mass.reshape(b_, t_)

    fn = jax.jit(per_pos)
    variants = args.variant or ["on:"]
    for var in variants:
        vname, _, spec = var.partition(":")
        sw = {}
        for s_ in [x for x in spec.split(",") if x]:
            k, v = s_.split("=", 1)
            sw[k] = float(v)
        consts = M.make_consts(cfg.model, **sw)
        say(f"{args.name}/{vname}: switches { {k: float(v) for k, v in consts['enable'].items() if np.ndim(v) == 0} }")
        out = {}
        with jax.default_matmul_precision("highest"):
            for b in batches:
                ces, bases, masses = [], [], []
                t1 = time.time()
                for lo in range(0, b.rows, args.chunk):
                    sub = {k: v[lo:lo + args.chunk] for k, v in b.arrays.items()}
                    ce, base, mass = jax.device_get(fn(params, consts, sub))
                    ces.append(np.asarray(ce)); bases.append(np.asarray(base)); masses.append(np.asarray(mass))
                ce = np.concatenate(ces); base = np.concatenate(bases); mass = np.concatenate(masses)
                w = np.asarray(b.arrays["w"], np.float32)
                out[f"{b.domain}/ce"] = ce
                out[f"{b.domain}/ce_base"] = base
                out[f"{b.domain}/mass"] = mass
                out[f"{b.domain}/w"] = w
                out[f"{b.domain}/tokens"] = np.asarray(b.arrays["tokens"])
                if "reg" in b.arrays:
                    out[f"{b.domain}/role"] = np.asarray(b.arrays["reg"])[..., M.ROLE_FIELD]
                loss = float((ce * w).sum() / w.sum())
                say(f"{args.name}/{vname}: {b.domain}: {b.rows} rows, loss {loss:.4f} over {w.sum():.0f} "
                    f"weighted tokens in {time.time() - t1:.0f} s")
        doms = [b.domain for b in batches]
        tot = sum(float((out[f'{d}/ce'] * out[f'{d}/w']).sum()) for d in doms)
        wt = sum(float(out[f'{d}/w'].sum()) for d in doms)
        say(f"{args.name}/{vname}: all: loss {tot / wt:.4f} over {wt:.0f}")
        meta = {"name": args.name, "variant": vname, "manifest": args.manifest, "ckpt": str(ck),
                "step": cman.get("step"), "config_hash": cman.get("config_hash"), "switches": sw,
                "ledger": (cman.get("meta") or {}).get("ledger"), "heldout_info": {
                    d: {k: v for k, v in x.items() if k != "stats"} for d, x in info["domains"].items()}}
        path = f"{args.out}_{vname}.npz"
        np.savez(path, meta=json.dumps(meta, default=str), **out)
        say(f"written {path}")


if __name__ == "__main__":
    main()
