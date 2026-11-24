"""Review item 40: at the graft point, are the gradients with respect to the shared parameters
equal under the source (final) and target (v9) configurations, and are the gradients of the new
leaves K, P_in, P_out exactly zero while those of U and G are not?

The source parameters are the stopped final run's checkpoint of step 29,722 (read from the
local snapshot N/snap_final_main16 without extracting it); the grafted
parameters are built as scripts/graft_run.py builds them (the v9 trainer's fresh tree for the live
run's effective configuration, seed 1, with every shared leaf replaced by the checkpoint's). Both
losses and gradients are computed on CPU in float32 with dense attention and no rematerialisation
(the overrides of the tool's --check), on real held-out rows (2 web rows and 2 chat rows of 1,024
tokens of bes.export.heldout_batches). Nothing is written except the JSON result.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = Path("/home/user/turingaward2027")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(HERE))
N = Path("/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad")


def say(m):
    print(time.strftime("%H:%M:%S ") + m, flush=True)


def main():
    import jax
    import jax.numpy as jnp
    import numpy as np

    from bes import export as EX
    from bes.config import load_config
    from bes.model import model as M
    from bes.tokenizer import load as load_tokenizer
    from bes.train import checkpoint as CK
    from bes.train import supervisor as SV
    import graft_run as G
    import snaptar

    os.chdir(REPO)
    fman = json.loads((N / "paper/lead_work/final_snap/run/manifest.json").read_text())
    vman = json.loads((N / "paper/data/_src/live/run/manifest.json").read_text())
    cfgs = {}
    for name, man in (("final", fman), ("v9", vman)):
        cfgs[name] = load_config(str(REPO / man["config"]), SV.run_overrides(man) + list(G.CHECK_OVERRIDES))
    say(f"seeds: final {cfgs['final'].run.seed}, v9 {cfgs['v9'].run.seed}")

    # source parameters
    leaves, cman, _ = snaptar.load_leaves(N / "snap_final_main16")     # the stopped run, step 29,722
    tf = jax.eval_shape(lambda: M.init_params(cfgs["final"].model, jax.random.PRNGKey(0)))
    paths_f, tmpl_f, treedef_f = CK.flatten_with_paths(tf)
    pf = jax.tree_util.tree_unflatten(treedef_f, [jnp.asarray(leaves["state/params/" + p]) for p in paths_f])
    say(f"source: step {cman['step']}, {len(paths_f)} leaves")

    # grafted parameters: the v9 trainer's fresh tree with every shared leaf replaced
    tree = G.fresh_tree(cfgs["v9"])
    pv_fresh = tree["state"].params
    paths_v, vals_v, treedef_v = CK.flatten_with_paths(pv_fresh)
    shared = [p for p in paths_v if p in set(paths_f)]
    new = [p for p in paths_v if p not in set(paths_f)]
    vals = []
    for p, v in zip(paths_v, vals_v):
        if p in set(paths_f):
            a = leaves["state/params/" + p]
            assert tuple(a.shape) == tuple(v.shape) and a.dtype == v.dtype, p
            vals.append(jnp.asarray(a))
        else:
            vals.append(jnp.asarray(v))
    pv = jax.tree_util.tree_unflatten(treedef_v, vals)
    del tree, leaves
    say(f"grafted: {len(shared)} shared leaves, new {new}")
    newstats = {p: {"max_abs": float(jnp.max(jnp.abs(v))), "shape": list(v.shape)} for p, v in zip(paths_v, vals) if p in new}
    say(f"new leaves at initialisation: {newstats}")

    tok = load_tokenizer()
    batches, _ = EX.heldout_batches(cfgs["v9"], tok, rows=8, domains=("web", "chat"), download=False, tok_dirs=[])
    result = {"source_step": cman["step"], "shared_leaves": len(shared), "new_leaves": new,
              "new_at_init": newstats, "batches": {}}
    for b in batches:
        batch = {k: v[:2] for k, v in b.arrays.items()}
        out = {}
        for name, params in (("final", pf), ("v9", pv)):
            mc = cfgs[name].model
            consts = M.make_consts(mc)
            t0 = time.time()
            with jax.default_matmul_precision("highest"):
                f = jax.jit(jax.value_and_grad(lambda p, c, bt, mc=mc: M.loss_fn(p, c, bt, mc)[0]))
                loss, g = f(params, consts, batch)
            loss = float(loss)
            gp, gl, _ = CK.flatten_with_paths(g)
            out[name] = (loss, dict(zip(gp, [np.asarray(x) for x in gl])))
            say(f"{b.domain}: {name} loss {loss:.7f} in {time.time() - t0:.0f} s")
        lf, gf = out["final"]
        lv, gv = out["v9"]
        rec = {"rows": 2, "tokens_per_row": int(b.arrays["w"].shape[1]),
               "weighted_tokens": float(np.asarray(batch["w"]).sum()),
               "loss_final": lf, "loss_v9": lv, "loss_difference": lv - lf, "shared": {}, "new": {}}
        worst_abs, worst_rel, identical = 0.0, 0.0, 0
        for p in shared:
            a, c = gf[p].astype(np.float64), gv[p].astype(np.float64)
            d = np.abs(c - a)
            mx = float(d.max())
            ref = float(np.abs(a).max())
            rel = mx / ref if ref > 0 else (0.0 if mx == 0 else float("inf"))
            l2 = float(np.linalg.norm(c - a) / max(np.linalg.norm(a), 1e-300))
            same = bool(np.array_equal(gf[p], gv[p]))
            identical += same
            worst_abs, worst_rel = max(worst_abs, mx), max(worst_rel, rel)
            rec["shared"][p] = {"max_abs_diff": mx, "max_rel_diff": rel, "l2_rel_diff": l2,
                                "bit_identical": same, "grad_max_abs": ref}
        for p in new:
            x = gv[p]
            rec["new"][p] = {"max_abs": float(np.abs(x).max()), "l2": float(np.linalg.norm(x.astype(np.float64))),
                             "exactly_zero": bool(not np.any(x))}
        rec["summary"] = {"shared_bit_identical": identical, "shared": len(shared),
                          "worst_max_abs_diff": worst_abs, "worst_max_rel_diff": worst_rel}
        result["batches"][b.domain] = rec
        say(f"{b.domain}: loss difference {lv - lf:+.3e}; shared gradients bit-identical {identical}/{len(shared)}; "
            f"worst max |diff| {worst_abs:.3e} (relative {worst_rel:.3e}); new: "
            + ", ".join(f"{p.split('/')[-2]}/{p.split('/')[-1]} zero={v['exactly_zero']} max={v['max_abs']:.3e}" for p, v in rec["new"].items()))
    (HERE / "graft_grad.json").write_text(json.dumps(result, indent=1))
    say("written graft_grad.json")


if __name__ == "__main__":
    main()
