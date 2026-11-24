"""Continue a run under another configuration that only ADDS parameters: graft its newest
training checkpoint into the new configuration's parameter tree and write a run directory that
``scripts/run_24h.py --resume`` continues from.

    python scripts/graft_run.py SRC_RUN DST_RUN --config configs/v9.py [--override run.seed=1]
        [--plan-override model.table_grad=sorted] [--check]

Made for 24 Nov 2026: at 0.47 of a ``configs/final.py`` run the configuration was changed to
``configs/v9.py``, which is the same dense looped core plus per-token pages and attention output
gains (both exactly zero at initialisation) and no written library. Training v9 from scratch in
the time left would have seen fewer tokens than the run already had; grafting keeps them.

What is carried over, leaf by leaf (a leaf is matched by its path and must have the same shape
and dtype in both trees):

* the parameters and the optimiser's slots of every leaf both configurations have (for MuonH the
  cached initial norms are part of the slot), the optimiser's update count and the step count;
* the trainer's time ledger and guard, so the wall-clock schedule ``p = T / (T + remaining)``,
  and with it the learning rate and the curriculum, continue where they were.

What is new: the leaves only the new configuration has, at its own initialisation with fresh
optimiser slots, and the data stream (a new loader journal: the mixture differs; give
``--override run.seed=N`` so that the large sources are not read in the same order again).

From the source run directory the new one takes the manifest (budget, deadline, the benchmark's
execution plan and horizon, the post-training plan, micro_split, the speed switches) with the
configuration replaced, the stage ledger (pre-flight, background and benchmark stay done), the
background tasks' markers and the benchmark's records. It does not take the training directory
(except ``run.json``), the logs, the export or the compilation cache. The source is only read.

``--check`` evaluates the loss of the source and of the grafted parameters on one small synthetic
batch under both configurations (dense attention, float32, CPU friendly) and fails unless they
agree: the graft must not change the function. With ``--cpu`` the process never touches an
accelerator (use it next to a running trainer).

The last line is ``GRAFT <json>``.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

#: files and directories of a run directory the new one starts with (when the source has them)
CARRIED = ("stages.json", "stage_table.json", "background.json", "state", "benchmark", "post_plan",
           "micro_split", "fast")
#: overrides of the check (the parameter trees do not depend on them)
CHECK_OVERRIDES = ("model.attn_impl=xla", "model.act_dtype=float32", "model.remat=false")


def _key(override: str) -> str:
    return override.split("=", 1)[0]


def merged_overrides(old: Sequence[str], new: Sequence[str]) -> list[str]:
    """``old`` with every entry of ``new`` in place of an entry of the same field, else appended."""
    out = list(old)
    for o in new:
        hit = [i for i, x in enumerate(out) if _key(x) == _key(o)]
        if hit:
            out[hit[-1]] = o
            for i in reversed(hit[:-1]):
                del out[i]
        else:
            out.append(o)
    return out


def new_manifest(man: dict, config: str, overrides: Sequence[str], plan_overrides: Sequence[str]) -> dict:
    """The source manifest with the configuration replaced (hashes are filled in by the caller)."""
    from bes.train.benchmark import Plan

    new = copy.deepcopy(man)
    path = Path(config)
    path = path if path.is_absolute() else REPO / path
    if not path.is_file():
        raise SystemExit(f"{config}: no such configuration")
    try:
        new["config"] = str(path.relative_to(REPO))
    except ValueError:
        new["config"] = str(path)
    new["config_abs"] = str(path.resolve())
    new["overrides"] = merged_overrides(man.get("overrides", []), overrides)
    bench = new.get("benchmark") or {}
    if not bench.get("complete"):
        raise SystemExit("the source run has no benchmark result: nothing to continue from")
    if plan_overrides:
        fields = {_key(o).split(".", 1)[1]: o.split("=", 1)[1] for o in plan_overrides}
        unknown = sorted(set(fields) - {"table_grad"})
        if unknown:
            raise SystemExit(f"--plan-override: only model.table_grad can be changed, not {unknown}")
        plan = Plan.from_dict(bench["plan"]).with_table(fields["table_grad"])
        bench["plan"], bench["plan_key"], bench["plan_overrides"] = plan.to_dict(), plan.key, plan.overrides()
        bench["grafted_plan"] = True                     # not the plan the benchmark timed
    return new


def graft_arrays(src_ckpt: Path, template: Any) -> tuple[dict, list[dict], str, dict]:
    """The host arrays of ``template`` (the new configuration's fresh ``{"state", "rng"}``) with
    every leaf the checkpoint ``src_ckpt`` also has replaced by the checkpoint's; the report says
    which leaves were copied, which are new and which of the source were left behind."""
    import numpy as np

    from bes.train import checkpoint as CK

    arrays, records, treedef = CK.to_host(template)
    src_man = CK.read_manifest(src_ckpt)
    src = {r["path"]: r for r in src_man["leaves"]}
    copied, fresh, mismatched = [], [], []
    with np.load(Path(src_ckpt) / CK.ARRAYS, allow_pickle=False) as z:
        for rec in records:
            p = rec["path"]
            if p.startswith("state/consts"):
                continue                                  # the new configuration's own switches
            if p not in src:
                fresh.append(p)
                continue
            s = src[p]
            if s["shape"] != rec["shape"] or s["dtype"] != rec["dtype"] or s["stored"] != rec["stored"] \
                    or s.get("key_impl") != rec.get("key_impl"):
                mismatched.append(f"{p}: {s['dtype']}{s['shape']} -> {rec['dtype']}{rec['shape']}")
                continue
            arr = z[p]
            if list(arr.shape) != s["shape"]:
                raise SystemExit(f"{src_ckpt}: {p} does not match its record")
            arrays[p] = np.array(arr, copy=True)
            copied.append(p)
    left = sorted(p for p in src if p not in {r["path"] for r in records} and not p.startswith("state/consts"))
    numbers = lambda paths: int(sum(int(np.prod(r["shape"])) for r in records if r["path"] in set(paths)))  # noqa: E731
    params = lambda paths: [p for p in paths if p.startswith("state/params/")]                        # noqa: E731
    report = {"copied": len(copied), "copied_params": len(params(copied)),
              "copied_param_numbers": numbers(params(copied)),
              "new_params": sorted(params(fresh)), "new_param_numbers": numbers(params(fresh)),
              "new_other": sorted(p for p in fresh if not p.startswith("state/params/")),
              "mismatched": mismatched, "left_behind": left,
              "source_step": int(src_man["step"]), "source_config_hash": src_man.get("config_hash")}
    return arrays, records, treedef, report


def fresh_tree(ecfg) -> Any:
    """``{"state", "rng"}`` as a new trainer of ``ecfg`` builds it (``Trainer.setup``)."""
    import jax

    from bes.model import model as M
    from bes.roles import role_of
    from bes.train import supervisor as SV
    from bes.train.train import init_state

    k_init, rng = jax.random.split(jax.random.key(ecfg.run.seed))
    consts = SV.make_consts_fn(ecfg.run.switches)(ecfg.model)
    state, _ = init_state(ecfg, M.init_params, role_of, k_init, consts)
    return {"state": state, "rng": rng}


def function_check(src_man: dict, dst_man: dict, src_ckpt: Path, dst_ckpt: Path, rows: int, t: int) -> dict:
    """Loss of the source and of the grafted parameters on one synthetic batch, each under its
    own configuration with :data:`CHECK_OVERRIDES`."""
    import jax
    import numpy as np

    import tpu_tune
    from bes.config import load_config
    from bes.model import model as M
    from bes.train import checkpoint as CK
    from bes.train import supervisor as SV

    out = {}
    batch = None
    for name, man, ckpt in (("source", src_man, src_ckpt), ("grafted", dst_man, dst_ckpt)):
        ecfg = load_config(str(REPO / man["config"]) if not Path(man["config"]).is_absolute() else man["config"],
                           SV.run_overrides(man) + list(CHECK_OVERRIDES))
        tree = fresh_tree(ecfg)
        got, _ = CK.load(ckpt, tree, device=True)
        mc = ecfg.model
        if batch is None:
            batch = jax.device_put(tpu_tune.synth_batch(mc, rows, t))
        consts = SV.make_consts_fn(ecfg.run.switches)(mc)
        loss, aux = jax.jit(lambda p, c, b, mc=mc: M.loss_fn(p, c, b, mc))(got["state"].params, consts, batch)
        out[name] = float(loss)
        del tree, got
    out["difference"] = out["grafted"] - out["source"]
    out["ok"] = bool(np.isfinite(out["source"]) and abs(out["difference"]) <= 1e-6 * max(1.0, abs(out["source"])))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", help="run directory to continue from (only read)")
    ap.add_argument("dst", help="run directory to write (must not exist)")
    ap.add_argument("--config", required=True, help="the new configuration file")
    ap.add_argument("--override", action="append", default=[], help="section.field=value added to (or "
                    "replacing an entry of) the run's command-line overrides, e.g. run.seed=1")
    ap.add_argument("--plan-override", action="append", default=[],
                    help="model.table_grad=MODE: change the table gradient mode of the benchmark's plan")
    ap.add_argument("--check", action="store_true", help="fail unless the grafted parameters give the "
                    "source's loss on a synthetic batch")
    ap.add_argument("--check-shape", default="2x256", help="ROWSxTOKENS of the check's batch")
    ap.add_argument("--note", default="", help="recorded in the manifest")
    ap.add_argument("--cpu", action="store_true", help="set JAX_PLATFORMS=cpu before JAX is imported "
                    "(always next to a running trainer: this tool needs no accelerator)")
    args = ap.parse_args(argv)
    if args.cpu:
        import os

        os.environ["JAX_PLATFORMS"] = "cpu"
        if "jax" in sys.modules:
            raise SystemExit("--cpu: JAX is already imported in this process")

    from bes.config import config_hash
    from bes.train import checkpoint as CK
    from bes.train import supervisor as SV

    say = lambda msg: print(msg, flush=True)                                   # noqa: E731
    src, dst = Path(args.src).resolve(), Path(args.dst).resolve()
    if dst.exists():
        raise SystemExit(f"{dst} exists: choose a new directory")
    man = SV.read_manifest(src)
    if not man:
        raise SystemExit(f"{src} holds no run (manifest.json)")
    info = CK.latest(src / "train" / "ckpt")
    if info is None:
        raise SystemExit(f"{src}/train/ckpt holds no checkpoint")
    src_ckpt_man = CK.read_manifest(info.path)
    src_ecfg = SV.effective_config(man)
    if src_ckpt_man.get("config_hash") != config_hash(src_ecfg):
        raise SystemExit(f"{info.path.name} was written under configuration {src_ckpt_man.get('config_hash')}, "
                         f"the run's effective configuration is {config_hash(src_ecfg)}")
    new = new_manifest(man, args.config, args.override, args.plan_override)
    ecfg = SV.effective_config(new)
    new["config_hash"] = config_hash(SV.base_config(new))
    new["effective"] = {"overrides": SV.run_overrides(new), "config_hash": config_hash(ecfg)}
    say(f"graft: {src} ({man['config']}, {config_hash(src_ecfg)}) -> {dst} ({new['config']}, "
        f"{config_hash(ecfg)}); checkpoint {info.path.name}")

    t0 = time.time()
    arrays, records, treedef, report = graft_arrays(info.path, fresh_tree(ecfg))
    if report["mismatched"]:
        raise SystemExit("leaves of both configurations differ in shape or dtype: " + "; ".join(report["mismatched"][:8]))
    if report["left_behind"]:
        say(f"graft: {len(report['left_behind'])} leaves of the source have no place in the new tree: "
            f"{report['left_behind'][:6]}")
    say(f"graft: {report['copied_params']} parameter leaves ({report['copied_param_numbers']:,} numbers) carried "
        f"over with their optimiser slots; new: {report['new_params']} ({report['new_param_numbers']:,} numbers)")

    meta = dict(src_ckpt_man.get("meta") or {})
    meta.update(cursor=0, reason="graft", grafted_from=str(info.path))
    tmp = dst.with_name(dst.name + f".graft-{int(time.time())}")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "train").mkdir(parents=True)
    out_info = CK.write(tmp / "train" / "ckpt", arrays, records, treedef, seq=0, step=report["source_step"],
                        config_hash=config_hash(ecfg), meta=meta)
    run_json = json.loads((src / "train" / "run.json").read_text())
    run_json["config_hash"] = config_hash(ecfg)
    (tmp / "train" / "run.json").write_text(json.dumps(run_json))
    for name in CARRIED:
        a = src / name
        if a.is_dir():
            shutil.copytree(a, tmp / name)
        elif a.is_file():
            shutil.copy2(a, tmp / name)
    led = meta.get("ledger") or {}
    t_train = float((led.get("seconds") or {}).get("train", 0.0))
    left = float(meta.get("train_end") or 0.0) - float(led.get("mark") or 0.0)
    p_graft = t_train / (t_train + left) if t_train > 0 and left > 0 else None
    new["graft"] = {"time": time.time(), "from_run": str(src), "from_config": man["config"],
                    "from_config_hash": config_hash(src_ecfg), "checkpoint": info.path.name,
                    "step": report["source_step"], "tokens": led.get("tokens"),
                    "p": None if p_graft is None else round(p_graft, 6),
                    "new_params": report["new_params"], "new_param_numbers": report["new_param_numbers"],
                    "copied_param_numbers": report["copied_param_numbers"], "note": args.note}
    new.pop("jax_cache", None)                            # configured again by the continued run
    SV.write_manifest(tmp, new)
    result = {"dst": str(dst), "checkpoint": out_info.path.name, "config_hash": config_hash(ecfg),
              "seconds": round(time.time() - t0, 1), **{k: report[k] for k in (
                  "copied_params", "copied_param_numbers", "new_params", "new_param_numbers", "left_behind",
                  "source_step")}}
    if args.check:
        rows, t = (int(x) for x in args.check_shape.lower().split("x"))
        chk = function_check(man, new, info.path, out_info.path, rows, t)
        result["check"] = chk
        say(f"graft: check on {rows} x {t} synthetic tokens: source loss {chk['source']:.7f}, grafted "
            f"{chk['grafted']:.7f}, difference {chk['difference']:+.2e}")
        if not chk["ok"]:
            shutil.rmtree(tmp, ignore_errors=True)
            say("GRAFT " + json.dumps({**result, "ok": False}))
            raise SystemExit("the grafted parameters do not give the source's loss: nothing was written")
    tmp.rename(dst)
    say("GRAFT " + json.dumps({**result, "ok": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
