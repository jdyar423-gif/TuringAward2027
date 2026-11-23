"""Held-out loss of the checkpoints of one or more runs of ``scripts/run_24h.py``, on the same rows.

    python scripts/midrun_eval.py --run runs/v9 --run runs/final --out runs/compare.json
    python scripts/midrun_eval.py --run runs/v9 --ckpts newest --device cpu     # next to a run that trains

The instrument of a head-to-head between two configurations that trained for a while on the same
machine (``docs/CHANGES.md`` section 19): every kept training checkpoint of every run is scored
on the held-out rows of the export's check (``bes.export.heldout_batches``: a ClimbMix file
beyond the download cap, the smol-smoltalk test split, the user turns of the proxy prompts),
with float32 activations, dense attention and ``highest`` matmul precision
(``bes.export.check_model_cfg``), so that two runs are compared on identical token rows with
identical arithmetic whatever their execution plan was.

Per checkpoint it reports the training tokens and training seconds stored with it (the run's
ledger), the weighted loss in nats per token overall and per domain, the loss of the tokens
predicted inside assistant turns (chat rows), and what the candidate tape buys. With two runs it
adds the paired difference at each run's newest checkpoint and, when their checkpoints allow
it, at matched training time and at matched training tokens (linear interpolation of the loss
in log tokens between the two neighbouring checkpoints; never an extrapolation).

``--device cpu`` hides the accelerator from JAX (``JAX_PLATFORMS=cpu``), so the script can run
while a run of ``scripts/run_24h.py`` owns the TPU; it is slow at full model size (minutes per
checkpoint). Without it the script takes the default device: stop the run first.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ROLE_ASSISTANT = 3          # bes.exact.registers.ROLE_ASSISTANT (checked in main)


def _say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def checkpoints(run: Path, which: str) -> list[Any]:
    """The complete training checkpoints of a run directory, oldest first (``newest``: the last)."""
    from bes.train import checkpoint as CK

    infos = CK.list_checkpoints(run / "train" / "ckpt")
    if not infos:
        raise FileNotFoundError(f"{run}: no training checkpoint under train/ckpt")
    return infos[-1:] if which == "newest" else infos


def run_config(run: Path):
    """``(cfg, manifest)``: the effective configuration when the benchmark has fixed it, else the
    base one (a run stopped before its benchmark has no checkpoint anyway)."""
    from bes.train import supervisor as SV

    man = SV.read_manifest(run)
    if man is None:
        raise FileNotFoundError(f"{run}: no manifest.json (not a run directory of scripts/run_24h.py)")
    cfg = SV.effective_config(man) if (man.get("benchmark") or {}).get("complete") else SV.base_config(man)
    return cfg, man


def metrics_fn(model_cfg):
    """One jitted metrics function of ``loss_fn`` for the check configuration of ``model_cfg``."""
    import jax

    from bes import export as EX
    from bes.model import model as M

    mcfg = EX.check_model_cfg(model_cfg)
    return jax.jit(lambda p, c, b: M.loss_fn(p, c, b, mcfg)[1])


def score(fn, params: Any, consts: Any, batches: Sequence[Any], chunk: int) -> dict[str, Any]:
    """Sums over the held-out rows: weighted cross-entropy overall, per domain and inside
    assistant turns, and the tape gain (``-log q(y) - ce``, weighted)."""
    import jax
    import numpy as np

    by: dict[str, dict[str, float]] = {}
    with jax.default_matmul_precision("highest"):
        for b in batches:
            acc = by.setdefault(b.domain, {"ce": 0.0, "w": 0.0, "ce_asst": 0.0, "w_asst": 0.0, "gain": 0.0})
            for lo in range(0, b.rows, chunk):
                sub = {k: v[lo:lo + chunk] for k, v in b.arrays.items()}
                m = jax.device_get(fn(params, consts, sub))
                w = float(m["weight"])
                acc["ce"] += float(m["ce_sum"])
                acc["w"] += w
                acc["gain"] += float(m["tape_gain"]) * w
                acc["ce_asst"] += float(np.asarray(m["ce_by_role"])[ROLE_ASSISTANT])
                acc["w_asst"] += float(np.asarray(m["weight_by_role"])[ROLE_ASSISTANT])
    out: dict[str, Any] = {"by_domain": {}}
    for dom, a in by.items():
        out["by_domain"][dom] = {
            "loss": a["ce"] / a["w"] if a["w"] > 0 else None, "weight": a["w"],
            "assistant_loss": a["ce_asst"] / a["w_asst"] if a["w_asst"] > 0 else None,
            "assistant_weight": a["w_asst"],
            "tape_gain": a["gain"] / a["w"] if a["w"] > 0 else None}
    ce, w = sum(a["ce"] for a in by.values()), sum(a["w"] for a in by.values())
    out["loss"] = ce / w if w > 0 else None
    out["weight"] = w
    return out


def evaluate_run(run: Path, args: argparse.Namespace, tok) -> dict[str, Any]:
    from bes import export as EX
    from bes.decode import respond
    from bes.train import checkpoint as CK
    from bes.train import supervisor as SV

    cfg, man = run_config(run)
    t0 = time.time()
    batches, info = EX.heldout_batches(cfg, tok, rows=args.rows, seq_len=args.seq_len or None,
                                       domains=tuple(args.domains.split(",")), download=not args.no_download,
                                       tok_dirs=[SV.tok_root(cfg, run) / "tok"], log=_say)
    if not batches:
        raise RuntimeError(f"{run}: no held-out rows could be built: {json.dumps(info['domains'])[:600]}")
    fn = metrics_fn(cfg.model)
    rows = {b.domain: b for b in batches}
    out: dict[str, Any] = {
        "run": str(run), "config": str(man.get("config")), "overrides": SV.run_overrides(man),
        "plan": (man.get("benchmark") or {}).get("plan_key"), "horizon_tokens": SV.horizon_of(man),
        "ladder": ((man.get("benchmark") or {}).get("ladder") or {}),
        "heldout": {d: {k: v for k, v in x.items() if k != "stats"} for d, x in info["domains"].items()},
        "seq_len": info["seq_len"], "checkpoints": []}
    _say(f"{run}: held-out rows built in {time.time() - t0:.0f} s: "
         + ", ".join(f"{d} {b.rows} rows" for d, b in rows.items()))
    for ck in checkpoints(run, args.ckpts):
        t1 = time.time()
        meta = (CK.read_manifest(ck.path).get("meta") or {})
        led = meta.get("ledger") or {}
        params, consts, linfo = respond.load_checkpoint(ck.path, cfg.model)
        res = score(fn, params, consts, batches, args.chunk)
        seconds = led.get("seconds") or {}
        rec = {"name": ck.name, "step": ck.step, "tokens": led.get("tokens"),
               "train_seconds": seconds.get("train") if isinstance(seconds, dict) else None,
               "reason": meta.get("reason"), "consts": linfo.get("consts"), **res,
               "eval_seconds": round(time.time() - t1, 1)}
        out["checkpoints"].append(rec)
        chat = res["by_domain"].get("chat") or {}
        _say(f"{run.name}: {ck.name}: {rec['tokens'] or 0:.3e} tokens, loss {res['loss']:.4f} ("
             + ", ".join(f"{d} {x['loss']:.4f}" for d, x in res["by_domain"].items())
             + (f"; chat assistant {chat['assistant_loss']:.4f}" if chat.get("assistant_loss") is not None else "")
             + f") in {rec['eval_seconds']:.0f} s")
    out["_rows"] = rows
    return out


def interpolate(points: Sequence[tuple[float, float]], x: float) -> float | None:
    """The loss at ``x`` by linear interpolation in ``log x`` between the neighbouring points
    ``(x_i, loss_i)``; ``None`` outside their range (no extrapolation) or without two points."""
    pts = sorted((float(a), float(b)) for a, b in points if a and a > 0 and b is not None)
    if x is None or x <= 0:
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return y1
            t = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return y0 + t * (y1 - y0)
    for xi, yi in pts:
        if abs(xi - x) <= 1e-9 * max(1.0, abs(x)):
            return yi
    return None


def _pick(rec: dict, key: str) -> float | None:
    """``loss`` | ``<domain>`` | ``<domain>.assistant`` of one checkpoint record."""
    if key == "loss":
        return rec.get("loss")
    dom, _, sub = key.partition(".")
    d = (rec.get("by_domain") or {}).get(dom) or {}
    return d.get("assistant_loss" if sub == "assistant" else "loss")


def compare(a: dict, b: dict) -> dict[str, Any]:
    """Paired differences ``b - a`` (negative: ``b`` is better) of two evaluated runs: at each
    run's newest checkpoint, at the training time both reached and at the training tokens both
    reached (the smaller of the two newest values, the other run interpolated)."""
    import numpy as np

    ra, rb = a["_rows"], b["_rows"]
    same = sorted(ra) == sorted(rb) and all(
        np.array_equal(ra[d].arrays["tokens"], rb[d].arrays["tokens"])
        and np.array_equal(ra[d].arrays["w"], rb[d].arrays["w"]) for d in ra)
    keys = ["loss"] + [d for d in ra] + [f"{d}.assistant" for d in ra if d == "chat"]
    la, lb = a["checkpoints"][-1], b["checkpoints"][-1]
    out: dict[str, Any] = {"a": a["run"], "b": b["run"], "same_rows": bool(same), "difference": "b - a",
                           "newest": {"a": la["name"], "b": lb["name"], "a_tokens": la["tokens"],
                                      "b_tokens": lb["tokens"], "a_train_seconds": la["train_seconds"],
                                      "b_train_seconds": lb["train_seconds"], "delta": {}}}
    for k in keys:
        va, vb = _pick(la, k), _pick(lb, k)
        out["newest"]["delta"][k] = None if va is None or vb is None else vb - va
    for axis in ("train_seconds", "tokens"):
        xa, xb = la.get(axis), lb.get(axis)
        if not xa or not xb:
            out[f"matched_{axis}"] = None
            continue
        x = min(float(xa), float(xb))
        m: dict[str, Any] = {axis: x, "a": {}, "b": {}, "delta": {}}
        for k in keys:
            va = interpolate([(c.get(axis), _pick(c, k)) for c in a["checkpoints"]], x)
            vb = interpolate([(c.get(axis), _pick(c, k)) for c in b["checkpoints"]], x)
            m["a"][k], m["b"][k] = va, vb
            m["delta"][k] = None if va is None or vb is None else vb - va
        out[f"matched_{axis}"] = m
    return out


def table(runs: Sequence[dict], cmp: dict | None) -> str:
    lines = []
    for r in runs:
        lines.append(f"{r['run']}  ({r['config']}; plan {r['plan'] or '-'})")
        lines.append(f"  {'checkpoint':<28} {'tokens':>10} {'train min':>9} {'loss':>8} "
                     + " ".join(f"{d:>8}" for d in r["_rows"]) + f" {'chat asst':>9}")
        for c in r["checkpoints"]:
            chat = (c["by_domain"].get("chat") or {}).get("assistant_loss")
            lines.append(f"  {c['name']:<28} {c['tokens'] or 0:>10.3e} "
                         f"{(c['train_seconds'] or 0) / 60:>9.1f} {c['loss']:>8.4f} "
                         + " ".join(f"{c['by_domain'][d]['loss']:>8.4f}" for d in r["_rows"])
                         + (f" {chat:>9.4f}" if chat is not None else f" {'-':>9}"))
    if cmp is not None:
        lines.append(f"difference b - a (b = {cmp['b']}, a = {cmp['a']}; negative: b is better); "
                     f"same rows: {cmp['same_rows']}")
        for name in ("newest", "matched_train_seconds", "matched_tokens"):
            m = cmp.get(name)
            if not m:
                lines.append(f"  {name}: not available")
                continue
            at = ""
            if name == "matched_train_seconds":
                at = f" at {m['train_seconds'] / 60:.1f} min of training"
            elif name == "matched_tokens":
                at = f" at {m['tokens']:.3e} tokens"
            lines.append(f"  {name}{at}: " + ", ".join(
                f"{k} {v:+.4f}" if v is not None else f"{k} n/a" for k, v in m["delta"].items()))
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="append", required=True, help="a run directory (repeat for a comparison)")
    ap.add_argument("--ckpts", default="all", choices=["all", "newest"],
                    help="every kept training checkpoint (default) or only the newest")
    ap.add_argument("--rows", type=int, default=64, help="held-out rows per domain (default 64)")
    ap.add_argument("--seq-len", type=int, default=0, help="row length (default: the run's base length)")
    ap.add_argument("--domains", default="web,chat,proxy")
    ap.add_argument("--chunk", type=int, default=8, help="rows per forward pass")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu"],
                    help="cpu: hide the accelerator from JAX (for use next to a run that trains)")
    ap.add_argument("--no-download", action="store_true", help="use only held-out files already on disk")
    ap.add_argument("--out", default=None, help="write the result as JSON")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    from bes.exact import registers
    from bes.tokenizer import load as load_tokenizer

    assert registers.ROLE_ASSISTANT == ROLE_ASSISTANT
    paths = [Path(r).resolve() for r in args.run]
    if args.out:
        args.out = str(Path(args.out).resolve())
    os.chdir(REPO)                       # the data paths of a configuration are relative to the code
    tok = load_tokenizer()
    runs = [evaluate_run(p, args, tok) for p in paths]
    cmp = compare(runs[0], runs[1]) if len(runs) == 2 else None
    print(table(runs, cmp))
    if args.out:
        result = {"time": time.time(), "runs": [{k: v for k, v in r.items() if k != "_rows"} for r in runs],
                  "comparison": cmp, "rows": args.rows, "device": args.device}
        tmp = Path(args.out + f".tmp-{os.getpid()}")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(result, indent=1, default=str))
        os.replace(tmp, args.out)
        _say(f"written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
