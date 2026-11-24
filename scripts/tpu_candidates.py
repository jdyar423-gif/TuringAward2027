"""The speed switches on the chip, in one short pause of the live run: which of the module-level
``FAST_*`` switches (default off, read at trace time; ``docs/CHANGES.md``) make the trainer's
micro-batch faster, with the loss and the gradients unchanged to float rounding.

    python scripts/tpu_candidates.py --config configs/final.py \\
        --override model.table_grad=take --override optim.micro_seqs=16 --micro-split 4 \\
        --out /content/bes_cli/candidates.json
    python scripts/tpu_candidates.py --smoke          # CPU, tiny configuration (a few minutes)

What is timed is what the trainer computes for one micro-batch: ``bes.train.train.accumulate``
over a ``[1, B, T]`` batch, cut on the device into ``--micro-split`` pieces exactly as the live
run cuts it (``value_and_grad`` of the loss of every piece and the float32 sum of the
gradients), at the trainer's shapes (``schedule.micro_shape``: 16 x 1024 and 8 x 2048 for
``configs/final.py`` with ``optim.micro_seqs=16``). The batches are synthetic
(``tpu_tune.synth_batch``, three packed segments per row) and the parameters freshly
initialised, so nothing is read from or
written to the run; nothing else may hold the device. The first program timed in a process is
about ten times slower for a few calls, so the first one runs ``--warmup-calls`` discarded
calls; every program then runs two more discarded calls and ``--calls`` timed ones (each waited
for), and the median counts.

Variants, in this order (most promising first by the engineers' expected effect, so that the
time budget cuts the least promising): per sequence length the baseline (all off), all on, each
switch alone; then, for every switch that was not at least ``--min-gain`` faster alone (or
failed), all on without it; last, only with ``--info``, ``FAST_SPLASH_IO`` with
``model.remat_keep=kernel`` (a configuration field: it changes the run's configuration hash, so
the live run cannot take it). A variant starts only if the time used so far plus the dearest
variant so far fits in ``--budget-s``; the rest are listed as skipped.

Every variant prints its compile seconds, ms per micro-batch, tokens per second, the loss and
its difference from the baseline's, the relative change of the gradient norm and the largest
difference of a gradient leaf relative to that leaf's largest entry (bit-exact switches give 0;
``FAST_CE`` differs at float32 rounding, which bfloat16 roundings carry through the backward
pass, up to about 1e-2 on small leaves whose gradient is a cancelling sum). A relative loss
difference above ``--loss-rtol`` or a gradient-norm difference above ``--grad-rtol`` is more
than float rounding: the variant is marked ``MISMATCH`` and never recommended.

The recommendation, per length and overall (``--long-share`` weights the lengths): among the
switches that are at least ``--min-gain`` faster alone, the measured combination (a single
switch, all on, or all on without the others) that is fastest; nothing if none is faster than
the baseline. The last line is ``CANDIDATES <json>``: every variant (``ms``, ``tokens_per_s``,
``loss_diff``, ...), the skipped ones, ``recommend`` and ``gain_pct``. To adopt a set, flip the
defaults of those switches in their modules (``integration.md``); no configuration field and no
checkpoint changes.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

#: (switch, module, expected gain in percent of a micro-batch from the engineers' compile-only
#: estimates for TPU v6e with libtpu 0.0.50: {(seq_len, cut into pieces): percent}); the numbers
#: only order the variants (the most promising first)
SWITCHES: tuple[tuple[str, str, dict], ...] = (
    ("FAST_RELU2", "bes.model.layers", {(1024, True): 2.5, (2048, True): 5.2, (1024, False): 0.5,
                                         (2048, False): 0.5}),
    ("FAST_SPLASH_IO", "bes.model.attention_splash", {(1024, True): 1.5, (2048, True): 1.2,
                                                      (1024, False): 1.5, (2048, False): 1.2}),
    ("FAST_CE", "bes.model.head", {(1024, True): 1.0, (2048, True): 0.95, (1024, False): 2.5,
                                   (2048, False): 2.5}),
)
NAMES = tuple(s[0] for s in SWITCHES)
LIVE_OVERRIDES = ("model.table_grad=take", "optim.micro_seqs=16")
SMOKE_OVERRIDES = ("model.attn_impl=splash", "model.act_dtype=bfloat16", "model.remat=true",
                   "model.table_grad=take", "model.clock_planes=0", "model.use_conf=false")


def expected(name: str, t: int, pieces: bool) -> float:
    table = dict((s[0], s[2]) for s in SWITCHES)[name]
    key = (1024 if t < 2048 else 2048, bool(pieces))
    return float(table.get(key, 1.0))


def set_switches(on: Sequence[str]) -> None:
    """Every switch of :data:`SWITCHES` to ``name in on`` (they are read at trace time). Where
    the checkout also has the attention kernel switch ``attention_splash.FAST_KERNEL`` (commit
    6bf9b6e and later; its default can come from ``BES_FAST_KERNEL``), it is kept at ``""``, the
    configured splash kernels, which every variant here is about."""
    for name, module, _ in SWITCHES:
        mod = importlib.import_module(module)
        if not hasattr(mod, name):
            raise SystemExit(f"tpu_candidates: {module}.{name} does not exist in this checkout "
                             f"({REPO}): apply the switches' patch first")
        setattr(mod, name, name in on)
    splash = importlib.import_module("bes.model.attention_splash")
    if hasattr(splash, "FAST_KERNEL"):
        splash.FAST_KERNEL = ""


def packed_batch(mcfg, rows: int, t: int, seed: int = 0, n_segments: int = 3) -> dict:
    """``tpu_tune.synth_batch`` with ``n_segments`` packed segments per row (random starts,
    ``BOS`` at every start), as the loader packs documents: the kernels' segment masks take part
    in the equality check against the baseline."""
    import numpy as np

    import tpu_tune
    from bes.config import special_id

    b = tpu_tune.synth_batch(mcfg, rows, t, seed)
    rng = np.random.default_rng(seed + 1)
    bos = special_id("<|bos|>", int(mcfg.vocab_size))
    seg = np.zeros((rows, t), np.int32)
    for r in range(rows):
        starts = np.sort(rng.choice(np.arange(16, t - 16), size=n_segments - 1, replace=False))
        for i, s in enumerate(starts):
            seg[r, s:] = i + 1
            b["tokens"][r, s] = bos
    b["seg"] = seg
    return b


@dataclasses.dataclass
class Variant:
    label: str
    t: int
    switches: tuple[str, ...]
    overrides: dict = dataclasses.field(default_factory=dict)   # model fields (information only)
    kind: str = "candidate"            # baseline | candidate | minus | info


def main(argv: Sequence[str] | None = None) -> int:
    t_start = time.time()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/final.py")
    ap.add_argument("--override", action="append", default=None,
                    help="section.field=value (repeatable; default: the live run's "
                         + " ".join(LIVE_OVERRIDES) + ")")
    ap.add_argument("--micro-split", default=None,
                    help="pieces per micro-batch as the live run cuts them: a number or "
                         "seq_len:pieces pairs, e.g. 1024:4,2048:8 (default 4; --smoke: 2)")
    ap.add_argument("--lengths", default="", help="comma-separated sequence lengths (default: "
                                                  "data.seq_len and data.seq_len_late)")
    ap.add_argument("--rows", default="", help="rows per micro-batch per length, e.g. 1024:16,2048:8 "
                                               "(default: the trainer's, schedule.micro_shape)")
    ap.add_argument("--budget-s", type=float, default=None,
                    help="seconds for the whole measurement, start-up included (default 200; --smoke: 300)")
    ap.add_argument("--calls", type=int, default=None, help="timed calls per variant (default 5; --smoke: 2)")
    ap.add_argument("--warmup-calls", type=int, default=None,
                    help="discarded calls of the first program (default 5; --smoke: 1)")
    ap.add_argument("--min-gain", type=float, default=0.01,
                    help="a switch is a candidate when it alone is at least this much faster (default 1%%)")
    ap.add_argument("--loss-rtol", type=float, default=1e-5,
                    help="largest relative loss difference that counts as float rounding")
    ap.add_argument("--grad-rtol", type=float, default=1e-4,
                    help="largest relative difference of the gradient norm that counts as float rounding")
    ap.add_argument("--long-share", type=float, default=0.6,
                    help="share of the remaining training at the long length, for the overall "
                         "recommendation (default 0.6: the live run of 24 Nov trains at T=2048 from "
                         "progress 0.67, about 12:30 UTC, to 18:34 UTC)")
    ap.add_argument("--info", action="store_true",
                    help="also time FAST_SPLASH_IO with model.remat_keep=kernel, last and for "
                         "information only (it changes the configuration hash)")
    ap.add_argument("--only", default="", help="comma-separated variant labels to run (testing)")
    ap.add_argument("--out", default=None, help="write the report (JSON) here")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU smoke test: configs/smoke.py with the live run's features (splash in "
                         "interpret mode, bfloat16, remat, take), tiny shapes")
    args = ap.parse_args(argv)
    if args.smoke:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        if args.config == "configs/final.py":
            args.config = "configs/smoke.py"
        overrides = list(SMOKE_OVERRIDES) + list(args.override or [])
        defaults = dict(micro_split="2", budget_s=300.0, calls=2, warmup_calls=1)
    else:
        overrides = list(LIVE_OVERRIDES if args.override is None else args.override)
        defaults = dict(micro_split="4", budget_s=200.0, calls=5, warmup_calls=5)
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    config = args.config if os.path.isabs(args.config) else str(REPO / args.config)

    import jax
    import jax.numpy as jnp
    import numpy as np

    from bes.config import load_config
    from bes.model import model as M
    from bes.train import schedule
    from bes.train import supervisor as SV
    from bes.train.train import accumulate, parse_micro_split, split_for

    say = lambda msg: print(msg, flush=True)                                   # noqa: E731
    cfg = load_config(config, overrides)
    mc = cfg.model
    if mc.attn_impl == "splash":
        from bes.model import attention_splash
        attention_splash.register_alias()
    set_switches(())
    split = parse_micro_split(args.micro_split)
    lengths = ([int(x) for x in args.lengths.split(",") if x.strip()] if args.lengths else
               sorted({int(cfg.data.seq_len), int(cfg.data.seq_len_late)}))
    rows_of = {t: schedule.micro_shape(cfg, t)[1] for t in lengths if
               (cfg.optim.micro_seqs * cfg.data.seq_len) % t == 0}
    if args.rows:
        rows_of.update({int(a): int(b) for a, b in (it.split(":") for it in args.rows.split(",") if it)})
    lengths = [t for t in lengths if t in rows_of]
    pieces = {t: split_for(split, t) for t in lengths}
    for t in lengths:
        if pieces[t] < 1 or rows_of[t] % pieces[t]:
            raise SystemExit(f"tpu_candidates: {pieces[t]} pieces do not divide {rows_of[t]} rows at T={t}")
    dev = jax.devices()[0]
    say(f"tpu_candidates: {os.path.relpath(config, REPO)} {overrides}, device {dev.device_kind}, "
        f"jax {jax.__version__}; micro-batches " + ", ".join(
            f"{rows_of[t]} x {t} in {pieces[t]} piece(s) of {rows_of[t] // pieces[t]} x {t}" for t in lengths)
        + f"; budget {args.budget_s:.0f} s")
    if dev.platform != "tpu":
        say("  (not a TPU: the times below say nothing about the chip)")

    consts = M.make_consts(mc, **SV.parse_switches(cfg.run.switches))
    params = M.init_params(mc, jax.random.key(0))
    batches = {t: jax.device_put({k: v[None] for k, v in packed_batch(mc, rows_of[t], t).items()})
               for t in lengths}

    @jax.jit
    def compare(g, g0):
        """(norm of g, norm of g0, largest |g - g0| / max |g0| over the leaves, its leaf)."""
        la, lb = jax.tree.leaves(g), jax.tree.leaves(g0)
        rel = jnp.stack([jnp.max(jnp.abs(a - b)) / jnp.maximum(jnp.max(jnp.abs(b)), 1e-30) for a, b in zip(la, lb)])
        norm = lambda ls: jnp.sqrt(sum(jnp.sum(jnp.square(x.astype(jnp.float32))) for x in ls))  # noqa: E731
        return norm(la), norm(lb), jnp.max(rel), jnp.argmax(rel)

    leaf_names = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(params)[0]]

    # ------------------------------------------------------------------ the variants
    def order(t: int) -> list[str]:
        return sorted(NAMES, key=lambda n: -expected(n, t, pieces[t] > 1))

    def label(switches: Sequence[str], t: int, extra: str = "") -> str:
        names = "+".join(s.replace("FAST_", "") for s in NAMES if s in switches) or "baseline"
        return f"T{t} {'all' if len(switches) == len(NAMES) and len(NAMES) > 1 else names}{extra}"

    plan: list[Variant] = []
    for t in lengths:
        plan.append(Variant(label((), t), t, (), kind="baseline"))
        plan.append(Variant(label(NAMES, t), t, NAMES))
        plan += [Variant(label((n,), t), t, (n,)) for n in order(t) if len(NAMES) > 1]
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    results: dict[str, dict] = {}
    skipped: list[str] = []
    base_out: dict[int, Any] = {}
    cost = {"max": 0.0}
    first = {"done": False}

    def run(v: Variant) -> dict | None:
        if only and v.kind != "baseline" and v.label not in only:
            return None
        if v.t not in base_out and v.kind != "baseline":
            skipped.append(v.label + " (no baseline at this length)")
            return None
        used = time.time() - t_start
        if used + cost["max"] > args.budget_s:
            skipped.append(v.label)
            say(f"  {v.label:34s} skipped: {used:.0f} s used, about {cost['max']:.0f} s more would pass "
                f"the budget of {args.budget_s:.0f} s")
            return None
        t0 = time.time()
        set_switches(v.switches)
        mcfg = dataclasses.replace(mc, **v.overrides) if v.overrides else mc
        k = pieces[v.t]

        def fn(p, c, b):                    # everything the trainer keeps: grads, loss, weight, metrics
            return accumulate(M.loss_fn, p, c, b, mcfg, micro_split=k)
        rec: dict[str, Any] = {"label": v.label, "kind": v.kind, "t": v.t, "rows": rows_of[v.t], "pieces": k,
                               "switches": list(v.switches), "overrides": dict(v.overrides)}
        try:
            c0 = time.perf_counter()
            compiled = jax.jit(fn).lower(params, consts, batches[v.t]).compile()
            rec["compile_s"] = time.perf_counter() - c0
            for _ in range(2 + (0 if first["done"] else max(0, args.warmup_calls))):
                out = compiled(params, consts, batches[v.t])
                jax.block_until_ready(out)
            first["done"] = True
            times = []
            for _ in range(max(1, args.calls)):
                c0 = time.perf_counter()
                out = compiled(params, consts, batches[v.t])
                jax.block_until_ready(out)
                times.append((time.perf_counter() - c0) * 1e3)
        except Exception as e:  # noqa: BLE001 - a variant that does not run is reported, never chosen
            rec.update(ok=False, error=f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}")
            say(f"  {v.label:34s} FAILED {rec['error'][:200]}")
            results[v.label] = rec
            cost["max"] = max(cost["max"], time.time() - t0)
            return rec
        finally:
            set_switches(())
        grads, loss = out[0], out[1]
        ms = statistics.median(times)
        rec.update(ok=True, ms=ms, ms_all=[round(x, 3) for x in times],
                   tokens_per_s=rows_of[v.t] * v.t / ms * 1e3, loss=float(loss))
        if v.kind == "baseline":
            base_out[v.t] = (grads, float(loss), ms)
            jax.block_until_ready(compare(grads, grads))               # compiled once, here
            rec.update(loss_diff=0.0, grad_norm_rel_diff=0.0, worst_leaf_rel_diff=0.0)
        else:
            g0, l0, ms0 = base_out[v.t]
            n1, n0, worst, idx = (np.asarray(x) for x in compare(grads, g0))
            rec.update(loss_diff=float(loss) - l0, gain_pct=100.0 * (1.0 - ms / ms0),
                       grad_norm_rel_diff=float(n1 / max(float(n0), 1e-30) - 1.0),
                       worst_leaf_rel_diff=float(worst),
                       worst_leaf=leaf_names[int(idx)] if float(worst) > 0 else "")
            if not (abs(rec["loss_diff"]) <= args.loss_rtol * abs(l0) and np.isfinite(float(n1))
                    and abs(rec["grad_norm_rel_diff"]) <= args.grad_rtol):
                rec.update(ok=False, error="MISMATCH: the loss or the gradient norm differs by more "
                                           "than float rounding")
        del out, grads
        rec["seconds"] = time.time() - t0
        cost["max"] = max(cost["max"], rec["seconds"])
        results[v.label] = rec
        flag = "" if rec["ok"] else "   MISMATCH"
        gain = f"{rec['gain_pct']:+6.2f}%" if "gain_pct" in rec else "       "
        say(f"  {v.label:34s} {ms:8.2f} ms {gain} {rec['tokens_per_s']:10,.0f} tok/s  loss {rec['loss']:.6f} "
            f"(diff {rec['loss_diff']:+.2e})  grad norm {rec['grad_norm_rel_diff']:+.1e}  worst leaf "
            f"{rec['worst_leaf_rel_diff']:.1e} {rec.get('worst_leaf', '')}  compile {rec['compile_s']:.1f} s  "
            f"[{min(times):.2f}-{max(times):.2f}]{flag}")
        return rec

    def single(name: str, t: int) -> dict | None:
        r = results.get(label((name,), t))
        return r if r and r.get("ok") else None

    def gain_alone(name: str, t: int) -> float | None:
        r = single(name, t)
        return None if r is None else r["gain_pct"] / 100.0

    for v in plan:
        run(v)
    # all on without each switch that was not at least --min-gain faster alone (or failed alone)
    if len(NAMES) > 2:
        for t in lengths:
            for name in order(t)[::-1]:
                r = results.get(label((name,), t))
                if r is not None and (not r.get("ok") or r["gain_pct"] / 100.0 < args.min_gain):
                    rest = tuple(n for n in NAMES if n != name)
                    run(Variant(label(rest, t), t, rest, kind="minus"))
    if args.info and "FAST_SPLASH_IO" in NAMES and getattr(mc, "remat", False) and lengths:
        t = lengths[0]
        run(Variant(label(("FAST_SPLASH_IO",), t, " remat_keep=kernel (info)"), t, ("FAST_SPLASH_IO",),
                    overrides={"remat_keep": "kernel"}, kind="info"))

    # ------------------------------------------------------------------ the recommendation
    def measured(t: int) -> dict[tuple[str, ...], float]:
        out = {}
        for r in results.values():
            if r["t"] == t and r.get("ok") and r["kind"] != "info":
                out[tuple(sorted(r["switches"]))] = r["ms"]
        return out

    recommend: dict[str, list[str]] = {}
    gains: dict[str, float] = {}
    for t in lengths:
        m = measured(t)
        if () not in m:
            continue
        cands = {n for n in NAMES if (gain_alone(n, t) or 0.0) >= args.min_gain}
        best = min((s for s in m if set(s) <= cands), key=lambda s: m[s])
        recommend[str(t)] = list(best) if m[best] < m[()] else []
        gains[str(t)] = 100.0 * (1.0 - m[tuple(sorted(recommend[str(t)]))] / m[()])
    if len(lengths) >= 2 and all(str(t) in recommend for t in lengths[:2]):
        t1, t2 = lengths[:2]
        m1, m2 = measured(t1), measured(t2)
        w = min(max(args.long_share, 0.0), 1.0)
        rel = {s: (1 - w) * m1[s] / m1[()] + w * m2[s] / m2[()] for s in m1 if s in m2}
        cands = {n for n in NAMES if (n,) in rel and rel[(n,)] <= 1.0 - args.min_gain}
        best = min((s for s in rel if set(s) <= cands), key=lambda s: rel[s])
        recommend["overall"] = list(best) if rel[best] < 1.0 else []
        gains["overall"] = 100.0 * (1.0 - rel[tuple(sorted(recommend["overall"]))])
    elif lengths and str(lengths[0]) in recommend:
        recommend["overall"] = recommend[str(lengths[0])]
        gains["overall"] = gains[str(lengths[0])]
    for key, sw in recommend.items():
        say(f"recommended ({key}): {', '.join(sw) or 'none'}  ({gains[key]:+.2f}% against all off)")
    if skipped:
        say(f"skipped ({len(skipped)}): {', '.join(skipped)}")
    report = {"device": dev.device_kind, "platform": dev.platform, "jax": jax.__version__,
              "config": os.path.relpath(config, REPO), "overrides": overrides, "micro_split": args.micro_split,
              "shapes": {str(t): [rows_of[t], t, pieces[t]] for t in lengths},
              "variants": list(results.values()), "skipped": skipped, "recommend": recommend,
              "gain_pct": gains, "min_gain": args.min_gain, "long_share": args.long_share,
              "seconds": round(time.time() - t_start, 1)}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1, default=str) + "\n")
    brief = dict(report, variants=[{k: r.get(k) for k in ("label", "kind", "t", "rows", "pieces", "switches",
                                                         "ok", "ms", "tokens_per_s", "gain_pct", "loss",
                                                         "loss_diff", "grad_norm_rel_diff", "worst_leaf_rel_diff",
                                                         "worst_leaf", "compile_s", "error") if k in r}
                                   for r in results.values()])
    say("CANDIDATES " + json.dumps(brief, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
