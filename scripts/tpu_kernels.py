"""Which attention kernel makes one piece of the training step fastest: one measurement on the
idle accelerator, for every value of the kernel switch ``bes.model.attention_splash.FAST_KERNEL``.

    python scripts/tpu_kernels.py --config configs/final.py \\
        --override model.table_grad=take --override optim.micro_seqs=4

The training step cuts each micro-batch into pieces on the device (``micro_split``): 4 rows of
1024 tokens now, 2 rows of 2048 later. ``optim.micro_seqs=4`` makes the micro-batch one such
piece, and this script times one ``value_and_grad`` of the loss on it at both lengths (``rows``
at ``data.seq_len``, the same tokens at ``data.seq_len_late``) for each kernel of ``--kernels``,
with a synthetic batch (three segments per row, ``BOS`` at every segment start, the loader's
fields) and freshly initialised parameters, so nothing else may hold the device. Per (kernel,
length): one compile, warm calls (more for the first program of the process, which runs slow
for a few calls on the TPU), then ``--calls`` rounds of ``--per-round`` calls issued back to back
(the device time of a call, without a host round trip per call; the median over the rounds). It
prints the loss and the largest gradient difference against the first kernel of the list (the
reference, ``""`` = the configured splash kernels) so that equality can be seen; the kernels
compute the same values to float rounding (``tests/model/test_attention_onetile.py``). No new
(kernel, length) starts after ``--budget-s`` seconds; every kernel is timed at the shorter
length before any at the longer one, and a ``BEST @ T`` line per length names the fastest kernel
that ran with the reference's values (loss within 1e-4 relative, and on every parameter leaf a
relative L2 gradient difference within 5%, or within 3 times that of the yardstick below when it
ran: far above the rounding differences, far below a wrong kernel).

An entry of ``--kernels`` may add the model's other speed switches (``bes.model.fast``) after
the kernel, joined by ``+``: ``onetile:save+relu2+ce`` times that kernel with ``FAST_RELU2`` and
``FAST_CE`` on (``+io``: ``FAST_SPLASH_IO``, for the configured splash kernels), so that one run
of this script ranks whole combinations. ``--steps`` adds, for the entries it names, the whole
accumulation step of the trainer (``bes.train.train.accumulate`` over all the pieces of a step,
as ``scripts/tpu_micro.py`` times it): the number the live rate follows. ``--step-rows 4,2``
times that step for several piece sizes (rows per piece at ``data.seq_len``; the same tokens per
piece at the other lengths), ``--step-lengths`` only at some lengths: which ``micro_split`` goes
with the chosen kernel.

``--smoke``: the tiny CPU configuration (``configs/smoke.py`` with the splash path, remat,
bfloat16 and interpret mode; pieces of 4 x 128 and 2 x 256 tokens), at most two rounds of one
call; checks that the script and the kernels run (CPU times in interpret mode mean nothing).

The last line is ``KERNELS <json>``: per (kernel, length) ``ms`` (median), ``tokens_per_s``,
``loss``, ``loss_diff``, ``grad_rel_diff`` / ``grad_rel_l2`` (the largest max|g - g_ref| /
max|g_ref| and |g - g_ref| / |g_ref| over the parameter leaves), ``compile_s`` or the reason it
failed, ``mosaic_calls`` (kernels in the compiled program: a ``save`` value has fewer than the
others, or the flag did not take) and ``speedup`` against the reference at the same length. For scale, the last default
kernel is a change of the splash tiles alone (``blocks:kvc=1024``, the kind of change the run
already took as "same values"): it moves the gradients of the bfloat16 model by about as much as
the other kernels (one rounding of a bfloat16 activation somewhere in 16 block applications); a
kernel far above it (or a loss difference beyond ~1e-4) computes something else and is not a
candidate. At 2048 tokens the configured kernels round dq once more than the one-tile kernels
(one bfloat16 dq per 1024-key tile, summed in XLA), so their gradient differences are somewhat
larger there than at 1024.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

#: the reference first, then the candidates by their static bundle counts (docs/TPU_NOTES.md,
#: "Attention kernels for one tile"; "onetile" before "onetile:save" because it changes only the
#: kernel, the flag also the rematerialisation), last a tile change of the configured kernels as
#: the yardstick of "same values to float rounding". Not in the list (``--kernels`` takes any
#: value): the weaker one-tile variants (``onetile:hp=1``, ``onetile:bs=128,hp=1``), tile and
#: kernel settings the static counts put behind the configured ones (``blocks:kvc=256``,
#: ``blocks:sep``), ``xlamap`` (cannot win) and ``tokamax:...`` (behind ``blocks:save`` by the
#: same counts, and needs ``pip install --no-deps tokamax==0.0.17`` on the runtime).
YARDSTICK = "blocks:kvc=1024"
DEFAULT_KERNELS = ("", "onetile", "onetile:save", "blocks:save", YARDSTICK)
SMOKE_OVERRIDES = ("model.attn_impl=splash", "model.remat=true", "model.act_dtype=bfloat16",
                   "model.table_grad=take", "optim.micro_seqs=4")


def segmented_batch(mcfg, rows: int, t: int, seed: int = 0, n_segments: int = 3) -> dict:
    """``tpu_tune.synth_batch`` with ``n_segments`` segments per row (random starts, ``BOS`` at
    each start): the segment mask of the kernels is exercised as in packed training rows."""
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


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(REPO / "configs" / "final.py"))
    ap.add_argument("--override", action="append", default=[], help="section.field=value")
    ap.add_argument("--kernels", default=None,
                    help="';'-separated FAST_KERNEL values, the first one is the reference "
                         f"(default: {';'.join(DEFAULT_KERNELS)})")
    ap.add_argument("--lengths", default="", help="comma-separated sequence lengths (default: "
                                                   "data.seq_len and data.seq_len_late)")
    ap.add_argument("--calls", type=int, default=5, help="timed rounds per (kernel, length) (median)")
    ap.add_argument("--per-round", type=int, default=10, help="calls issued back to back per round")
    ap.add_argument("--steps", default="", help="';'-separated entries of --kernels that are also timed as "
                                                "a whole accumulation step ('all': every entry)")
    ap.add_argument("--step-calls", type=int, default=2, help="timed whole steps per (entry, length, rows)")
    ap.add_argument("--step-rows", default="", help="comma-separated rows per piece at data.seq_len for the "
                                                    "whole steps (default: optim.micro_seqs)")
    ap.add_argument("--step-lengths", default="", help="comma-separated lengths of the whole steps (default: all)")
    ap.add_argument("--budget-s", type=float, default=600.0, help="no new (kernel, length) after this")
    ap.add_argument("--smoke", action="store_true", help="tiny configuration on CPU (interpret mode)")
    args = ap.parse_args(argv)

    import jax
    import jax.numpy as jnp
    import numpy as np

    from bes.config import load_config
    from bes.model import attention_splash as S
    from bes.model import fast as F
    from bes.model import model as M
    from bes.train import supervisor as SV
    from bes.train.train import accumulate

    t_start = time.time()
    overrides = list(args.override)
    if args.smoke:
        args.config = str(REPO / "configs" / "smoke.py")
        overrides = list(SMOKE_OVERRIDES) + overrides
        args.calls, args.per_round = min(args.calls, 2), 1
    kernels = DEFAULT_KERNELS if args.kernels is None else tuple(args.kernels.split(";"))

    def setting_of(entry: str) -> dict:
        """``"onetile:save+relu2+ce"`` -> the switches of ``bes.model.fast`` (all of them: an
        entry says everything that is on)."""
        kern, *flags = entry.split("+")
        unknown = [f for f in flags if f not in F.FLAGS]
        if unknown:
            raise ValueError(f"--kernels: unknown switch {unknown[0]!r} in {entry!r} (one of {', '.join(F.FLAGS)})")
        S.parse_switch(kern)
        return {"kernel": kern, **{name: name in flags for name in F.FLAGS}}

    for k in kernels:
        setting_of(k)                                       # unknown names fail before any work
    whole = set(kernels) if args.steps.strip() == "all" else {x for x in args.steps.split(";") if x in kernels}
    if args.steps.strip() not in ("", "all") and whole != set(args.steps.split(";")):
        raise SystemExit("--steps names an entry that --kernels does not have")
    defaults = F.current()
    step_rows = [int(x) for x in args.step_rows.split(",") if x.strip()]
    step_lengths = {int(x) for x in args.step_lengths.split(",") if x.strip()}
    cfg = load_config(args.config, overrides)
    mc = cfg.model
    rows1, t1, t2 = int(cfg.optim.micro_seqs), int(cfg.data.seq_len), int(cfg.data.seq_len_late)
    per_step = int(cfg.optim.tokens_per_step)
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()] or sorted({t1, t2})
    dev = jax.devices()[0].device_kind
    on_tpu = dev.lower().startswith("tpu")
    say = lambda msg: print(msg, flush=True)                                   # noqa: E731
    say(f"tpu_kernels: {args.config} {overrides}, device {dev}, piece {rows1} x {t1} "
        f"(lengths {lengths}), kernels {list(kernels)}")
    consts = M.make_consts(mc, **SV.parse_switches(cfg.run.switches))
    params = M.init_params(mc, jax.random.key(0))
    batches = {t: jax.device_put(segmented_batch(mc, max(1, rows1 * t1 // t), t)) for t in lengths}
    ref: dict[int, tuple[float, object]] = {}
    out: list[dict] = []
    first = True

    def grad_rel_diff(g, g_ref) -> tuple[float, float]:
        """Over the parameter leaves: the largest max|g - g_ref| / max|g_ref| and the largest
        |g - g_ref|_2 / |g_ref|_2."""
        worst_max = worst_l2 = 0.0
        for a, b in zip(jax.tree.leaves(g), jax.tree.leaves(g_ref)):
            a, b = a.astype(jnp.float32), b.astype(jnp.float32)
            scale, norm = float(jnp.max(jnp.abs(b))), float(jnp.linalg.norm(b.ravel()))
            if scale > 0:
                worst_max = max(worst_max, float(jnp.max(jnp.abs(a - b))) / scale)
                worst_l2 = max(worst_l2, float(jnp.linalg.norm((a - b).ravel())) / norm)
        return worst_max, worst_l2

    for t in lengths:
        for kern in kernels:
            label = f"{kern or '(configured)'} @ {t}"
            rows = max(1, rows1 * t1 // t)
            if time.time() - t_start > args.budget_s:
                say(f"  budget of {args.budget_s:.0f} s used: {label} not timed")
                out.append({"kernel": kern, "t": t, "rows": rows, "ok": False, "reason": "budget"})
                continue
            batch = batches[t]
            jax.clear_caches()                              # no trace of another kernel is reused
            F.apply(setting_of(kern))                       # read when the program is traced
            steps: list[dict] = []
            try:
                f = jax.jit(lambda p, b: jax.value_and_grad(lambda q: M.loss_fn(q, consts, b, mc)[0])(p))
                t0 = time.perf_counter()
                comp = f.lower(params, batch).compile()
                comp_s = time.perf_counter() - t0
                # Mosaic kernels in the compiled program (0 in interpret mode): with ``save`` one
                # forward kernel fewer per block (the loop step does not run it again)
                n_mosaic = comp.as_text().count('custom_call_target="tpu_custom_call"')
                got = comp(params, batch)
                jax.block_until_ready(got)
                for _ in range(4 if (first and on_tpu) else 1):  # the first program runs slow at first
                    got = comp(params, batch)
                    jax.block_until_ready(got)
                first = False
                times = []
                for _ in range(max(1, args.calls)):
                    t0 = time.perf_counter()
                    for _ in range(max(1, args.per_round)):
                        got = comp(params, batch)
                    jax.block_until_ready(got)
                    times.append((time.perf_counter() - t0) * 1e3 / max(1, args.per_round))
                for r1 in (step_rows or [rows1]) if kern in whole and (not step_lengths or t in step_lengths) else ():
                    # the trainer's accumulation over all the pieces of a step (the same piece
                    # repeated), as scripts/tpu_micro.py times it
                    r = max(1, r1 * t1 // t)
                    if per_step % (r * t):
                        continue
                    a = per_step // (r * t)
                    one = batch if r == rows else jax.device_put(segmented_batch(mc, r, t))
                    many = {k: jnp.broadcast_to(v, (a, *v.shape)) for k, v in one.items()}
                    fs = jax.jit(lambda p, b: accumulate(M.loss_fn, p, consts, b, mc)[:3])
                    t0 = time.perf_counter()
                    acc = fs(params, many)
                    jax.block_until_ready(acc)
                    step_comp = time.perf_counter() - t0
                    acc = fs(params, many)
                    jax.block_until_ready(acc)
                    t0 = time.perf_counter()
                    for _ in range(max(1, args.step_calls)):
                        acc = fs(params, many)
                    jax.block_until_ready(acc)
                    ms_step = (time.perf_counter() - t0) / max(1, args.step_calls) * 1e3
                    steps.append({"a": a, "rows": r, "ms": ms_step, "tokens_per_s": per_step / ms_step * 1e3,
                                  "loss": float(acc[1]), "compile_s": round(step_comp, 1)})
                    del fs, acc, many, one
            except Exception as e:  # noqa: BLE001 - a kernel that does not run is reported, not chosen
                msg = f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}"
                say(f"  {label:40s} FAILED {msg}")
                out.append({"kernel": kern, "t": t, "rows": rows, "ok": False, "reason": msg})
                continue
            finally:
                F.apply(defaults)
            loss, grads = float(got[0]), got[1]
            ms = statistics.median(times)
            rec = {"kernel": kern, "t": t, "rows": rows, "ok": True, "ms": ms,
                   "ms_all": [round(x, 3) for x in times], "tokens_per_s": rows * t / ms * 1e3,
                   "loss": loss, "compile_s": round(comp_s, 1), "mosaic_calls": n_mosaic}
            if steps:
                rec["steps"] = steps
            if t not in ref:
                ref[t] = (loss, grads, ms)
                rec["reference"] = True
            else:
                rec["loss_diff"] = loss - ref[t][0]
                rec["grad_rel_diff"], rec["grad_rel_l2"] = grad_rel_diff(grads, ref[t][1])
                rec["speedup"] = ref[t][2] / ms
            extra = (f"  dloss {rec['loss_diff']:+.1e}  grad diff max {rec['grad_rel_diff']:.1e} "
                     f"l2 {rec['grad_rel_l2']:.1e}  x{rec['speedup']:.3f}") if "speedup" in rec else "  (reference)"
            say(f"  {label:40s} {ms:9.2f} ms  {rec['tokens_per_s']:10,.0f} tok/s  loss {loss:.6f}"
                f"{extra}  (compile {comp_s:.0f} s, {n_mosaic} Mosaic calls)")
            for st in steps:
                say(f"  {'':24s} whole step {st['a']:4d} x {st['rows']:2d} x {t:<5d} {st['ms']:9.1f} ms  "
                    f"{st['tokens_per_s']:10,.0f} tok/s  loss {st['loss']:.6f}  (compile {st['compile_s']:.0f} s)")
            out.append(rec)
            del comp, f, got
    for t in lengths:
        ran = [r for r in out if r["t"] == t and r["ok"]]
        yard = [r["grad_rel_l2"] for r in ran if r["kernel"] == YARDSTICK and "grad_rel_l2" in r]
        tol = max([0.05] + [3.0 * y for y in yard])
        same = [r for r in ran if r.get("reference") or (abs(r["loss_diff"]) <= 1e-4 * abs(ref[t][0])
                                                         and r["grad_rel_l2"] <= tol)]
        for r in ran:
            if r not in same:
                say(f"  NOT EQUAL @ {t}: {r['kernel']} (dloss {r['loss_diff']:+.1e}, grad l2 {r['grad_rel_l2']:.1e})")
        if same:
            best = min(same, key=lambda r: r["ms"])
            say(f"BEST @ {t}: {best['kernel'] or '(configured)'} {best['ms']:.2f} ms "
                f"x{best.get('speedup', 1.0):.3f} vs the reference")
    say("KERNELS " + json.dumps({"results": out, "device": dev, "seconds": round(time.time() - t_start, 1),
                                 "config": args.config, "overrides": overrides}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
