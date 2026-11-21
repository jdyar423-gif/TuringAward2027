"""Compile-only checks for one TPU v6e chip (SPEC section 5, plan gate "AOT").

No TPU is needed: libtpu's compiler is driven from a CPU host through
``jax.experimental.topologies.get_topology_desc(platform="tpu", topology_name="v6e:1x1",
chips_per_host_bounds=(1, 1, 1), num_slices=1)``. Every program is lowered with abstract
arguments (``jax.ShapeDtypeStruct``) placed on the topology's device and compiled for it. Each
result reports lowering and compile time, ``memory_analysis()`` (the compiler's peak estimate of
device memory), ``cost_analysis()`` (flops, bytes accessed) and counts of selected ops in the
optimised HLO (scatter, gather, while loops, Mosaic kernels) and in the lowered StableHLO.

Programs
--------
``loss``     ``value_and_grad(loss_fn)`` of one micro-batch ``[B, T]`` (SPEC 4.4)
``train``    the whole step of ``bes.train.train.make_train_step``: ``A`` micro-batches,
             gradient accumulation, optimiser update, donated state (SPEC 5)
``decode``   ``bes.model.generate.decode_step`` for ``B`` rows and ``Tmax`` cache slots
``prefill``  ``bes.model.generate.prefill`` of one chunk of ``C`` positions

Command line (on a CPU host, or on the TPU VM itself)::

    python -m bes.train.aot --config configs/final.py --programs loss,decode \\
        --impl xla,splash --shapes 1024x32,2048x16 --remat on --json aot.json
    python -m bes.train.aot --override model.n_loops=1 ...      # a reduced model first

The exit status is 1 when any program fails to compile or its peak exceeds ``--limit-gib``.
An in-process guard aborts the run when the host process exceeds ``--max-rss-gb``.
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import os
import re
import resource
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

GIB = 2 ** 30
#: the plan's gate: peak device memory below 28 GiB of the 32 GiB of HBM of one v6e chip
HBM_LIMIT_GIB = 28.0
#: libtpu reads these at start-up; on a host without TPU metadata they silence its warnings
TPU_ENV = {"TPU_ACCELERATOR_TYPE": "v6e-1", "TPU_WORKER_HOSTNAMES": "localhost",
           "TPU_SKIP_MDS_QUERY": "1"}
#: regular expressions counted in the optimised HLO text
HLO_OPS = {
    "scatter": r"\sscatter(?:-start)?\(",
    "gather": r"\sgather\(",
    "while": r"\swhile\(",
    "dynamic_update_slice": r"\sdynamic-update-slice\(",
    "mosaic_kernel": r'custom_call_target="tpu_custom_call"',
    "custom_call": r"\scustom-call\(",
    "all_reduce": r"\sall-reduce(?:-start)?\(",
}
#: regular expressions counted in the lowered StableHLO text (op position only, not the
#: ``#stablehlo.scatter<...>`` attributes)
STABLEHLO_OPS = {"scatter": r'(?<![#\w])"?stablehlo\.scatter"?\(', "while": r'(?<![#\w])"?stablehlo\.while"?\(',
                 "gather": r'(?<![#\w])"?stablehlo\.gather"?\(', "mosaic_kernel": r"tpu_custom_call"}


# --------------------------------------------------------------------------- target
def tpu_device(topology: str = "v6e:1x1"):
    """The first device of a TPU topology description (compile target, no hardware)."""
    for k, v in TPU_ENV.items():
        os.environ.setdefault(k, v)
    from jax.experimental import topologies
    topo = topologies.get_topology_desc(platform="tpu", topology_name=topology,
                                        chips_per_host_bounds=(1, 1, 1), num_slices=1)
    return topo.devices[0]


def abstract(tree: Any, sharding) -> Any:
    """``jax.ShapeDtypeStruct`` with ``sharding`` for every leaf of ``tree``."""
    import jax
    import numpy as np
    return jax.tree.map(lambda a: jax.ShapeDtypeStruct(np.shape(a), a.dtype, sharding=sharding), tree)


# ------------------------------------------------------------------------- programs
def micro_batch_spec(b: int, t: int) -> dict:
    """Abstract micro-batch of SPEC section 3 (without the ``A`` axis)."""
    import jax
    import jax.numpy as jnp

    from bes.config import CAND_SLOTS, N_CAND_FEAT, N_CLOCKS, N_REG
    s = jax.ShapeDtypeStruct
    return {"tokens": s((b, t + 1), jnp.int32), "w": s((b, t), jnp.float32),
            "seg": s((b, t), jnp.int32), "reg": s((b, t, N_REG), jnp.int16),
            "clock": s((b, t, N_CLOCKS), jnp.int32), "cand_tok": s((b, t, CAND_SLOTS), jnp.int32),
            "cand_feat": s((b, t, CAND_SLOTS, N_CAND_FEAT), jnp.int8)}


def feature_rows_spec(b: int) -> dict:
    """Abstract decoding feature rows ``[B, ...]``."""
    import jax
    import jax.numpy as jnp

    from bes.config import CAND_SLOTS, N_CAND_FEAT, N_CLOCKS, N_REG
    s = jax.ShapeDtypeStruct
    return {"reg": s((b, N_REG), jnp.int16), "clock": s((b, N_CLOCKS), jnp.int32),
            "cand_tok": s((b, CAND_SLOTS), jnp.int32),
            "cand_feat": s((b, CAND_SLOTS, N_CAND_FEAT), jnp.int8)}


def _model_shapes(mcfg):
    import jax

    from bes.model import model as M
    params = jax.eval_shape(functools.partial(M.init_params, mcfg), jax.random.PRNGKey(0))
    consts = jax.eval_shape(lambda: M.make_consts(mcfg))
    return params, consts


def loss_program(mcfg, b: int, t: int) -> tuple[Callable, tuple]:
    """Jitted ``value_and_grad(loss_fn)`` and its abstract arguments."""
    import jax

    from bes.model import model as M
    params, consts = _model_shapes(mcfg)
    fn = jax.jit(jax.value_and_grad(functools.partial(M.loss_fn, cfg=mcfg), has_aux=True))
    return fn, (params, consts, micro_batch_spec(b, t))


def train_program(cfg, a: int, b: int, t: int) -> tuple[Callable, tuple]:
    """The training step of ``bes.train.train`` (donated state) and its abstract arguments."""
    import jax
    import jax.numpy as jnp

    from bes.model import model as M
    from bes.optim import build as build_optimizer
    from bes.train import train as T
    mcfg = cfg.model
    params, _ = _model_shapes(mcfg)
    opt = build_optimizer(cfg, params)
    state = jax.eval_shape(
        lambda p: T.TrainState(step=jnp.zeros((), jnp.int32), params=p, opt_state=opt.init(p),
                               consts=M.make_consts(mcfg)), params)
    step = T.make_train_step(M.loss_fn, opt, cfg)
    return step, (state, T.batch_spec((a, b, t)), jax.ShapeDtypeStruct((), jnp.float32))


def decode_program(mcfg, b: int, tmax: int) -> tuple[Callable, tuple]:
    """Jitted ``decode_step`` (donated cache) and its abstract arguments."""
    import jax
    import jax.numpy as jnp

    from bes.model import generate as G
    params, consts = _model_shapes(mcfg)
    cache = jax.eval_shape(lambda: G.init_cache(mcfg, b, tmax))
    fn = jax.jit(functools.partial(G.decode_step, cfg=mcfg), donate_argnums=(2,))
    return fn, (params, consts, cache, jax.ShapeDtypeStruct((b,), jnp.int32), feature_rows_spec(b))


def prefill_program(mcfg, b: int, chunk: int, tmax: int) -> tuple[Callable, tuple]:
    """Jitted ``prefill`` of one chunk (donated cache) and its abstract arguments."""
    import jax

    from bes.model import generate as G
    params, consts = _model_shapes(mcfg)
    cache = jax.eval_shape(lambda: G.init_cache(mcfg, b, tmax))
    f = micro_batch_spec(b, chunk)
    feats = {k: v for k, v in f.items() if k != "w"}
    feats["tokens"] = jax.ShapeDtypeStruct((b, chunk), f["tokens"].dtype)
    fn = jax.jit(functools.partial(G.prefill, cfg=mcfg), donate_argnums=(2,))
    return fn, (params, consts, cache, feats)


# ------------------------------------------------------------------------- analysis
@dataclass
class AotResult:
    """Outcome of one compile-only check (sizes in bytes)."""
    program: str
    impl: str
    shape: str
    remat: bool
    ok: bool
    lower_s: float = 0.0
    compile_s: float = 0.0
    peak_bytes: int = 0
    memory: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    hlo_ops: dict = field(default_factory=dict)
    stablehlo_ops: dict = field(default_factory=dict)
    host_rss_gb: float = 0.0
    error: str = ""

    @property
    def peak_gib(self) -> float:
        return self.peak_bytes / GIB

    def row(self) -> str:
        if not self.ok and self.error:
            return f"{self.program:8s} {self.impl:6s} {self.shape:12s} remat={int(self.remat)} FAILED: {self.error}"
        flops = self.cost.get("flops", float("nan"))
        return (f"{self.program:8s} {self.impl:6s} {self.shape:12s} remat={int(self.remat)} "
                f"compile {self.compile_s:6.1f}s peak {self.peak_gib:6.2f} GiB "
                f"(args {self.memory.get('argument_size_in_bytes', 0) / GIB:.2f}, "
                f"temp {self.memory.get('temp_size_in_bytes', 0) / GIB:.2f}, "
                f"out {self.memory.get('output_size_in_bytes', 0) / GIB:.2f}, "
                f"alias {self.memory.get('alias_size_in_bytes', 0) / GIB:.2f}) "
                f"flops {flops:.3e} scatter {self.hlo_ops.get('scatter', 0)} "
                f"while {self.hlo_ops.get('while', 0)} mosaic {self.hlo_ops.get('mosaic_kernel', 0)} "
                f"host-rss {self.host_rss_gb:.2f} GB")


def count_ops(text: str, patterns: dict[str, str] | None = None, *, regex: bool = True) -> dict[str, int]:
    """Occurrences of each pattern in an HLO / StableHLO text."""
    patterns = HLO_OPS if patterns is None else patterns
    if regex:
        return {k: len(re.findall(p, text)) for k, p in patterns.items()}
    return {k: text.count(p) for k, p in patterns.items()}


def memory_stats(compiled) -> tuple[int, dict[str, int]]:
    """``(peak bytes, all integer fields of memory_analysis())``. The peak is the compiler's
    ``peak_memory_in_bytes`` when present, else ``argument + output + temp - alias``."""
    ma = compiled.memory_analysis()
    if ma is None:
        return 0, {}
    fields = {a: int(getattr(ma, a)) for a in dir(ma)
              if a.endswith("_in_bytes") and isinstance(getattr(ma, a), int)}
    peak = fields.get("peak_memory_in_bytes", 0)
    if not peak:
        peak = (fields.get("argument_size_in_bytes", 0) + fields.get("output_size_in_bytes", 0)
                + fields.get("temp_size_in_bytes", 0) - fields.get("alias_size_in_bytes", 0))
    return peak, fields


def cost_stats(compiled) -> dict[str, float]:
    """``flops``, ``bytes accessed``, ``transcendentals`` and ``optimal_seconds`` of
    ``cost_analysis()`` (whichever are reported)."""
    ca = compiled.cost_analysis()
    if isinstance(ca, (list, tuple)):
        ca = ca[0] if ca else {}
    ca = ca or {}
    keep = ("flops", "bytes accessed", "transcendentals", "optimal_seconds")
    return {k: float(ca[k]) for k in keep if k in ca}


def host_rss_gb() -> float:
    """Peak resident set size of this process so far (GB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def compile_program(name: str, fn: Callable, args: tuple, device, *, impl: str = "",
                    shape: str = "", remat: bool = False) -> AotResult:
    """Lower ``fn`` with abstract ``args`` on ``device`` and compile it."""
    import jax
    sharding = jax.sharding.SingleDeviceSharding(device)
    res = AotResult(program=name, impl=impl, shape=shape, remat=remat, ok=False)
    try:
        t0 = time.time()
        lowered = fn.lower(*abstract(args, sharding))
        t1 = time.time()
        res.stablehlo_ops = count_ops(lowered.as_text(), STABLEHLO_OPS)
        compiled = lowered.compile()
        t2 = time.time()
    except Exception as exc:  # noqa: BLE001 - the report records any compile failure
        res.error = f"{type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}"
        res.host_rss_gb = host_rss_gb()
        return res
    res.lower_s, res.compile_s = t1 - t0, t2 - t0
    res.peak_bytes, res.memory = memory_stats(compiled)
    res.cost = cost_stats(compiled)
    res.hlo_ops = count_ops(compiled.as_text())
    res.host_rss_gb = host_rss_gb()
    res.ok = True
    return res


def within_limit(res: AotResult, limit_gib: float = HBM_LIMIT_GIB) -> bool:
    return res.ok and res.peak_gib < limit_gib


# ----------------------------------------------------------------------- RSS guard
class RssGuard:
    """Aborts the process (exit status 3) when its resident set exceeds ``limit_gb``.

    The compiler cannot be interrupted from Python, so a daemon thread polls ``/proc`` and
    calls ``os._exit``; use it only in dedicated processes.
    """

    def __init__(self, limit_gb: float, period_s: float = 0.5):
        self.limit_gb, self.period_s = float(limit_gb), float(period_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def current_gb() -> float:
        try:
            with open("/proc/self/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 2 ** 20
        except OSError:
            pass
        return 0.0

    def _run(self) -> None:
        while not self._stop.wait(self.period_s):
            if self.current_gb() > self.limit_gb:
                print(f"aot: resident memory above {self.limit_gb:.1f} GB, aborting", flush=True)
                os._exit(3)

    def __enter__(self) -> "RssGuard":
        if self.limit_gb > 0:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()


# -------------------------------------------------------------------------- driver
def parse_pairs(text: str) -> list[tuple[int, int]]:
    """``"1024x32,2048x16"`` -> ``[(1024, 32), (2048, 16)]``."""
    out = []
    for item in filter(None, (s.strip() for s in text.split(","))):
        a, _, b = item.partition("x")
        out.append((int(a), int(b)))
    return out


def model_variant(mcfg, impl: str, remat: bool):
    """The model configuration compiled for the TPU (never Pallas interpret mode)."""
    if impl == "splash":
        from bes.model import attention_splash
        attention_splash.register_alias()
    return dataclasses.replace(mcfg, attn_impl=impl, remat=remat, splash_interpret="false")


def run(cfg, programs: Sequence[str], impls: Sequence[str], shapes: Sequence[tuple[int, int]],
        remats: Sequence[bool], *, micro: int = 8, decode: Sequence[tuple[int, int]] = ((8, 2048),),
        prefill_chunk: int = 256, device=None, log: Callable[[str], None] = print) -> list[AotResult]:
    """Compile every requested program; ``shapes`` are ``(T, B)`` pairs, ``decode`` are
    ``(B, Tmax)`` pairs. Returns the results in order."""
    device = tpu_device() if device is None else device
    results: list[AotResult] = []

    def go(name, builder, impl, shape, remat):
        fn, args = builder()
        res = compile_program(name, fn, args, device, impl=impl, shape=shape, remat=remat)
        log(res.row())
        results.append(res)

    for remat in remats:
        for impl in impls:
            mcfg = model_variant(cfg.model, impl, remat)
            for t, b in shapes:
                if "loss" in programs:
                    go("loss", lambda: loss_program(mcfg, b, t), impl, f"T{t}xB{b}", remat)
                if "train" in programs:
                    full = dataclasses.replace(cfg, model=mcfg)
                    go("train", lambda: train_program(full, micro, b, t), impl,
                       f"A{micro}xT{t}xB{b}", remat)
    base = model_variant(cfg.model, "xla", False)      # generation uses its own cache attention
    for b, tmax in decode:
        if "decode" in programs:
            go("decode", lambda: decode_program(base, b, tmax), "cache", f"B{b}xS{tmax}", False)
        if "prefill" in programs:
            go("prefill", lambda: prefill_program(base, 1, prefill_chunk, tmax), "cache",
               f"C{prefill_chunk}xS{tmax}", False)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="configs/final.py")
    p.add_argument("--override", action="append", default=[], help="e.g. model.n_loops=1")
    p.add_argument("--programs", default="loss,decode", help="loss,train,decode,prefill")
    p.add_argument("--impl", default="xla,splash")
    p.add_argument("--shapes", default="1024x32,2048x16", help="TxB pairs (loss, train)")
    p.add_argument("--remat", default="config", choices=("config", "on", "off", "both"))
    p.add_argument("--micro", type=int, default=8, help="micro-batches per train step (A)")
    p.add_argument("--decode", default="8x2048", help="BxTmax pairs (decode, prefill)")
    p.add_argument("--prefill-chunk", type=int, default=256)
    p.add_argument("--topology", default="v6e:1x1")
    p.add_argument("--limit-gib", type=float, default=HBM_LIMIT_GIB)
    p.add_argument("--max-rss-gb", type=float, default=6.0)
    p.add_argument("--json", default="")
    args = p.parse_args(argv)

    import jax
    try:                                   # compile-only: never initialise a TPU runtime here
        jax.config.update("jax_platforms", os.environ.get("JAX_PLATFORMS") or "cpu")
    except Exception:  # noqa: BLE001 - backends already initialised
        pass
    from bes.config import load_config
    cfg = load_config(args.config, args.override)
    remats = {"config": [cfg.model.remat], "on": [True], "off": [False], "both": [False, True]}[args.remat]
    progs = [s.strip() for s in args.programs.split(",") if s.strip()]
    impls = [s.strip() for s in args.impl.split(",") if s.strip()]
    t0 = time.time()
    with RssGuard(args.max_rss_gb):
        dev = tpu_device(args.topology)
        print(f"aot: target {dev.device_kind} ({args.topology}), config {args.config} "
              f"{' '.join(args.override)}", flush=True)
        results = run(cfg, progs, impls, parse_pairs(args.shapes), remats, micro=args.micro,
                      decode=[(t, b) for t, b in parse_pairs(args.decode)],
                      prefill_chunk=args.prefill_chunk, device=dev,
                      log=lambda s: print(s, flush=True))
    bad = [r for r in results if not within_limit(r, args.limit_gib)]
    print(f"aot: {len(results) - len(bad)}/{len(results)} within {args.limit_gib:g} GiB "
          f"in {time.time() - t0:.0f}s; host peak RSS {host_rss_gb():.2f} GB", flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump([dataclasses.asdict(r) for r in results], fh, indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
