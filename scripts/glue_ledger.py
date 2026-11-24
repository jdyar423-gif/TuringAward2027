"""Glue ledger: every primitive application of one training micro-batch, with its output bytes,
its forward / recomputed / backward phase and the ``bes/`` source line it came from, read off the
traced program (no device, no parameter arrays: everything is abstract).

    cd <checkout>                           # bes is imported from the CURRENT WORKING DIRECTORY
    python <path to>/glue_ledger.py                     # configs/final.py, the live run's overrides, T=1024
    python <path to>/glue_ledger.py --seq-len 2048      # 8 x 2048 (the trainer's micro-batch at T=2048)
    python <path to>/glue_ledger.py --set bes.model.attention.FAST_X=1 --json /tmp/fast_x.json
    python <path to>/glue_ledger.py --set bes.model.attention.FAST_X=1 --diff ledger_baseline_1024.json
    python <path to>/glue_ledger.py --explain attention.py:377     # every equation of one line

In the development container prefix it with ``taskset -c 2,3 nice -n 10``. It is fast: tracing
the real model takes about 1 s at 16 x 1024 and at 8 x 2048 (CPU, jax 0.11.1), the whole run
2-4 s and about 250 MB of memory; it never compiles and never touches an accelerator (it forces
``JAX_PLATFORMS=cpu``) and writes no ``__pycache__`` into the checkout.

What is traced (the program ``bes/train/train.py`` runs for one micro-batch): ``--program
trainer`` (default) is ``train.accumulate`` with ``loss_fn = bes.model.model.loss_fn`` over a
single micro-batch, i.e. ``jax.value_and_grad(loss_fn, has_aux=True)`` and the float32 gradient
accumulation of the trainer's scan body; the parameters come from ``jax.eval_shape`` of
``init_params``, the constants from ``make_consts`` with ``run.switches`` (passed as inputs, as
the trainer's state does) and the batch from ``train.batch_spec`` (the SPEC fields and dtypes),
so rematerialisation, bfloat16 and the splash kernels are as configured. What lies outside the
micro-batch scan (the accumulator's zeros, the final division) runs once per step and is reported
on one line, outside the totals. ``--program vg`` traces ``value_and_grad`` of the loss alone (as
``scripts/tpu_tune.py`` times it), ``--program forward`` the loss alone. On CPU the splash kernels
are traced in interpret mode; the program is the same apart from that flag.

The walk enters every sub-program (``scan``, ``while``, ``cond``, ``checkpoint``/remat,
``custom_jvp``/``custom_vjp``, inner ``jit``, ``pallas_call``) and multiplies by its static trip
count (``scan`` length: 8 for the loops; the grid inside a Pallas kernel; a ``while`` counts once
and is flagged; of a ``cond`` the heaviest branch counts). Phase: ``fwd`` (the forward pass),
``remat`` (the forward recomputed inside the backward pass: name stack ``rematted_computation``),
``bwd`` (transposed: name stack ``transpose(...)``). Source line: the innermost frame inside
``bes/`` of the equation's traceback (``--depth 2`` keys by the two innermost, i.e. a helper
and its caller); equations of a shared inner ``jit`` (``jnp.where``, ``jnp.einsum``, the splash
wrapper) are charged to the line that called it.

Columns. ``out``: the output bytes of every application (x trips), what the program would write
if nothing were fused. ``pass``: the pass over memory the primitive stands for: its output,
except for reductions (the operand read), scatters and ``dynamic_update_slice`` (the update
written); split by phase. ``mat``: the part of the pass that is materialised under a simple
fusion rule: XLA fuses element-wise chains, converts, selects, slices, pads, concatenations and
broadcasts into their consumers, so a result counts only when it must be written to memory,
i.e. when it reaches a matrix product, a Pallas kernel, a loop carry or loop input, a
gather/scatter/sort or an output of the program (a constant broadcast passed to a loop as a loop
constant is sunk into it); a reduction counts the read of its fused sources (stopping at values
stored in a narrower float type, such as bfloat16 activations, and at materialised values). A
value with several consumers may still be written by XLA, so ``pass`` is the upper and ``mat``
a lower-ish bracket. Glue: the memory-bound classes (slice, concatenate, pad, transpose,
gather/scatter, select_n, materialised broadcast, float32<->bfloat16 convert, reductions,
sort/top_k, copy); its total is line (3). Free: reshapes, fused broadcasts and transposes at a
product (a transpose of a product's output nothing else reads, or one only products read: XLA
makes it the product's layout or folds it into the dimension numbers; the transpose inside
``jnp.einsum(..., "...->bhtk")`` is one). ``scan xs/ys`` (reads of a loop's stacked inputs and
writes of its stacked outputs) is listed apart and is not in the glue total. The matrix products
are split by operand and result dtypes. What runs inside a Pallas kernel works on VMEM tiles,
not HBM; it is listed apart with its FLOPs (an upper bound: fully masked tiles are skipped at run
time). Unknown primitives are listed under "other" with their bytes. Section (2) ranks source
lines by glue pass bytes, (2b) by materialised bytes outside the matrix products (glue and
element-wise). This is a ledger of the traced program, not of XLA's fusions: it shows where the
passes are and what a change removes (``--diff``); the chip decides.

Options: ``--set module.NAME=VALUE`` (repeatable) sets a module-level attribute before tracing,
coerced to the type of its current value (it must exist, and must be read at call time, e.g.
``attention.FAST_X``, not imported by value); ``--override section.field=value`` (repeatable) is
added to ``--base-overrides`` (default: the live run's ``model.table_grad=take
optim.micro_seqs=16``); ``--json FILE`` writes the ledger; ``--diff FILE`` prints the change
against an earlier ``--json`` (by class, and by source function, which survives line shifts);
``--explain TEXT`` lists every equation charged to a source key containing TEXT, with its class,
phase, applications, tensor and name stack. A tiny configuration for quick checks: ``--config
configs/smoke.py --override model.attn_impl=splash --override model.act_dtype=bfloat16
--override model.remat=true``.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import importlib
import json
import linecache
import math
import os
import sys
import time
from collections import defaultdict
from typing import Any, Sequence

sys.dont_write_bytecode = True                  # do not leave __pycache__ in other checkouts
os.environ["JAX_PLATFORMS"] = "cpu"             # tracing only: never touch an accelerator

LIVE_OVERRIDES = "model.table_grad=take optim.micro_seqs=16"
PEAK_TFLOPS = 814.0                             # bf16 matrix products as measured (docs/TPU_NOTES.md)

# --------------------------------------------------------------------------- classes
ELEMENTWISE = frozenset("""
add sub mul div rem pow integer_pow neg sign floor ceil round is_finite exp exp2 log log1p expm1
tanh logistic sin cos tan asin acos atan atan2 sinh cosh asinh acosh atanh sqrt rsqrt cbrt square
abs max min nextafter clamp and or xor not eq ne lt le gt ge shift_left shift_right_logical
shift_right_arithmetic population_count clz erf erfc erf_inv lgamma digamma igamma igammac
igamma_grad_a random_gamma_grad bessel_i0e bessel_i1e regularized_incomplete_beta polygamma zeta
add_any real imag conj complex reduce_precision cosine sine tangent mul_add
""".split())
SLICE = frozenset("slice dynamic_slice dynamic_update_slice split".split())
CONCAT = frozenset("concatenate stack".split())
TRANSPOSE = frozenset("transpose rev".split())
GATHER = frozenset("""gather scatter scatter-add scatter_add scatter-mul scatter_mul scatter-min
scatter_min scatter-max scatter_max scatter_sub scatter-sub scatter_apply select_and_scatter_add
select_and_scatter select_and_gather_add""".split())
REDUCE = frozenset("""reduce_sum reduce_max reduce_min reduce_prod reduce_and reduce_or reduce_xor
argmax argmin reduce cumsum cumprod cummax cummin cumlogsumexp reduce_window reduce_window_sum
reduce_window_max reduce_window_min""".split())
SORT = frozenset("sort top_k approx_top_k".split())
GENERATED = frozenset("broadcast_in_dim iota tile".split())
FREE = frozenset("""reshape squeeze expand_dims stop_gradient optimization_barrier device_put
bitcast_convert_type name""".split())
COPY = frozenset("copy copy_p".split())
MATMUL = frozenset("dot_general conv_general_dilated ragged_dot ragged_dot_general".split())
#: containers XLA inlines: what their outputs feed decides whether something inside materialises
TRANSPARENT = frozenset("""jit pjit closed_call core_call remat2 checkpoint custom_jvp_call
custom_vjp_call custom_vjp_call_jaxpr custom_lin named_call""".split())
#: leaves whose sub-programs are combiners, not computations
LEAF_WITH_COMBINER = GATHER | REDUCE | SORT
#: consumers into which XLA fuses a broadcast (anything else, or a loop/kernel input, materialises it)
FUSES_BROADCAST = (ELEMENTWISE | SLICE | TRANSPOSE | FREE | COPY | GENERATED | CONCAT
                   | {"select_n", "convert_element_type", "pad"}
                   | {"reduce_sum", "reduce_max", "reduce_min", "reduce_prod", "reduce_and",
                      "reduce_or", "argmax", "argmin"})

C_DOT, C_KERNEL, C_ELEM = "dot_general", "pallas kernel", "elementwise"
C_SLICE, C_CONCAT, C_PAD, C_TRANS = "slice/dynamic_slice", "concatenate", "pad", "transpose"
C_GATHER, C_SELECT, C_BMAT = "gather/scatter", "select_n (where)", "broadcast, materialised"
C_CONV, C_RED, C_SORT, C_COPY = "convert f32<->bf16", "reductions", "sort/top_k", "copy"
C_FREE, C_BFUSED, C_SCAN, C_OTHER = "reshape (free)", "broadcast, fused", "scan xs/ys", "other"
C_TFOLD = "transpose at a product"
GLUE = (C_SLICE, C_CONCAT, C_PAD, C_TRANS, C_GATHER, C_SELECT, C_BMAT, C_CONV, C_RED, C_SORT, C_COPY)
SECTIONS = (("matrix products", (C_DOT, C_KERNEL)), ("fusable element-wise arithmetic", (C_ELEM,)),
            ("memory-bound (the glue)", GLUE), ("free (fused or folded by XLA)", (C_FREE, C_BFUSED, C_TFOLD)),
            ("loop boundaries (not in the glue total)", (C_SCAN,)), ("other (unknown primitives)", (C_OTHER,)))
PHASES = ("fwd", "remat", "bwd")
MB = 1e6
NO_FRAME = (("(no bes frame)", 0, ""),)
#: containers whose sub-program jax traces once per function and shapes and shares between callers
CACHEABLE = frozenset("jit pjit closed_call core_call custom_jvp_call custom_vjp_call custom_vjp_call_jaxpr".split())


@dataclasses.dataclass(frozen=True)
class Ctx:
    mult: int = 1                    # applications per micro-batch of one equation here (trip counts)
    path: str = ""                   # accumulated name stack
    where: tuple = ()                # source frames charged when an equation has none of its own
    site: tuple = ()                 # call site of the innermost enclosing cacheable container
    up: tuple = ()                   # enclosing transparent containers: ((eqn, parent jaxpr), ...)
    kernel: str = ""                 # name of the enclosing Pallas kernel ("" = HBM program)
    scope: str = "micro"             # "micro" (per micro-batch) or "step" (once per step)
    flags: tuple = ()                # notes such as "while: trips unknown"


@dataclasses.dataclass
class Rec:
    cls: str
    prim: str
    phase: str
    scope: str
    kernel: str
    n: int                           # applications (count x trips)
    out: float                       # output bytes, all applications
    pas: float                       # pass bytes, all applications
    mat: float                       # the pass bytes whose result must be in memory (fusion rule)
    flops: float
    aval: str                        # the tensor of the pass, e.g. "bf16[16,1024,768]"
    aval_bytes: int
    where: tuple
    note: str = ""
    path: str = ""                   # name stack (for --explain)


# --------------------------------------------------------------------------- helpers
def _is_jaxpr(x) -> bool:
    return hasattr(x, "eqns") and hasattr(x, "invars")


def _as_jaxpr(x):
    """A ``Jaxpr`` from a ``Jaxpr`` or a ``ClosedJaxpr``, else None."""
    if _is_jaxpr(x):
        return x
    inner = getattr(x, "jaxpr", None)
    return inner if _is_jaxpr(inner) else None


def _subjaxprs(eqn):
    """``(param name, Jaxpr)`` of every sub-program in an equation's parameters."""
    for k, v in eqn.params.items():
        j = _as_jaxpr(v)
        if j is not None:
            yield k, j
        elif isinstance(v, (tuple, list)):
            for i, x in enumerate(v):
                j = _as_jaxpr(x)
                if j is not None:
                    yield f"{k}[{i}]", j


def _is_var(v) -> bool:
    return type(v).__name__ not in ("Literal", "DropVar") and hasattr(v, "aval")


def _nbytes(aval) -> int:
    shape, dtype = getattr(aval, "shape", None), getattr(aval, "dtype", None)
    if shape is None or dtype is None:
        return 0
    try:
        item = np_dtype(dtype).itemsize
    except Exception:  # noqa: BLE001 - extended dtypes (PRNG keys, ...)
        item = int(getattr(dtype, "itemsize", 0) or 0)
    n = 1
    for d in shape:
        if not isinstance(d, int):
            try:
                d = int(d)
            except Exception:  # noqa: BLE001 - symbolic dimension
                return 0
        n *= d
    return n * item


def np_dtype(dtype):
    import numpy as np
    return np.dtype(dtype)


_SHORT = {"float32": "f32", "bfloat16": "bf16", "float16": "f16", "float64": "f64", "int32": "i32",
          "int16": "i16", "int8": "i8", "uint8": "u8", "uint32": "u32", "bool": "bool", "int64": "i64"}


def _aval_str(aval) -> str:
    dtype = str(getattr(aval, "dtype", "?"))
    shape = getattr(aval, "shape", None)
    if shape is None:
        return type(aval).__name__
    return f"{_SHORT.get(dtype, dtype)}[{','.join(str(d) for d in shape)}]"


def _is_float(dtype) -> bool:
    try:
        import jax.numpy as jnp
        return bool(jnp.issubdtype(dtype, jnp.floating))
    except Exception:  # noqa: BLE001
        return False


def _scan_split(eqn):
    """``(num_consts, num_carry)`` of a scan equation (jax 0.11 ``ft_in`` or older params)."""
    p = eqn.params
    if "num_consts" in p and "num_carry" in p:
        return int(p["num_consts"]), int(p["num_carry"])
    parts = p["ft_in"].unpack()
    return len(parts[0]), len(parts[1])


def _dot_flops(eqn) -> float:
    try:
        (lc, _), _ = eqn.params["dimension_numbers"]
        lhs = eqn.invars[0].aval.shape
        k = 1
        for d in lc:
            k *= int(lhs[d])
        out = 1
        for d in eqn.outvars[0].aval.shape:
            out *= int(d)
        return 2.0 * out * k
    except Exception:  # noqa: BLE001
        return 0.0


def _phase(path: str) -> str:
    if "rematted_computation" in path:
        return "remat"
    if "transpose(" in path:
        return "bwd"
    return "fwd"


# --------------------------------------------------------------------------- the walk
class Ledger:
    def __init__(self, repo: str, bes_dir: str, depth: int, a_scan_file: str | None):
        self.repo, self.bes_dir, self.depth = repo, bes_dir.rstrip(os.sep) + os.sep, max(1, depth)
        self.a_scan_file = a_scan_file          # trainer program: the micro-batch scan's file
        self.recs: list[Rec] = []
        self.notes: list[str] = []
        self.n_eqns = 0
        self._uses: dict[int, dict] = {}
        self._defs: dict[int, dict] = {}
        self._mat: dict[tuple, bool] = {}
        self._keep: list = []                   # keeps walked jaxprs alive (ids are cache keys)
        self.found_micro_scan = False

    # ------------------------------------------------------------------ source frames
    def frames(self, eqn) -> tuple:
        tb = getattr(eqn.source_info, "traceback", None)
        if tb is None:
            return ()
        out = []
        try:
            for f in tb.frames:                                     # innermost first
                fn = f.file_name
                if fn.startswith(self.bes_dir):
                    out.append((os.path.relpath(fn, self.repo), int(f.line_num), f.function_name))
        except Exception:  # noqa: BLE001
            return ()
        return tuple(out)

    def where(self, eqn, ctx: Ctx) -> tuple:
        """The equation's innermost ``bes/`` frames, or what it inherits: an equation without a
        bes frame (jax internals, library kernels) is charged to the enclosing equation's line.
        Below a cacheable container (an inner jit, custom_jvp, custom_vjp: traced once and
        shared by every caller with the same shapes) the own frames count only when they pass
        through that container's call site; otherwise they may be another caller's."""
        own = self.frames(eqn)
        if not own:
            return ctx.where or NO_FRAME
        if ctx.site and ctx.site[0] != NO_FRAME[0] and not any(f[:2] == ctx.site[0][:2] for f in own):
            return ctx.where
        return own[:self.depth]

    # ------------------------------------------------------------------ materialisation
    def uses(self, jaxpr) -> dict:
        key = id(jaxpr)
        if key not in self._uses:
            u: dict = defaultdict(list)
            for eqn in jaxpr.eqns:
                for i, v in enumerate(eqn.invars):
                    if _is_var(v):
                        u[v].append((eqn, i))
            for i, v in enumerate(jaxpr.outvars):
                if _is_var(v):
                    u[v].append((None, i))
            self._uses[key] = u
            self._keep.append(jaxpr)
        return self._uses[key]

    def sub_mat_out(self, eqn, jaxpr, mat_out: frozenset, const_like: bool = False) -> frozenset:
        return frozenset(i for i, v in enumerate(eqn.outvars)
                         if _is_var(v) and self.materialised(v, jaxpr, mat_out, const_like))

    def materialised(self, var, jaxpr, mat_out: frozenset, const_like: bool = False, depth: int = 0) -> bool:
        """Does ``var`` (defined in ``jaxpr``) have to exist in memory: a non-fusing consumer, a
        loop carry or loop input, a kernel input, or an output that its context keeps.
        ``const_like``: a broadcast of a literal or an iota; passed to a loop as a loop constant
        it is sunk back into the loop body by XLA and is not materialised."""
        key = (id(jaxpr), mat_out, var, const_like)
        if key in self._mat:
            return self._mat[key]
        self._mat[key] = True                               # cycle guard (cannot happen)
        res = False
        for eqn, i in self.uses(jaxpr).get(var, ()):
            if eqn is None:
                if i in mat_out:
                    res = True
                    break
                continue
            name = eqn.primitive.name
            if name in FUSES_BROADCAST:
                continue
            if name == "scan" and const_like:
                try:
                    if i < _scan_split(eqn)[0]:
                        continue                            # a loop constant: sunk into the body
                except Exception:  # noqa: BLE001
                    pass
            if name in TRANSPARENT and depth < 50:
                subs = [j for _, j in _subjaxprs(eqn)]
                if len(subs) == 1 and len(subs[0].invars) == len(eqn.invars):
                    sub = subs[0]
                    if self.materialised(sub.invars[i], sub, self.sub_mat_out(eqn, jaxpr, mat_out),
                                         const_like, depth + 1):
                        res = True
                        break
                    continue
            res = True
            break
        self._mat[key] = res
        return res

    # ------------------------------------------------------------------ records
    def add(self, cls, prim, eqn, ctx: Ctx, where, out_b, pass_b, flops=0.0, note="", aval=None, mat_b=None):
        if aval is None:                                    # the largest output
            aval = max((v.aval for v in eqn.outvars if hasattr(v, "aval")), key=_nbytes, default=None)
        mat_b = pass_b if mat_b is None else mat_b
        self.recs.append(Rec(cls, prim, _phase(ctx.path), ctx.scope, ctx.kernel, ctx.mult,
                             float(out_b) * ctx.mult, float(pass_b) * ctx.mult, float(mat_b) * ctx.mult,
                             float(flops) * ctx.mult,
                             _aval_str(aval) if aval is not None else "", _nbytes(aval) if aval is not None else 0,
                             where, note or ";".join(ctx.flags), ctx.path))

    def defs(self, jaxpr) -> dict:
        key = id(jaxpr)
        if key not in self._defs:
            self._defs[key] = {v: eqn for eqn in jaxpr.eqns for v in eqn.outvars if _is_var(v)}
        return self._defs[key]

    def feeds_products(self, var, jaxpr, depth: int = 0) -> bool:
        """Every consumer of ``var`` is a matrix product (through converts and reshapes): XLA
        folds a transpose there into the product's dimension numbers."""
        uses = self.uses(jaxpr).get(var, ())
        if not uses or depth > 8:
            return False
        for eqn, _ in uses:
            if eqn is None:
                return False
            name = eqn.primitive.name
            if name in MATMUL:
                continue
            if name in ("convert_element_type", "reshape", "squeeze", "expand_dims") and \
                    all(self.feeds_products(v, jaxpr, depth + 1) for v in eqn.outvars if _is_var(v)):
                continue
            return False
        return True

    def fused_read(self, vars_, jaxpr, up: tuple = ()) -> int:
        """Bytes a fusion rooted at a reduction reads for ``vars_``: the distinct sources reached
        through the element-wise producers, up-casts, selects, broadcasts and reshapes that XLA fuses
        into it, also across inner jits and remat regions (``up``: the enclosing transparent
        containers). Sources: a value stored in a narrower float type (the result of a down-cast,
        e.g. an activation in bfloat16), a value something else materialises, a slice, a
        transpose, a product, a kernel or loop output, a loop input; a broadcast or iota operand
        counts with its own (small) size."""
        through = ELEMENTWISE | FREE | GENERATED | {"convert_element_type", "select_n"}
        seen, srcs = set(), {}
        stack = [(v, jaxpr, tuple(up), True) for v in vars_ if _is_var(v)]
        while stack and len(seen) < 5000:
            v, j, path, first = stack.pop()
            key = (id(j), v)
            if key in seen:
                continue
            seen.add(key)
            eqn = self.defs(j).get(v)
            if eqn is not None and eqn.primitive.name in through:
                stop = False
                if not first:
                    if eqn.primitive.name == "convert_element_type" and _is_var(eqn.invars[0]):
                        src_dt, dst_dt = eqn.invars[0].aval.dtype, eqn.outvars[0].aval.dtype
                        stop = (_is_float(src_dt) and _is_float(dst_dt)
                                and np_dtype(dst_dt).itemsize < np_dtype(src_dt).itemsize)
                    if not stop and eqn.primitive.name not in GENERATED:
                        stop = self.materialised(v, j, frozenset())
                if not stop:
                    stack.extend((x, j, path, False) for x in eqn.invars if _is_var(x))
                    continue
                srcs[key] = _nbytes(v.aval)
                continue
            if eqn is not None and eqn.primitive.name in TRANSPARENT:
                subs = [s for _, s in _subjaxprs(eqn)]
                if len(subs) == 1 and len(subs[0].outvars) == len(eqn.outvars):
                    x = subs[0].outvars[eqn.outvars.index(v)]
                    if _is_var(x):
                        stack.append((x, subs[0], path + ((eqn, j),), first))
                    continue
            if eqn is None and path:
                ceqn, pj = path[-1]
                try:
                    idx = j.invars.index(v)
                except ValueError:
                    idx = -1
                if 0 <= idx < len(ceqn.invars) and len(j.invars) == len(ceqn.invars):
                    x = ceqn.invars[idx]
                    if _is_var(x):
                        stack.append((x, pj, path[:-1], first))
                    continue
            srcs[key] = _nbytes(v.aval)
        return sum(srcs.values())

    def folded_transpose(self, eqn, jaxpr) -> bool:
        """A transpose of a product's output that nothing else reads (it becomes the product's
        output layout), or one that only products read (TransposeFolding)."""
        src = eqn.invars[0]
        if _is_var(src):
            prod = self.defs(jaxpr).get(src)
            if prod is not None and prod.primitive.name in MATMUL and len(self.uses(jaxpr).get(src, ())) == 1:
                return True
        return all(self.feeds_products(v, jaxpr) for v in eqn.outvars if _is_var(v))

    def classify(self, eqn, jaxpr, ctx: Ctx, mat_out) -> tuple[str, float, float, float]:
        """``(class, output bytes, pass bytes, materialised pass bytes)`` of one application of a
        leaf equation. Materialised (the fusion rule of :meth:`materialised`): a result that must be
        written to memory; a reduction always reads its operand."""
        name = eqn.primitive.name
        out_b = sum(_nbytes(v.aval) for v in eqn.outvars if hasattr(v, "aval"))
        in_b = [(_nbytes(v.aval) if hasattr(v, "aval") else 0) for v in eqn.invars]

        def mat(b, const_like=False):
            if ctx.kernel:
                return b
            return b if any(_is_var(v) and self.materialised(v, jaxpr, mat_out, const_like)
                            for v in eqn.outvars) else 0
        if name in MATMUL:
            return C_DOT, out_b, out_b, out_b
        if name in ELEMENTWISE:
            return C_ELEM, out_b, out_b, mat(out_b)
        if name == "convert_element_type":
            src = eqn.invars[0].aval.dtype if hasattr(eqn.invars[0], "aval") else None
            dst = eqn.outvars[0].aval.dtype
            if src is not None and str(src) == str(dst):
                return C_FREE, out_b, out_b, 0
            if src is not None and _is_float(src) and _is_float(dst):
                return C_CONV, out_b, out_b, mat(out_b)
            return C_ELEM, out_b, out_b, mat(out_b)
        if name in SLICE:
            if name == "dynamic_update_slice":
                upd = in_b[1] if len(in_b) > 1 else out_b
                return C_SLICE, out_b, upd, upd
            return C_SLICE, out_b, out_b, mat(out_b)
        if name in CONCAT:
            return C_CONCAT, out_b, out_b, mat(out_b)
        if name == "pad":
            return C_PAD, out_b, out_b, mat(out_b)
        if name in TRANSPOSE:
            if not ctx.kernel and name == "transpose" and self.folded_transpose(eqn, jaxpr):
                return C_TFOLD, out_b, out_b, 0
            return C_TRANS, out_b, out_b, mat(out_b)
        if name in GATHER:
            if name.startswith("scatter") and len(in_b) >= 3:
                return C_GATHER, out_b, in_b[2], in_b[2]
            return C_GATHER, out_b, out_b, mat(out_b)
        if name == "select_n":
            return C_SELECT, out_b, out_b, mat(out_b)
        if name in REDUCE:
            b = max(in_b + [out_b])
            if ctx.kernel:
                return C_RED, out_b, b, b
            n_ops = max(1, len(eqn.invars) // 2) if name == "reduce" else 1    # reduce: operands, inits
            return C_RED, out_b, b, self.fused_read(eqn.invars[:n_ops], jaxpr, ctx.up)
        if name in SORT:
            b = max(in_b + [out_b])
            return C_SORT, out_b, b, b
        if name in COPY:
            return C_COPY, out_b, out_b, mat(out_b)
        if name in FREE:
            return C_FREE, out_b, out_b, 0
        if name in GENERATED:
            if ctx.kernel:
                return C_BMAT, out_b, out_b, out_b
            const_like = name == "iota" or not any(_is_var(v) for v in eqn.invars)
            m = mat(out_b, const_like)
            return (C_BMAT if m else C_BFUSED), out_b, out_b, m
        return C_OTHER, out_b, out_b, out_b

    # ------------------------------------------------------------------ walk
    def walk(self, jaxpr, ctx: Ctx, mat_out: frozenset | None = None):
        if mat_out is None:
            mat_out = frozenset(range(len(jaxpr.outvars)))
        self.uses(jaxpr)
        for eqn in jaxpr.eqns:
            self.n_eqns += 1
            try:
                self.visit(eqn, jaxpr, ctx, mat_out)
            except Exception as e:  # noqa: BLE001 - never fail on a strange equation
                name = getattr(eqn.primitive, "name", "?")
                try:
                    out_b = sum(_nbytes(v.aval) for v in eqn.outvars if hasattr(v, "aval"))
                except Exception:  # noqa: BLE001
                    out_b = 0
                self.add(C_OTHER, name, eqn, ctx, ctx.where or NO_FRAME, out_b, out_b,
                         note=f"walk error {type(e).__name__}: {str(e)[:120]}")
                self.notes.append(f"{name}: {type(e).__name__}: {str(e)[:200]}")

    def visit(self, eqn, jaxpr, ctx: Ctx, mat_out):
        name = eqn.primitive.name
        ns = str(eqn.source_info.name_stack)
        path = ctx.path + ("/" + ns if ns else "")
        where = self.where(eqn, ctx)
        here = dataclasses.replace(ctx, path=path, where=where)
        if name in ("scan", "while", "cond", "pallas_call"):
            here = dataclasses.replace(here, up=())         # their inputs are loop or kernel state

        if name == "scan":
            length = int(eqn.params["length"])
            body = _as_jaxpr(eqn.params["jaxpr"])
            scope = ctx.scope
            if self.a_scan_file and not ctx.kernel and ctx.scope == "step" \
                    and where and where[0][0] == self.a_scan_file and not self.found_micro_scan:
                scope, self.found_micro_scan = "micro", True         # the trainer's micro-batch scan
                length = 1                                           # one micro-batch per ledger
            here = dataclasses.replace(here, scope=scope)
            try:
                nc, nk = _scan_split(eqn)
                xs = sum(_nbytes(v.aval) for v in eqn.invars[nc + nk:] if hasattr(v, "aval"))
                ys = sum(_nbytes(v.aval) for v in eqn.outvars[nk:] if hasattr(v, "aval"))
                if scope == "micro" and ctx.scope == "step":
                    xs, ys = xs / max(1, int(eqn.params["length"])), ys / max(1, int(eqn.params["length"]))
                self.add(C_SCAN if not ctx.kernel else C_SLICE, "scan xs/ys", eqn, here, where, xs + ys, xs + ys,
                         note=f"length {eqn.params['length']}")
            except Exception as e:  # noqa: BLE001
                self.notes.append(f"scan at {where[0][0]}:{where[0][1]}: xs/ys not counted ({type(e).__name__})")
            self.walk(body, dataclasses.replace(here, mult=ctx.mult * length))
            return
        if name == "while":
            note = "while: trips unknown, counted once"
            self.notes.append(f"while loop at {where[0][0]}:{where[0][1]}: {note}")
            flagged = dataclasses.replace(here, flags=ctx.flags + (note,))
            for _, sub in _subjaxprs(eqn):
                self.walk(sub, flagged)
            return
        if name == "cond":
            branches = [j for j in (_as_jaxpr(b) for b in eqn.params.get("branches", ())) if j is not None]
            best, best_b = [], -1.0
            for b in branches:
                saved, self.recs = self.recs, []
                self.walk(b, here)
                got = sum(r.pas for r in self.recs)
                if got > best_b:
                    best, best_b = self.recs, got
                self.recs = saved
            self.recs.extend(best)
            return
        if name == "pallas_call":
            kname = str(eqn.params.get("name") or eqn.params.get("name_and_src_info") or "pallas_call")
            kname = kname.split(" ")[0]
            grid = getattr(eqn.params.get("grid_mapping"), "grid", ()) or ()
            try:
                trips = math.prod(int(g) for g in grid)
                flags = ctx.flags
            except Exception:  # noqa: BLE001 - dynamic grid
                trips, flags = 1, ctx.flags + ("pallas grid unknown, counted once",)
            body = _as_jaxpr(eqn.params.get("jaxpr"))
            flops = 0.0                                      # all applications (x ctx.mult)
            if body is not None:
                saved, self.recs = self.recs, []
                self.walk(body, dataclasses.replace(here, kernel=kname, mult=ctx.mult * trips, flags=flags))
                inner, self.recs = self.recs, saved
                flops = sum(r.flops for r in inner if r.cls == C_DOT)
                self.recs.extend(inner)
            out_b = sum(_nbytes(v.aval) for v in eqn.outvars if hasattr(v, "aval"))
            self.add(C_KERNEL, kname, eqn, here, where, out_b, out_b, flops / max(1, ctx.mult))
            self.recs[-1].aval = " ".join(_aval_str(v.aval) for v in eqn.outvars if hasattr(v, "aval"))
            return
        if name not in LEAF_WITH_COMBINER:
            subs = list(_subjaxprs(eqn))
            if subs:
                if name in TRANSPARENT:
                    sub_mo = self.sub_mat_out(eqn, jaxpr, mat_out) if len(subs) == 1 else None
                else:
                    sub_mo = None
                    self.notes.append(f"{name} at {where[0][0]}:{where[0][1]}: unknown container, "
                                      f"its {len(subs)} sub-program(s) counted once")
                inner = dataclasses.replace(here, site=where) if name in CACHEABLE else here
                if name in TRANSPARENT and len(subs) == 1:
                    inner = dataclasses.replace(inner, up=ctx.up + ((eqn, jaxpr),))
                else:
                    inner = dataclasses.replace(inner, up=())
                for _, sub in subs:
                    mo = sub_mo if sub_mo is not None and len(sub.outvars) == len(eqn.outvars) else None
                    self.walk(sub, inner, mo)
                return
        cls, out_b, pass_b, mat_b = self.classify(eqn, jaxpr, here, mat_out)
        flops = _dot_flops(eqn) if cls == C_DOT else 0.0
        if cls == C_DOT and not here.kernel:                # operand and result dtypes
            try:
                dts = [_SHORT.get(str(v.aval.dtype), str(v.aval.dtype)) for v in eqn.invars[:2]]
                name = f"{name}[{dts[0]},{dts[1]}->{_SHORT.get(str(eqn.outvars[0].aval.dtype), '?')}]"
            except Exception:  # noqa: BLE001
                pass
        aval = None
        if pass_b != out_b:                                 # the operand read or the update written
            ins = [v.aval for v in eqn.invars if hasattr(v, "aval")]
            aval = next((a for a in ins if _nbytes(a) == pass_b), None)
        self.add(cls, name, eqn, here, where, out_b, pass_b, flops, aval=aval, mat_b=mat_b)


# --------------------------------------------------------------------------- set-up
def git_state(repo: str) -> str:
    """``<short HEAD>`` plus ``+changes in bes/ configs/`` when they differ from it (read-only git)."""
    import subprocess
    try:
        head = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "--no-optional-locks", "-C", repo, "status", "--porcelain", "--",
                                "bes", "configs"], capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "?"
    n = len([ln for ln in dirty.splitlines() if ln.strip()])
    return (head or "?") + (f" +{n} changed file(s) in bes/ configs/" if n else "")


def import_bes(repo: str):
    repo = os.path.abspath(repo)
    if not os.path.isfile(os.path.join(repo, "bes", "__init__.py")):
        raise SystemExit(f"glue_ledger: no bes package in {repo}: run it from the root of a checkout "
                         "(or pass --repo)")
    sys.path.insert(0, repo)
    import bes
    got = os.path.dirname(os.path.abspath(bes.__file__))
    if got != os.path.join(repo, "bes"):
        raise SystemExit(f"glue_ledger: bes was imported from {got}, not from {repo}")
    return got


def _coerce(old: Any, text: str) -> Any:
    if isinstance(old, bool):
        t = text.strip().lower()
        if t in ("1", "true", "yes", "on"):
            return True
        if t in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{text!r} is not a boolean")
    if isinstance(old, int):
        return int(text)
    if isinstance(old, float):
        return float(text)
    if isinstance(old, str):
        return text
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def apply_sets(items: Sequence[str]) -> list[str]:
    """``module.attr=value`` -> setattr on the imported module (longest importable prefix)."""
    done = []
    for item in items:
        target, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set {item!r}: expected module.NAME=VALUE")
        parts = target.strip().split(".")
        obj, rest = None, []
        for k in range(len(parts) - 1, 0, -1):
            try:
                obj = importlib.import_module(".".join(parts[:k]))
            except ImportError:
                continue
            rest = parts[k:]
            break
        if obj is None:
            raise SystemExit(f"--set {item!r}: no importable module in {target!r}")
        for name in rest[:-1]:
            obj = getattr(obj, name)
        attr = rest[-1]
        if not hasattr(obj, attr):
            raise SystemExit(f"--set {item!r}: {'.'.join(parts[:-1])} has no attribute {attr!r} "
                             "(add the switch first, or check the name)")
        old = getattr(obj, attr)
        new = _coerce(old, value.strip())
        setattr(obj, attr, new)
        done.append(f"{target.strip()} = {new!r} (was {old!r})")
    return done


def build(args):
    """The traced program, its abstract arguments and a description."""
    import jax

    from bes.config import load_config
    from bes.model import model as M
    from bes.train import schedule
    from bes.train import train as TR

    overrides = [o for o in args.base_overrides.split() if o] + list(args.override)
    cfg = load_config(args.config, overrides)
    sets = apply_sets(args.set)
    mc = cfg.model
    t = int(args.seq_len or cfg.data.seq_len)
    try:
        _, rows_cfg, _ = schedule.micro_shape(cfg, t)
    except ValueError:
        rows_cfg = int(cfg.optim.micro_seqs)
    rows = int(args.rows or rows_cfg)
    if mc.attn_impl == "splash":
        from bes.model import attention_splash
        attention_splash.register_alias()
    try:
        from bes.train import supervisor as SV
        consts = SV.make_consts_fn(cfg.run.switches)(mc)
    except ImportError:
        sw = {}
        for it in str(cfg.run.switches).replace(",", " ").split():
            k, _, v = it.partition(":")
            sw[k] = float(v)
        consts = M.make_consts(mc, **sw)
    params = jax.eval_shape(lambda k: M.init_params(mc, k), jax.random.key(int(cfg.run.seed)))
    consts_a = TR.abstract(consts)
    spec = TR.batch_spec((1, rows, t))
    if args.program == "trainer":
        def fn(p, c, b):
            return TR.accumulate(M.loss_fn, p, c, b, mc)
        batch = spec
    else:
        batch = {k: jax.ShapeDtypeStruct(v.shape[1:], v.dtype) for k, v in spec.items()}
        if args.program == "vg":
            def fn(p, c, b):
                return jax.value_and_grad(M.loss_fn, has_aux=True)(p, c, b, mc)
        else:
            def fn(p, c, b):
                return M.loss_fn(p, c, b, mc)
    n_params = sum(math.prod(x.shape) for x in jax.tree.leaves(params))
    desc = {"config": args.config, "overrides": overrides, "sets": sets, "program": args.program,
            "rows": rows, "seq_len": t, "rows_trainer": rows_cfg, "act_dtype": mc.act_dtype,
            "remat": bool(mc.remat), "remat_keep": mc.remat_keep, "attn_impl": mc.attn_impl,
            "table_grad": mc.table_grad, "loss_chunk": int(mc.loss_chunk), "n_loops": int(mc.n_loops),
            "n_blocks": int(mc.n_blocks), "d_model": int(mc.d_model), "params": int(n_params),
            "switches": cfg.run.switches, "jax": jax.__version__}
    return fn, (params, consts_a, batch), desc


# --------------------------------------------------------------------------- report
def _agg():
    return {"count": 0, "out": 0.0, "pass": 0.0, "mat": 0.0, "flops": 0.0, **{p: 0.0 for p in PHASES}}


#: classes whose materialised results are writes outside the matrix products (ranking (2b))
WRITES = GLUE + (C_ELEM,)


def summarise(led: Ledger, top: int) -> dict:
    classes: dict[str, dict] = defaultdict(_agg)
    other: dict[str, dict] = defaultdict(_agg)
    kernels: dict[str, dict] = defaultdict(_agg)
    inside: dict[str, dict] = defaultdict(_agg)
    step: dict[str, dict] = defaultdict(_agg)
    lines: dict[tuple, dict] = {}
    for r in led.recs:
        if r.scope != "micro":
            a = step[r.cls]
        elif r.kernel:
            a = inside[f"{r.kernel}: {r.prim}"]
        else:
            a = classes[r.cls]
        micro = r.scope == "micro" and not r.kernel
        for tgt in ([a, other[r.prim]] if (r.cls == C_OTHER and micro)
                    else [a, kernels[r.prim]] if (r.cls in (C_KERNEL, C_DOT) and micro) else [a]):
            tgt["count"] += r.n
            tgt["out"] += r.out
            tgt["pass"] += r.pas
            tgt["mat"] += r.mat
            tgt["flops"] += r.flops
            tgt[r.phase] += r.pas
        if r.scope != "micro" or r.kernel:
            continue
        ln = lines.get(r.where)
        if ln is None:
            ln = lines[r.where] = {"where": [list(f) for f in r.where], "glue": 0.0, "glue_mat": 0.0,
                                   "writes_mat": 0.0, "out": 0.0, "flops": 0.0, **{p: 0.0 for p in PHASES},
                                   "by_class": defaultdict(float), "mat_by_class": defaultdict(float),
                                   "prims": {}, "mat_aval": "", "mat_aval_bytes": 0}
        ln["out"] += r.out
        ln["flops"] += r.flops
        ln["by_class"][r.cls] += r.pas
        if r.mat:
            ln["mat_by_class"][r.cls] += r.mat
        if r.cls in WRITES and r.mat:
            ln["writes_mat"] += r.mat
            if r.aval_bytes > ln["mat_aval_bytes"]:
                ln["mat_aval"], ln["mat_aval_bytes"] = r.aval, r.aval_bytes
        if r.cls in GLUE:
            ln["glue"] += r.pas
            ln["glue_mat"] += r.mat
            ln[r.phase] += r.pas
            pr = ln["prims"].setdefault(r.prim, {"count": 0, "pass": 0.0, "mat": 0.0, "aval": "", "aval_bytes": 0})
            pr["count"] += r.n
            pr["pass"] += r.pas
            pr["mat"] += r.mat
            if r.aval_bytes > pr["aval_bytes"]:
                pr["aval"], pr["aval_bytes"] = r.aval, r.aval_bytes
    glue = {p: sum(classes[c][p] for c in GLUE) for p in PHASES}
    glue["total"] = sum(classes[c]["pass"] for c in GLUE)
    glue["mat"] = sum(classes[c]["mat"] for c in GLUE)
    glue["count"] = sum(classes[c]["count"] for c in GLUE)
    glue["writes_mat"] = sum(classes[c]["mat"] for c in WRITES)
    ranked = sorted(lines.values(), key=lambda x: (-x["glue"], -x["writes_mat"]))
    for ln in ranked:
        f = ln["where"][0]
        path = os.path.join(led.repo, f[0]) if f[1] else ""
        ln["code"] = linecache.getline(path, f[1]).strip() if path else ""
        ln["by_class"] = dict(ln["by_class"])
        ln["mat_by_class"] = dict(ln["mat_by_class"])
    return {"classes": dict(classes), "other": dict(other), "kernels": dict(kernels),
            "inside_kernels": dict(inside), "per_step": dict(step), "glue": glue,
            "lines": ranked, "top": top}


def _mb(x: float) -> str:
    return f"{x / MB:,.0f}"


def _loc(where) -> str:
    return f"{where[0][0]}:{where[0][1]} {where[0][2]}" + "".join(
        f" < {g[0].split('/')[-1]}:{g[1]} {g[2]}" for g in where[1:])


def render(meta: dict, s: dict, out) -> None:
    w = out.write
    c = s["classes"]
    d = meta["desc"]
    w(f"glue ledger: {d['config']} [{' '.join(d['overrides']) or 'no overrides'}], program {d['program']}, "
      f"micro-batch {d['rows']} x {d['seq_len']} (trainer: {d['rows_trainer']} x {d['seq_len']}), "
      f"{d['act_dtype']}, remat {d['remat']}{' keep=' + d['remat_keep'] if d['remat_keep'] else ''}, "
      f"attention {d['attn_impl']}, table_grad {d['table_grad']}, {d['n_blocks']} blocks x {d['n_loops']} loops, "
      f"d {d['d_model']}, {d['params'] / 1e6:.1f}M stored parameters\n")
    w(f"bes from {meta['bes']} (git {meta.get('git', '?')}); jax {d['jax']}; trace {meta['trace_s']:.1f} s, "
      f"walk {meta['walk_s']:.1f} s; {meta['n_eqns']:,} equations walked; {meta['stamp']}\n")
    for item in d["sets"]:
        w(f"set: {item}\n")
    if d["switches"]:
        w(f"switches: {d['switches']}\n")
    w("\n(1) by primitive class, per micro-batch: applications (x trips); MB (1e6 bytes) of output, of the "
      "pass (output; operand for reductions, update for scatters), of the pass by phase, and the part of the "
      "pass that is materialised (mat: written to memory, fusion rule in the help)\n")
    hdr = (f"  {'class':34s} {'count':>9s} {'out MB':>10s} {'pass MB':>10s} {'fwd':>9s} {'remat':>9s} "
           f"{'bwd':>9s} {'mat MB':>9s} {'GFLOP':>9s}\n")
    w(hdr)

    def row(label, a, indent=2):
        fl = f"{a['flops'] / 1e9:9,.1f}" if a["flops"] else f"{'':9s}"
        w(f"{'':{indent}s}{label:{36 - indent}s} {a['count']:9,d} {_mb(a['out']):>10s} {_mb(a['pass']):>10s} "
          f"{_mb(a['fwd']):>9s} {_mb(a['remat']):>9s} {_mb(a['bwd']):>9s} {_mb(a['mat']):>9s} {fl}\n")

    for title, names in SECTIONS:
        w(f"  {title}\n")
        if names == (C_OTHER,):
            if not s["other"]:
                w("    (none)\n")
            for prim, a in sorted(s["other"].items(), key=lambda kv: -kv[1]["pass"]):
                row(prim, a, 4)
            continue
        for name in names:
            row(name, c.get(name, _agg()), 4)
            if name in (C_KERNEL, C_DOT):
                pre = "dot_general[" if name == C_DOT else ""
                subs = [(k, a) for k, a in s["kernels"].items() if k.startswith("dot_general[") == bool(pre)]
                for k, a in sorted(subs, key=lambda kv: -kv[1]["flops"]):
                    row(k[len("dot_general"):][:30] if pre else k[:30], a, 6)
        if names == GLUE:
            tot = _agg()
            for name in GLUE:
                for k in tot:
                    tot[k] += c.get(name, _agg())[k]
            row("= glue (sum of the above)", tot, 4)
    if s["inside_kernels"]:
        w("  inside Pallas kernels (VMEM tiles, not HBM; x grid, fully masked tiles included)\n")
        tot = defaultdict(lambda: _agg())
        for k, a in s["inside_kernels"].items():
            kern = k.split(": ")[0]
            for f in a:
                tot[kern][f] += a[f]
        for kern, a in sorted(tot.items(), key=lambda kv: -kv[1]["flops"]):
            row(kern[:32], a, 4)
    if s["per_step"]:
        ps = sum(a["pass"] for a in s["per_step"].values())
        parts = ", ".join(f"{k} {_mb(a['pass'])}" for k, a in sorted(s["per_step"].items(), key=lambda kv: -kv[1]["pass"])[:4])
        w(f"  once per step, outside the micro-batch scan (not in the totals): {_mb(ps)} MB ({parts})\n")

    w(f"\n(2) top {s['top']} source lines by memory-bound pass MB per micro-batch (fwd/remat/bwd; mat = "
      f"materialised), with the primitives they produce (MB, applications x trips, largest tensor)\n")
    for i, ln in enumerate(s["lines"][:s["top"]], 1):
        if ln["glue"] <= 0:
            break
        w(f"{i:3d} {_mb(ln['glue']):>8s} MB  ({_mb(ln['fwd'])}/{_mb(ln['remat'])}/{_mb(ln['bwd'])}, "
          f"mat {_mb(ln['glue_mat'])})  {_loc(ln['where'])}\n")
        prims = sorted(ln["prims"].items(), key=lambda kv: -kv[1]["pass"])
        w("      " + ", ".join(f"{p} {_mb(a['pass'])} ({a['count']}x, {a['aval']})" for p, a in prims[:5]) + "\n")
        if ln.get("code"):
            w(f"      | {ln['code'][:110]}\n")
    n2 = max(10, s["top"] // 2) if s["top"] > 0 else 0
    by_w = sorted((ln for ln in s["lines"] if ln["writes_mat"] > 0), key=lambda x: -x["writes_mat"])[:n2]
    if n2:
        w(f"\n(2b) top {n2} source lines by materialised MB outside the matrix products: glue and element-wise "
          f"results that must be written (their consumer is a product, a kernel, a loop boundary or an "
          f"output) and the fused reads of reductions\n")
    for i, ln in enumerate(by_w, 1):
        parts = ", ".join(f"{k} {_mb(v)}" for k, v in sorted(ln["mat_by_class"].items(), key=lambda kv: -kv[1])
                          if k in WRITES and v >= 0.5 * MB)
        w(f"{i:3d} {_mb(ln['writes_mat']):>8s} MB  {_loc(ln['where'])}  [{parts}; largest {ln['mat_aval']}]\n")
        if ln.get("code"):
            w(f"      | {ln['code'][:110]}\n")
    g = s["glue"]
    dot = c.get(C_DOT, _agg())
    w(f"\n(3) glue bytes per micro-batch: {_mb(g['total'])} MB in {g['count']:,} applications (fwd {_mb(g['fwd'])}, "
      f"remat {_mb(g['remat'])}, bwd {_mb(g['bwd'])}), of which materialised {_mb(g['mat'])} MB; for scale: "
      f"matrix products {dot['flops'] / 1e12:,.2f} TFLOP = {dot['flops'] / (PEAK_TFLOPS * 1e12) * 1e3:.1f} ms "
      f"at {PEAK_TFLOPS:.0f} TFLOP/s, their outputs {_mb(dot['out'])} MB\n")
    for n in meta["notes"][:20]:
        w(f"note: {n}\n")


def render_explain(led: Ledger, pattern: str, out) -> None:
    """Every application charged to a source key containing ``pattern`` (e.g. ``attention.py:377``)."""
    w = out.write
    rows = [r for r in led.recs if pattern in " ".join(f"{f[0]}:{f[1]} {f[2]}" for f in r.where)]
    w(f"\n--explain {pattern!r}: {len(rows)} equations (pass and materialised MB per micro-batch, "
      f"applications x trips, name stack)\n")
    for r in sorted(rows, key=lambda r: (-r.pas, r.prim)):
        loc = ", ".join(f"{f[0].split('/')[-1]}:{f[1]}" for f in r.where)
        tag = f"{r.scope}{' kernel ' + r.kernel if r.kernel else ''}"
        w(f"  {r.pas / MB:9,.1f} {r.mat / MB:9,.1f} {r.prim:22s} {r.cls:26s} {r.phase:5s} {r.n:5d}x {r.aval:26s} "
          f"{loc} [{tag}]{' ' + r.note if r.note else ''} | {r.path[-110:]}\n")


def render_diff(base: dict, cur: dict, out, top: int = 25) -> None:
    """Classes (exact) and source functions (robust to line shifts) against an earlier ``--json``."""
    w = out.write
    bs, cs = base["summary"], cur["summary"]
    bd = base["meta"]["desc"]
    w(f"\ndiff against {bd.get('config')} {bd.get('overrides')} {bd.get('sets')} ({bd['rows']} x {bd['seq_len']}, "
      f"{bd.get('program')}, {base['meta'].get('stamp', '')}, bes from {base['meta'].get('bes')})\n")
    if (bd["rows"], bd["seq_len"]) != (cur["meta"]["desc"]["rows"], cur["meta"]["desc"]["seq_len"]):
        w("  WARNING: the micro-batch shapes differ\n")
    w(f"  {'class (pass MB; mat MB)':34s} {'base':>10s} {'now':>10s} {'delta':>10s} {'mat base':>10s} "
      f"{'mat now':>10s} {'delta':>10s}\n")

    def line(label, b, a, bm, am):
        left = (f"{_mb(b):>10s} {_mb(a):>10s} {(a - b) / MB:+10,.0f}" if b is not None
                else f"{'':>10s} {'':>10s} {'':>10s}")
        w(f"  {label:34s} {left} {_mb(bm):>10s} {_mb(am):>10s} {(am - bm) / MB:+10,.0f}\n")
    names = [n for _, ns in SECTIONS for n in ns if n != C_OTHER]
    for n in names:
        b, a = (s["classes"].get(n, {}) for s in (bs, cs))
        if b.get("pass") or a.get("pass"):
            line(n, b.get("pass", 0.0), a.get("pass", 0.0), b.get("mat", 0.0), a.get("mat", 0.0))
    bg, ag = bs["glue"], cs["glue"]
    line("= glue", bg["total"], ag["total"], bg.get("mat", 0.0), ag.get("mat", 0.0))
    w(f"  {'':34s} {'':>10s} {'':>10s} {(ag['total'] - bg['total']) / max(bg['total'], 1) * 100:+9.1f}% "
      f"{'':>10s} {'':>10s} {(ag.get('mat', 0) - bg.get('mat', 0)) / max(bg.get('mat', 0), 1) * 100:+9.1f}%\n")
    w(f"  {'glue applications':34s} {bg.get('count', 0):10,d} {ag.get('count', 0):10,d} "
      f"{ag.get('count', 0) - bg.get('count', 0):+10,d}\n")
    line("mat. writes outside products", None, None, bg.get("writes_mat", 0.0), ag.get("writes_mat", 0.0))
    bf = bs["classes"].get(C_DOT, {}).get("flops", 0.0)
    af = cs["classes"].get(C_DOT, {}).get("flops", 0.0)
    w(f"  {'dot_general GFLOP':34s} {bf / 1e9:10,.1f} {af / 1e9:10,.1f} {(af - bf) / 1e9:+10,.1f}\n")

    def by_func(s):
        acc: dict[str, list] = defaultdict(lambda: [0.0, 0.0])
        for ln in s["lines"]:
            f = ln["where"][0]
            acc[f"{f[0]} {f[2]}"][0] += ln["glue"]
            acc[f"{f[0]} {f[2]}"][1] += ln.get("writes_mat", 0.0)
        return acc
    fb, fa = by_func(bs), by_func(cs)
    zero = [0.0, 0.0]

    def change(k):
        return max(abs(fa.get(k, zero)[0] - fb.get(k, zero)[0]), abs(fa.get(k, zero)[1] - fb.get(k, zero)[1]))
    keys = sorted(set(fb) | set(fa), key=lambda k: -change(k))
    w("  by source function (line numbers may move between checkouts), largest changes: glue pass MB "
      "base -> now, materialised writes MB base -> now\n")
    for k in keys[:top]:
        if change(k) < 0.5 * MB:
            break
        b, a = fb.get(k, zero), fa.get(k, zero)
        w(f"    {(a[0] - b[0]) / MB:+9,.0f} ({_mb(b[0]):>7s} -> {_mb(a[0]):>7s})   {(a[1] - b[1]) / MB:+9,.0f} "
          f"({_mb(b[1]):>7s} -> {_mb(a[1]):>7s})  {k}\n")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n\n".join(__doc__.split("\n\n")[1:]))
    ap.add_argument("--repo", default=".", help="checkout whose bes package is traced (default: the "
                                                "current working directory)")
    ap.add_argument("--config", default="configs/final.py", help="relative to --repo (default %(default)s)")
    ap.add_argument("--base-overrides", default=LIVE_OVERRIDES,
                    help="overrides applied first (default: the live run's, %(default)r; '' for none)")
    ap.add_argument("--override", action="append", default=[], help="section.field=value (repeatable)")
    ap.add_argument("--set", action="append", default=[], help="module.NAME=VALUE (repeatable)")
    ap.add_argument("--seq-len", type=int, default=0, help="sequence length (default: data.seq_len)")
    ap.add_argument("--rows", type=int, default=0,
                    help="rows per micro-batch (default: the trainer's, micro_seqs * seq_len / T)")
    ap.add_argument("--program", choices=("trainer", "vg", "forward"), default="trainer")
    ap.add_argument("--top", type=int, default=40, help="source lines listed (default %(default)s)")
    ap.add_argument("--depth", type=int, default=1, help="bes frames per source key (default 1)")
    ap.add_argument("--json", default=None, help="write the ledger here")
    ap.add_argument("--diff", default=None, help="compare with a ledger written by --json")
    ap.add_argument("--explain", action="append", default=[],
                    help="list every equation charged to a source key containing this text, e.g. "
                         "attention.py:377 (repeatable)")
    args = ap.parse_args(argv)

    repo = os.path.abspath(args.repo)
    bes_dir = import_bes(repo)
    if not os.path.isabs(args.config) and os.path.exists(os.path.join(repo, args.config)):
        args.config = os.path.join(repo, args.config)
    import jax

    t0 = time.perf_counter()
    fn, fargs, desc = build(args)
    desc["config"] = os.path.relpath(args.config, repo) if args.config.startswith(repo) else args.config
    closed = jax.make_jaxpr(fn)(*fargs)
    trace_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    led = Ledger(repo, bes_dir, args.depth,
                 os.path.join("bes", "train", "train.py") if args.program == "trainer" else None)
    led.walk(closed.jaxpr, Ctx(scope="step" if args.program == "trainer" else "micro"))
    if args.program == "trainer" and not led.found_micro_scan:
        led.notes.append("the trainer's micro-batch scan was not found: everything is counted once")
        for r in led.recs:
            r.scope = "micro"
    walk_s = time.perf_counter() - t1
    summ = summarise(led, args.top)
    meta = {"desc": desc, "bes": bes_dir, "git": git_state(repo), "trace_s": trace_s, "walk_s": walk_s,
            "n_eqns": led.n_eqns, "notes": sorted(set(led.notes)),
            "stamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
    render(meta, summ, sys.stdout)
    for pattern in args.explain:
        render_explain(led, pattern, sys.stdout)
    doc = {"meta": meta, "summary": summ}
    if args.diff:
        with open(args.diff) as f:
            render_diff(json.load(f), json.loads(json.dumps(doc, default=str)), sys.stdout)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(doc, f, indent=1, default=str)
            f.write("\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
