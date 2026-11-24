"""TPU attention path (``cfg.attn_impl == "splash"``): the Pallas splash-attention kernels of
``jax.experimental.pallas.ops.tpu.splash_attention`` as shipped with jax 0.11.1.

Contract (identical to the dense path ``bes.model.attention.attend_xla`` for self-attention):
``q, k, v: [B, T, H, head_dim]`` arrive full width with the reserved bias dims, the sink keys
and values and the confidence dim already assembled (``bes.model.sinks``); the logit is the plain
dot product (no softmax scale: the learned gain lives in ``q``); a key is visible when it is not
later than the query and has the same segment id. The calibrated sink is an ordinary key (the
``BOS`` position, whose bias dim carries ``sink_logit``), so the kernel's own ``sinks`` argument
(a value-less extra logit per head) is not used, and neither is its logit soft cap.

What the installed kernel provides and how it is used here:

* per-example kernels ``[H, T, head_dim]`` (``make_splash_mha_single_device``), vmapped over the
  batch; a static causal mask (``CausalMask``, computed inside the kernel, fully masked blocks
  skipped) combined with per-example ``SegmentIds``;
* block sizes from the config (``splash_block_q``, ``splash_block_kv``,
  ``splash_block_kv_compute``), shrunk to the largest multiple of 128 dividing the padded
  length; the backward kernels use the same tiles (``splash_fused_bwd`` selects the fused
  dQ/dK/dV kernel);
* lengths that are not a multiple of 128 are padded with an extra segment id (padded queries
  see only padded keys, so no softmax row is empty; real queries never see them);
* the kernel's custom VJP (forward residual = log-sum-exp) gives the gradients;
* ``interpret`` mode (``splash_interpret``: ``"auto"`` = interpret unless the default backend is
  a TPU) runs the same kernels on CPU, which the tests use against the dense path.

Precision: the kernel computes ``q.k`` with float32 accumulation of the operand dtype (bfloat16 in
training), the softmax statistics in float32 and ``p @ v`` with ``p`` in float32 and ``v`` cast to
float32; the output is returned in the dtype of ``q``.

Imported lazily by ``bes.model.attention`` so CPU runs never touch Pallas.

Other kernels for the same attention, behind the module switch :data:`FAST_KERNEL` (read at trace
time; the default ``""`` is the path above). Each computes the same output and gradients to float
rounding (``tests/model/test_attention_onetile.py``); ``scripts/tpu_kernels.py`` times them on the
chip:

* ``"onetile[:bs=256,hp=..,max_t=2048]"``: ``bes.model.attention_onetile``, one grid step per
  (row, heads) slice that walks the causal triangle in strips of ``bs`` rows (``hp`` heads per
  grid step, default 2 up to 1024 tokens, else 1); longer sequences than ``max_t`` (default
  :data:`ONETILE_MAX_T`) keep the splash kernels;
* ``"blocks:q=..,kv=..,kvc=..[,sep][,klayout=seq]"``: these splash kernels with other settings
  (tiles, separate dq/dkv kernels, operand layouts) than the configuration's;
* ``"tokamax[:block=1024,kvc=..,diag=1,grid=2]"``: the splash kernel of tokamax 0.0.17
  (``bes.model.attention_tokamax``; ``pip install --no-deps tokamax==0.0.17``);
* ``"xlamap"``: the dense XLA formulation one (row, head) slice at a time (``lax.map``);
* the flag ``save`` on any of them (``"onetile:save"``, ``"blocks:save"``): the rematerialised
  loop step (``bes.model.controller.run_loops``) keeps the forward kernel's output and
  log-sum-exp (:func:`remat_policy`), so its backward pass does not run the forward kernel a
  second time (same values: the kernel is deterministic).
"""
from __future__ import annotations

import functools
import os

import jax
import jax.numpy as jnp
import numpy as np

LANES = 128            # TPU lane width: block sizes and padded lengths are multiples of it
PAD_SEGMENT = -2       # segment id of padding positions (real ids are >= 0, cache gaps use -1)

#: kernel switch (module docstring); read at trace time, so set it before a program is traced
#: (a function JAX has already traced keeps the kernel it was traced with). Initial value from
#: the environment variable ``BES_FAST_KERNEL`` (unset: the configured kernels), so that a
#: process can be started with another kernel without a change of code or configuration.
FAST_KERNEL = os.environ.get("BES_FAST_KERNEL", "")
#: name of the forward residuals for the ``save`` flag (``jax.ad_checkpoint.checkpoint_name``)
RESIDUAL_NAME = "attention_residuals"
#: longest (padded) sequence the one-tile kernels take (the whole slice and its strip temporaries
#: live in VMEM; 2048 is compiled for the chip); longer ones use the splash kernels
ONETILE_MAX_T = 2048
KERNELS = ("", "onetile", "blocks", "tokamax", "xlamap")


def parse_switch(value: str | None = None) -> tuple[str, dict[str, str], frozenset[str]]:
    """``"name:key=value,flag,..."`` -> ``(name, {key: value}, {flags})`` (default: the switch)."""
    s = (FAST_KERNEL if value is None else value).strip()
    name, _, rest = s.partition(":")
    name = name.strip()
    if name not in KERNELS:
        raise ValueError(f"FAST_KERNEL: unknown kernel {name!r} in {s!r} (one of {KERNELS})")
    opts, flags = {}, set()
    for item in filter(None, (x.strip() for x in rest.split(","))):
        key, eq, val = item.partition("=")
        if eq:
            opts[key.strip()] = val.strip()
        else:
            flags.add(key)
    return name, opts, frozenset(flags)


#: name prefixes of the forward attention kernels (``pallas_call`` names) the ``save`` flag keeps
FORWARD_KERNELS = ("onetile_fwd", "splash_mha_fwd")


def remat_policy():
    """The ``jax.checkpoint`` policy the switch asks of a rematerialised loop step: ``None``
    (recompute everything, the configured behaviour) unless it carries the flag ``save``.

    With ``save``, the named forward residuals (:data:`RESIDUAL_NAME`) and the forward attention
    kernels themselves are saveable, nothing else. The kernels have to be: JAX never removes a
    ``pallas_call`` as dead code (its DCE rule keeps every kernel), so a kernel that is not
    saveable is run again in the backward pass even when all of its outputs were saved. Marked
    saveable, it is not recomputed, and only the values the backward pass reads become residuals:
    the named output and log-sum-exp (bfloat16 ``[B, H, T, hd]`` and float32 ``[B, H, T]``, x8
    for the one-tile kernels), not the splash kernel's other outputs (its running maximum, sum
    and output, and the 128-lane log-sum-exp)."""
    if "save" not in parse_switch()[2]:
        return None
    names = jax.checkpoint_policies.save_only_these_names(RESIDUAL_NAME)

    def keep_attention_forward(prim, *args, **params) -> bool:
        if getattr(prim, "name", "") == "pallas_call":
            return str(params.get("name") or "").startswith(FORWARD_KERNELS)
        return bool(names(prim, *args, **params))
    return keep_attention_forward


#: Speed switch, read at trace time (plain Python ``if``); False keeps the program of the live run
#: (tests/model/test_splash_io.py: the same outputs and gradients, bit for bit).
#:
#: FAST_SPLASH_IO: :func:`attend_major` calls the library's own kernel bodies through pallas_call
#: set-ups of this module (:func:`_io_forward`, :func:`_io_dkv`) that keep the kernels'
#: accumulators in VMEM scratch and pass the log-sum-exp compactly. As shipped (jax 0.11.1/0.11.2),
#: the forward call declares its m, l and o accumulators as outputs (``TODO(sharadmv): convert m/l
#: to be scratch``), so every example writes them to HBM, and writes the log-sum-exp lane-broadcast
#: as ``float32[H, T, 128]`` that the backward pass slices to ``[H, T]`` and broadcasts again to
#: ``[H, 8, T]``; the fused backward call does the same with its dq/dk/dv accumulators. At 16 x
#: 1024 per block application that is 75 MB written for nothing by each forward call (the forward
#: pass and its recomputation), a 50 MB read to slice the log-sum-exp and 25 MB by the backward
#: call. Here the forward call writes the log-sum-exp once as ``[H, 8, T]`` (the layout the
#: backward kernel reads, transposed in the kernel from the tile the library computes) and nothing
#: else. The training program compiled for TPU v6e moves 66.0 instead of 70.0 GB through HBM per
#: 16 x 1024 micro-batch (68.7 instead of 71.9 GB at 8 x 2048), with the same peak memory.
FAST_SPLASH_IO = False


def _sa():
    from jax.experimental.pallas.ops.tpu import splash_attention
    return splash_attention


def _sak():
    """The library module that holds the splash kernel bodies (jax 0.11.1 and 0.11.2 ship the
    same file apart from type-checker comments)."""
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
    return splash_attention_kernel


def available() -> bool:
    """True when the installed jax ships the splash-attention kernels."""
    try:
        _sa()
    except ImportError:
        return False
    return True


def register_alias() -> None:
    """Make ``import bes.model.splash`` resolve to this module when no such module exists.

    ``bes.model.attention`` selects the splash path with ``from bes.model import splash``; this
    module is ``bes.model.attention_splash``. Until that import names this module, callers that
    run ``attn_impl="splash"`` (the compile-only checks, training) call this first; afterwards it
    is a no-op.
    """
    import importlib.util
    import sys
    if "bes.model.splash" in sys.modules or importlib.util.find_spec("bes.model.splash") is not None:
        return
    sys.modules["bes.model.splash"] = sys.modules[__name__]


def use_interpret(cfg) -> bool:
    """Interpret mode from ``cfg.splash_interpret`` (``"auto"``: not on a TPU backend)."""
    mode = str(getattr(cfg, "splash_interpret", "auto")).lower()
    if mode in ("1", "true", "yes", "on"):
        return True
    if mode in ("0", "false", "no", "off"):
        return False
    if mode != "auto":
        raise ValueError(f"splash_interpret must be auto/true/false, got {mode!r}")
    return jax.default_backend() != "tpu"


def padded_length(t: int) -> int:
    """Sequence length rounded up to a multiple of 128."""
    return -(-int(t) // LANES) * LANES


def _tile(requested: int, length: int) -> int:
    """Largest multiple of 128 that is <= ``requested`` and divides ``length``."""
    b = max(LANES, min(int(requested), length) // LANES * LANES)
    while length % b:
        b -= LANES
    return b


def block_sizes(cfg, t: int):
    """``BlockSizes`` for a padded length ``t`` (a multiple of 128), forward and backward."""
    sa = _sa()
    bq = _tile(getattr(cfg, "splash_block_q", 512), t)
    bkv = _tile(getattr(cfg, "splash_block_kv", 512), t)
    bkc = _tile(getattr(cfg, "splash_block_kv_compute", 512), bkv)
    fused = bool(getattr(cfg, "splash_fused_bwd", False))
    return sa.BlockSizes(
        block_q=bq, block_kv=bkv, block_kv_compute=bkc,
        block_q_dkv=bq, block_kv_dkv=bkv, block_kv_dkv_compute=bkc,
        block_q_dq=None if fused else bq, block_kv_dq=None if fused else bkv,
        use_fused_bwd_kernel=fused)


def switched_block_sizes(cfg, t: int, opts: dict[str, str], flags: frozenset[str]):
    """:func:`block_sizes` with the overrides of a ``"blocks:..."`` switch: ``q``, ``kv``, ``kvc``
    (forward tiles; the backward tiles follow unless ``qdkv``, ``kvdkv``, ``kvcdkv``, ``qdq``,
    ``kvdq`` are given), ``sep`` / ``fused`` (backward kernels), ``qlayout`` / ``klayout`` /
    ``vlayout`` = ``seq`` or ``head`` (operand layouts the kernels enforce)."""
    sa = _sa()
    base = block_sizes(cfg, t)
    bq = _tile(int(opts.get("q", base.block_q)), t)
    bkv = _tile(int(opts.get("kv", base.block_kv)), t)
    bkc = _tile(int(opts.get("kvc", base.block_kv_compute)), bkv)
    fused = base.use_fused_bwd_kernel
    if "sep" in flags:
        fused = False
    if "fused" in flags:
        fused = True
    qd = _tile(int(opts.get("qdkv", bq)), t)
    kvd = _tile(int(opts.get("kvdkv", bkv)), t)
    kcd = _tile(int(opts.get("kvcdkv", bkc)), kvd)
    layout = {"seq": sa.QKVLayout.SEQ_MINOR, "head": sa.QKVLayout.HEAD_DIM_MINOR}
    return sa.BlockSizes(
        block_q=bq, block_kv=bkv, block_kv_compute=bkc,
        block_q_dkv=qd, block_kv_dkv=kvd, block_kv_dkv_compute=kcd,
        block_q_dq=None if fused else _tile(int(opts.get("qdq", bq)), t),
        block_kv_dq=None if fused else _tile(int(opts.get("kvdq", bkv)), t),
        use_fused_bwd_kernel=fused,
        q_layout=layout[opts.get("qlayout", "head")], k_layout=layout[opts.get("klayout", "head")],
        v_layout=layout[opts.get("vlayout", "head")])


@functools.lru_cache(maxsize=None)
def make_kernel(t: int, n_heads: int, bs, interpret: bool, residual_name: str | None = None):
    """Per-example kernel for ``n_heads`` heads and length ``t`` (multiple of the blocks).

    Built under ``jax.ensure_compile_time_eval`` so that its mask-info arrays are concrete even
    when the first request comes from inside a trace (otherwise the cache would keep tracers of
    that trace and the next program using the same shape would fail)."""
    sa = _sa()
    mask = sa.MultiHeadMask([sa.CausalMask((t, t)) for _ in range(n_heads)])
    kw = {} if residual_name is None else {"residual_checkpoint_name": residual_name}
    with jax.ensure_compile_time_eval():
        return sa.make_splash_mha_single_device(mask, block_sizes=bs, interpret=interpret, **kw)


def attend(q: jax.Array, k: jax.Array, v: jax.Array, seg: jax.Array, cfg, *,
           interpret: bool | None = None) -> jax.Array:
    """Causal same-segment self-attention with the splash kernels.

    q, k, v: ``[B, T, H, head_dim]`` (any float dtype, normally the activation dtype);
    seg: int ``[B, T]`` segment ids (negative ids are allowed and only see equal ids).
    Returns ``[B, T, H, head_dim]`` in the dtype of ``q``.
    """
    qt, kt, vt = (jnp.swapaxes(a, 1, 2) for a in (q, k, v))                   # [B, H, T, hd]
    return jnp.swapaxes(attend_major(qt, kt, vt, seg, cfg, interpret=interpret), 1, 2)


def attend_major(q: jax.Array, k: jax.Array, v: jax.Array, seg: jax.Array, cfg, *,
                 interpret: bool | None = None) -> jax.Array:
    """:func:`attend` on head-major operands: ``q, k, v: [B, H, T, head_dim]``, the layout of the
    kernels, so that a caller that builds its operands in this layout pays no transpose (on TPU a
    transpose of an activation costs about as much as the matrix product that made it:
    docs/TPU_NOTES.md). Returns ``[B, H, T, head_dim]`` in the dtype of ``q``.
    """
    b, h, t, hd = q.shape
    tp = padded_length(t)
    seg = jnp.asarray(seg, jnp.int32)
    if tp != t:
        pad = ((0, 0), (0, 0), (0, tp - t), (0, 0))
        q, k, v = (jnp.pad(a, pad) for a in (q, k, v))
        seg = jnp.pad(seg, ((0, 0), (0, tp - t)), constant_values=PAD_SEGMENT)
    interp = use_interpret(cfg) if interpret is None else bool(interpret)
    name, opts, flags = parse_switch()
    res_name = RESIDUAL_NAME if "save" in flags else None
    if name == "onetile" and tp > int(opts.get("max_t", ONETILE_MAX_T)):
        name = ""                                   # longer than compiled and checked: the splash kernels
    if name == "onetile":
        from bes.model import attention_onetile as onetile
        y = onetile.attend_major(q, k, v, seg, bs=int(opts.get("bs", 256)), interpret=interp,
                                 residual_name=res_name, heads_per_step=int(opts.get("hp", 0)) or None)
        return y[:, :, :t]
    if name == "tokamax":
        from bes.model import attention_tokamax as tkx
        y = tkx.attend_major(q, k, v, seg, block=int(opts.get("block", 1024)),
                             kvc=int(opts["kvc"]) if "kvc" in opts else None,
                             diag_skip=opts.get("diag", "0") not in ("0", "false"),
                             diag_grid=int(opts.get("grid", 2)), interpret=interp, residual_name=res_name)
        return y[:, :, :t]
    if name == "xlamap":
        from bes.model import attention_onetile as onetile
        return onetile.attend_xla_slices(q, k, v, seg)[:, :, :t]
    sa = _sa()
    bs = switched_block_sizes(cfg, tp, opts, flags) if name == "blocks" else block_sizes(cfg, tp)
    if FAST_SPLASH_IO and name == "" and res_name is None:   # trace-time switch; the configured kernels only
        return jax.vmap(_io_attention(tp, h, bs, interp))(q, k, v, seg)[:, :, :t]
    kernel = make_kernel(tp, h, bs, interp) if res_name is None else make_kernel(tp, h, bs, interp, res_name)
    ids = sa.SegmentIds(q=seg, kv=seg)

    def one(q1, k1, v1, s1):
        return kernel(q1, k1, v1, segment_ids=s1)
    return jax.vmap(one)(q, k, v, ids)[:, :, :t]


# ------------------------------------------------------- FAST_SPLASH_IO: the kernels' own I/O
# The pallas_call set-ups below follow jax.experimental.pallas.ops.tpu.splash_attention.
# splash_attention_kernel (_splash_attention_forward, _splash_attention_bwd_dkv; Apache License
# 2.0, the JAX authors) for the case this model uses: multi-head (not MQA), segment ids, no sinks,
# no soft cap, head-dim-minor layouts. The kernel bodies are the library's, called unchanged; only
# what is an output and what is VMEM scratch differs, so the arithmetic is the same.

def _io_forward(info, q, k, v, seg, bs, *, mask_value, mask_function, interpret):
    """Forward kernel for one example: ``q [H, T, hd]``, ``k, v [H, T, hd]``, seg int32 ``[T]``.
    Returns ``(o [H, T, hd] in the dtype of q, lse float32[H, 8, T])``: the log-sum-exp of every
    query row repeated on 8 sublanes, the layout the backward kernel reads."""
    sak = _sak()
    nl, ns = sak.NUM_LANES, sak.NUM_SUBLANES
    h, t, hd = q.shape
    hd_v = v.shape[-1]
    bq, bkv, bkc = bs.block_q, bs.block_kv, bs.block_kv_compute
    per_kv = h // k.shape[0]
    grid_width = info.data_next.shape[-1] if info.data_next is not None else k.shape[1] // bkv
    nxt = sak._next_nonzero

    def q_map(hh, i, j, *_):
        return hh, i, 0

    def kv_map(hh, i, j, data_next_ref, block_mask_ref, mask_next_ref=None):
        next_j, *_ = nxt(hh, i, j, data_next_ref, block_mask_ref, mask_next_ref)
        return sak._div(hh, per_kv), next_j, 0

    def mask_map(hh, i, j, data_next_ref, block_mask_ref, mask_next_ref=None):
        _, next_m, *_ = nxt(hh, i, j, data_next_ref, block_mask_ref, mask_next_ref)
        return next_m, 0, 0

    def q_seg_map(hh, i, j, *_):
        return i, 0

    def kv_seg_map(hh, i, j, data_next_ref, block_mask_ref, mask_next_ref=None):
        next_j, *_ = nxt(hh, i, j, data_next_ref, block_mask_ref, mask_next_ref)
        return 0, next_j

    in_specs = [_spec((None, bq, hd), q_map), _spec((None, bkv, hd), kv_map),
                _spec((None, bkv, hd_v), kv_map), _spec((bq, nl), q_seg_map),
                _spec((ns, bkv), kv_seg_map), None,
                _spec((None, bq, bkv), mask_map) if info.partial_mask_blocks is not None else None]
    q_seg = jax.lax.broadcast_in_dim(seg, (t, nl), (0,))
    kv_seg = jax.lax.broadcast_in_dim(seg, (ns, k.shape[1]), (1,))
    if info.q_sequence is not None:
        q_sequence = jax.lax.broadcast_in_dim(info.q_sequence, (t, nl), (0,))
        in_specs.append(_spec((bq, nl), q_seg_map))
    else:
        q_sequence = None
        in_specs.append(None)
    body = functools.partial(
        sak.flash_attention_kernel, mask_value=mask_value, grid_width=grid_width, bq=bq, bkv=bkv,
        bkv_compute=bkc, head_dim_v=hd_v, q_layout=bs.q_layout, k_layout=bs.k_layout,
        v_layout=bs.v_layout, attn_logits_soft_cap=None, mask_function=mask_function)

    def kernel(data_next_ref, block_mask_ref, mask_next_ref, q_ref, k_ref, v_ref, q_seg_ref,
               kv_seg_ref, sinks_ref, mask_ref, q_sequence_ref, o_ref, lse_ref,
               m_scratch, l_scratch, o_scratch, lse_scratch):
        body(data_next_ref, block_mask_ref, mask_next_ref, q_ref, k_ref, v_ref, q_seg_ref,
             kv_seg_ref, sinks_ref, mask_ref, q_sequence_ref, m_scratch, l_scratch, o_scratch,
             o_ref, lse_scratch)

        @_pallas().when(_pallas().program_id(2) == grid_width - 1)
        def _():                      # the lane-broadcast [bq, 128] tile -> 8 rows of [bq]
            lse_ref[...] = lse_scratch[...].T[:ns, :]

    def lse_map(hh, i, j, *_):
        return hh, 0, i

    pltpu = _pltpu()
    f32 = jnp.float32
    return _pallas().pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3, in_specs=in_specs,
            out_specs=[_spec((None, bq, hd_v), q_map), _spec((None, ns, bq), lse_map)],
            grid=(h, t // bq, grid_width),
            scratch_shapes=[pltpu.VMEM((bq, nl), f32), pltpu.VMEM((bq, nl), f32),
                            pltpu.VMEM((bq, hd_v), f32), pltpu.VMEM((bq, nl), f32)]),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary", "arbitrary")),
        out_shape=[jax.ShapeDtypeStruct((h, t, hd_v), q.dtype), jax.ShapeDtypeStruct((h, ns, t), f32)],
        name="splash_mha_fwd_io", interpret=interpret,
    )(info.data_next, info.block_mask, info.mask_next, q, k, v, q_seg, kv_seg, None,
      info.partial_mask_blocks, q_sequence)


def _io_dkv(info, q, k, v, seg, lse, do, di, bs, *, mask_value, mask_function, interpret):
    """Fused backward kernel for one example (``bs.use_fused_bwd_kernel``): ``lse float32[H, 8,
    T]`` from :func:`_io_forward`, ``di float32[H, T]``. Returns ``(dq, dk, dv)`` in the dtypes of
    ``q, k, v``; the accumulators are VMEM scratch instead of outputs."""
    sak = _sak()
    nl, ns = sak.NUM_LANES, sak.NUM_SUBLANES
    h, t, hd = q.shape
    hd_v = v.shape[-1]
    n_kv, t_kv, _ = k.shape
    bq, bkv, bkc = bs.block_q_dkv, bs.block_kv_dkv, bs.block_kv_dkv_compute
    per_kv = h // n_kv
    grid_width = info.data_next.shape[-2] if info.data_next is not None else t // bq
    nxt = sak._next_nonzero

    def next_i(kv_index, head_index, q_index, data_next_ref, block_mask_ref, mask_next_ref=None):
        return nxt(head_index, q_index, kv_index, data_next_ref, block_mask_ref, mask_next_ref,
                   next_i=True)

    def o_map(kv_index, head_index, q_index, *refs):
        return head_index, next_i(kv_index, head_index, q_index, *refs)[0], 0

    def kv_map(kv_index, head_index, *_):
        return sak._div(head_index, per_kv), kv_index, 0

    def mask_map(kv_index, head_index, q_index, *refs):
        return next_i(kv_index, head_index, q_index, *refs)[1], 0, 0

    def q_seg_map(kv_index, head_index, q_index, *refs):
        return 0, next_i(kv_index, head_index, q_index, *refs)[0]

    def lse_map(kv_index, head_index, q_index, *refs):
        return head_index, 0, next_i(kv_index, head_index, q_index, *refs)[0]

    in_specs = [_spec((None, bq, hd), o_map), _spec((None, bkv, hd), kv_map),
                _spec((None, bkv, hd_v), kv_map), _spec((ns, bq), q_seg_map),
                _spec((bkv, nl), lambda kv_index, *_: (kv_index, 0)), None,
                _spec((None, ns, bq), lse_map), _spec((None, bq, hd_v), o_map),
                _spec((None, ns, bq), lse_map),
                _spec((None, bkv, bq), mask_map) if info.partial_mask_blocks is not None else None]
    q_seg = jax.lax.broadcast_in_dim(seg, (ns, t), (1,))
    kv_seg = jax.lax.broadcast_in_dim(seg, (t_kv, nl), (0,))
    di = jnp.broadcast_to(jnp.expand_dims(di, -2), (h, ns, t))
    if info.q_sequence is not None:
        in_specs.append(_spec((ns, bq), q_seg_map))
        q_sequence = jax.lax.broadcast_in_dim(info.q_sequence, (ns, t), (1,))
    else:
        in_specs.append(None)
        q_sequence = None
    body = functools.partial(
        sak._flash_attention_dkv_kernel, mask_value=mask_value, num_q_heads=h, num_kv_heads=n_kv,
        is_mqa=False, grid_width=grid_width, bq=bq, bkv_compute=bkc, attn_logits_soft_cap=None,
        q_layout=bs.q_layout, k_layout=bs.k_layout, v_layout=bs.v_layout, bkv=bkv,
        mask_function=mask_function)
    pltpu = _pltpu()
    f32 = jnp.float32
    with_dq_scratch = bkv != bkc
    scratch = ([pltpu.VMEM((bq, hd), f32)] if with_dq_scratch else []) + \
        [pltpu.VMEM((bkv, hd), f32), pltpu.VMEM((bkv, hd_v), f32)]

    def kernel(data_next_ref, block_mask_ref, mask_next_ref, q_ref, k_ref, v_ref, q_seg_ref,
               kv_seg_ref, sinks_ref, lse_ref, do_ref, di_ref, mask_ref, q_sequence_ref,
               dq_ref, dk_ref, dv_ref, *scratch_refs):
        dq_scratch = scratch_refs[0] if with_dq_scratch else None
        dk_scratch, dv_scratch = scratch_refs[-2:]
        body(data_next_ref, block_mask_ref, mask_next_ref, q_ref, k_ref, v_ref, q_seg_ref,
             kv_seg_ref, sinks_ref, lse_ref, do_ref, di_ref, mask_ref, q_sequence_ref,
             dq_scratch, dk_scratch, dv_scratch, dq_ref, dk_ref, dv_ref)

    def dq_map(kv_index, head_index, q_index, *_):
        return kv_index, head_index, q_index, 0

    dq_parts, dk, dv = _pallas().pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3, in_specs=in_specs,
            out_specs=[_spec((None, None, bq, hd), dq_map), _spec((None, bkv, hd), kv_map),
                       _spec((None, bkv, hd_v), kv_map)],
            grid=(t_kv // bkv, h, grid_width), scratch_shapes=scratch),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary", "arbitrary", "arbitrary")),
        out_shape=[jax.ShapeDtypeStruct((t_kv // bkv,) + q.shape, q.dtype),
                   jax.ShapeDtypeStruct(k.shape, k.dtype), jax.ShapeDtypeStruct(v.shape, v.dtype)],
        name="splash_mha_dkv_io", interpret=interpret,
    )(info.data_next, info.block_mask, info.mask_next, q, k, v, q_seg, kv_seg, None, lse, do, di,
      info.partial_mask_blocks, q_sequence)
    return dq_parts.sum(axis=0), dk, dv


def _pallas():
    from jax.experimental import pallas
    return pallas


def _pltpu():
    from jax.experimental.pallas import tpu
    return tpu


def _spec(block_shape, index_map):
    return _pallas().BlockSpec(block_shape, index_map)


@functools.lru_cache(maxsize=None)
def _io_attention(t: int, n_heads: int, bs, interpret: bool):
    """Per-example attention ``(q, k, v [H, t, hd], seg int32 [t]) -> o`` of ``FAST_SPLASH_IO``:
    the kernels of :func:`make_kernel` (same masks, tiles and mask value) through
    :func:`_io_forward` and, for the fused backward, :func:`_io_dkv`; with separate backward
    kernels the library's backward runs on the ``[H, t]`` log-sum-exp."""
    sak = _sak()
    lib = make_kernel(t, n_heads, bs, interpret)
    kw = lib.kwargs
    fwd_info, dkv_info, dq_info = lib.fwd_mask_info, lib.dkv_mask_info, lib.dq_mask_info
    common = dict(mask_value=kw["mask_value"], mask_function=kw["mask_function"], interpret=interpret)

    @jax.custom_vjp
    def attn(q, k, v, seg):
        return _io_forward(fwd_info, q, k, v, seg, bs, **common)[0]

    def attn_fwd(q, k, v, seg):
        o, lse = _io_forward(fwd_info, q, k, v, seg, bs, **common)
        return o, (q, k, v, seg, o, lse)

    def attn_bwd(res, do):
        q, k, v, seg, o, lse = res
        if bs.use_fused_bwd_kernel:
            di = jnp.einsum("hsd,hsd->hs", o.astype(jnp.float32), do.astype(jnp.float32))
            dq, dk, dv = _io_dkv(dkv_info, q, k, v, seg, lse, do, di, bs, **common)
        else:
            ids = sak.SegmentIds(q=seg, kv=seg)
            res_lib = (q, k, v, ids, None, o, lse[:, 0, :], dq_info, dkv_info)
            grads = sak._splash_attention_bwd(False, kw["mask_value"], False, bs, None,
                                              kw["mask_function"], None, interpret, res_lib, do)
            dq, dk, dv = grads[3:6]
        return dq, dk, dv, None

    attn.defvjp(attn_fwd, attn_bwd)
    # inside a jit, as the library's call: without it XLA fused the query prologue around the
    # kernels differently at 8 x 2048 (compiled for v6e: 1.3 GB saved instead of 3.3 GB, and
    # 0.2 GB more peak memory); with it the rest of the program compiles as before
    return jax.jit(attn)


def attend_like_xla(q: jax.Array, k: jax.Array, v: jax.Array, seg_q: jax.Array,
                    seg_k: jax.Array, pos_q: jax.Array, pos_k: jax.Array,
                    sink_k: jax.Array | None = None, *, cfg=None):
    """The signature of ``bes.model.attention.attend_xla`` for self-attention.

    Requires ``Tq == Tk`` with the same segments and positions ``arange(T)`` (the static causal
    mask is on slot order); positions and segments are not checked inside a trace. Returns
    ``(y float32[B, T, H, hd], None)``: the kernel does not expose the probabilities (with the
    confidence dim on, the sink mass is ``1 - y[..., -1]``).
    """
    del seg_k, pos_q, pos_k, sink_k
    if q.shape[1] != k.shape[1]:
        raise ValueError("the splash path is self-attention only (Tq == Tk)")
    y = attend(q, k, v, seg_q, cfg if cfg is not None else _DefaultCfg())
    return y.astype(jnp.promote_types(q.dtype, jnp.float32)), None


class _DefaultCfg:
    splash_block_q = splash_block_kv = splash_block_kv_compute = 512
    splash_fused_bwd = False
    splash_interpret = "auto"


def reference(q, k, v, seg) -> np.ndarray:
    """Float64 numpy reference of the same attention (tests and diagnostics)."""
    q, k, v = (np.asarray(a, np.float64) for a in (q, k, v))
    seg = np.asarray(seg)
    t = q.shape[1]
    logits = np.einsum("bqhd,bkhd->bhqk", q, k)
    mask = (np.arange(t)[None, :, None] >= np.arange(t)[None, None, :]) & \
        (seg[:, :, None] == seg[:, None, :])
    logits = np.where(mask[:, None], logits, -np.inf)
    p = np.exp(logits - logits.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return np.einsum("bhqk,bkhd->bqhd", p, v)


def flops(b: int, t: int, h: int, hd: int, causal: bool = True) -> float:
    """Forward matmul FLOPs of the attention core (``q.k`` and ``p.v``); half when causal."""
    f = 4.0 * b * h * t * t * hd
    return f / 2 if causal else f


__all__ = ["available", "register_alias", "use_interpret", "padded_length", "block_sizes", "make_kernel", "attend",
           "attend_major", "attend_like_xla", "reference", "flops", "LANES", "FAST_KERNEL", "parse_switch",
           "remat_policy", "switched_block_sizes"]
