"""Causal same-segment attention for sequences that fit one tile: Pallas TPU kernels with one grid
step per (row, heads) slice, selected by ``attention_splash.FAST_KERNEL = "onetile[:bs=256,hp=2]"``.

The splash kernels (``jax.experimental.pallas.ops.tpu.splash_attention``) at the training shapes
(``T = 1024`` or ``2048`` tokens, tiles of 1024) see one or two tiles per slice and pay the
machinery of a general block-sparse kernel on each: an online softmax over the inner key/value
steps, a causal mask computed on every element of the tile (half of which it then throws away),
the full tile through the matrix units and the vector units although half of it is above the
diagonal, its running maximum, sum and output written to HBM as kernel outputs, and the
log-sum-exp residual written 128 lanes wide. These kernels instead hold the whole slice
(``q, k, v: [T, head_dim]``) in VMEM and walk it in strips of ``BS`` rows that only cover the
causal triangle (at ``BS = 256`` and ``T = 1024``: 62.5% of the score elements):

* forward (rows = queries): strip ``i`` holds queries ``[i BS, (i+1) BS)`` against keys
  ``[0, (i+1) BS)``, as an off-diagonal piece (no causal mask: only the segment compare) and the
  diagonal block (causal and segment). The softmax of the strip is exact in one pass (the row
  maximum of the whole strip, no rescaling), ``p @ v`` as in the splash kernel (``p`` in float32,
  ``v`` cast to float32), the output scaled by ``1 / l`` and rounded to the dtype of ``q``. The
  log-sum-exp leaves the kernel as a lane-dense row ``[8, T]`` (one transpose of a ``[BS, 128]``
  block per strip), the layout the backward kernel reads.
* backward (rows = keys, the layout of the splash ``dkv`` kernel): strip ``j`` holds keys
  ``[j BS, (j+1) BS)`` against queries ``[j BS, T)``: ``p^T = exp(s^T - lse)``, ``dv = p^T @ do``,
  ``dp^T = v @ do^T``, ``ds^T = (dp^T - di) p^T``, ``dk = ds^T @ q`` complete per strip, and
  ``dq += ds @ k`` accumulated in float32 over the strips. Casts as in the splash kernel: ``p`` and
  ``ds`` to the input dtype before their products, accumulation in float32, outputs rounded once
  to the input dtype.

Same values as the splash kernels to float rounding (sums in another order; ``tests/model/
test_attention_onetile.py`` compares forward and gradients against them in interpret mode).
Masking value, segment semantics and the absence of a softmax scale are those of the splash path
(``bes.model.attention_splash``). The whole slice and the strip temporaries live in VMEM: at
``T = 2048``, ``BS = 256`` about 20 MiB in the backward kernel (the scoped VMEM limit of the
chip is 32 MiB); both training shapes were compiled for the chip with its libtpu from a CPU host
(``docs/TPU_NOTES.md``, "Attention kernels for one tile", with the static bundle counts that chose
the variants below).

Also here: :func:`attend_xla_slices`, the dense XLA formulation computed slice by slice with
``lax.map`` (``FAST_KERNEL = "xlamap"``), with the splash numerics.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

LANES = 128
SUBLANES = 8
#: the splash kernels' mask value (``splash_attention_kernel.DEFAULT_MASK_VALUE``)
MASK_VALUE = -0.7 * float(np.finfo(np.float32).max)
NT = (((1,), (1,)), ((), ()))           # a @ b.T
NN = (((1,), (0,)), ((), ()))           # a @ b
TN = (((0,), (0,)), ((), ()))           # a.T @ b


def _pallas():
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    return pl, pltpu


def strip_rows(t: int, requested: int = 256) -> int:
    """Strip height: the largest multiple of 128 that is <= ``requested`` and divides ``t``."""
    b = max(LANES, min(int(requested), int(t)) // LANES * LANES)
    while t % b:
        b -= LANES
    return b


# ----------------------------------------------------------------------------- forward kernel
def _lane_fold(op, arrays):
    """Element-wise ``op`` over the 128-lane columns of ``arrays`` (each ``[rows, 128 m]``):
    ``[rows, 128]``, so that one cross-lane reduction finishes the row reduction of all of them."""
    acc = None
    for a in arrays:
        for c in range(a.shape[-1] // LANES):
            col = a[:, c * LANES:(c + 1) * LANES]
            acc = col if acc is None else op(acc, col)
    return acc


# Variants of the kernel bodies for the compile-only comparison (module constants read at trace
# time; the defaults are the ones chosen from the static bundle counts, docs/TPU_NOTES.md).
FWD_ORDER = "heads"     # "seq": strip by strip; "allqk": all score products first (-25% bundles);
                        # "pipe": one strip ahead (-21%); "heads": the scores of head h+1
                        # interleaved with the softmax of head h (another -4%)
BWD_ORDER = "pipe"      # "seq": piece by piece; "pipe": score products of the next piece first
                        # (-22%); "ipipe": "pipe" with the heads alternating (+3%)
DS_CAST_FIRST = True    # backward: round ds to the input dtype before its transpose (same values, -2%)
HEADS_PER_STEP = 0      # heads per grid step; 0: 2 up to T = 1024, else 1 (one head's schedule fills
                        # the gaps of the other: -5% bundles per head at T = 1024, -1.5% at 2048)


def heads_per_grid_step(h: int, t: int, requested: int | None = None) -> int:
    """Heads per grid step: ``requested`` (or :data:`HEADS_PER_STEP`), 0 = automatic, clipped
    to a divisor of ``h``."""
    hp = int(HEADS_PER_STEP if requested is None else requested)
    if hp <= 0:
        hp = 2 if t <= 1024 else 1
    hp = max(1, min(hp, h))
    while h % hp:
        hp -= 1
    return hp


def _fwd_kernel(segc_ref, segr_ref, q_ref, k_ref, v_ref, o_ref, *lse_ref, t: int, bs: int, hp: int):
    f32 = jnp.float32
    n = t // bs
    causal = lax.broadcasted_iota(jnp.int32, (bs, bs), 1) <= lax.broadcasted_iota(jnp.int32, (bs, bs), 0)

    def scores(hh, i):
        """Masked scores of strip i of head hh: [diagonal block, keys before the strip], row max."""
        r0 = i * bs
        q = q_ref[hh, r0:r0 + bs, :]
        qid = segc_ref[r0:r0 + bs, :]                                   # [bs, 128] lane-broadcast
        s_d = lax.dot_general(q, k_ref[hh, r0:r0 + bs, :], NT, preferred_element_type=f32)
        vis_d = jnp.logical_and(jnp.tile(qid, (1, bs // LANES)) == segr_ref[0:1, r0:r0 + bs], causal)
        pieces = [jnp.where(vis_d, s_d, MASK_VALUE)]
        if i:
            s_o = lax.dot_general(q, k_ref[hh, 0:r0, :], NT, preferred_element_type=f32)
            vis_o = jnp.tile(qid, (1, r0 // LANES)) == segr_ref[0:1, 0:r0]
            pieces.append(jnp.where(vis_o, s_o, MASK_VALUE))
        m = jnp.max(_lane_fold(jnp.maximum, pieces), axis=-1, keepdims=True)
        return pieces, m

    def vf(hh, a, b):
        return v_ref[hh, a:b, :].astype(f32)

    def finish(hh, i, pieces, m):
        r0 = i * bs
        ps = [jnp.exp(s - m) for s in pieces]
        l = jnp.sum(_lane_fold(jnp.add, ps), axis=-1, keepdims=True)
        o = lax.dot_general(ps[0], vf(hh, r0, r0 + bs), NN, preferred_element_type=f32)
        if i:
            o = o + lax.dot_general(ps[1], vf(hh, 0, r0), NN, preferred_element_type=f32)
        o_ref[hh, r0:r0 + bs, :] = (o * (1.0 / l)).astype(o_ref.dtype)
        if lse_ref:
            lse = jnp.broadcast_to(jnp.log(l) + m, (bs, LANES))         # [bs, 128]
            lse_ref[0][hh, :, r0:r0 + bs] = lse.T[0:SUBLANES, :]        # row layout [8, bs]

    work = [(hh, i) for hh in range(hp) for i in range(n)]
    if FWD_ORDER == "seq":
        for hh, i in work:
            finish(hh, i, *scores(hh, i))
    elif FWD_ORDER == "allqk":
        got = [scores(hh, i) for hh, i in work]
        for (hh, i), g in zip(work, got):
            finish(hh, i, *g)
    elif FWD_ORDER == "heads":
        # the scores of head h+1 (stores) between the softmax and products of head h (vector units)
        got = {(0, i): scores(0, i) for i in range(n)}
        for hh in range(hp):
            for i in range(n):
                if hh + 1 < hp:
                    got[hh + 1, i] = scores(hh + 1, i)
                finish(hh, i, *got.pop((hh, i)))
    elif FWD_ORDER == "pipe":                                           # one strip ahead
        nxt = scores(*work[0])
        for w, (hh, i) in enumerate(work):
            cur = nxt
            if w + 1 < len(work):
                nxt = scores(*work[w + 1])
            finish(hh, i, *cur)
    else:
        raise ValueError(f"FWD_ORDER={FWD_ORDER!r}")


def _forward(q, k, v, segc, segr, *, bs: int, hp: int, save_lse: bool, interpret: bool):
    """q, k, v: [B, H, T, hd]; segc: int32 [B, T, 128]; segr: int32 [B, 8, T].
    Returns o [B, H, T, hd] (dtype of q) and, with save_lse, lse float32 [B, H, 8, T]."""
    pl, pltpu = _pallas()
    b, h, t, hd = q.shape
    slab = pl.BlockSpec((None, hp, t, hd), lambda i, j: (i, j, 0, 0))
    in_specs = [pl.BlockSpec((None, t, LANES), lambda i, j: (i, 0, 0)),
                pl.BlockSpec((None, SUBLANES, t), lambda i, j: (i, 0, 0)),
                slab, slab, slab]
    out_shape = [jax.ShapeDtypeStruct((b, h, t, hd), q.dtype)]
    out_specs = [slab]
    if save_lse:
        out_shape.append(jax.ShapeDtypeStruct((b, h, SUBLANES, t), jnp.float32))
        out_specs.append(pl.BlockSpec((None, hp, SUBLANES, t), lambda i, j: (i, j, 0, 0)))
    outs = pl.pallas_call(
        functools.partial(_fwd_kernel, t=t, bs=bs, hp=hp),
        grid=(b, h // hp), in_specs=in_specs, out_specs=out_specs, out_shape=out_shape,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        interpret=interpret, name=f"onetile_fwd{'_lse' if save_lse else ''}_t{t}_bs{bs}_hp{hp}",
    )(segc, segr, q, k, v)
    return (outs[0], outs[1]) if save_lse else outs[0]


# ---------------------------------------------------------------------------- backward kernel
def _bwd_kernel(segc_ref, segr_ref, q_ref, k_ref, v_ref, do_ref, lse_ref, di_ref,
                dq_ref, dk_ref, dv_ref, dq_acc, *, t: int, bs: int, hp: int):
    f32 = jnp.float32
    n = t // bs
    # diagonal block in the key-major layout: key row r, query column c, visible when r <= c
    causal = lax.broadcasted_iota(jnp.int32, (bs, bs), 0) <= lax.broadcasted_iota(jnp.int32, (bs, bs), 1)
    dq_acc[...] = jnp.zeros_like(dq_acc)
    # pieces in order: strip j (keys [j bs, (j+1) bs)) x its diagonal block, then the queries after it
    pieces = []
    for hh in range(hp):
        for j in range(n):
            c0 = j * bs
            pieces.append((hh, j, c0, c0 + bs, True))
            if c0 + bs < t:
                pieces.append((hh, j, c0 + bs, t, False))
    if BWD_ORDER == "ipipe":                                            # heads alternate piece by piece
        pieces.sort(key=lambda pc: (pc[1], pc[2], pc[0]))

    def products(piece):
        """Score and output-gradient products of a piece (the matrix units only)."""
        hh, j, q0, q1, _ = piece
        c0 = j * bs
        s_t = lax.dot_general(k_ref[hh, c0:c0 + bs, :], q_ref[hh, q0:q1, :], NT, preferred_element_type=f32)
        dp_t = lax.dot_general(v_ref[hh, c0:c0 + bs, :], do_ref[hh, q0:q1, :], NT, preferred_element_type=f32)
        return s_t, dp_t                                                # [bs, w] each

    grads = {}

    def finish(piece, s_t, dp_t):
        hh, j, q0, q1, diag = piece
        c0 = j * bs
        w = q1 - q0
        k = k_ref[hh, c0:c0 + bs, :]
        q = q_ref[hh, q0:q1, :]
        do = do_ref[hh, q0:q1, :]
        vis = jnp.tile(segc_ref[c0:c0 + bs, :], (1, w // LANES)) == segr_ref[0:1, q0:q1]
        if diag:
            vis = jnp.logical_and(vis, causal)
        s_t = jnp.where(vis, s_t, MASK_VALUE)
        p_t = jnp.exp(s_t - lse_ref[hh, 0:1, q0:q1])
        dv_p = lax.dot_general(p_t.astype(do.dtype), do, NN, preferred_element_type=f32)
        ds_t = (dp_t - di_ref[hh, 0:1, q0:q1]) * p_t
        ds_lo = ds_t.astype(do.dtype)
        dk_p = lax.dot_general(ds_lo, q, NN, preferred_element_type=f32)
        ds_tt = ds_lo.T if DS_CAST_FIRST else ds_t.T.astype(k.dtype)
        dq_acc[hh, q0:q1, :] += lax.dot_general(ds_tt, k, NN, preferred_element_type=f32)
        if (hh, j) in grads:
            dk_o, dv_o = grads[hh, j]
            dk_p, dv_p = dk_o + dk_p, dv_o + dv_p
        grads[hh, j] = (dk_p, dv_p)
        if q1 == t:                                                    # last piece of strip j
            dk_ref[hh, c0:c0 + bs, :] = dk_p.astype(dk_ref.dtype)
            dv_ref[hh, c0:c0 + bs, :] = dv_p.astype(dv_ref.dtype)

    if BWD_ORDER not in ("seq", "pipe", "ipipe"):
        raise ValueError(f"BWD_ORDER={BWD_ORDER!r}")
    if BWD_ORDER == "seq":
        for pc in pieces:
            finish(pc, *products(pc))
    else:                                                               # "pipe": one piece ahead
        nxt = products(pieces[0])
        for i, pc in enumerate(pieces):
            cur = nxt
            if i + 1 < len(pieces):
                nxt = products(pieces[i + 1])
            finish(pc, *cur)
    dq_ref[...] = dq_acc[...].astype(dq_ref.dtype)


def _backward(q, k, v, segc, segr, lse, di, do, *, bs: int, hp: int, interpret: bool):
    pl, pltpu = _pallas()
    b, h, t, hd = q.shape
    slab = pl.BlockSpec((None, hp, t, hd), lambda i, j: (i, j, 0, 0))
    row = pl.BlockSpec((None, hp, SUBLANES, t), lambda i, j: (i, j, 0, 0))
    in_specs = [pl.BlockSpec((None, t, LANES), lambda i, j: (i, 0, 0)),
                pl.BlockSpec((None, SUBLANES, t), lambda i, j: (i, 0, 0)),
                slab, slab, slab, slab, row, row]
    out_shape = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in (q, k, v)]
    return pl.pallas_call(
        functools.partial(_bwd_kernel, t=t, bs=bs, hp=hp),
        grid=(b, h // hp), in_specs=in_specs, out_specs=[slab, slab, slab], out_shape=out_shape,
        scratch_shapes=[pltpu.VMEM((hp, t, hd), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        interpret=interpret, name=f"onetile_bwd_t{t}_bs{bs}_hp{hp}",
    )(segc, segr, q, k, v, do, lse, di)


# ------------------------------------------------------------------------------- custom VJP
def _seg_layouts(seg):
    """int32 [B, T] -> ([B, T, 128] lane-broadcast, [B, 8, T] sublane-broadcast)."""
    b, t = seg.shape
    return (lax.broadcast_in_dim(seg, (b, t, LANES), (0, 1)),
            lax.broadcast_in_dim(seg, (b, SUBLANES, t), (0, 2)))


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7))
def _attend(q, k, v, seg, bs, hp, interpret, residual_name):
    segc, segr = _seg_layouts(seg)
    return _forward(q, k, v, segc, segr, bs=bs, hp=hp, save_lse=False, interpret=interpret)


def _attend_fwd(q, k, v, seg, bs, hp, interpret, residual_name):
    segc, segr = _seg_layouts(seg)
    o, lse = _forward(q, k, v, segc, segr, bs=bs, hp=hp, save_lse=True, interpret=interpret)
    if residual_name:
        from jax import ad_checkpoint
        o = ad_checkpoint.checkpoint_name(o, residual_name)
        lse = ad_checkpoint.checkpoint_name(lse, residual_name)
    return o, (q, k, v, seg, o, lse)


def _attend_bwd(bs, hp, interpret, residual_name, res, do):
    q, k, v, seg, o, lse = res
    segc, segr = _seg_layouts(seg)
    # the splash backward's di (an XLA reduction outside the kernel), as a row per (b, h)
    di = jnp.einsum("bhsd,bhsd->bhs", o.astype(jnp.float32), do.astype(jnp.float32))
    b, h, t = di.shape
    di = lax.broadcast_in_dim(di, (b, h, SUBLANES, t), (0, 1, 3))
    dq, dk, dv = _backward(q, k, v, segc, segr, lse, di, do.astype(q.dtype), bs=bs, hp=hp, interpret=interpret)
    return dq, dk, dv, None


_attend.defvjp(_attend_fwd, _attend_bwd)


def attend_major(q: jax.Array, k: jax.Array, v: jax.Array, seg: jax.Array, *, bs: int = 256,
                 interpret: bool = False, residual_name: str | None = None,
                 heads_per_step: int | None = None) -> jax.Array:
    """Causal same-segment attention on head-major operands ``q, k, v: [B, H, T, hd]`` (``T`` a
    multiple of 128) with int ``seg [B, T]``; returns ``[B, H, T, hd]`` in the dtype of ``q``.
    ``bs``: strip height (clipped to a multiple of 128 dividing ``T``); ``heads_per_step``: heads
    per grid step (:func:`heads_per_grid_step`); ``residual_name``: name the forward residuals (output
    and log-sum-exp) for a ``jax.checkpoint`` policy."""
    _, h, t, _ = q.shape
    if t % LANES:
        raise ValueError(f"the one-tile kernels need T % {LANES} == 0, got {t}")
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError(f"q, k, v must share a dtype, got {q.dtype}, {k.dtype}, {v.dtype}")
    return _attend(q, k, v, jnp.asarray(seg, jnp.int32), strip_rows(t, bs),
                   heads_per_grid_step(h, t, heads_per_step), bool(interpret), residual_name)


# ----------------------------------------------------------------------- XLA, slice by slice
def _xla_scores(q1, k1, s1):
    t = q1.shape[0]
    s = lax.dot_general(q1, k1, NT, preferred_element_type=jnp.float32)
    vis = (jnp.arange(t)[None, :] <= jnp.arange(t)[:, None]) & (s1[:, None] == s1[None, :])
    return jnp.where(vis, s, MASK_VALUE)


def _xla_fwd_slice(args):
    q1, k1, v1, s1 = args                                                # [T, hd] x3, [T]
    s = _xla_scores(q1, k1, s1)
    m = jnp.max(s, axis=-1, keepdims=True)
    p = jnp.exp(s - m)
    l = jnp.sum(p, axis=-1, keepdims=True)
    o = lax.dot_general(p, v1.astype(jnp.float32), NN, precision=lax.Precision.HIGHEST,
                        preferred_element_type=jnp.float32)
    return (o * (1.0 / l)).astype(q1.dtype), (jnp.log(l) + m)[:, 0]


def _xla_bwd_slice(args):
    q1, k1, v1, s1, lse, di, do = args
    p = jnp.exp(_xla_scores(q1, k1, s1) - lse[:, None])
    dv = lax.dot_general(p.astype(do.dtype), do, TN, preferred_element_type=jnp.float32)
    dp = lax.dot_general(do, v1, NT, preferred_element_type=jnp.float32)
    ds = ((dp - di[:, None]) * p).astype(do.dtype)
    dk = lax.dot_general(ds, q1, TN, preferred_element_type=jnp.float32)
    dq = lax.dot_general(ds, k1, NN, preferred_element_type=jnp.float32)
    return dq.astype(q1.dtype), dk.astype(k1.dtype), dv.astype(v1.dtype)


def _flat_slices(q, k, v, seg):
    b, h, t, hd = q.shape
    flat = lambda a: a.reshape(b * h, *a.shape[2:])                    # noqa: E731
    segs = jnp.broadcast_to(jnp.asarray(seg, jnp.int32)[:, None, :], (b, h, t)).reshape(b * h, t)
    return flat(q), flat(k), flat(v), segs


@jax.custom_vjp
def attend_xla_slices(q: jax.Array, k: jax.Array, v: jax.Array, seg: jax.Array) -> jax.Array:
    """The same attention in plain XLA, one (row, head) slice at a time (``lax.map``, so that a
    score matrix is ``[T, T]``, never ``[B, H, T, T]``), with the splash numerics: float32 logits
    from the operand dtype, the splash mask value, ``p @ v`` with ``p`` and ``v`` in float32 at
    ``HIGHEST`` precision (a float32 product as inside the kernel; the default precision of a
    float32 product on the TPU is one bfloat16 pass), the output rounded to the dtype of ``q``;
    the backward of the splash kernels (``p`` and ``ds`` rounded to the operand dtype before
    their products, float32 accumulation), slice by slice again."""
    b, h, t, hd = q.shape
    out, _ = lax.map(_xla_fwd_slice, _flat_slices(q, k, v, seg))
    return out.reshape(b, h, t, hd)


def _xla_fwd(q, k, v, seg):
    b, h, t, hd = q.shape
    out, lse = lax.map(_xla_fwd_slice, _flat_slices(q, k, v, seg))
    out = out.reshape(b, h, t, hd)
    return out, (q, k, v, seg, out, lse)


def _xla_bwd(res, do):
    q, k, v, seg, o, lse = res
    b, h, t, hd = q.shape
    di = jnp.einsum("bhsd,bhsd->bhs", o.astype(jnp.float32), do.astype(jnp.float32)).reshape(b * h, t)
    fq, fk, fv, fs = _flat_slices(q, k, v, seg)
    dq, dk, dv = lax.map(_xla_bwd_slice, (fq, fk, fv, fs, lse, di,
                                          do.astype(q.dtype).reshape(b * h, t, hd)))
    shape = (b, h, t, hd)
    return dq.reshape(shape), dk.reshape(shape), dv.reshape(shape), None


attend_xla_slices.defvjp(_xla_fwd, _xla_bwd)


__all__ = ["attend_major", "attend_xla_slices", "strip_rows", "MASK_VALUE"]
