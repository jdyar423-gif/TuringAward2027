"""``attention_splash.FAST_SPLASH_IO``: the splash kernels with their accumulators in VMEM scratch
and the log-sum-exp passed as ``[H, 8, T]`` compute the same values as the library's set-up, bit
for bit (the kernel bodies are the library's; only what is an output differs). Interpret mode on
CPU, as the other splash tests; the switch is flipped with ``monkeypatch`` and every function is
traced afresh for each value (the switch is read at trace time)."""
from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _model_testlib import make_batch, random_params, tiny_cfg
from bes.model import attention_splash as S
from bes.model import model as M

pytestmark = pytest.mark.skipif(not S.available(), reason="splash attention not installed")


def _both(monkeypatch, fn):
    """``fn()`` with the switch off and on (each call traces anew)."""
    out = {}
    for flag in (False, True):
        monkeypatch.setattr(S, "FAST_SPLASH_IO", flag)
        out[flag] = jax.tree.map(np.asarray, fn())
    return out[False], out[True]


# (length, (block_q, block_kv, block_kv_compute), fused backward, dtype): no dq accumulator in the
# fused backward kernel (block_kv == block_kv_compute) and with one (the shipped 1024/1024/512
# case), a padded length, bfloat16, separate dq/dkv kernels (the library's backward on the [H, T]
# log-sum-exp), several query tiles.
KERNEL_CASES = [
    (128, (128, 128, 128), True, "float32"),
    (256, (256, 256, 128), True, "float32"),
    (256, (256, 256, 128), True, "bfloat16"),
    (200, (128, 128, 128), True, "bfloat16"),
    (256, (128, 256, 128), False, "float32"),
    (384, (128, 128, 128), True, "bfloat16"),
]


@pytest.mark.parametrize("t,tiles,fused,dtype", KERNEL_CASES)
def test_kernel_output_and_gradients_are_bitwise_equal(monkeypatch, t, tiles, fused, dtype):
    cfg = dataclasses.replace(tiny_cfg(attn_impl="splash"), splash_block_q=tiles[0], splash_block_kv=tiles[1],
                              splash_block_kv_compute=tiles[2], splash_fused_bwd=fused)
    rng = np.random.default_rng(t)
    b, h, hd = 2, 2, 128
    q, k, v = (jnp.asarray(rng.standard_normal((b, h, t, hd)), dtype) for _ in range(3))
    seg = np.zeros((b, t), np.int32)
    seg[1, t // 3:] = 1                                  # two packed segments in the second row
    seg = jnp.asarray(seg)
    r = jnp.asarray(rng.standard_normal((b, h, t, hd)), jnp.float32)

    def run():
        f = lambda q, k, v: jnp.sum(r * S.attend_major(q, k, v, seg, cfg).astype(jnp.float32))
        return (S.attend_major(q, k, v, seg, cfg),) + jax.grad(f, argnums=(0, 1, 2))(q, k, v)

    ref, new = _both(monkeypatch, run)
    assert new[0].shape == (b, h, t, hd) and new[0].dtype == jnp.dtype(dtype)
    for name, a, c in zip(("y", "dq", "dk", "dv"), ref, new):
        np.testing.assert_array_equal(c, a, err_msg=name)
    assert np.any(np.asarray(ref[1], np.float32) != 0)


def test_the_kernels_write_only_their_results(monkeypatch):
    """The program of the switch, in a gradient (as in training): the forward kernel has two
    outputs (``o`` and ``float32[H, 8, T]``) instead of five (the m, l, o accumulators, ``o`` and
    ``float32[H, T, 128]``), the backward kernel three (dq, dk, dv) instead of five."""
    cfg = tiny_cfg(attn_impl="splash")
    q = k = v = jax.ShapeDtypeStruct((2, 2, 128, 128), jnp.bfloat16)
    seg = jnp.zeros((2, 128), jnp.int32)
    shapes = {}
    for flag in (False, True):
        monkeypatch.setattr(S, "FAST_SPLASH_IO", flag)
        f = lambda q, k, v: jnp.sum(S.attend_major(q, k, v, seg, cfg).astype(jnp.float32))
        jaxpr = jax.make_jaxpr(jax.grad(f, argnums=(0, 1, 2)))(q, k, v)
        calls = [e for e in _eqns(jaxpr.jaxpr) if e.primitive.name == "pallas_call"]
        assert len(calls) == 2, flag                    # forward and fused backward, vmapped
        shapes[flag] = [[(tuple(o.aval.shape), str(o.aval.dtype)) for o in e.outvars] for e in calls]
    acc, act, f32 = ((2, 128, 128), "float32"), ((2, 2, 128, 128), "bfloat16"), ((2, 2, 128, 128), "float32")
    assert shapes[False] == [[acc, acc, acc, act, f32], [acc, acc, ((2, 1, 2, 128, 128), "bfloat16"), act, act]]
    assert shapes[True] == [[act, ((2, 2, 8, 128), "float32")], [((2, 1, 2, 128, 128), "bfloat16"), act, act]]


def _eqns(jaxpr):
    """Every equation of a jaxpr and of the jaxprs in its equations' parameters."""
    for e in jaxpr.eqns:
        yield e
        for p in e.params.values():
            for sub in (p if isinstance(p, (list, tuple)) else (p,)):
                inner = getattr(sub, "jaxpr", sub)      # ClosedJaxpr -> Jaxpr
                if hasattr(inner, "eqns"):
                    yield from _eqns(inner)


def test_keeping_the_kernel_outputs_with_the_switch_changes_no_value(monkeypatch):
    """``model.remat_keep="kernel"`` keeps the outputs of the Pallas kernels across the
    rematerialised loop step; with the switch those are ``o`` and the compact log-sum-exp only
    (28 MB per block application at 16 x 1024 instead of 100 MB). Same loss and gradients as the
    live run's program (switch off, full recomputation), bit for bit, and the backward pass runs
    no forward kernel again."""
    base = tiny_cfg(n_loops=3, attn_impl="splash", act_dtype="bfloat16", remat=True)
    params = random_params(base, seed=21)
    consts = M.make_consts(base)
    batch = make_batch(base, B=2, T=40, seed=22, seg_starts=((0,), (0, 17)))
    out, kernels = {}, {}
    for flag, keep in ((False, ""), (True, "kernel"), (True, "")):
        monkeypatch.setattr(S, "FAST_SPLASH_IO", flag)
        cfg = dataclasses.replace(base, remat_keep=keep)
        fn = jax.value_and_grad(lambda p, c=cfg: M.loss_fn(p, consts, batch, c)[0])
        out[flag, keep] = jax.tree.map(np.asarray, jax.jit(fn)(params))
        kernels[flag, keep] = sum(e.primitive.name == "pallas_call" for e in _eqns(jax.make_jaxpr(fn)(params).jaxpr))
    for key in ((True, "kernel"), (True, "")):
        np.testing.assert_array_equal(out[key][0], out[False, ""][0])
        for a, c in zip(jax.tree.leaves(out[False, ""][1]), jax.tree.leaves(out[key][1])):
            np.testing.assert_array_equal(c, a)
    assert kernels[True, "kernel"] < kernels[True, ""], kernels


@pytest.mark.parametrize("act", ["bfloat16", "float32"])
def test_model_loss_and_every_gradient_are_bitwise_equal(monkeypatch, act):
    """The live run's code path on a small model: splash attention, rematerialised loops, sink,
    tape, registers and value embeddings on, packed segments, a length that is padded to 128."""
    cfg = tiny_cfg(n_loops=2, attn_impl="splash", act_dtype=act, remat=True, table_grad="take")
    params = random_params(cfg, seed=31)
    consts = M.make_consts(cfg)
    batch = make_batch(cfg, B=2, T=40, seed=32, seg_starts=((0,), (0, 17)))

    def run():
        fn = jax.jit(jax.value_and_grad(functools.partial(M.loss_fn, cfg=cfg), has_aux=True))
        (loss, metrics), grads = fn(params, consts, batch)
        return loss, metrics["sink_mass"], grads

    (l0, m0, g0), (l1, m1, g1) = _both(monkeypatch, run)
    assert np.isfinite(l0)
    np.testing.assert_array_equal(l1, l0)
    np.testing.assert_array_equal(m1, m0)
    leaves0, tree0 = jax.tree_util.tree_flatten_with_path(g0)
    leaves1, tree1 = jax.tree_util.tree_flatten_with_path(g1)
    assert tree0 == tree1
    for (path, a), (_, c) in zip(leaves0, leaves1):
        np.testing.assert_array_equal(c, a, err_msg=jax.tree_util.keystr(path))
