"""The kernels behind ``attention_splash.FAST_KERNEL`` compute the values of the configured splash
kernels to float rounding: the one-tile kernels (``bes.model.attention_onetile``), the splash
kernels with other settings (``blocks:...``), the XLA formulation slice by slice (``xlamap``) and
the tokamax splash kernel (``tokamax:...``, when tokamax is installed); forward and gradients of
``q, k, v`` with packed segments, in float32 and with the bfloat16 operands of the model; the
whole model's loss and gradients; and the ``save`` flag keeps the forward residuals so that the
gradient program runs the forward kernel once instead of twice.

What "float rounding" means here. In float32 the kernels agree to 1e-5 of the output scale
(measured: 1e-7 to 8e-7). With bfloat16 operands the kernels round their output, ``p`` and
``ds`` to bfloat16 at the same points as the splash kernels; sums in another order then move a
value across a rounding boundary now and then (and a gradient element that is a cancelling sum
inherits the flip of one of its bfloat16 terms): the largest difference stays under one
bfloat16 unit at the scale of the array (2^-8 of its largest magnitude; measured 0.8-3.5e-4),
at most 1% of the elements differ at all (measured 0.01-0.05%), and the error against a float32
dense reference is that of the splash kernels (within 10% RMS). A change of the splash tiles
alone (``blocks:kvc=128``, the kind of change the run already took as "same values") is tested
alongside as the yardstick and shows the same differences.
"""
from __future__ import annotations

import dataclasses
import functools
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _model_testlib import make_batch, random_params, tiny_cfg
from bes.model import attention, attention_onetile as OT, attention_splash as S, model as M, sinks

pytestmark = pytest.mark.skipif(not S.available(), reason="splash attention not installed")

try:
    from bes.model import attention_tokamax as TK
    HAVE_TOKAMAX = TK.available()
except ImportError:  # pragma: no cover
    HAVE_TOKAMAX = False


class _Cfg:
    """The kernel settings of ``configs/final.py`` (tiles clipped to the length), interpret mode."""
    splash_block_q = splash_block_kv = 1024
    splash_block_kv_compute = 512
    splash_fused_bwd = True
    splash_interpret = "true"


@pytest.fixture
def switch():
    """Set ``FAST_KERNEL`` for one test and restore the default afterwards."""
    def set_(value: str):
        S.FAST_KERNEL = value
    yield set_
    S.FAST_KERNEL = ""


def _inputs(b: int, h: int, t: int, hd: int, seed: int, dtype):
    """Model-like head-major operands: unit-norm content dims with a per-head gain on the
    queries, the sink key at every segment start (bias dim carrying the sink logit), sink values
    and the confidence dim; three segments in odd rows."""
    rng = np.random.default_rng(seed)
    dc = hd - 2
    qc = rng.standard_normal((b, t, h, dc)).astype(np.float32)
    kc = rng.standard_normal((b, t, h, dc)).astype(np.float32)
    qc = qc / np.linalg.norm(qc, axis=-1, keepdims=True) * 4.0
    kc = kc / np.linalg.norm(kc, axis=-1, keepdims=True)
    v = rng.standard_normal((b, t, h, hd)).astype(np.float32)
    seg = np.zeros((b, t), np.int32)
    sink = np.zeros((b, t), np.float32)
    sink[:, 0] = 1.0
    for r in range(1, b, 2):
        for i, s in enumerate(sorted(rng.choice(np.arange(8, t - 8), size=2, replace=False))):
            seg[r, s:] = i + 1
            sink[r, s] = 1.0
    q, k, v = sinks.assemble(jnp.asarray(qc), jnp.asarray(kc), jnp.asarray(v), jnp.asarray(sink),
                             jnp.asarray(rng.standard_normal(h), jnp.float32),
                             jnp.asarray(rng.standard_normal((h, hd - 1)), jnp.float32), 1.0, 2, jnp.float32)
    q, k, v = (jnp.swapaxes(a, 1, 2).astype(dtype) for a in (q, k, v))           # [B, H, T, hd]
    r = jnp.asarray(rng.standard_normal((b, h, t, hd)), jnp.float32)
    return q, k, v, jnp.asarray(seg), r


def _run(value: str, q, k, v, seg, r):
    """Output and gradients of sum(r * y) of ``attend_major`` with the switch set to ``value``."""
    S.FAST_KERNEL = value
    try:
        fn = functools.partial(S.attend_major, seg=seg, cfg=_Cfg(), interpret=True)
        y = jax.jit(fn)(q, k, v)
        g = jax.jit(jax.grad(lambda a, b, c: jnp.sum(fn(a, b, c).astype(jnp.float32) * r), argnums=(0, 1, 2)))(q, k, v)
    finally:
        S.FAST_KERNEL = ""
    return [np.asarray(x, np.float32) for x in (y, *g)]


def _dense32(q, k, v, seg, r):
    """Float32 dense reference (operands upcast): output and gradients."""
    def fn(a, b, c):
        a, b, c = (jnp.swapaxes(x.astype(jnp.float32), 1, 2) for x in (a, b, c))
        pos = jnp.broadcast_to(jnp.arange(a.shape[1], dtype=jnp.int32), seg.shape)
        return jnp.swapaxes(attention.attend_xla(a, b, c, seg, seg, pos, pos)[0], 1, 2)
    y = fn(q, k, v)
    g = jax.grad(lambda a, b, c: jnp.sum(fn(a, b, c) * r), argnums=(0, 1, 2))(q.astype(jnp.float32),
                                                                             k.astype(jnp.float32),
                                                                             v.astype(jnp.float32))
    return [np.asarray(x, np.float32) for x in (y, *g)]


def assert_same_to_rounding(got, want, dense, dtype, what=""):
    """The criterion of the module docstring for one array."""
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    scale = float(np.max(np.abs(want)))
    d = np.abs(got - want)
    if dtype == jnp.float32:
        assert d.max() <= 1e-5 * scale, f"{what}: max diff {d.max():.3g} at scale {scale:.3g}"
        return
    assert d.max() <= 2.0 ** -8 * scale, f"{what}: max diff {d.max() / scale:.3g} of the scale"
    assert np.mean(d > 0) <= 0.01, f"{what}: {np.mean(d > 0):.2%} of the elements differ"
    err_got = np.sqrt(np.mean((got - dense) ** 2))
    err_want = np.sqrt(np.mean((want - dense) ** 2))
    assert err_got <= 1.1 * err_want + 1e-7 * scale, f"{what}: rms error {err_got:.3g} vs splash {err_want:.3g}"


CANDIDATES = ["onetile", "onetile:bs=128,hp=1", "onetile:bs=256,hp=1", "xlamap", "blocks:kvc=128",
              "blocks:sep", "onetile:save",
              pytest.param("tokamax", marks=pytest.mark.skipif(not HAVE_TOKAMAX, reason="tokamax not installed")),
              pytest.param("tokamax:diag=1,grid=4",
                           marks=pytest.mark.skipif(not HAVE_TOKAMAX, reason="tokamax not installed"))]


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("value", CANDIDATES)
def test_kernels_match_the_splash_kernels(value, dtype):
    q, k, v, seg, r = _inputs(2, 2, 512, 128, seed=3, dtype=dtype)
    want = _run("", q, k, v, seg, r)
    got = _run(value, q, k, v, seg, r)
    dense = _dense32(q, k, v, seg, r) if dtype == jnp.bfloat16 else None
    for name, a, b, c in zip(("out", "dq", "dk", "dv"), got, want, dense or [None] * 4):
        assert a.shape == b.shape and np.all(np.isfinite(a)), name
        assert_same_to_rounding(a, b, c, dtype, f"{value} {name}")


@pytest.mark.parametrize("t,bs", [(128, 256), (200, 256), (384, 128), (640, 256)])
def test_one_tile_kernels_against_float64_and_dense(t, bs):
    """Lengths that are one strip, padded (200 -> 256), three strips, and not a multiple of the
    strip (640 at 256 -> strips of 128); heads per step 1 and 2; float32 against the float64
    reference and the dense gradients."""
    q, k, v, seg, r = _inputs(2, 4, t, 128, seed=t, dtype=jnp.float32)
    for hp in (1, 2):
        got = _run(f"onetile:bs={bs},hp={hp}", q, k, v, seg, r)
        ref = S.reference(*(jnp.swapaxes(a, 1, 2) for a in (q, k, v)), seg)
        np.testing.assert_allclose(got[0], np.swapaxes(ref, 1, 2), atol=2e-5)
        dense = _dense32(q, k, v, seg, r)
        for a, b in zip(got[1:], dense[1:]):
            np.testing.assert_allclose(a, b, atol=2e-4, rtol=1e-4)
    assert OT.strip_rows(640, 256) == 128 and OT.strip_rows(1024, 256) == 256 and OT.strip_rows(128, 256) == 128
    assert OT.heads_per_grid_step(6, 1024) == 2 and OT.heads_per_grid_step(6, 2048) == 1
    assert OT.heads_per_grid_step(6, 1024, 4) == 3


def test_the_default_switch_is_the_configured_splash_path():
    """``FAST_KERNEL = ""``: bit for bit the vmapped splash kernel with the configured tiles (the
    code path before the switch existed)."""
    q, k, v, seg, _ = _inputs(2, 2, 256, 128, seed=5, dtype=jnp.bfloat16)
    sa = S._sa()
    kernel = S.make_kernel(256, 2, S.block_sizes(_Cfg(), 256), True)
    old = jax.vmap(lambda a, b, c, s: kernel(a, b, c, segment_ids=s))(q, k, v, sa.SegmentIds(q=seg, kv=seg))
    assert S.FAST_KERNEL == ""
    np.testing.assert_array_equal(np.asarray(S.attend_major(q, k, v, seg, _Cfg(), interpret=True), np.float32),
                                  np.asarray(old, np.float32))


def test_switch_values_and_the_remat_policy(switch):
    assert S.parse_switch("") == ("", {}, frozenset())
    assert S.parse_switch("onetile:bs=128,hp=1,save") == ("onetile", {"bs": "128", "hp": "1"}, frozenset({"save"}))
    assert S.parse_switch("blocks:kvc=1024,sep") == ("blocks", {"kvc": "1024"}, frozenset({"sep"}))
    with pytest.raises(ValueError, match="unknown kernel"):
        S.parse_switch("flash")
    assert S.remat_policy() is None                     # the default: recompute everything
    switch("onetile:save")
    assert S.remat_policy() is not None
    bs = S.switched_block_sizes(_Cfg(), 1024, {"kvc": "1024"}, frozenset({"sep"}))
    assert (bs.block_q, bs.block_kv, bs.block_kv_compute, bs.use_fused_bwd_kernel) == (1024, 1024, 1024, False)
    assert (bs.block_q_dq, bs.block_kv_dq) == (1024, 1024)


def _model(value: str, cfg, params, consts, batch):
    S.FAST_KERNEL = value
    try:
        fn = jax.value_and_grad(lambda p: M.loss_fn(p, consts, batch, cfg)[0])
        n_kernels = str(jax.make_jaxpr(fn)(params)).count("pallas_call")
        loss, grads = jax.jit(fn)(params)
    finally:
        S.FAST_KERNEL = ""
    return float(loss), grads, n_kernels


@pytest.mark.slow
def test_model_loss_and_gradients_with_every_kernel():
    """The tiny model (remat on, two loops) with each kernel: the loss and the gradients of
    every parameter of the default path; ``save`` holds fewer kernels in the gradient program
    (the forward kernel is not run again in the backward pass of a loop step)."""
    cfg = tiny_cfg(n_loops=2, attn_impl="splash", remat=True)
    params = random_params(cfg, seed=31)
    consts = M.make_consts(cfg)
    batch = make_batch(cfg, B=2, T=40, seed=32, seg_starts=((0,), (0, 17)))
    l0, g0, k0 = _model("", cfg, params, consts, batch)
    for value in ("onetile", "onetile:save", "blocks:save", "xlamap"):
        loss, grads, kernels = _model(value, cfg, params, consts, batch)
        np.testing.assert_allclose(loss, l0, rtol=1e-6, err_msg=value)
        for a, b in zip(jax.tree.leaves(grads), jax.tree.leaves(g0)):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=2e-6, rtol=1e-4, err_msg=value)
        if value.endswith(":save"):
            base = _model(value.split(":")[0].replace("blocks", ""), cfg, params, consts, batch)[2]
            assert kernels < base, (value, kernels, base)


@pytest.mark.skipif(not HAVE_TOKAMAX, reason="tokamax not installed")
def test_tokamax_kernel_loads_without_the_tokamax_package():
    mods = TK.kernel_modules()
    assert hasattr(mods.splash_attention_kernel, "make_splash_mha_single_device")
    assert "tokamax" not in sys.modules                 # flax, qwix, xprof, ... were not imported
    with pytest.raises(ValueError):                     # the diagonal skip needs a pure causal mask
        mods.splash_attention_kernel.make_splash_mha_single_device(
            np.ones((256, 256), bool), config=mods.splash_attention_kernel.SplashConfig(
                block_q=256, block_kv=256, block_kv_compute=256, qk_diag_skip=True))
