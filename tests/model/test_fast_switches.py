"""The speed switches together: ``attention_splash.FAST_SPLASH_IO``, ``layers.FAST_RELU2`` and
``head.FAST_CE`` all on at once give the loss and the gradients of all of them off.

``value_and_grad(loss_fn)`` (loss, every metric, every gradient leaf) on a tiny configuration with
the live run's features (``configs/final.py``: splash attention, here in interpret mode, bfloat16
activations, rematerialised loops, sink, candidate tape in and out, registers, value embeddings,
clocks and confidence off; ``table_grad="take"``), with splash tiles that exercise what the
shipped 1024/1024/512 tiles do at T=1024: a fused backward kernel with its dq accumulator
(``block_kv != block_kv_compute``), several query blocks, a padded length (200 -> 256) and
packed segments. What they do at T=2048 (two key blocks, the upper one causally skipped for the
first query block, with the dq accumulator) is the kernel test at the end, bit for bit.

Two of the switches are exact and one is not, so the comparison is split:

* the exact ones (FAST_SPLASH_IO, FAST_RELU2) on against all off: bit for bit;
* all on against FAST_CE alone: bit for bit (the exact switches change nothing whatever the head);
* all on against all off: FAST_CE's float32 rounding of the softmax (the log-sum-exp without the
  row maximum), which bfloat16 roundings of the cotangents can carry through the backward pass:
  the loss and the metrics to ``1e-6`` relative; every gradient leaf to one bfloat16 rounding
  step (``2^-7`` of its largest entry) in bfloat16, the unembedding to one step per loss chunk
  (with several chunks the reference rounds its running sum over the chunks to bfloat16 after
  every chunk, the fast path once), and ``5e-6`` in float32. ``tests/model/test_io_speed.py``
  has the head alone at tighter tolerances.

Measured (CPU, jax 0.11.1, about 55 s per case): the exact switches and "all against FAST_CE
alone" bit for bit in every case; all against all off: bfloat16 with one chunk, the loss and all
59 gradient leaves bit for bit (metrics 1.4e-7); with two chunks the unembedding 2.4e-3 of its
largest entry, the other 58 leaves bit for bit; float32, the loss bit for bit, the worst leaf
6.2e-7 (``emb/tok``).
"""
from __future__ import annotations

import contextlib
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _model_testlib import make_batch, random_params, tiny_cfg
from bes.model import attention_splash, head, layers, model as M

pytestmark = pytest.mark.skipif(not attention_splash.available(), reason="splash attention not installed")

#: every switch and its module
SWITCHES = {"FAST_SPLASH_IO": attention_splash, "FAST_RELU2": layers, "FAST_CE": head}
EXACT = ("FAST_SPLASH_IO", "FAST_RELU2")
#: the live run's model (configs/final.py with table_grad=take) at a tiny size
LIVE = dict(attn_impl="splash", act_dtype="bfloat16", remat=True, clock_planes=0, use_conf=False,
            table_grad="take")
#: tiles 128 / 256 / 128 at a padded length of 256: two query blocks and a key block of two
#: compute steps (the fused backward kernel's dq accumulator); the kernel test below adds
#: causally skipped blocks to it
TILES = dict(max_seq=256, splash_block_q=128, splash_block_kv=256, splash_block_kv_compute=128,
             splash_fused_bwd=True)
BF16_STEP = 2.0 ** -7          # one bfloat16 rounding step relative to the entry (upper bound)


@contextlib.contextmanager
def switches(on):
    """The switches ``on`` on, the others off; the attention kernel switch of later commits
    (``attention_splash.FAST_KERNEL``), where it exists, at ``""``: the configured kernels."""
    old = {name: getattr(mod, name) for name, mod in SWITCHES.items()}
    kernel = getattr(attention_splash, "FAST_KERNEL", None)
    try:
        for name, mod in SWITCHES.items():
            setattr(mod, name, name in on)
        if kernel is not None:
            attention_splash.FAST_KERNEL = ""
        yield
    finally:
        for name, mod in SWITCHES.items():
            setattr(mod, name, old[name])
        if kernel is not None:
            attention_splash.FAST_KERNEL = kernel


def _run(cfg, params, consts, batch, on):
    """``value_and_grad(loss_fn)`` traced afresh with the switches ``on`` (read at trace time),
    and the printed program."""
    with switches(on):
        vg = jax.value_and_grad(lambda p, c, b: M.loss_fn(p, c, b, cfg), has_aux=True)
        text = str(jax.make_jaxpr(vg)(params, consts, batch))
        out = jax.tree.map(np.asarray, jax.jit(vg)(params, consts, batch))
    return out, text


def _leaves(out):
    (_, _), grads = out
    return jax.tree_util.tree_flatten_with_path(grads)[0]


def _rel(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-30))


def _assert_bitwise(got, want, what):
    (lg, mg), _ = got
    (lw, mw), _ = want
    np.testing.assert_array_equal(lg, lw, err_msg=f"{what}: loss")
    for k in mw:
        np.testing.assert_array_equal(mg[k], mw[k], err_msg=f"{what}: metric {k}")
    fg, fw = _leaves(got), _leaves(want)
    assert [p for p, _ in fg] == [p for p, _ in fw]
    for (path, a), (_, b) in zip(fg, fw):
        assert a.dtype == b.dtype and a.shape == b.shape
        np.testing.assert_array_equal(a, b, err_msg=f"{what}: gradient {jax.tree_util.keystr(path)}")


def _assert_rounding(got, want, grad_tol, what, overrides=None):
    """Loss and metrics to 1e-6 relative; every gradient leaf to ``grad_tol`` of its largest
    entry (``overrides``: ``{leaf path: tolerance}``)."""
    (lg, mg), _ = got
    (lw, mw), _ = want
    assert np.isfinite(float(lw))
    assert abs(float(lg) - float(lw)) <= 1e-6 * abs(float(lw)), (what, float(lg), float(lw))
    for k in mw:
        np.testing.assert_allclose(mg[k], mw[k], rtol=1e-6, atol=1e-7, err_msg=f"{what}: metric {k}")
    fg, fw = _leaves(got), _leaves(want)
    assert [p for p, _ in fg] == [p for p, _ in fw]
    for (path, a), (_, b) in zip(fg, fw):
        name = jax.tree_util.keystr(path)
        assert a.dtype == b.dtype and a.shape == b.shape
        assert np.all(np.isfinite(a)), (what, name)
        assert _rel(a, b) <= (overrides or {}).get(name, grad_tol), (what, name, _rel(a, b))


def _case(act: str, loss_chunk: int):
    cfg = tiny_cfg(n_loops=3, **dict(LIVE, act_dtype=act), **TILES, loss_chunk=loss_chunk)
    params = random_params(cfg, seed=41)
    consts = M.make_consts(cfg)
    batch = make_batch(cfg, B=2, T=200, seed=42, seg_starts=((0,), (0, 77)))
    return cfg, params, consts, batch


@pytest.mark.slow
@pytest.mark.parametrize("act,loss_chunk", [("bfloat16", 512), ("bfloat16", 256), ("float32", 512)])
def test_all_switches_on_give_the_loss_and_gradients_of_all_off(act, loss_chunk):
    """400 rows: one loss chunk (512, as every training piece of the live run) or two (256, as an
    unsplit 16 x 1024 micro-batch at the default ``loss_chunk`` of 8,192)."""
    cfg, params, consts, batch = _case(act, loss_chunk)
    runs = {key: _run(cfg, params, consts, batch, on) for key, on in
            (("off", ()), ("exact", EXACT), ("ce", ("FAST_CE",)), ("all", tuple(SWITCHES)))}
    texts = {key: text for key, (_, text) in runs.items()}
    out = {key: o for key, (o, _) in runs.items()}
    # every switch reached the traced program: the kernels of FAST_SPLASH_IO, the custom VJP of
    # FAST_RELU2 (inside the rematerialised loop step), the row maximum FAST_CE drops
    for key in ("exact", "all"):
        assert "splash_mha_fwd_io" in texts[key] and "bwd=_relu2_bwd" in texts[key], key
    for key in ("off", "ce"):
        assert "splash_mha_fwd_io" not in texts[key] and "bwd=_relu2_bwd" not in texts[key], key
    assert texts["ce"].count("reduce_max") < texts["off"].count("reduce_max")
    assert texts["all"].count("reduce_max") < texts["exact"].count("reduce_max")
    _assert_bitwise(out["exact"], out["off"], f"{act}/{loss_chunk}: exact switches vs off")
    _assert_bitwise(out["all"], out["ce"], f"{act}/{loss_chunk}: all vs FAST_CE alone")
    n_chunks = -(-400 // loss_chunk)
    if act == "bfloat16":
        tol, out_tol = BF16_STEP, n_chunks * BF16_STEP
    else:
        tol = out_tol = 5e-6
    _assert_rounding(out["all"], out["off"], tol, f"{act}/{loss_chunk}: all vs off",
                     overrides={"['head']['out']": out_tol})


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_kernels_with_the_dq_accumulator_and_skipped_blocks_are_bitwise_equal(dtype):
    """``FAST_SPLASH_IO`` on the case of the shipped tiles at T=2048 (1024 / 1024 / 512: two
    query and two key blocks, the second key block skipped for the first query block, the fused
    backward kernel's dq accumulator) at a quarter of the size: tiles 128 / 256 / 128 at a padded
    length of 512 (400 -> 512, four query blocks, two key blocks), two packed segments. Output,
    dq, dk and dv bit for bit (``tests/model/test_splash_io.py`` covers these two cases apart)."""
    cfg = dataclasses.replace(tiny_cfg(attn_impl="splash"), **TILES)
    rng = np.random.default_rng(7)
    b, h, t, hd = 2, 2, 400, 128
    q, k, v = (jnp.asarray(rng.standard_normal((b, h, t, hd)), dtype) for _ in range(3))
    seg = np.zeros((b, t), np.int32)
    seg[1, 150:] = 1
    seg = jnp.asarray(seg)
    r = jnp.asarray(rng.standard_normal((b, h, t, hd)), jnp.float32)

    def run():
        def f(q, k, v):
            return jnp.sum(r * attention_splash.attend_major(q, k, v, seg, cfg).astype(jnp.float32))
        y = attention_splash.attend_major(q, k, v, seg, cfg)
        return jax.tree.map(np.asarray, (y,) + jax.grad(f, argnums=(0, 1, 2))(q, k, v))

    with switches(()):
        ref = run()
    with switches(("FAST_SPLASH_IO",)):
        text = str(jax.make_jaxpr(lambda q: attention_splash.attend_major(q, k, v, seg, cfg))(q))
        new = run()
    assert "splash_mha_fwd_io" in text
    bs = attention_splash.block_sizes(cfg, attention_splash.padded_length(t))
    assert (bs.block_q, bs.block_kv, bs.block_kv_compute) == (128, 256, 128) and bs.use_fused_bwd_kernel
    for name, a, c in zip(("y", "dq", "dk", "dv"), ref, new):
        np.testing.assert_array_equal(c, a, err_msg=name)
        assert np.any(np.asarray(a, np.float32) != 0), name
