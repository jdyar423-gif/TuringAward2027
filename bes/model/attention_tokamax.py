"""The splash kernel of tokamax 0.0.17 (``attention_splash.FAST_KERNEL = "tokamax[:...]"``).

tokamax (Apache-2.0, ``pip install tokamax``) is Google's Pallas kernel library; 0.0.17 is its
last release for jax 0.11 (0.0.18 needs jax >= 0.12). Its splash kernel differs from the one
shipped with jax in what matters at our shapes: the running maximum, sum and output are VMEM
scratch instead of kernel outputs, the causal mask is computed only on blocks that need it, and
``qk_diag_skip`` / ``sv_diag_skip`` skip the score and value products of the sub-tiles above the
diagonal of a square diagonal block (bit-exact: those entries are masked anyway; the element-wise
work still covers the whole tile).

The package's ``__init__`` imports its autotuner, benchmarks and quantisation layers (flax,
qwix, xprof, tensorboardX: 69 packages in all). The kernel itself needs none of them, so
:func:`kernel_modules` loads the four files of the kernel (``reference``,
``splash_attention_mask``, ``splash_attention_mask_info``, ``splash_attention_kernel``) from the
installed package under a private module name, with their package-absolute imports pointed at
each other: ``pip install --no-deps tokamax==0.0.17`` (one 2.7 MB pure-Python wheel) is all a
runtime needs.

Numerics: ``use_base2_exp=False`` (the default ``True`` multiplies ``q`` by ``log2 e`` inside the
kernel, a different rounding), no logit cap or constant maximum; otherwise the computation of the
jax splash kernel (float32 scores from the operand dtype, ``p @ v`` with ``p`` and ``v`` in
float32, ``p`` and ``ds`` rounded to the operand dtype in the backward), so the values agree to
float rounding (``tests/model/test_attention_onetile.py``).
"""
from __future__ import annotations

import functools
import importlib.util
import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp

VERSION = "0.0.17"
_PKG = "bes_tokamax_splash"          # private module name of the loaded kernel files
_SRC = "tokamax._src.ops.experimental.tpu.splash_attention"
_FILES = ("splash_attention_mask", "splash_attention_mask_info", "reference", "splash_attention_kernel")


def available() -> bool:
    return importlib.util.find_spec("tokamax") is not None


@functools.lru_cache(maxsize=None)
def kernel_modules() -> types.SimpleNamespace:
    """The kernel files of the installed tokamax package, loaded without ``import tokamax``."""
    spec = importlib.util.find_spec("tokamax")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("tokamax is not installed (pip install --no-deps tokamax==0.0.17)")
    root = Path(list(spec.submodule_search_locations)[0])
    d = root / "_src" / "ops" / "experimental" / "tpu" / "splash_attention"
    try:
        from importlib.metadata import version as dist_version
        version = dist_version("tokamax")                  # reads the metadata, imports nothing
    except Exception:  # noqa: BLE001 - a copy without metadata
        version = "?"
    if version != VERSION:
        raise ImportError(f"tokamax {version} found, the kernel switch was checked with {VERSION}")
    import jax.extend.core  # noqa: F401 - the kernel file uses jax.extend.core.Primitive (imported by tokamax's own __init__)
    pkg = sys.modules.get(_PKG)
    if pkg is None:
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = []
        sys.modules[_PKG] = pkg
    for name in _FILES:
        full = f"{_PKG}.{name}"
        if full in sys.modules:
            continue
        src = (d / f"{name}.py").read_text()
        # ``base`` re-exports SegmentIds, the mask value and the residual types of ``reference``
        src = src.replace(f"from {_SRC} import base\n", f"from {_PKG} import reference as base\n")
        src = src.replace(f"from {_SRC} import ", f"from {_PKG} import ")
        if "tokamax." in src.replace("tokamax._src.ops.experimental.tpu.splash_attention", ""):
            raise ImportError(f"{name}.py of tokamax {version} imports other tokamax modules")
        mod = types.ModuleType(full)
        mod.__file__ = str(d / f"{name}.py")
        sys.modules[full] = mod
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)   # noqa: S102 - the installed wheel's own code
        setattr(pkg, name, mod)
    return types.SimpleNamespace(**{n: sys.modules[f"{_PKG}.{n}"] for n in _FILES})


@functools.lru_cache(maxsize=None)
def make_kernel(t: int, block: int, kvc: int, diag_skip: bool, diag_grid: int, interpret: bool,
                residual_name: str | None):
    m = kernel_modules()
    sk = m.splash_attention_kernel
    config = sk.SplashConfig(
        block_q=block, block_kv=block, block_kv_compute=kvc,
        block_q_dkv=block, block_kv_dkv=block, block_kv_dkv_compute=kvc,
        use_fused_bwd_kernel=True, use_base2_exp=False, fuse_reciprocal=True,
        qk_diag_skip=diag_skip, sv_diag_skip=diag_skip, qk_diag_grid=diag_grid,
        residual_checkpoint_name=residual_name, interpret=interpret)
    with jax.ensure_compile_time_eval():
        return sk.make_splash_mha_single_device(m.splash_attention_mask.CausalMask((t, t)), config=config)


def attend_major(q: jax.Array, k: jax.Array, v: jax.Array, seg: jax.Array, *, block: int = 1024,
                 kvc: int | None = None, diag_skip: bool = False, diag_grid: int = 2,
                 interpret: bool = False, residual_name: str | None = None) -> jax.Array:
    """Causal same-segment attention on ``q, k, v: [B, H, T, hd]`` (``T`` a multiple of 128) with
    the tokamax splash kernel, one kernel per example (vmapped, as the jax splash path)."""
    b, h, t, hd = q.shape
    block = min(int(block), t)
    while t % block:
        block -= 128
    kvc = block if kvc is None else min(int(kvc), block)
    if diag_skip:
        kvc = block                                # the skip needs square single-step blocks
    kernel = make_kernel(t, block, kvc, bool(diag_skip), int(diag_grid), bool(interpret), residual_name)
    seg_ids = kernel_modules().reference.SegmentIds
    seg = jnp.asarray(seg, jnp.int32)

    def one(q1, k1, v1, s1):
        return kernel(q1, k1, v1, segment_ids=seg_ids(q=s1, kv=s1))
    return jax.vmap(one)(q, k, v, seg)


__all__ = ["available", "kernel_modules", "attend_major", "VERSION"]
