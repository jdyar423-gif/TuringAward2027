"""Compile-only TPU checks (``bes.train.aot``).

Fast tests run the analysis helpers and every program builder on the CPU backend. The tests
marked ``slow`` compile for a TPU v6e topology description with the installed libtpu (no TPU
needed; one to several minutes each and up to about 3 GB of host memory)."""
from __future__ import annotations

import dataclasses
import importlib.util
import os
import sys

import jax
import jax.numpy as jnp
import pytest

from bes.config import load_config
from bes.train import aot

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HAS_LIBTPU = importlib.util.find_spec("libtpu") is not None


def _cfg(name: str):
    return load_config(os.path.join(ROOT, "configs", f"{name}.py"))


def _tiny_model(cfg):
    return dataclasses.replace(cfg.model, vocab_size=512, d_model=64, n_heads=2, head_dim=32,
                               ve_dim=32, head_rank=32, n_loops=2, max_seq=64, clock_planes=3,
                               cand_dim=16, critic_hidden=16, loss_chunk=64, act_dtype="float32")


def test_parse_pairs_and_count_ops():
    assert aot.parse_pairs("1024x32, 2048x16") == [(1024, 32), (2048, 16)]
    text = ("  %scatter.1 = f32[4] scatter(%a, %b, %c), to_apply=%add\n"
            "  %w = (s32[]) while(%t), condition=%c, body=%b\n"
            "  %g = f32[2] gather(%x, %i)\n"
            '  %k = f32[8] custom-call(%q), custom_call_target="tpu_custom_call"\n')
    ops = aot.count_ops(text)
    assert ops["scatter"] == 1 and ops["while"] == 1 and ops["gather"] == 1
    assert ops["mosaic_kernel"] == 1 and ops["custom_call"] == 1 and ops["all_reduce"] == 0
    shlo = ('%7 = "stablehlo.scatter"(%0, %6) <{d = #stablehlo.scatter<inserted_window_dims = [0]>}>\n'
            '%8:3 = stablehlo.while(%iterArg = %7) : tensor<16xf32>\n')
    assert aot.count_ops(shlo, aot.STABLEHLO_OPS) == {"scatter": 1, "while": 1, "gather": 0,
                                                       "mosaic_kernel": 0}
    assert aot.count_ops("a b a", {"a": "a"}, regex=False) == {"a": 2}


def test_analysis_of_cpu_compiled_functions():
    """The helpers read real compiled objects: a scatter-add and a scan are found, the memory
    and cost analyses are parsed."""
    cpu = jax.devices("cpu")[0]

    def f(x, idx):
        y = jnp.zeros(16).at[idx].add(x)
        return jax.lax.scan(lambda c, v: (c + v, None), 0.0, y)[0]
    args = (jax.ShapeDtypeStruct((8,), jnp.float32), jax.ShapeDtypeStruct((8,), jnp.int32))
    res = aot.compile_program("toy", jax.jit(f), args, cpu)
    assert res.ok, res.error
    assert res.hlo_ops["scatter"] >= 1 and res.hlo_ops["while"] + res.stablehlo_ops["while"] >= 1
    assert res.stablehlo_ops["scatter"] == 1
    assert res.peak_bytes > 0 and "argument_size_in_bytes" in res.memory
    assert aot.within_limit(res) and not aot.within_limit(res, limit_gib=0.0)
    assert "toy" in res.row()
    bad = aot.compile_program("broken", jax.jit(lambda x: x @ x), (jax.ShapeDtypeStruct((3, 4), jnp.float32),), cpu)
    assert not bad.ok and "TypeError" in bad.error and "FAILED" in bad.row()


def test_program_builders_on_cpu():
    """Every program lowers and compiles on CPU for a tiny model (plumbing of the abstract
    arguments: parameters, constants, batch, cache, feature rows, train state)."""
    cfg = _cfg("smoke")
    mcfg = _tiny_model(cfg)
    cpu = jax.devices("cpu")[0]
    progs = {
        "loss": aot.loss_program(mcfg, 2, 16),
        "decode": aot.decode_program(mcfg, 3, 32),
        "prefill": aot.prefill_program(mcfg, 1, 8, 32),
        "train": aot.train_program(dataclasses.replace(cfg, model=mcfg), 2, 2, 16),
    }
    for name, (fn, args) in progs.items():
        res = aot.compile_program(name, fn, args, cpu)
        assert res.ok, (name, res.error)
        assert res.hlo_ops["scatter"] == 0, name          # one-hot table gradients, no scatter
        assert res.peak_bytes > 0


def test_rss_guard_is_inert_below_its_limit():
    with aot.RssGuard(1e6, period_s=0.01) as g:
        assert g.current_gb() > 0
    with aot.RssGuard(0.0):
        pass


def test_model_variant_never_uses_interpret_mode():
    m = aot.model_variant(_cfg("final").model, "splash", True)
    assert m.attn_impl == "splash" and m.remat and m.splash_interpret == "false"
    assert "bes.model.splash" in sys.modules      # the alias for the core's import


needs_tpu_compiler = pytest.mark.skipif(not HAS_LIBTPU, reason="libtpu is not installed")


@pytest.mark.slow
@needs_tpu_compiler
@pytest.mark.parametrize("impl", ["xla", "splash"])
def test_tpu_compile_of_a_reduced_model(impl):
    """Smoke-sized model, one micro-batch, for the v6e target: compiles, no scatter, the splash
    path contains Mosaic kernels and the dense path none."""
    cfg = _cfg("smoke")
    dev = aot.tpu_device()
    assert "v6" in dev.device_kind.lower()
    with aot.RssGuard(6.0):
        mcfg = aot.model_variant(dataclasses.replace(cfg.model, act_dtype="bfloat16"), impl, True)
        res = aot.compile_program("loss", *aot.loss_program(mcfg, 4, 128), dev, impl=impl)
    assert res.ok, res.error
    assert res.hlo_ops["scatter"] == 0
    assert (res.hlo_ops["mosaic_kernel"] > 0) == (impl == "splash")
    assert aot.within_limit(res)


@pytest.mark.slow
@needs_tpu_compiler
def test_tpu_compile_of_the_final_shapes():
    """The final configuration (bf16, splash attention, remat) for both training shapes and a
    decode step with 8 rows and 2048 slots: every peak below 28 GiB, no scatter (about 10
    minutes on 2 cores)."""
    cfg = _cfg("final")
    with aot.RssGuard(6.0):
        results = aot.run(cfg, ["loss", "decode"], ["splash"], [(1024, 32), (2048, 16)], [True],
                          decode=[(8, 2048)], log=lambda s: None)
    assert len(results) == 3
    for res in results:
        assert res.ok, res.error
        assert aot.within_limit(res), res.row()
        assert res.hlo_ops["scatter"] == 0, res.row()
