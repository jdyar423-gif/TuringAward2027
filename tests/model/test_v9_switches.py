"""The speed switches of the v9 token pages together (``bes.model.fast``: ``flat``, ``rows``,
``tokn``, with ``relu2``): the trainer's accumulation gives the loss and every gradient leaf of
the reference path. ``flat`` and ``rows`` are reshapes and a kept residual (bit for bit),
``tokn`` reorders float32 sums (rounding); each has its own test file, this one checks that they
do not disturb each other in one program, with remat and pieces, as the live run uses them."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _model_testlib import make_batch, tiny_cfg
from bes.model import fast as F
from bes.model import model as M
from bes.train import train as TR

ALL = {"flat": True, "rows": True, "tokn": True, "relu2": True}


def _setup(dtype: str):
    mcfg = tiny_cfg(gain_o=True, tok_neurons=4, tok_dim=32, table_grad="take", remat=True, act_dtype=dtype)
    params = M.init_params(mcfg, jax.random.key(3))
    keys = iter(jax.random.split(jax.random.key(5), 8))
    params["page"]["val"] = 0.1 * jax.random.normal(next(keys), params["page"]["val"].shape)
    params["gain"]["o"] = 0.1 * jax.random.normal(next(keys), params["gain"]["o"].shape)
    parts = [make_batch(mcfg, B=2, T=16, seed=7 + i) for i in range(2)]
    batch = {k: jnp.asarray(np.stack([p[k] for p in parts])) for k in parts[0]}
    return mcfg, params, M.make_consts(mcfg), batch


def _run(mcfg, params, consts, batch, setting: dict, split: int):
    before = F.apply(setting)
    try:
        fn = jax.jit(lambda p, c, b: TR.accumulate(M.loss_fn, p, c, b, mcfg, split))
        return jax.device_get(fn(params, consts, batch))
    finally:
        F.apply(before)


@pytest.mark.parametrize("dtype,tol", [("float32", 2e-5), ("bfloat16", 2.0 ** -6)])
def test_the_page_switches_together_give_the_reference_accumulation(dtype, tol):
    mcfg, params, consts, batch = _setup(dtype)
    off = {name: False for name in ALL}
    for split in (1, 2):
        ref = _run(mcfg, params, consts, batch, off, split)
        got = _run(mcfg, params, consts, batch, ALL, split)
        assert jax.tree.structure(ref) == jax.tree.structure(got)
        worst = 0.0
        for a, b in zip(jax.tree.leaves(ref), jax.tree.leaves(got)):
            a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
            assert a.shape == b.shape
            scale = float(np.max(np.abs(a), initial=0.0))
            if scale > 0:
                worst = max(worst, float(np.max(np.abs(a - b))) / scale)
        print(f"{dtype} micro_split={split}: largest difference relative to a leaf's largest entry {worst:.3g}")
        assert worst <= tol
        assert float(np.max(np.abs(ref[0]["page"]["key"]))) > 0.0 and float(np.max(np.abs(ref[0]["page"]["val"]))) > 0.0
    assert F.current() == {"kernel": F.current()["kernel"], **{name: False for name in F.FLAGS}}
