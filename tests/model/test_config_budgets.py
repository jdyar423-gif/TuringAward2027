"""Budgets of the shipped configurations (see docs/ACCOUNTING.md)."""
import functools
import os

import jax
import pytest

from bes import budget
from bes.config import load_config
from bes.model import model as M

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _budget(name, assets=None):
    cfg = load_config(os.path.join(ROOT, "configs", f"{name}.py"))
    shapes = jax.eval_shape(functools.partial(M.init_params, cfg.model), jax.random.PRNGKey(0))
    return cfg, budget.count(shapes, cfg.model, assets)


@pytest.mark.parametrize("name", ["final", "strict", "no_library"])
def test_standard_reading_is_under_the_caps(name):
    cfg, b = _budget(name, budget.PLAN_ASSETS if name != "no_library" else None)
    assert b.total <= 100e6
    assert b.active_step <= 20e6
    assert cfg.model.remat          # required to fit device memory (docs/TPU_NOTES.md)


def test_strict_config_is_under_20m_in_the_strict_reading_too():
    """The budget's window formula is an upper bound of the measured read set; 840 distinct
    tokens is the largest count observed in 2048-token windows of real text."""
    _, final = _budget("final")
    _, strict = _budget("strict")
    assert strict.active_strict(1024, 480) <= 20e6
    assert strict.active_strict(2048, 840) <= 20e6
    assert final.active_strict(2048, 840) > 20e6          # why the strict variant exists
    assert final.active_step - strict.active_step == pytest.approx(
        2 * 2 * 768 * (3072 - 2752) + 768 * (256 - 128) + (256 - 128), abs=1)
