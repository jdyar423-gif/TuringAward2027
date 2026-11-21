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
    """With the library that is actually built: every stored table and the analytical worst
    case of one retrieval's read set (docs/E1_COVERAGE.md)."""
    cfg, b = _budget(name, budget.V2_ASSETS if name != "no_library" else None)
    assert b.total <= 100e6
    assert b.active_step <= 20e6
    assert cfg.model.remat          # required to fit device memory (docs/TPU_NOTES.md)


def test_built_library_assets_are_what_the_documents_say():
    v2 = budget.V2_ASSETS
    assert v2.library_numbers + v2.index_numbers == 63_768_391       # docs/E1_COVERAGE.md
    assert v2.library_read + v2.index_read == 245_223                # Library.read_bound total
    assert v2.library_read + v2.index_read <= 300_000                # LibraryCfg.max_read_numbers


def test_strict_config_is_under_20m_in_the_strict_reading_too():
    """The budget's window formula is an upper bound of the measured read set; 840 distinct
    tokens is the largest count observed in 2048-token windows of real text. The built library's
    worst-case retrieval read is included."""
    _, final = _budget("final", budget.V2_ASSETS)
    _, strict = _budget("strict", budget.V2_ASSETS)
    assert strict.active_strict(1024, 480) <= 20e6
    assert strict.active_strict(2048, 840) <= 20e6
    assert final.active_strict(2048, 840) > 20e6          # why the strict variant exists
    assert final.active_step - strict.active_step == pytest.approx(
        2 * 2 * 768 * (3072 - 2688) + 768 * (256 - 128) + (256 - 128), abs=1)


def test_recorded_library_assets_match_the_library_on_disk():
    """V2_ASSETS is a recorded copy of one build; a rebuilt library must not drift silently."""
    from bes.config import LibraryCfg
    lib_dir = os.path.join(ROOT, LibraryCfg().library_dir)
    if not os.path.exists(os.path.join(lib_dir, "library.json")):
        pytest.skip("the built library is not present")
    from bes.library.retrieve import Library
    live = budget.assets_of(Library.load(lib_dir))
    v2 = budget.V2_ASSETS
    assert live.library_numbers + live.index_numbers == v2.library_numbers + v2.index_numbers
    assert live.library_read + live.index_read == v2.library_read + v2.index_read
