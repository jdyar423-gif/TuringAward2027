"""A run that continues an exported model: ``bes/train/init_from.py``, the continuation's
learning-rate settings (``OptimCfg.lr_mult``, ``tokens_before``) and ``configs/m6v.py``."""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib
import io
import json
import tarfile
from pathlib import Path

import jax
import numpy as np
import pytest

from bes import budget
from bes import export as E
from bes.config import Cfg, ModelCfg, load_config
from bes.model import model as M
from bes.roles import role_of
from bes.train import init_from as IF
from bes.train import schedule as S
from bes.train.checkpoint import flatten_with_paths
from bes.train.train import init_state

B = importlib.import_module("bes.optim.build")
REPO = Path(__file__).resolve().parents[2]
MODEL = ModelCfg(d_model=64, n_heads=2, head_dim=32, ve_dim=32, head_rank=32, n_loops=2, max_seq=64,
                 clock_planes=3, cand_dim=16, critic_hidden=16, gain_o=True, tok_neurons=2, tok_dim=8)


def _trained_like(cfg: ModelCfg, seed: int = 5):
    """An initialisation with every zero-initialised table made non-zero (a trained model has
    no all-zero leaf; a loader that kept the initialisation would be seen)."""
    params = M.init_params(cfg, jax.random.PRNGKey(seed))
    rng = np.random.default_rng(seed)
    return jax.tree.map(lambda a: np.asarray(a) + rng.normal(0, 0.05, np.shape(a)).astype(np.float32), params)


def _write_export(d: Path, params) -> tuple[dict, list]:
    d.mkdir(parents=True)
    arrays, choices = E.quantize(params)
    E.write_npz(d / E.WEIGHTS16, arrays)
    (d / E.WEIGHTS_JSON).write_text(json.dumps({"format": E.FORMAT, "leaves": [c.to_dict() for c in choices]}))
    (d / "config.py").write_text("# stand-in\n")
    return arrays, choices


def _pack(src_parent: Path, name: str, out: Path, part_bytes: int = 4096) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        tar.add(src_parent / name, arcname=name)
    blob = buf.getvalue()
    out.mkdir(parents=True)
    parts = []
    for i in range(0, len(blob), part_bytes):
        chunk = blob[i:i + part_bytes]
        pname = f"x.tar.{len(parts):03d}"
        (out / pname).write_bytes(chunk)
        parts.append({"name": pname, "bytes": len(chunk), "sha256": hashlib.sha256(chunk).hexdigest()})
    (out / IF.PARTS).write_text(json.dumps({"name": "x", "bytes": len(blob),
                                            "sha256": hashlib.sha256(blob).hexdigest(), "parts": parts}))


def _leaves(tree):
    paths, leaves, _ = flatten_with_paths(tree)
    return dict(zip(paths, [np.asarray(x) for x in leaves]))


def test_an_export_directory_is_found_directly_or_one_level_down(tmp_path):
    _write_export(tmp_path / "a" / "export", _trained_like(MODEL))
    assert IF.export_dir(tmp_path / "a" / "export") == tmp_path / "a" / "export"
    assert IF.export_dir(tmp_path / "a") == tmp_path / "a" / "export"
    with pytest.raises(IF.InitFromError):
        IF.export_dir(tmp_path / "nothing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(IF.InitFromError):                       # neither weights nor parts
        IF.export_dir(tmp_path / "empty", tmp_path / "cache")


def test_parts_are_checked_unpacked_once_and_refused_when_damaged(tmp_path):
    _write_export(tmp_path / "src" / "export", _trained_like(MODEL))
    _pack(tmp_path / "src", "export", tmp_path / "release" / "mymodel")
    cache = tmp_path / "cache"
    got = IF.export_dir(tmp_path / "release" / "mymodel", cache)
    assert got == cache / "init" / "mymodel" / "export" and (got / E.WEIGHTS16).is_file()
    assert (got / E.WEIGHTS16).read_bytes() == (tmp_path / "src" / "export" / E.WEIGHTS16).read_bytes()
    (got / "sentinel").write_text("kept")                       # a second call does not unpack again
    assert IF.export_dir(tmp_path / "release" / "mymodel", cache) == got and (got / "sentinel").is_file()
    # a damaged part is refused before anything is replaced
    part = tmp_path / "release" / "mymodel" / "x.tar.001"
    raw = bytearray(part.read_bytes()); raw[10] ^= 0xFF
    bad = tmp_path / "release" / "bad"
    bad.mkdir()
    for p in (tmp_path / "release" / "mymodel").iterdir():
        (bad / p.name).write_bytes(p.read_bytes())
    (bad / "x.tar.001").write_bytes(bytes(raw))
    with pytest.raises(IF.InitFromError, match="SHA-256"):
        IF.export_dir(bad, cache)
    (bad / "x.tar.001").unlink()
    with pytest.raises(IF.InitFromError, match="missing"):
        IF.export_dir(bad, cache)


def test_the_init_function_returns_the_exported_weights_in_new_arrays(tmp_path):
    params = _trained_like(MODEL)
    arrays, choices = _write_export(tmp_path / "export", params)
    want = _leaves(E.dequantize(arrays, choices, MODEL))
    fn = IF.make_init_fn(tmp_path / "export", tmp_path / "cache")
    a, b = fn(MODEL, jax.random.PRNGKey(0)), fn(MODEL, jax.random.PRNGKey(1))
    got = _leaves(a)
    assert set(got) == set(want) and len(got) == fn.info["leaves"]
    for p in want:
        np.testing.assert_array_equal(got[p], want[p], err_msg=p)
        assert np.any(got[p] != 0), p
    assert fn.info["numbers"] == sum(v.size for v in want.values())
    assert all(x is not y for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


def test_another_architecture_is_refused(tmp_path):
    _write_export(tmp_path / "export", _trained_like(MODEL))
    without_pages = dataclasses.replace(MODEL, tok_neurons=0)
    with pytest.raises(IF.InitFromError, match="model configuration"):
        IF.load_params(tmp_path / "export", without_pages, tmp_path / "cache")
    wider = dataclasses.replace(MODEL, tok_dim=16)
    with pytest.raises(IF.InitFromError, match="model configuration"):
        IF.load_params(tmp_path / "export", wider, tmp_path / "cache")


def test_the_trainer_state_starts_from_the_export_with_an_empty_optimiser(tmp_path):
    params = _trained_like(MODEL)
    arrays, choices = _write_export(tmp_path / "export", params)
    cfg = dataclasses.replace(Cfg(), model=MODEL)
    state, opt = init_state(cfg, IF.make_init_fn(tmp_path / "export", tmp_path / "cache"), role_of,
                            jax.random.PRNGKey(0))
    want = _leaves(E.dequantize(arrays, choices, MODEL))
    got = _leaves(state.params)
    for p in want:
        np.testing.assert_array_equal(got[p], want[p], err_msg=p)
    assert int(state.step) == 0
    fresh, _ = init_state(cfg, M.init_params, role_of, jax.random.PRNGKey(0))
    for x, y in zip(jax.tree_util.tree_leaves(state.opt_state), jax.tree_util.tree_leaves(fresh.opt_state)):
        if np.asarray(y).dtype.kind == "f" and not np.any(np.asarray(y)):
            assert not np.any(np.asarray(x))                     # moments start at zero as in a new run


def test_lr_mult_scales_every_group_and_tokens_before_enters_the_data_factor():
    base = S.with_horizon(Cfg(), 3.0e9)
    cont = dataclasses.replace(base, optim=dataclasses.replace(base.optim, lr_mult=0.25, tokens_before=9.0e9))
    whole = S.with_horizon(Cfg(), 12.0e9)
    assert B.data_factor(cont) == pytest.approx(B.data_factor(whole))
    a, b, w = B.group_lrs(base), B.group_lrs(cont), B.group_lrs(whole)
    for g in a:
        if g == "matrix":
            assert b[g] == pytest.approx(0.25 * w[g])
        else:
            assert b[g] == pytest.approx(0.25 * a[g])
    assert B.group_lrs(Cfg()) == B.group_lrs(dataclasses.replace(Cfg(), optim=dataclasses.replace(
        Cfg().optim, tokens_before=5e9)))                        # no horizon: no data correction at all
    assert {k: float(v) for k, v in B.lr_inputs(cont).items()} == pytest.approx(
        {k: v for k, v in b.items() if k != "frozen"})


def test_the_continuation_follows_one_schedule_over_both_runs():
    """The 24-hour rule for a run of ``T0 + T1`` tokens (peak, linear to a tenth of it), read from
    the point where the continuation begins: the same rates at its start and at its end."""
    cfg = load_config(str(REPO / "configs" / "m6v.py"))
    v9 = load_config(str(REPO / "configs" / "v9.py"))
    t0, t1 = cfg.optim.tokens_before, 3.7e9
    whole = S.with_horizon(v9, t0 + t1)
    now = S.with_horizon(cfg, t1)
    p0 = t0 / (t0 + t1)
    full = dataclasses.replace(cfg.optim, warmup_frac=0.0)
    for g in ("matrix", "embed", "page", "gain", "unembed"):
        start = B.group_lrs(now)[g] * S.lr_factor(0.0, full)
        end = B.group_lrs(now)[g] * S.lr_factor(1.0, full)
        assert start == pytest.approx(B.group_lrs(whole)[g] * S.lr_factor(p0, v9.optim), rel=0.02), g
        assert end == pytest.approx(B.group_lrs(whole)[g] * S.lr_factor(1.0, v9.optim), rel=0.02), g
    assert S.lr_factor(0.0, cfg.optim) == 0.0 and S.lr_factor(cfg.optim.warmup_frac, cfg.optim) > 0.95


def test_m6v_is_the_v9_model_on_the_six_hour_stream():
    cfg = load_config(str(REPO / "configs" / "m6v.py"))
    v9 = load_config(str(REPO / "configs" / "v9.py"))
    m6 = load_config(str(REPO / "configs" / "m6.py"))
    assert dataclasses.replace(cfg.model, table_grad=v9.model.table_grad) == v9.model
    # the six-hour data stream, and exemplars also under the library's own floor for 60% of the samples
    assert dataclasses.replace(cfg.data, exemplar_floor=0.0, exemplar_floor_frac=0.0) == m6.data
    assert (cfg.data.exemplar_floor, cfg.data.exemplar_floor_frac) == (0.35, 0.6)
    assert cfg.decode == m6.decode and cfg.decode.exemplar_floor == 0.0       # inference: the library's floors
    assert cfg.run.init_from == "release/main16-v9-before-dpo" and cfg.run.ship_library == "select"
    assert dataclasses.replace(cfg.run, name=m6.run.name, out_dir=m6.run.out_dir, init_from="",
                               init_sha256="", fast=m6.run.fast, ship_library=m6.run.ship_library) == m6.run
    readme = (REPO / "release" / "main16-v9-before-dpo" / "README.md").read_text()
    assert cfg.run.init_sha256 in readme and len(cfg.run.init_sha256) == 64
    shapes = jax.eval_shape(functools.partial(M.init_params, cfg.model), jax.random.PRNGKey(0))
    b = budget.count(shapes, cfg.model)
    assert b.total == 98_036_104 and b.active_step == 19_631_240     # docs/V9.md section 5
    from bes.model import fast
    assert fast.parse(cfg.run.fast) == {"kernel": "onetile:save", "relu2": True, "ce": False,
                                        "flat": True, "rows": True, "tokn": True}


def test_the_release_parts_hold_the_weights_the_run_starts_from():
    parts = REPO / "release" / "main16-v9-before-dpo"
    if not (parts / IF.PARTS).is_file():
        pytest.skip("the release parts are not in this checkout")
    meta = json.loads((parts / IF.PARTS).read_text())
    assert sum(p["bytes"] for p in meta["parts"]) == meta["bytes"] == 196_352_000
    assert all((parts / p["name"]).stat().st_size == p["bytes"] for p in meta["parts"])


def test_the_canary_trains_an_initialisation_at_the_reference_rates():
    """A continuation lowers its rates for trained weights; the start-up canary (an
    initialisation, ten steps) keeps those of a run of its own."""
    from bes.train import benchmark as BM

    cfg = S.with_horizon(load_config(str(REPO / "configs" / "m6v.py")), 3.7e9)
    ref = dataclasses.replace(cfg, optim=dataclasses.replace(cfg.optim, lr_mult=1.0, tokens_before=0.0))
    got = BM.canary_group_lr(cfg)
    assert {k: float(v) for k, v in got.items()} == {k: float(v) for k, v in B.lr_inputs(ref).items()}
    assert float(got["matrix"]) > 3 * float(B.lr_inputs(cfg)["matrix"])
