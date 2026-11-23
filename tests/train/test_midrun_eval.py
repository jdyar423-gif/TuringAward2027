"""``scripts/midrun_eval.py``: the arithmetic of the comparison (the scoring itself is
``bes.model.model.loss_fn`` on ``bes.export.heldout_batches`` rows, tested with the export)."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("bes_midrun_eval", REPO / "scripts" / "midrun_eval.py")
ME = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ME)


def test_interpolation_is_linear_in_log_tokens_and_never_extrapolates():
    pts = [(1e6, 5.0), (4e6, 4.0), (16e6, 3.5)]
    assert ME.interpolate(pts, 2e6) == pytest.approx(4.5)             # halfway in log tokens
    assert ME.interpolate(pts, 4e6) == pytest.approx(4.0)
    assert ME.interpolate(pts, 8e6) == pytest.approx(3.75)
    assert ME.interpolate(pts, 16e6) == pytest.approx(3.5)
    assert ME.interpolate(pts, 5e5) is None and ME.interpolate(pts, 2e7) is None
    assert ME.interpolate([(4e6, 4.0)], 4e6) == pytest.approx(4.0)    # one checkpoint: its own point only
    assert ME.interpolate([(4e6, 4.0)], 3e6) is None
    assert ME.interpolate([(0, 9.0), (None, 1.0), (4e6, None)], 1.0) is None     # nothing usable
    assert ME.interpolate(pts, 0) is None and ME.interpolate(pts, None) is None


def _ckpt(name, tokens, seconds, loss, chat, asst):
    return {"name": name, "tokens": tokens, "train_seconds": seconds, "loss": loss,
            "by_domain": {"web": {"loss": loss + 0.2, "assistant_loss": None},
                          "chat": {"loss": chat, "assistant_loss": asst}}}


def _run(name, ckpts, tokens=None):
    rows = np.arange(12, dtype=np.int32).reshape(3, 4) if tokens is None else tokens
    batch = SimpleNamespace(arrays={"tokens": rows, "w": np.ones((3, 3), np.float32)})
    return {"run": name, "config": f"configs/{name}.py", "plan": "splash-remat1-ms32-onehot",
            "checkpoints": ckpts, "_rows": {"web": batch, "chat": batch}}


def test_the_comparison_pairs_the_runs_at_the_time_and_the_tokens_both_reached():
    a = _run("a", [_ckpt("a1", 1e6, 600.0, 5.0, 4.8, 4.6), _ckpt("a2", 4e6, 2400.0, 4.0, 3.8, 3.6)])
    b = _run("b", [_ckpt("b1", 1e6, 1200.0, 4.9, 4.7, 4.5), _ckpt("b2", 2e6, 2400.0, 4.3, 4.1, 3.9)])
    cmp = ME.compare(a, b)
    assert cmp["same_rows"] is True and cmp["difference"] == "b - a"
    assert cmp["newest"]["delta"]["loss"] == pytest.approx(0.3)       # b is behind at its newest checkpoint
    assert cmp["newest"]["delta"]["chat.assistant"] == pytest.approx(0.3)
    t = cmp["matched_train_seconds"]                                 # both reached 2400 s: the same numbers
    assert t["train_seconds"] == 2400.0 and t["delta"]["loss"] == pytest.approx(0.3)
    n = cmp["matched_tokens"]                                        # at b's 2e6 tokens a is interpolated: 4.5
    assert n["tokens"] == 2e6 and n["a"]["loss"] == pytest.approx(4.5)
    assert n["delta"]["loss"] == pytest.approx(-0.2)                 # per token, b is ahead
    assert n["delta"]["chat"] == pytest.approx(4.1 - 4.3)
    # runs scored on other rows are flagged, and a run without ledger values has no matched points
    other = _run("c", [_ckpt("c1", None, None, 4.0, 3.9, 3.7)], tokens=np.zeros((3, 4), np.int32))
    cmp = ME.compare(a, other)
    assert cmp["same_rows"] is False and cmp["matched_tokens"] is None and cmp["matched_train_seconds"] is None
    assert cmp["newest"]["delta"]["loss"] == pytest.approx(0.0)
    text = ME.table([a, b], ME.compare(a, b))
    assert "matched_tokens at 2.000e+06 tokens" in text and "chat.assistant" in text and "same rows: True" in text
    assert "newest: loss +0.3000" in text and "matched_train_seconds at 40.0 min of training" in text
