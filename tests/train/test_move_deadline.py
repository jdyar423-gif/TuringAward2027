"""Moving the deadline of a run that continues on another machine (``bes.train.supervisor.
move_deadline``, ``scripts/run_24h.py --move-deadline``)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from bes.train import schedule
from bes.train import stages as ST
from bes.train import supervisor as SV

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("bes_run_24h_move", REPO / "scripts" / "run_24h.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

T0 = 1_000_000.0
BUDGET = 24 * 3600.0


def _run(tmp_path: Path) -> Path:
    out = tmp_path / "run"
    (out / "train").mkdir(parents=True)
    SV.write_manifest(out, {"format": SV.FORMAT, "config": str(REPO / "configs" / "final.py"), "overrides": [],
                            "budget": {"t0": T0, "deadline": T0 + BUDGET, "budget_s": BUDGET}})
    ST.StageLedger(out / "stages.json", t0=T0, deadline=T0 + BUDGET, order=ST.ORDER).save()
    (out / "train" / "run.json").write_text(json.dumps({"t0": T0, "deadline": T0 + BUDGET, "config_hash": "x"}))
    return out


def test_the_deadline_moves_in_every_record_and_the_schedule_goes_on_where_it_was(tmp_path):
    out = _run(tmp_path)
    post_s = SV.base_config(SV.read_manifest(out)).run.post_reserve_hours * 3600.0
    # eight hours of training, then the machine is gone for 90 minutes
    died, back = T0 + 9 * 3600.0, T0 + 10.5 * 3600.0
    trained = 8 * 3600.0
    p_before = schedule.progress(trained, died, T0 + BUDGET - post_s)
    res = SV.move_deadline(out, T0 + BUDGET + 5400.0, now=back, reason="another runtime")
    assert res["from"] == T0 + BUDGET and res["moved_s"] == 5400.0 and res["left_s"] == pytest.approx(15 * 3600.0)
    assert res["train_end"] == pytest.approx(T0 + BUDGET + 5400.0 - post_s)
    assert set(res["files"]) == {"stages.json", "train/run.json", "manifest.json"}
    man = SV.read_manifest(out)
    assert man["budget"]["deadline"] == T0 + BUDGET + 5400.0 and man["budget"]["budget_s"] == BUDGET
    assert man["budget"]["moved"] == [{"time": back, "from": T0 + BUDGET, "to": T0 + BUDGET + 5400.0,
                                       "reason": "another runtime"}]
    assert json.loads((out / "train" / "run.json").read_text())["deadline"] == T0 + BUDGET + 5400.0
    # the stage ledger takes the moved deadline (and still refuses another one)
    led = ST.StageLedger(out / "stages.json", deadline=T0 + BUDGET + 5400.0, order=ST.ORDER)
    assert led.deadline == T0 + BUDGET + 5400.0 and led.t0 == T0
    with pytest.raises(ValueError):
        ST.StageLedger(out / "stages.json", deadline=T0 + BUDGET, order=ST.ORDER)
    # moved by the downtime, the progress continues where it stopped; unmoved, it would have jumped
    assert schedule.progress(trained, back, res["train_end"]) == pytest.approx(p_before)
    assert schedule.progress(trained, back, T0 + BUDGET - post_s) > p_before + 0.03
    # earlier is allowed too (a machine that does not live as long as the run had left): compressed
    res = SV.move_deadline(out, T0 + 20 * 3600.0, now=back)
    assert res["moved_s"] < 0 and len(SV.read_manifest(out)["budget"]["moved"]) == 2
    assert schedule.progress(trained, back, res["train_end"]) > p_before


def test_a_deadline_that_is_not_in_the_future_and_a_directory_without_a_run_are_refused(tmp_path):
    out = _run(tmp_path)
    with pytest.raises(ValueError, match="not in the future"):
        SV.move_deadline(out, T0 + 3600.0, now=T0 + 7200.0)
    with pytest.raises(ValueError, match="not in the future"):
        SV.move_deadline(out, float("nan"), now=T0)
    assert SV.read_manifest(out)["budget"] == {"t0": T0, "deadline": T0 + BUDGET, "budget_s": BUDGET}
    with pytest.raises(ValueError, match="no run"):
        SV.move_deadline(tmp_path / "nothing", T0 + BUDGET, now=T0)
    # a run that died before its stage ledger and its trainer existed: only the manifest changes
    (out / "stages.json").unlink()
    (out / "train" / "run.json").unlink()
    assert SV.move_deadline(out, T0 + BUDGET + 60.0, now=T0)["files"] == ["manifest.json"]


def test_the_command_line_takes_an_absolute_time_or_minutes_and_runs_nothing(tmp_path, capsys):
    assert R.parse_deadline("1800000000", 5.0) == 1_800_000_000.0
    assert R.parse_deadline("+45", 1000.0) == 1000.0 + 2700.0 and R.parse_deadline(" -30 ", 5000.0) == 3200.0
    with pytest.raises(SystemExit):
        R.parse_deadline("soon", 0.0)
    out = _run(tmp_path)
    far = 4_000_000_000.0
    assert R.main(["--out", str(out), "--move-deadline", str(far)]) == 0
    assert SV.read_manifest(out)["budget"]["deadline"] == far and "moved by" in capsys.readouterr().err
    assert R.main(["--out", str(out), "--move-deadline", "+90"]) == 0
    assert SV.read_manifest(out)["budget"]["deadline"] == far + 5400.0
    assert not (out / "stage_table.json").exists() and not (out / "jax_cache").exists()      # nothing ran
    with pytest.raises(SystemExit, match="not in the future"):
        R.main(["--out", str(out), "--move-deadline", "12345"])
    with pytest.raises(SystemExit, match="no run"):
        R.main(["--out", str(tmp_path / "empty"), "--move-deadline", "+5"])
