"""``python -m bes.eval.proxy`` end to end with stand-in generators and a stand-in judge."""
from __future__ import annotations

import json

from bes.eval.proxy import runner
from bes.eval.proxy.__main__ import build_spec, main, ours_presets
from bes.eval.proxy.common import read_jsonl


def _cache_lines(root):
    return {p.name: len(read_jsonl(p)) for p in (root / "cache").glob("*.jsonl")}


def test_all_end_to_end_and_resumable(tmp_path, capsys):
    args = ["--root", str(tmp_path), "all", "--ours", "standin-plain", "--opponent", "standin-verbose",
            "--judge", "standin-longer", "--split", "dev", "--categories", "math,instruct", "--limit", "14"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "| ours-vs-standin-verbose-common-dev | judge-standin-longer | 14 |" in out
    summary = (tmp_path / "reports" / "summary-dev.md").read_text()
    assert "opponent `standin-verbose" in summary and "ours `standin-plain" in summary
    report = json.loads((tmp_path / "reports" / "ab" / "ours-vs-standin-verbose-common-dev--judge-standin-longer.json").read_text())
    assert report["overall"]["n"] == 14 and report["overall"]["losses"] >= 1
    assert set(report["by_category"]) <= {"math", "instruct"}
    before = _cache_lines(tmp_path)
    assert main(args) == 0          # everything is cached: no new generations, same report
    assert _cache_lines(tmp_path) == before
    exact = list((tmp_path / "reports" / "exact").glob("*.json"))
    assert len(exact) == 2
    data = json.loads(exact[0].read_text())
    assert data["summary"]["n_prompts"] == 14 and data["summary"]["math"]["n"] > 0


def test_individual_commands(tmp_path, capsys):
    root = ["--root", str(tmp_path)]
    assert main(root + ["run", "--model", "standin-plain", "--label", "a", "--split", "test", "--limit", "6"]) == 0
    assert main(root + ["run", "--model", "standin-loop", "--label", "b", "--split", "test", "--limit", "6"]) == 0
    runs = runner.list_runs(tmp_path)
    assert len(runs) == 2
    x = next(r for r in runs if r.startswith("standin-plain"))
    y = next(r for r in runs if r.startswith("standin-loop"))
    assert main(root + ["exact", y, "--split", "test"]) == 0
    assert "looping tail 1.0" in capsys.readouterr().out
    assert main(root + ["pairs", "--x", x, "--y", y, "--name", "ab", "--split", "test"]) == 0
    assert main(root + ["judge", "--pairs", "ab", "--judge", "standin-first"]) == 0
    assert main(root + ["score", "--pairs", "ab"]) == 0
    out = capsys.readouterr().out
    assert "judge-standin-first" in out and "| overall | 6 | 0/0/6 |" in out
    assert main(root + ["pairs", "--x", x, "--y", y, "--name", "ab2", "--split", "test", "--seed", "1"]) == 0
    assert main(root + ["judge", "--pairs", "ab2", "--judge", "standin-first"]) == 0
    capsys.readouterr()
    assert main(root + ["compare", "--a", "ab", "--b", "ab2", "--source", "judge-standin-first"]) == 0
    paired = json.loads(capsys.readouterr().out)
    assert (paired["n"], paired["same"], paired["p"]) == (6, 6, 1.0)   # a position-biased judge: all ties
    assert main(root + ["power", "--delta", "0.05"]) == 0
    assert capsys.readouterr().out.strip() == "783"
    assert main(["prompts", "--split", "dev"]) == 0
    assert '"total": 200' in capsys.readouterr().out


def test_build_spec_for_registry_and_ours(tmp_path):
    spec = build_spec("standin-plain", None, ["temperature=0.2"], None, 3, 4, "logprob_mean", 1, None)
    assert spec.settings.temperature == 0.2 and spec.settings.samples == 4 and spec.seed == 3
    ours = build_spec("ours", "k1", [], "ours-k1", 0, None, None, 1, None,
                      generator="bes.eval.proxy.run_ours:standin", generator_args={"style": "plain"})
    assert ours.key == "standin-plain" and ours.settings.extra_dict["particles"] == 1
    assert set(ours_presets()) == {"default", "k1", "nolib"}
