"""The supervisor's machinery: background processes (restart with back-off, a failing child,
dependencies, no orphans), the run lock, the core plan, switches and the fallback ladder, the
effective configuration, the library gate and self-check, sample labels, the download plan,
the device check; and (slow) the CPU rehearsal of scripts/run_24h.py with a kill and a resume."""
from __future__ import annotations

import dataclasses
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from bes.config import Cfg, load_config
from bes.train import supervisor as SV
from bes.train.schedule import FakeClock, stage_at

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable


# ---------------------------------------------------------------- background processes
def _script(tmp: Path, name: str, body: str) -> Path:
    p = tmp / f"{name}.py"
    p.write_text(textwrap.dedent(body))
    return p


def test_failing_child_is_restarted_with_backoff_then_succeeds(tmp_path):
    flaky = _script(tmp_path, "flaky", f"""
        import sys, pathlib
        p = pathlib.Path({str(tmp_path / 'count')!r})
        n = int(p.read_text()) if p.exists() else 0
        p.write_text(str(n + 1))
        print("run", n + 1, flush=True)
        sys.exit(0 if n + 1 >= 3 else 7)
    """)
    bad = _script(tmp_path, "bad", "import sys; print('always fails'); sys.exit(5)")
    ok = _script(tmp_path, "ok", "print('done')")
    ps = SV.ProcessSupervisor(tmp_path / "logs", tmp_path / "bg.json", period_s=0.05)
    ps.add(SV.TaskSpec("flaky", (PY, str(flaky)), restarts=4, backoff_s=0.05, nice=0))
    ps.add(SV.TaskSpec("bad", (PY, str(bad)), restarts=2, backoff_s=0.05, nice=0))
    ps.add(SV.TaskSpec("after_flaky", (PY, str(ok)), after=("flaky",), nice=0))
    ps.add(SV.TaskSpec("after_bad", (PY, str(ok)), after=("bad",), nice=0))
    assert ps.wait(timeout=60)
    st = ps.status()
    assert st["flaky"]["state"] == "finished" and st["flaky"]["runs"] == 3
    assert st["flaky"]["exit_codes"] == [7, 7, 0]
    assert st["bad"]["state"] == "failed" and st["bad"]["runs"] == 3          # 1 + 2 restarts
    assert "exit code 5" in st["bad"]["error"]
    assert st["after_flaky"]["state"] == "finished"
    assert st["after_bad"]["state"] == "failed" and "dependency bad failed" in st["after_bad"]["error"]
    log = (tmp_path / "logs" / "flaky.log").read_text()
    assert log.count("===== start") == 3 and "run 3" in log
    saved = json.loads((tmp_path / "bg.json").read_text())["tasks"]
    assert saved["flaky"]["state"] == "finished"
    # keep_finished: a resumed supervisor does not start a finished task again ...
    ps2 = SV.ProcessSupervisor(tmp_path / "logs", tmp_path / "bg.json", period_s=0.05,
                               keep_finished=True)
    ps2.add(SV.TaskSpec("flaky", (PY, str(flaky)), restarts=4, backoff_s=0.05))
    ps2.poll()
    assert ps2.finished("flaky") and ps2.status()["flaky"]["from_previous_run"]
    assert (tmp_path / "count").read_text() == "3"
    # ... by default it runs it again (idempotent tasks, the disk may have lost their work)
    ps3 = SV.ProcessSupervisor(tmp_path / "logs", tmp_path / "bg.json", period_s=0.05)
    ps3.add(SV.TaskSpec("flaky", (PY, str(flaky)), restarts=4, backoff_s=0.05))
    assert ps3.wait(["flaky"], timeout=30) and ps3.finished("flaky")
    assert (tmp_path / "count").read_text() == "4"


def test_backoff_doubles_and_stop_terminates_children(tmp_path):
    clock = FakeClock(0.0)
    bad = _script(tmp_path, "bad", "import sys; sys.exit(1)")
    ps = SV.ProcessSupervisor(tmp_path / "logs", None, clock=clock)
    ps.add(SV.TaskSpec("bad", (PY, str(bad)), restarts=3, backoff_s=10.0, backoff_max_s=25.0))
    delays = []
    for _ in range(4):
        ps.poll()                                       # start
        ps.procs["bad"].wait()
        ps.poll()                                       # reap: back-off
        st = ps.status()["bad"]
        if st["state"] == "failed":
            break
        delays.append(st["next_start"] - clock())
        clock.advance(st["next_start"] - clock())
    assert delays == [10.0, 20.0, 25.0]
    assert ps.failed("bad")
    sleeper = _script(tmp_path, "sleeper", "import time; time.sleep(120)")
    ps2 = SV.ProcessSupervisor(tmp_path / "logs2", None, period_s=0.05)
    ps2.add(SV.TaskSpec("sleep", (PY, str(sleeper)), nice=0))
    ps2.start()
    t_end = time.time() + 20
    while ps2.status()["sleep"]["state"] != "running" and time.time() < t_end:
        time.sleep(0.05)
    pid = ps2.status()["sleep"]["pid"]
    t0 = time.time()
    ps2.stop(timeout=10)
    assert time.time() - t0 < 10
    with pytest.raises(OSError):
        os.kill(pid, 0)
    assert ps2.status()["sleep"]["state"] == "stopped"


def test_children_die_with_the_supervisor(tmp_path):
    """A supervisor killed with SIGKILL leaves no background process behind."""
    pidfile = tmp_path / "child.pid"
    inner = _script(tmp_path, "inner", f"""
        import os, time, pathlib
        from bes.train.supervisor import child_setup
        child_setup(0, (), int(os.environ["BES_SUPERVISOR_PID"]))
        pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid()))
        time.sleep(120)
    """)
    outer = _script(tmp_path, "outer", f"""
        import time
        from bes.train.supervisor import ProcessSupervisor, TaskSpec
        ps = ProcessSupervisor({str(tmp_path / 'logs')!r}, None, period_s=0.05)
        ps.add(TaskSpec("inner", ({PY!r}, {str(inner)!r}), nice=0))
        ps.start()
        time.sleep(120)
    """)
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    proc = subprocess.Popen([PY, str(outer)], env=env, cwd=str(ROOT))
    t_end = time.time() + 30
    while not pidfile.exists() and time.time() < t_end:
        time.sleep(0.05)
    assert pidfile.exists(), "the child did not start"
    child = int(pidfile.read_text())
    proc.kill()
    proc.wait()
    t_end = time.time() + 10
    alive = True
    while alive and time.time() < t_end:
        try:
            os.kill(child, 0)
            with open(f"/proc/{child}/stat") as f:      # a zombie counts as gone
                alive = f.read().split()[2] != "Z"
        except (OSError, FileNotFoundError):
            alive = False
        time.sleep(0.05)
    assert not alive, "the background child outlived its supervisor"


def test_run_lock_refuses_a_second_supervisor(tmp_path):
    lock = SV.RunLock(tmp_path / ".lock").acquire()
    code = ("import sys; from bes.train.supervisor import RunLock, RunLocked\n"
            f"try:\n RunLock({str(tmp_path / '.lock')!r}).acquire()\nexcept RunLocked as e:\n"
            " print(e); sys.exit(9)\n")
    out = subprocess.run([PY, "-c", code], capture_output=True, text=True, cwd=str(ROOT))
    assert out.returncode == 9 and str(os.getpid()) in out.stdout
    lock.release()
    out = subprocess.run([PY, "-c", code], capture_output=True, text=True, cwd=str(ROOT))
    assert out.returncode == 0


# ---------------------------------------------------------------------- core plan
@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 44, 96])
def test_core_plan_invariants(n):
    cp = SV.core_plan(n)
    ids = set(range(n))
    for name in ("trainer", "tokenize", "downloads", "library", "sidecar", "opponents"):
        part = getattr(cp, name)
        assert part and set(part) <= ids, name
    assert 2 <= cp.loader_workers <= 6 and cp.loader_max >= cp.loader_workers
    assert cp.opponent_workers >= 1 and cp.opponent_threads in (1, 2)
    assert cp.judge_threads >= 1
    if n >= 16:                     # enough cores: background shares do not touch the trainer's
        bg = set(cp.tokenize) | set(cp.downloads) | set(cp.library) | set(cp.sidecar) | set(cp.opponents)
        assert not bg & set(cp.trainer)
        assert len(bg) + len(cp.trainer) + cp.loader_workers <= n
    assert SV.CorePlan.from_dict(json.loads(json.dumps(cp.to_dict()))) == cp
    if n == 44:
        assert (cp.loader_workers, cp.loader_max, len(cp.tokenize)) == (6, 12, 4)


# --------------------------------------------------------- switches, ladder, config
def test_switches_and_the_ladder():
    assert SV.parse_switches("sink:0 conf:0, src_5:0.25") == {"sink": 0.0, "conf": 0.0, "src_5": 0.25}
    assert SV.parse_switches("") == {}
    with pytest.raises(ValueError):
        SV.parse_switches("sink=0")
    assert SV.format_switches({"sink": 0.0, "src_5": 0.25}) == "sink:0 src_5:0.25"
    groups = [g for g, _, _ in SV.LADDER]
    # the v9 paths first (attention gate, selective attention, token gains, token neurons), then
    # plan section 6
    assert groups == ["gate", "sel", "gain", "page", "sink", "conf", "ngram", "library", "tilt",
                      "registers", "calc"]
    assert SV.ladder_rung(0) == ({}, [], [])
    sw, ov, g = SV.ladder_rung(4)
    assert g == groups[:4] and sw == {"gate": 0.0, "sel": 0.0, "gain": 0.0, "page": 0.0} and ov == []
    sw, ov, g = SV.ladder_rung(8)
    assert g == groups[:8] and sw == {"gate": 0.0, "sel": 0.0, "gain": 0.0, "page": 0.0,
                                      "sink": 0.0, "conf": 0.0, "cand_in": 0.0}
    assert ov == ["data.library_frac_base=0", "data.library_frac_chat=0"]
    sw, _, _ = SV.ladder_rung(len(SV.LADDER))
    M = pytest.importorskip("bes.model.model")
    # the whole ladder switches off every path of the safe configuration
    safe = SV.safe_switches()
    assert set(M.SWITCHES) <= set(sw) | {"kind"} and all(v == 0 for v in sw.values())
    assert set(safe) == set(M.SWITCHES) | {"kind"}
    for name, base in (("safe", "final"), ("v9_safe", "v9")):
        cfg = load_config(str(ROOT / "configs" / f"{name}.py"))
        assert SV.parse_switches(cfg.run.switches) == safe, name
        assert cfg.data.library_frac_base == 0 and cfg.data.library_frac_chat == 0
        ref = load_config(str(ROOT / "configs" / f"{base}.py"))
        assert cfg.model == ref.model and cfg.optim == ref.optim      # same graph, same tree


def test_effective_configuration_round_trip():
    man = {"config": str(ROOT / "configs" / "smoke.py"), "overrides": ["run.deadline_hours=0.2"],
           "benchmark": {"plan_overrides": ["model.remat=true", "optim.micro_seqs=4",
                                            "model.attn_impl=xla", "model.table_grad=take"],
                         "ladder": {"rung": 2, "switches": "sink:0 conf:0", "overrides": []},
                         "horizon_tokens": 1.234567e7}}
    ov = SV.run_overrides(man)
    assert ov[0] == "run.deadline_hours=0.2" and ov[-1] == "run.switches=sink:0 conf:0"
    assert SV.horizon_of(man) == 1.234567e7 and SV.horizon_of({}) is None
    cfg = SV.effective_config(man)
    assert cfg.model.remat and cfg.optim.micro_seqs == 4 and cfg.model.table_grad == "take"
    assert cfg.run.switches == "sink:0 conf:0" and cfg.optim.horizon_tokens == 1.234567e7
    assert SV.effective_config(man) == cfg                 # a resume rebuilds the same config
    assert SV.base_config(man).optim.horizon_tokens == 0.0
    gen = SV.generator_overrides(man)
    assert gen.split(",") == ["model.remat=true", "model.attn_impl=xla", "model.table_grad=take"]
    rung0 = dict(man, benchmark=dict(man["benchmark"], ladder={"rung": 0, "switches": None,
                                                                "overrides": []}))
    assert not any(o.startswith("run.switches") for o in SV.run_overrides(rung0))


def test_a_v9_run_never_asks_for_the_library(tmp_path, monkeypatch):
    """configs/v9.py has no library (it derives from configs/no_library.py): the library task
    writes its marker without building one, the run's library accessors answer without looking
    for one, the library gate passes every stage with the library fractions at 0, and the
    effective configuration of a v9 benchmark result (plan, ladder rung) keeps the v9 model. Of
    the v9 rungs, the configuration has the token gains and the token neurons (no gate, no
    selective attention: their rungs change nothing for it)."""
    import runpy

    import numpy as np

    from bes.train import benchmark as B

    M = pytest.importorskip("bes.model.model")
    v9 = ROOT / "configs" / "v9.py"
    cfg = load_config(str(v9))
    assert not cfg.data.use_library and not cfg.decode.use_library
    assert cfg.data.library_frac_base == 0 and cfg.data.library_frac_chat == 0
    monkeypatch.setattr(runpy, "run_path", lambda *a, **kw: pytest.fail("a v9 run builds no library"))
    out = tmp_path / "run"
    SV.write_manifest(out, {"budget": {"t0": time.time()}})
    paths = SV.RunPaths(out)
    assert SV.task_library(paths, cfg, SV.core_plan(2), print) == 0
    mk = SV.read_marker(paths, "library")
    assert mk["path"] is None and "no library" in mk["reason"]
    R = _run24()
    run = R.Run.__new__(R.Run)
    run.cfg, run.paths = cfg, paths
    assert run.library_ready() is None and run.library_path() == ""
    fl = FakeLoader()
    gate = SV.LibraryGate(fl, run.library_ready, check_every_s=0.0)
    for i, p in enumerate((0.0, 0.5, 0.95)):
        gate(i, stage_at(p, cfg))
    assert not gate.active and not fl.libs and all(b == 0 and c == 0 for _, b, c in fl.calls)
    # the benchmark's plan and a ladder rung through the effective configuration
    plan = B.Plan.of(cfg)
    assert plan.key == "splash-remat1-ms32-sorted"
    man = {"config": str(v9), "overrides": [],
           "benchmark": {"plan_overrides": plan.overrides(),
                         "ladder": {"rung": 3, "switches": "gate:0 sel:0 gain:0", "overrides": []},
                         "horizon_tokens": 2.0e10}}
    ecfg = SV.effective_config(man)
    assert ecfg.model == cfg.model and ecfg.run.switches == "gate:0 sel:0 gain:0"
    assert SV.generator_overrides(man).split(",") == [o for o in plan.overrides()
                                                      if o.startswith("model.")]
    on = {k for k, v in M.make_consts(cfg.model)["enable"].items() if np.ndim(v) == 0 and float(v)}
    assert {"gain", "page"} <= on and not {"sel", "gate"} & on
    rung = M.make_consts(cfg.model, **SV.parse_switches(ecfg.run.switches))["enable"]
    assert float(rung["gain"]) == 0.0 and float(rung["page"]) == 1.0
    assert SV.format_switches(SV.ladder_rung(3)[0]) == ecfg.run.switches


def test_make_consts_fn_applies_the_switches():
    M = pytest.importorskip("bes.model.model")
    cfg = load_config(str(ROOT / "configs" / "smoke.py"))
    c = SV.make_consts_fn("sink:0 reg:0")(cfg.model)
    assert float(c["enable"]["sink"]) == 0.0 and float(c["enable"]["reg"]) == 0.0
    assert float(c["enable"]["conf"]) == 1.0
    full = M.make_consts(cfg.model)
    import jax
    assert jax.tree.structure(c) == jax.tree.structure(full)     # same compiled program


# ---------------------------------------------------------------------- library gate
class FakeLoader:
    def __init__(self):
        self.calls, self.libs = [], []

    def __call__(self, index, stage):
        self.calls.append((index, stage.library_frac_base, stage.library_frac_chat))
        return {"index": index}

    def set_library(self, lib):
        self.libs.append(lib)

    def stats(self):
        return {"block": None}


def test_library_gate_holds_the_fractions_at_zero_until_ready():
    cfg = load_config(str(ROOT / "configs" / "smoke.py"))
    stage = stage_at(0.1, cfg)
    assert stage.library_frac_base > 0 and stage.library_frac_chat > 0
    clock = FakeClock(0.0)
    ready = {"lib": None, "checks": 0}

    def is_ready():
        ready["checks"] += 1
        return ready["lib"]

    fl = FakeLoader()
    gate = SV.LibraryGate(fl, is_ready, check_every_s=30.0, clock=clock)
    for i in range(3):
        gate(i, stage)
        clock.advance(10.0)
    assert ready["checks"] == 1                            # throttled to one check per 30 s
    assert all(b == 0 and c == 0 for _, b, c in fl.calls)
    ready["lib"] = "LIB"
    gate(3, stage)                                         # 30 s since the last check
    assert fl.libs == ["LIB"] and gate.active and gate.activated_at_index == 3
    assert fl.calls[-1][1:] == (stage.library_frac_base, stage.library_frac_chat)
    clock.advance(100.0)
    gate(4, stage)
    assert fl.libs == ["LIB"]                              # set once, never swapped
    fl2 = FakeLoader()
    gate2 = SV.LibraryGate(fl2, lambda: pytest.fail("not asked"), active=True)
    gate2(0, stage)
    assert fl2.calls[0][1] == stage.library_frac_base and not fl2.libs


def test_library_self_check(tmp_path):
    from bes.library.index import IndexCfg
    from bes.library.retrieve import Library

    passages = [("DNA", "DNA is the molecule that carries genetic information."),
                ("Paris", "Paris is the capital city of France.")]
    Library.build(passages, str(tmp_path / "lib"), index_cfg=IndexCfg(budget_numbers=8000, n_bits=10,
                                                                     max_list=32))
    rep = SV.library_self_check(tmp_path / "lib", min_probe_hits=0)
    assert rep["ok"] and rep["n_wiki"] == 2 and rep["fingerprint"]["n_passages"] == 2
    rep = SV.library_self_check(tmp_path / "lib", cap=10, min_probe_hits=0)
    assert not rep["ok"] and "above the cap" in rep["reasons"][0]
    rep = SV.library_self_check(tmp_path / "missing")
    assert not rep["ok"] and rep["reasons"]
    real = ROOT / "data_cache" / "library" / "lib_v2"
    if (real / "library.json").exists():
        rep = SV.library_self_check(real)
        assert rep["ok"], rep["reasons"]
        assert rep["probe_hits"] >= 1 and rep["total_numbers"] <= 64_000_000


def _settings_library(path: Path, sources: tuple[str, ...], permissive: bool) -> Path:
    """A tiny working library (rule gate) with a qa layer whose entries come from ``sources`` and
    the permissive flag a build records in ``library.json``."""
    from bes.library import qa as Q
    from bes.library.index import IndexCfg
    from bes.library.pack import PackCfg
    from bes.library.retrieve import GateCfg, Library, RetrieveCfg

    docs = [("DNA", "DNA is the molecule that carries genetic information."),
            ("Napoleon", "Napoleon was a French military leader and emperor."),
            ("Paris", "Paris is the capital city of France.")]
    rows = [("Who wrote Hamlet?", "William Shakespeare"), ("What is the capital of Peru?", "Lima"),
            ("Who painted the Mona Lisa?", "Leonardo da Vinci")]
    pairs = [Q.QAPair(q, a, sources[i % len(sources)], i) for i, (q, a) in enumerate(rows)]
    store = Q.QAStore.build(pairs, Q.QACfg(min_cov_q=0.0), PackCfg(block_raw_bytes=128),
                            meta={"permissive_only": permissive})
    path.mkdir(parents=True, exist_ok=True)
    Library.build(docs, str(path), index_cfg=IndexCfg(budget_numbers=8000, n_bits=10, max_list=32),
                  cfg=RetrieveCfg(gate=GateCfg(min_score=0.5)), count_tokens=lambda s: len(s.split()),
                  meta={"build": {"permissive_only": permissive}}, qa=store)
    return path


def test_library_task_rebuilds_a_library_built_with_other_settings(tmp_path, monkeypatch):
    """library-3 / tools-6: a permissive-only run finds the library a non-permissive run (the
    canary) left behind; it is moved aside and rebuilt with --permissive-only, the rebuilt one
    records its settings, and a library whose recorded tokenizer or LibraryCfg differ is rebuilt
    too. Before: any library that passed the self-check was used as it was (TriviaQA and
    WebQuestions entries, licence unknown, in a permissive-only run)."""
    import runpy

    from bes.config import LibraryCfg

    monkeypatch.chdir(tmp_path)                       # LibraryCfg.library_dir is relative: here
    lib_dir = Path(LibraryCfg().library_dir)
    _settings_library(lib_dir, ("nq_open_train", "trivia_qa_train", "web_questions_train"), permissive=False)
    out = tmp_path / "run"
    SV.write_manifest(out, {"budget": {"t0": time.time()}})
    paths = SV.RunPaths(out)
    perm = load_config(str(ROOT / "configs" / "smoke.py"), ["data.permissive_only=true"])
    builds: list[list[str]] = []

    def fake_build(path, run_name=None):                # scripts/build_library.py, as the task calls it
        argv = list(sys.argv)
        builds.append(argv)
        dest = Path(argv[argv.index("--out") + 1])
        permissive = "--permissive-only" in argv
        _settings_library(dest, ("nq_open_train",) if permissive else
                          ("nq_open_train", "trivia_qa_train", "web_questions_train"), permissive)

    monkeypatch.setattr(runpy, "run_path", fake_build)
    cp = SV.core_plan(2)
    assert SV.task_library(paths, perm, cp, print) == 0
    assert len(builds) == 1 and "--permissive-only" in builds[0]
    aside = [p for p in lib_dir.parent.iterdir() if p.name.startswith("lib_v2.aside-")]
    assert len(aside) == 1 and SV.recorded_library_settings(aside[0])["permissive_only"] is False
    got = SV.recorded_library_settings(lib_dir)
    assert got["recorded"] and got["permissive_only"] is True and got["qa_stored"] == ["nq_open_train"]
    assert got["tokenizer_sha256"] == SV.library_settings(perm)["tokenizer_sha256"]
    mk = SV.read_marker(paths, "library")
    assert mk["path"] == str(lib_dir) and mk["settings"]["qa_stored"] == ["nq_open_train"]
    # the same settings again: used as it is
    assert SV.task_library(paths, perm, cp, print) == 0 and len(builds) == 1
    # a recorded tokenizer or LibraryCfg that differs from the run's: rebuilt
    rec = json.loads((lib_dir / SV.LIBRARY_SETTINGS).read_text())
    (lib_dir / SV.LIBRARY_SETTINGS).write_text(json.dumps({**rec, "tokenizer_sha256": "0" * 64}))
    assert any("tokenizer" in r for r in SV.library_settings_mismatch(lib_dir, SV.library_settings(perm)))
    assert SV.task_library(paths, perm, cp, print) == 0 and len(builds) == 2
    rec = json.loads((lib_dir / SV.LIBRARY_SETTINGS).read_text())
    (lib_dir / SV.LIBRARY_SETTINGS).write_text(json.dumps({**rec, "library_cfg": {**rec["library_cfg"],
                                                                                   "gate_precision": 0.1}}))
    assert any("LibraryCfg (fields ['gate_precision'])" in r
               for r in SV.library_settings_mismatch(lib_dir, SV.library_settings(perm)))
    assert SV.task_library(paths, perm, cp, print) == 0 and len(builds) == 3
    # a library built by hand (no record): reused when what it records matches, unless refused
    (lib_dir / SV.LIBRARY_SETTINGS).unlink()
    assert SV.task_library(paths, perm, cp, print) == 0 and len(builds) == 3
    strict = load_config(str(ROOT / "configs" / "smoke.py"), ["data.permissive_only=true",
                                                               "run.library_reuse_unrecorded=false"])
    assert SV.task_library(paths, strict, cp, print) == 0 and len(builds) == 4
    assert (lib_dir / SV.LIBRARY_SETTINGS).is_file()


# ------------------------------------------------------------------- sample labels
def test_label_sample_scores():
    lab = SV.label_sample("Explain gravity in exactly two sentences.",
                          "Gravity pulls masses together. It keeps planets in orbit.", "stop")
    assert lab["n_constraints"] >= 1 and lab["violations"] == [] and lab["score"] == 1.0
    lab = SV.label_sample("Explain gravity in exactly two sentences.",
                          "Gravity pulls masses together.", "stop")
    assert lab["violations"] and lab["score"] < 1.0
    assert SV.label_sample("Hi", "", "stop")["score"] == 0.0                      # empty
    assert SV.label_sample("Hi", "Hello there, how can I help?", "length")["score"] == 0.0  # cut off
    ok = SV.label_sample("What is 12 times 3?", "12 times 3 is 36. The answer is 36.", "stop", "36")
    bad = SV.label_sample("What is 12 times 3?", "The answer is 35.", "stop", "36")
    assert ok["math_correct"] and ok["score"] == 1.0
    assert bad["math_correct"] is False and bad["score"] == 0.0
    plain = SV.label_sample("Tell me about cats.", "Cats are small domestic animals.", "stop")
    assert plain["score"] == 1.0 and plain["n_constraints"] == 0


# ------------------------------------------------------------------ plans and checks
def test_download_plan():
    from bes.data import sources

    cfg = Cfg()
    plan = SV.download_plan(cfg)
    assert plan[0] == ("climbmix", 1, 1)
    first = [n for n, m, ph in plan if ph == 1]
    chat_like = {n for k in ("chat", "rag", "math") for n in sources.mixture_weights(k)}
    assert set(first) == {"climbmix"} | chat_like
    phase2 = [n for n, m, ph in plan if ph == 2]
    assert phase2[0] == "climbmix" and "ultrafeedback" in phase2
    assert all(m is None for n, m, ph in plan if ph == 2)
    perm = SV.download_plan(dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, permissive_only=True)))
    assert perm[0][0] == "fineweb_edu" and all(not sources.get(n).nonpermissive for n, _, _ in perm)
    smoke = load_config(str(ROOT / "configs" / "smoke.py"))
    assert SV.download_plan(smoke) == [("climbmix", 1, 1), ("smoltalk_constraints", 1, 1)]
    with pytest.raises(KeyError):
        SV.download_plan(dataclasses.replace(smoke, run=dataclasses.replace(smoke.run,
                                                                           download_sources="nope:1")))
    assert SV.disk_need_gb(cfg) > SV.disk_need_gb(smoke) > 0


def test_disk_estimate_is_stated_and_counts_the_raw_deletion():
    cfg = Cfg()
    on = SV.disk_need(cfg)
    off = SV.disk_need(dataclasses.replace(cfg, run=dataclasses.replace(cfg.run,
                                                                       delete_raw_after_tokenize=False)))
    for est in (on, off):
        parts = est["shards_gb"] + est["raw_left_gb"] + est["tokenize_temp_gb"] + est["other_gb"]
        assert est["total_gb"] == pytest.approx(parts, abs=0.01) and est["rule"]
        assert est["shards_gb"] == pytest.approx(1.4 * est["downloads_gb"], abs=0.01)
    assert off["raw_left_gb"] == off["downloads_gb"] and not off["delete_raw_after_tokenize"]
    assert on["delete_raw_after_tokenize"] and 0 < on["raw_left_gb"] < 0.1 * on["downloads_gb"]
    assert on["total_gb"] < off["total_gb"] and SV.disk_need_gb(cfg) == on["total_gb"]


def test_disk_estimate_counts_the_token_pages_of_v9():
    """configs/v9.py: its token pages (gain/o and page/*, the ledger rows "gain" and "page") are
    in every checkpoint the run can hold (run.keep_ckpts kept and one being written), 12 bytes
    per number (float32 and two Adam moments); configurations without pages keep the fixed
    allowance."""
    import functools

    import jax

    from bes import budget
    from bes.model import model as M

    v9 = load_config(str(ROOT / "configs" / "v9.py"))
    final = load_config(str(ROOT / "configs" / "final.py"))
    shapes = jax.eval_shape(functools.partial(M.init_params, v9.model), jax.random.PRNGKey(0))
    led = budget.count(shapes, v9.model).ledger()
    assert SV.page_numbers(v9.model) == led["gain"][0] + led["page"][0] == 63_504_384
    assert SV.page_numbers(final.model) == 0
    a, b = SV.disk_need(v9), SV.disk_need(final)
    assert b["other_gb"] == SV.DISK_OTHER_GB and "token pages" not in b["rule"]
    assert a["other_gb"] == pytest.approx(SV.DISK_OTHER_GB + 4 * 12 * 63_504_384 / 1e9, abs=1e-3)
    assert "token pages" in a["rule"]
    assert a["total_gb"] - b["total_gb"] == pytest.approx(a["other_gb"] - b["other_gb"], abs=0.01)


def _fake_source(data_dir, name, n_files, t_files, t0, sizes=None):
    """Raw files of ``name`` under ``data_dir`` as ``bes.data.download`` leaves them: a cached
    listing, the files, a download manifest recording file ``i`` at time ``t_files[i]``."""
    from bes.data import download, sources

    src = sources.get(name)
    prefix = download.static_prefix(src.files[0])
    stem = src.files[0].replace("*", "{:05d}")
    files = [download.RemoteFile(stem.format(i), (sizes or [100 + i] * n_files)[i]) for i in range(n_files)]
    download.source_listing(src, data_dir, lister=lambda repo, rev, pre: ("c0", files))
    rdir = download.repo_dir(src, data_dir)
    for f, t in zip(files, t_files):
        (rdir / f.path).parent.mkdir(parents=True, exist_ok=True)
        (rdir / f.path).write_bytes(b"x" * f.size)
        with open(rdir / download.MANIFEST, "a") as fh:
            fh.write(json.dumps({"repo": src.repo, "commit": "c0", "path": f.path, "size": f.size,
                                 "sha256": None, "time": time.strftime("%Y-%m-%dT%H:%M:%S",
                                                                       time.localtime(t))}) + "\n")
    return rdir, files


def test_a_file_two_sources_read_stays_until_both_have_their_shard(tmp_path):
    """synth and synth_rag read the same raw files. In the run of 23 Nov the one that finished
    first deleted them, the other got two shards of four, and on the next machine not the same
    two: the continued training could not find a shard of its journal."""
    from bes.data import download, shards, tokenize_worker

    out, data = tmp_path / "run", tmp_path / "data_cache"
    t0 = time.time() - 100.0
    SV.write_manifest(out, {"budget": {"t0": t0}})
    paths = SV.RunPaths(out)
    cfg = dataclasses.replace(Cfg(), run=dataclasses.replace(Cfg().run, data_dir=str(data),
                                                             tok_root="run"))
    root = SV.tok_root(cfg, out)
    rdir, files = _fake_source(data, "synth", 4, [t0 + 1] * 4, t0)
    assert download.repo_dir("synth_rag", str(data)) == rdir

    def commit(name, i):
        shards.append_manifest(tokenize_worker.tok_dir(name, root),
                               {"shard": i, "max_records": None,
                                "input": {"path": files[i].path, "size": files[i].size}})

    for i in range(4):
        commit("synth", i)
    assert SV.prune_raw(paths, cfg, "synth", print) == []            # synth_rag has none yet
    assert SV.prune_raw(paths, cfg, "synth_rag", print) == []
    commit("synth_rag", 0)
    commit("synth_rag", 1)
    assert SV.prune_raw(paths, cfg, "synth_rag", print) == [files[0].path, files[1].path]
    assert [(rdir / f.path).exists() for f in files] == [False, False, True, True]
    commit("synth_rag", 2)
    commit("synth_rag", 3)
    # the newest file stays for the sampling prompts of synth, whichever source prunes
    assert SV.prune_raw(paths, cfg, "synth_rag", print) == [files[2].path]
    assert SV.prune_raw(paths, cfg, "synth", print) == []
    assert (rdir / files[3].path).exists()


def test_the_marker_of_an_earlier_downloads_task_does_not_end_the_tokenisation(tmp_path):
    """A continued run starts the downloads again (on a recycled machine they fetch files
    again); the marker of the earlier process must not let the tokenisation end before them."""
    out = tmp_path / "run"
    SV.write_manifest(out, {"budget": {"t0": time.time()}})
    paths = SV.RunPaths(out)
    assert not SV._downloads_over(paths)
    SV.write_marker(paths, "downloads")                 # no process supervisor: the marker counts
    assert SV._downloads_over(paths)
    state = lambda **kw: SV.atomic_write_json(paths.background, {"tasks": {"downloads": kw}})  # noqa: E731
    state(state="waiting", started=None)
    assert not SV._downloads_over(paths)
    state(state="running", started=time.time() + 5)     # the marker is older than this task
    assert not SV._downloads_over(paths)
    SV.write_marker(paths, "downloads")
    state(state="running", started=time.time() - 5)
    assert SV._downloads_over(paths)
    state(state="finished", started=time.time())
    assert SV._downloads_over(paths)


def test_raw_files_are_deleted_once_their_whole_shard_is_committed(tmp_path, monkeypatch):
    from bes.data import download, shards, tokenize_worker

    out, data = tmp_path / "run", tmp_path / "data_cache"
    t0 = time.time() - 100.0
    SV.write_manifest(out, {"budget": {"t0": t0}})
    paths = SV.RunPaths(out)
    cfg = dataclasses.replace(Cfg(), run=dataclasses.replace(Cfg().run, data_dir=str(data),
                                                             tok_root="run"))
    root = SV.tok_root(cfg, out)
    # web text: 0 owned and whole, 1 owned but cut at max_records, 2 from before the run, 3 owned
    # without a shard yet
    rdir, files = _fake_source(data, "climbmix", 4, [t0 + 5, t0 + 6, t0 - 500, t0 + 7], t0)
    tdir = tokenize_worker.tok_dir("climbmix", root)
    for i, mr in ((0, None), (1, 3000), (2, None)):
        shards.append_manifest(tdir, {"shard": i, "max_records": mr,
                                      "input": {"path": files[i].path, "size": files[i].size}})
    # chat with messages: every file owned and whole; the newest stays for the sampling prompts
    cdir, cfiles = _fake_source(data, "smoltalk_constraints", 3, [t0 + 1] * 3, t0)
    for i, f in enumerate(cfiles):
        shards.append_manifest(tokenize_worker.tok_dir("smoltalk_constraints", root),
                               {"shard": i, "max_records": None,
                                "input": {"path": f.path, "size": f.size}})
    off = dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, delete_raw_after_tokenize=False))
    assert SV.prune_raw(paths, off, "climbmix", print) == []
    assert SV.prune_raw(paths, cfg, "climbmix", print) == [files[0].path]
    assert [(rdir / f.path).exists() for f in files] == [False, True, True, True]
    assert SV.prune_raw(paths, cfg, "smoltalk_constraints", print) == [f.path for f in cfiles[:2]]
    assert [(cdir / f.path).exists() for f in cfiles] == [False, False, True]
    # the downloads do not fetch a deleted file whose whole shard is committed again
    fetched = []
    monkeypatch.setattr(download, "download_file", lambda repo, commit, rf, d, **kw: fetched.append(rf.path))
    assert SV._download_source(paths, cfg, "climbmix", None, print) == 4
    assert sorted(fetched) == [f.path for f in files[1:]]
    called = []
    monkeypatch.setattr(download, "download_source", lambda *a, **kw: called.append(a) or [1, 2])
    assert SV._download_source(paths, off, "climbmix", None, print) == 2 and called


def test_check_devices():
    cpu = {"backend": "cpu", "jax": SV.PINNED_JAX, "devices": [{"kind": "cpu"}]}
    with pytest.raises(SV.PreflightError, match="no TPU"):
        SV.check_devices(cpu, allow_cpu=False)
    assert any("rehearsal" in w for w in SV.check_devices(cpu, allow_cpu=True))
    tpu = {"backend": "tpu", "jax": SV.PINNED_JAX, "devices": [{"kind": "TPU v6 lite"}]}
    assert SV.check_devices(tpu, allow_cpu=False) == []
    v5 = {"backend": "tpu", "jax": "0.4.0", "devices": [{"kind": "TPU v5 lite"}] * 4}
    w = SV.check_devices(v5, allow_cpu=False)
    assert len(w) == 3


def test_manifest_and_markers(tmp_path):
    paths = SV.RunPaths(tmp_path)
    SV.write_manifest(tmp_path, {"a": 1, "x": float("inf")})
    assert SV.read_manifest(tmp_path) == {"a": 1, "x": "inf"}
    SV.update_manifest(tmp_path, b={"c": Path("/q")})
    assert SV.read_manifest(tmp_path)["b"] == {"c": "/q"}
    SV.write_marker(paths, "library", path="/lib")
    assert SV.read_marker(paths, "library")["path"] == "/lib"
    assert SV.read_marker(paths, "none") is None
    cfg = load_config(str(ROOT / "configs" / "smoke.py"))
    assert SV.tok_root(cfg, tmp_path) == tmp_path / "data"
    assert SV.proxy_root(cfg, tmp_path) == tmp_path / "proxy"
    assert SV.tok_root(Cfg(), tmp_path) == Path("data_cache")
    assert SV.proxy_root(Cfg(), tmp_path) == Path("data_cache") / "proxy"
    assert SV.code_commit(ROOT)["commit"]


# ------------------------------------------------------------- the CPU rehearsal (slow)
def _wait_for(pred, timeout, every=1.0):
    t_end = time.time() + timeout
    while time.time() < t_end:
        if pred():
            return True
        time.sleep(every)
    return False


@pytest.mark.slow
def test_cpu_rehearsal_with_kill_and_resume(tmp_path):
    """scripts/run_24h.py on configs/smoke.py with a 12-minute budget: killed (SIGKILL) in the
    middle of training, resumed with --resume, every stage ends finished or skipped with a
    reason, the budget holds and the export exists."""
    out = tmp_path / "smoke"
    env = dict(os.environ, JAX_PLATFORMS="cpu", OMP_NUM_THREADS="2", PYTHONPATH=str(ROOT))
    # the whole process tree on two cores (the machine limits of this container)
    cores = ",".join(str(c) for c in sorted(os.sched_getaffinity(0))[-2:])
    pin = ["taskset", "-c", cores] if subprocess.run(["which", "taskset"], capture_output=True).returncode == 0 else []
    script = [PY, str(ROOT / "scripts" / "run_24h.py")]
    cmd = pin + script + ["--config", str(ROOT / "configs" / "smoke.py"), "--allow-cpu",
                          "--budget-minutes", "12", "--out", str(out),
                          # the raw samples in data_cache are shared: never delete them in a test
                          "--override", "run.delete_raw_after_tokenize=false"]
    log1 = open(tmp_path / "first.log", "w")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=log1, stderr=subprocess.STDOUT)

    def trained_a_while():
        ck = out / "train" / "ckpt"
        return ck.exists() and len([p for p in ck.iterdir() if p.name.startswith("ckpt-")]) >= 2

    assert _wait_for(trained_a_while, 9 * 60), (tmp_path / "first.log").read_text()[-3000:]
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    stages = json.loads((out / "stages.json").read_text())
    assert stages["stages"]["train"]["status"] == "started"
    log2 = open(tmp_path / "second.log", "w")
    res = subprocess.run(pin + script + ["--out", str(out), "--resume", "--allow-cpu"], cwd=str(ROOT),
                         env=env, stdout=log2, stderr=subprocess.STDOUT, timeout=20 * 60)
    text = (tmp_path / "second.log").read_text()
    assert res.returncode == 0, text[-4000:]
    stages = json.loads((out / "stages.json").read_text())
    st = stages["stages"]
    assert all(st[n]["status"] in ("finished", "skipped") for n in st), {n: st[n]["status"] for n in st}
    assert all(st[n]["reason"] for n in st if st[n]["status"] == "skipped")
    assert st["train"]["attempts"] == 2 and st["train"]["status"] == "finished"
    assert st["train"]["interrupted"] == 1 and st["train"]["down_s"] > 0
    led = json.loads((out / "train" / "ledger.json").read_text())
    assert led["restarts"] == 1 and led["seconds"]["lost"] > 0    # the same budget, time lost
    assert st["export"]["status"] == "finished"
    assert (out / "final").exists()
    man = json.loads((out / "manifest.json").read_text())
    assert man["benchmark"]["complete"] and man["benchmark"]["horizon_tokens"] > 0
    # the budget: the export finished within the budget plus at most the export's own overrun
    assert st["export"]["finished"] <= stages["deadline"] + 60.0


# ------------------------------------------------------------- scripts/run_24h.py (quick)
def _run24():
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_24h_under_test", ROOT / "scripts" / "run_24h.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_24h_knows_which_shards_of_the_journal_this_machine_lacks(tmp_path):
    """A training continued on a recycled machine reads the shards its loader journal lists;
    the train stage waits until the tokenisation has made them again (24 Nov: it did not, and
    the first batch failed with 'the continued document cannot be read')."""
    from bes.data import shards

    R = _run24()
    journal = tmp_path / "loader_journal.jsonl"
    assert R._journal_shards(journal) == {}
    recs = [{"type": "header", "settings": {}},
            {"type": "block", "id": 0, "new_shards": {"climbmix": [[0, 10, 1000, False, 5], [1, 12, 1200, False, 6]]}},
            {"type": "block", "id": 1, "new_shards": {}},
            {"type": "block", "id": 2, "new_shards": {"synth_rag": [[0, 7, 700, True, 9], [3, 8, 800, True, 9]]}}]
    journal.write_text("".join(json.dumps(r) + "\n" for r in recs) + '{"type": "block", "new_shards": {"torn')
    want = R._journal_shards(journal)
    assert {k: [e[0] for e in v] for k, v in want.items()} == {"climbmix": [0, 1], "synth_rag": [0, 3]}
    root = tmp_path / "data"
    assert R._missing_shards(root, want) == ["climbmix/0", "climbmix/1", "synth_rag/0", "synth_rag/3"]
    for name, sid, seg, tok in (("climbmix", 0, 10, 1000), ("climbmix", 1, 12, 1200), ("synth_rag", 3, 8, 800)):
        (root / "tok" / name).mkdir(parents=True, exist_ok=True)
        shards.append_manifest(root / "tok" / name, {"shard": sid, "n_segments": seg, "n_tokens": tok})
    assert R._missing_shards(root, want) == ["synth_rag/0"]
    shards.append_manifest(root / "tok" / "synth_rag", {"shard": 0, "n_segments": 7, "n_tokens": 701})
    with pytest.raises(RuntimeError, match="the data differs"):
        R._missing_shards(root, want)


def test_run_24h_takes_a_post_training_plan_changed_while_the_run_is_under_way(tmp_path):
    """24 Nov: sampling got through 3 % of its prompts in its 50 minutes and the judged comparison
    is redone outside the run, so their time went to training. The configuration (and its hash)
    stays; the plan comes from <run>/post_plan and travels in the manifest."""
    from bes.train import stages as ST

    R = _run24()
    out = tmp_path / "run"
    SV.write_manifest(out, {"budget": {"t0": 0.0}})
    run_cfg = dataclasses.replace(Cfg().run, post_reserve_hours=3.6667, post_minutes="critic=0,buffer=20")
    said = []
    assert R.post_plan_run(run_cfg, out, said.append) is run_cfg and not said
    (out / "post_plan").write_text("sample=3 critic=0 prefopt=60\nselect=25 judge=0 export=5 buffer=15\n")
    got = R.post_plan_run(run_cfg, out, said.append)
    assert got.post_reserve_hours == pytest.approx(108 / 60) and "108 minutes" in said[-1]
    assert SV.read_manifest(out)["post_plan"] == "sample=3 critic=0 prefopt=60 select=25 judge=0 export=5 buffer=15"
    specs = ST.stage_specs(got, 20 * 3600.0)
    assert specs["judge"].planned_s == 0 and specs["sample"].planned_s == pytest.approx(180.0)
    assert specs["prefopt"].planned_s == pytest.approx(3600.0)
    old = ST.stage_specs(run_cfg, 20 * 3600.0)
    assert specs["train"].planned_s - old["train"].planned_s == pytest.approx(3.6667 * 3600 - 108 * 60)
    (out / "post_plan").unlink()                         # the next machine: the manifest carries it
    assert R.post_plan_run(run_cfg, out).post_reserve_hours == pytest.approx(1.8)
    (out / "post_plan").write_text("sample=many")        # not a plan: the earlier one stays
    assert R.post_plan_run(run_cfg, out, said.append).post_reserve_hours == pytest.approx(1.8)
    assert any("ignored" in m for m in said)


def test_run_24h_applies_the_speed_switches_of_a_run(tmp_path):
    """The kernel and the hand-written backward passes a run's programs are traced with come
    from <run>/fast, travel in the manifest, and are applied before any stage traces a program;
    a text that is not a setting changes nothing."""
    from bes.model import fast

    R = _run24()
    out = tmp_path / "run"
    SV.write_manifest(out, {"budget": {"t0": 0.0}})
    before, said = fast.current(), []
    assert before == {"kernel": "", **{name: False for name in fast.FLAGS}}
    try:
        assert R.fast_switches_run(out, said.append) == "" and not said and fast.current() == before
        (out / "fast").write_text("kernel=onetile:save  relu2=1\nce=1\n")
        assert R.fast_switches_run(out, said.append) == "kernel=onetile:save relu2=1 ce=1"
        assert fast.current() == {**before, "kernel": "onetile:save", "relu2": True, "ce": True}
        assert SV.read_manifest(out)["fast"] == "kernel=onetile:save relu2=1 ce=1" and "onetile:save" in said[-1]
        fast.apply(before)
        (out / "fast").unlink()                              # the next machine: the manifest carries it
        assert R.fast_switches_run(out) == "kernel=onetile:save relu2=1 ce=1"
        assert fast.current()["kernel"] == "onetile:save" and fast.current()["ce"]
        fast.apply(before)
        for bad in ("kernel=warp9", "relu2=maybe", "turbo=1", "relu2"):
            (out / "fast").write_text(bad)                   # not a setting: the earlier one stays
            assert R.fast_switches_run(out, said.append) == "kernel=onetile:save relu2=1 ce=1"
            assert "ignored" in said[-2] and bad.split("=")[0] in said[-2]
            fast.apply(before)
        (out / "fast").write_text("kernel= relu2=0 ce=0")    # back to the reference path
        assert R.fast_switches_run(out, said.append) == "kernel= relu2=0 ce=0" and fast.current() == before
    finally:
        fast.apply(before)
    assert fast.describe({"relu2": True, "kernel": "blocks:kvc=256,sep"}) == "kernel=blocks:kvc=256,sep relu2=1"
    assert fast.parse("kernel=onetile:bs=128,hp=1 io=on") == {"kernel": "onetile:bs=128,hp=1", "io": True}


def test_run_24h_refuses_a_second_run_and_a_resume_without_a_run(tmp_path):
    R = _run24()
    smoke = str(ROOT / "configs" / "smoke.py")
    with pytest.raises(SystemExit, match="no run"):
        R.main(["--out", str(tmp_path / "a"), "--resume", "--allow-cpu"])
    out = tmp_path / "b"
    args = R.parse_args(["--config", smoke, "--out", str(out), "--budget-minutes", "12"])
    man = R.new_manifest(args, out)
    assert man["overrides"] == ["run.deadline_hours=0.2"]
    assert man["budget"]["budget_s"] == pytest.approx(720.0)
    assert man["budget"]["deadline"] - man["budget"]["t0"] == pytest.approx(720.0)
    assert man["config_hash"] and man["code"]["commit"]
    SV.write_manifest(out, man)
    with pytest.raises(SystemExit, match="already holds a run"):
        R.main(["--config", smoke, "--out", str(out), "--allow-cpu"])
    with pytest.raises(SystemExit, match="budget of a run is fixed"):
        R.main(["--out", str(out), "--resume", "--allow-cpu", "--budget-minutes", "30"])
    with pytest.raises(SystemExit, match="post-training reserve"):
        R.new_manifest(R.parse_args(["--config", smoke, "--budget-minutes", "2"]), out)
    # the run lock: a live supervisor in the directory refuses another one
    lock = SV.RunLock(SV.RunPaths(out).lock).acquire()
    try:
        with pytest.raises(SV.RunLocked):
            R.main(["--out", str(out), "--resume", "--allow-cpu"])
    finally:
        lock.release()


def test_run_24h_persistent_compilation_cache_is_guarded(tmp_path, monkeypatch):
    """The cache is enabled under <run>/jax_cache; an option this JAX does not know is logged and
    the run goes on without the cache (before: the exception stopped the supervisor)."""
    import jax
    from jax._src import compilation_cache as cc

    R = _run24()
    old = (jax.config.jax_compilation_cache_dir, jax.config.jax_persistent_cache_min_compile_time_secs)
    said = []
    try:
        info = R.configure_jax(SV.RunPaths(tmp_path / "a"), log=said.append)
        assert info["enabled"] is True and (tmp_path / "a" / "jax_cache").is_dir()
        assert jax.config.jax_compilation_cache_dir == str(tmp_path / "a" / "jax_cache")
        real = jax.config.update

        def update(name, value):
            if name == "jax_persistent_cache_min_compile_time_secs":
                raise AttributeError("unrecognized config option")
            return real(name, value)

        monkeypatch.setattr(jax.config, "update", update)
        info = R.configure_jax(SV.RunPaths(tmp_path / "b"), log=said.append)
        assert info["enabled"] is False and "unrecognized" in info["reason"]
        assert jax.config.jax_compilation_cache_dir is None
        assert any("going on without it" in m for m in said)
    finally:
        monkeypatch.undo()
        jax.config.update("jax_compilation_cache_dir", old[0])
        jax.config.update("jax_persistent_cache_min_compile_time_secs", old[1])
        cc.reset_cache()


def test_run_24h_finds_a_parameters_only_final_checkpoint(tmp_path):
    import numpy as np

    from bes.train import checkpoint as CK

    R = _run24()
    run = R.Run.__new__(R.Run)
    run.paths = SV.RunPaths(tmp_path)
    tree = {"state": {"params": {"a": np.zeros(2, np.float32)}}}
    CK.save(tmp_path / "train" / "ckpt", tree, step=30, config_hash="h")
    assert run.train_checkpoint().name == "ckpt-000001-step-000000030"
    CK.save(tmp_path / "train" / "ckpt_params", tree, step=40, config_hash="h")
    assert run.train_checkpoint().parent.name == "ckpt_params"     # the later final state
    CK.save(tmp_path / "train" / "ckpt", tree, step=40, config_hash="h")
    assert run.train_checkpoint().parent.name == "ckpt"            # a tie: the full one


def test_run_24h_health_samples_are_paired_budgeted_and_use_the_decode_settings(monkeypatch):
    """The DPO health sampler: the start's call stops at its time budget (at least min_prompts),
    every later call samples exactly those prompts with the same seeds (paired statistics), at
    the run's own reply limit, with the run's temperature and self-evidence weight."""
    import types

    from bes.decode import lm as LM
    from bes.decode import smc
    from bes.model import generate as G

    R = _run24()
    cfg = load_config(str(ROOT / "configs" / "smoke.py"))
    made, calls = [], []

    class FakeLM:
        def __init__(self, params, consts, model_cfg, tok, **kw):
            made.append(kw)

    def fake_decode(lm, prompt, **kw):
        calls.append((tuple(prompt), kw["seed"], kw["settings"].temperature, kw["max_new_tokens"]))
        n = 3 + len(prompt) % 5
        return types.SimpleNamespace(particles=[types.SimpleNamespace(tokens=list(range(n)))])

    monkeypatch.setattr(LM, "OurLM", FakeLM)
    monkeypatch.setattr(smc, "decode", fake_decode)
    monkeypatch.setattr(G, "jitted", lambda model_cfg: "decoder")
    run = R.Run.__new__(R.Run)
    run._tok = object()
    prompts = [[1] * (i + 1) for i in range(10)]
    fn = run._sample_stats_fn(cfg, {}, prompts, cfg.decode.max_new_tokens, budget_s=0.0,
                              min_prompts=3)
    start, cand = fn("p0"), fn("p1")
    assert start["n"] == cand["n"] == 3 and start["max_new"] == cfg.decode.max_new_tokens
    assert [c[:2] for c in calls[:3]] == [c[:2] for c in calls[3:]] == \
        [(tuple(prompts[i]), 7 + i) for i in range(3)]
    assert all(c[2] == cfg.decode.temperature and c[3] == cfg.decode.max_new_tokens for c in calls)
    assert all(kw["self_evidence_weight"] == cfg.decode.self_evidence_weight for kw in made)
    assert start["len"] == cand["len"]


def test_run_24h_sampling_prompts_come_from_the_held_out_split_first(tmp_path, monkeypatch):
    """The preference-sampling prompts are the loader's held-out conversations (never trained
    on); a source with too few of them is topped up from its raw rows, marked heldout=False
    (before: every prompt came from the training universe)."""
    from bes.data import download, loader as L, samples as SM
    from bes.tokenizer import load as load_tokenizer

    R = _run24()
    cfg = load_config(str(ROOT / "configs" / "smoke.py"))     # one chat source with messages
    cfg = dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, data_dir=str(tmp_path / "dc")))
    tok = load_tokenizer()
    held = [SM.conversation_sample([{"role": "user", "content": f"Tell me about cats, part {k}."},
                                    {"role": "assistant", "content": "Cats purr."}], tok)
            for k in range(3)]
    asked = []
    monkeypatch.setattr(L, "heldout_samples",
                        lambda root, name, n, seed=0, **kw: asked.append((Path(root), name, n)) or held)
    raw = [{"messages": [{"role": "user", "content": f"What is {k} plus {k}?"},
                         {"role": "assistant", "content": f"It is {2 * k}."}]} for k in range(9)]
    monkeypatch.setattr(download, "completed_files", lambda name, root: [download.LocalFile(
        name, 0, "r", "c", "f.parquet", str(tmp_path / "f.parquet"), 1)])
    monkeypatch.setattr(download, "iter_records", lambda path, cols, batch_rows=256: iter([raw]))
    run = R.Run.__new__(R.Run)
    run.cfg, run.paths, run._tok, run.sup = cfg, SV.RunPaths(tmp_path / "run"), tok, None
    items = run._sample_prompts(8)
    assert asked == [(SV.tok_root(cfg, tmp_path / "run"), "smoltalk_constraints", 8)]
    assert [it["heldout"] for it in items] == [True] * 3 + [False] * 5
    assert len({it["id"] for it in items}) == 8 and items[0]["id"].startswith("smoltalk_constraints/heldout/")
    assert items[0]["messages"][-1]["content"].startswith("Tell me about cats") or \
        items[0]["kind"] == "constraint"
    assert items[3]["id"] == "smoltalk_constraints/0/0" and "plus" in items[3]["messages"][-1]["content"]


def _long_constraint_reply() -> tuple[str, str]:
    """A user message with an 'exactly 7 paragraphs' constraint and a reply of more than 2000
    characters that satisfies it, its seventh paragraph after character 2000."""
    import random

    words = ("stone tower raven moon gate hall candle shadow wind chapel ivy lantern bell crypt banner "
             "river forest mist owl thunder rain iron oak velvet portrait staircase cellar wolf frost ember").split()
    rng = random.Random(0)
    paras = []
    for _ in range(7):
        paras.append(" ".join(
            "The {} near the {} watched the {} while the {} and the {} waited by the {}.".format(*rng.sample(words, 6))
            for _ in range(5)))
    reply = "\n\n".join(paras)
    assert len(reply) > 2000 and reply.index(paras[6]) > 2000
    return "Describe a gothic castle at night in exactly 7 paragraphs.", reply


def test_run_24h_sampling_labels_every_response_on_the_text_of_its_tokens(tmp_path, monkeypatch):
    """data-2: the data set's own reply is labelled on the whole text of the tokens a preference
    pair is built from. Before: on its first 2000 characters, so a correct reply of 7 paragraphs
    scored 0 and became the rejected side against a particle that scored 1."""
    import types

    from bes.config import END
    from bes.data import prefs
    from bes.data.chat_format import display_text
    from bes.decode import lm as LM
    from bes.decode import smc
    from bes.model import generate as G
    from bes.tokenizer import load as load_tokenizer

    R = _run24()
    tok = load_tokenizer()
    # a reply the budget cut: seven paragraphs, the last one unfinished, END in the last slot
    cut_text = "\n\n".join(f"Paragraph {i} of the old castle on the hill." for i in range(1, 7)) + \
        "\n\nParagraph 7 of the old castle on"
    cut = tok.encode(cut_text) + [END]
    cfg = load_config(str(ROOT / "configs" / "smoke.py"), [f"decode.max_new_tokens={len(cut)}"])
    user, reply = _long_constraint_reply()
    short7 = "\n\n".join(f"Paragraph {i} of the castle." for i in range(1, 8))
    item = {"id": "smoltalk_constraints/0/0", "source": "smoltalk_constraints", "heldout": False,
            "messages": [{"role": "user", "content": user}], "reference": None, "kind": "constraint",
            "reply": reply}
    particles = [types.SimpleNamespace(tokens=tok.encode(short7) + [END], finish="stop"),
                 types.SimpleNamespace(tokens=tok.encode("One paragraph only.") + [END], finish="stop"),
                 types.SimpleNamespace(tokens=cut, finish="stop")]
    assert SV.label_sample(user, cut_text, "stop")["score"] == 1.0      # only the budget shows the cut
    from bes.decode import respond

    monkeypatch.setattr(G, "jitted", lambda model_cfg: "decoder")
    monkeypatch.setattr(LM, "OurLM", lambda *a, **k: object())
    monkeypatch.setattr(smc, "decode", lambda lm, prompt, **kw: types.SimpleNamespace(particles=particles))
    asked = []                                          # retrieval as Responder.reply does it
    monkeypatch.setattr(respond.Responder, "retrieve",
                        lambda self, text, history=None: asked.append((text, history)) or None)
    run = R.Run.__new__(R.Run)
    run.cfg, run.paths, run._tok, run.sup, run.relabelled = cfg, SV.RunPaths(tmp_path), tok, None, 0
    run.man = {}
    run.effective = lambda: cfg
    run.latest_params = lambda ecfg, prefer_post=True: (None, None, {"path": "ckpt", "step": 1}, None, "train")
    run.library_ready = lambda: None
    run._sample_prompts = lambda limit: [item]
    ctx = types.SimpleNamespace(should_stop=lambda: False, check=lambda: None)
    run.stage_sample(ctx)
    rec = json.loads((tmp_path / "post" / "samples.jsonl").read_text().splitlines()[0])
    ref = next(p for p in rec["particles"] if p["origin"] == "reference")
    full = display_text(ref["tokens"], tok).strip()
    assert ref["score"] == SV.label_sample(user, full, "stop")["score"] == 1.0
    assert [p["score"] for p in rec["particles"]] == [1.0, 0.0, 0.0, 1.0]
    assert rec["particles"][2]["finish"] == "length"     # as Responder.reply reports a reply at its budget
    assert asked == [(user, [])]                         # the earlier user turns (none here)
    assert len(full) > 2000 and len(ref["text"]) == 2000 and ref["text_chars"] == len(full)
    for p in rec["particles"]:                         # every score is the label of its own tokens
        assert p["text_sha1"] == R._text_sha1(display_text(p["tokens"], tok).strip())
        assert p["score"] == SV.label_sample(user, display_text(p["tokens"], tok).strip(), p["finish"])["score"]
    pairs = prefs.pairs_from_scores(rec["prompt"], [p["tokens"] for p in rec["particles"]],
                                    [p["score"] for p in rec["particles"]], source="sampled")
    assert pairs and all(list(q.rejected) != ref["tokens"] for q in pairs)


def test_run_24h_stale_sample_labels_are_computed_again_from_the_tokens():
    """data-2: a record written before labels were tied to the tokens (no text_sha1; the reference
    scored on a 2000-character cut) is labelled again from its tokens before any pair is built."""
    from bes.data import prefs
    from bes.data.chat_format import display_text
    from bes.tokenizer import load as load_tokenizer

    R = _run24()
    tok = load_tokenizer()
    user, reply = _long_constraint_reply()
    ids = tok.encode(reply)
    good = tok.encode("\n\n".join(f"Paragraph {i}." for i in range(1, 8)))
    old = {"user": user, "reference": None, "prompt": [1, 2, 3], "particles": [
        {"tokens": good, "score": 1.0, "finish": "stop", "origin": "particle"},
        {"tokens": ids, "score": 0.0, "finish": "stop", "origin": "reference", "text": reply[:2000]}]}
    show = lambda t: display_text(t, tok)                                   # noqa: E731
    assert prefs.pairs_from_scores(old["prompt"], [p["tokens"] for p in old["particles"]],
                                   [p["score"] for p in old["particles"]])          # the wrong pair
    fixed, n = R._scored_samples([old], show)
    assert n == 2 and [p["score"] for p in fixed[0]["particles"]] == [1.0, 1.0]
    assert not prefs.pairs_from_scores(old["prompt"], [p["tokens"] for p in fixed[0]["particles"]],
                                       [p["score"] for p in fixed[0]["particles"]])
    again, n = R._scored_samples(fixed, show)          # labels that match their tokens are kept
    assert n == 0 and again == fixed


def _decision_run(R, tmp_path, cfg, lib):
    run = R.Run.__new__(R.Run)
    run.cfg, run.paths, run.sup, run.man = cfg, SV.RunPaths(tmp_path / "run"), None, {}
    run.log = lambda m: None
    run.library_path = lambda: "lib" if lib is not None else ""
    run.library_ready = lambda: lib
    return run


def _ctx(deadline: float):
    import types

    return types.SimpleNamespace(deadline=deadline, now=time.time, stop_requested=lambda: False)


def test_run_24h_library_ships_only_on_a_significant_factual_gain_without_a_dev_loss(tmp_path, monkeypatch):
    """The decode selection's library decision (plan section 2, EVAL_PROTOCOL sections 8 and 10)
    with the real factual measurement (bes.eval.proxy.factual) driven by its stand-in
    generator: shipped only when the library helps significantly more questions than it hurts
    and the dev exact score does not drop; no measurement means no library (before: chosen by a
    dev exact score that cannot see retrieval, ties kept the library)."""
    import types

    from bes.eval.proxy import factual as F

    R = _run24()
    cfg = load_config(str(ROOT / "configs" / "smoke.py"))
    qs = [F.Question(F.question_id("popqa", f"Who founded city {i}?"), "popqa", f"Who founded city {i}?",
                     (f"founder {i}",)) for i in range(30)]
    qfile = tmp_path / "questions.jsonl"
    F.save_questions(qfile, qs, {"test": True})

    def measure(behaviour: dict, scores: dict, name: str):
        (tmp_path / f"{name}.json").write_text(json.dumps(behaviour))
        gen = F.standin_factual(str(qfile), behaviour=str(tmp_path / f"{name}.json"), key=f"standin-{name}")
        monkeypatch.setattr(F, "prepare", lambda root, **kw: (qfile, list(qs)))
        lib = types.SimpleNamespace(retrieve=lambda q: "passage" if behaviour[q]["fires"] else None)
        run = _decision_run(R, tmp_path / name, cfg, lib)
        return run._library_decision(_ctx(time.time() + 600), cfg, gen, tmp_path / name / "proxy",
                                     R._decode_settings(cfg), scores, 4)

    # the library answers 20 questions the model cannot, misleads on 2 (one it knew): shipped
    helps = {q.q: {"memory": False, "fires": i < 21, "misleads": i == 20} for i, q in enumerate(qs)}
    helps[qs[29].q] = {"memory": True, "fires": True, "misleads": True}      # hurt: right only without
    dec = measure(helps, {"default": 0.6, "nolib": 0.6, "k1": 0.5}, "helps")
    assert dec["ship"] is True, dec["reason"]
    fa = dec["factual"]
    assert fa["n_paired"] == 30 and fa["helped"] == 20 and fa["hurt"] == 1 and fa["p"] < 0.05
    assert fa["fired"] == 22 and fa["order"] == "gate-fired questions first"
    assert (tmp_path / "helps" / "proxy" / "reports" / "factual" / "run-factual.md").is_file()
    # the same factual gain but a lower dev exact score with the library: not shipped
    dec = measure(helps, {"default": 0.55, "nolib": 0.6}, "devloss")
    assert dec["ship"] is False and "dev exact score lower with the library" in dec["reason"]
    # a small gain is no evidence: not shipped (ties never keep the library)
    small = {q.q: {"memory": False, "fires": i < 3, "misleads": i == 2} for i, q in enumerate(qs)}
    dec = measure(small, {"default": 0.6, "nolib": 0.6}, "small")
    assert dec["ship"] is False and "no significant factual gain (helped 2, hurt 0" in dec["reason"]
    # no measurement (it fails, or no time is left): not shipped, the error is recorded
    run = _decision_run(R, tmp_path / "fail", cfg, types.SimpleNamespace(retrieve=lambda q: None))
    monkeypatch.setattr(F, "prepare", lambda root, **kw: (_ for _ in ()).throw(OSError("no evaluation files")))
    dec = run._library_decision(_ctx(time.time() + 600), cfg, None, tmp_path, [], {"default": 1.0, "nolib": 0.0}, 4)
    assert dec["ship"] is False and "no evaluation files" in dec["factual_error"]
    dec = run._library_decision(_ctx(time.time() - 1), cfg, None, tmp_path, [], {"default": 1.0, "nolib": 0.0}, 4)
    assert dec["ship"] is False and "no time left" in dec["factual_error"]
    # no library at all
    dec = _decision_run(R, tmp_path / "none", cfg, None)._library_decision(_ctx(time.time() + 600), cfg, None,
                                                                          tmp_path, [], {}, 0)
    assert dec == {**dec, "ship": False, "reason": "no library was built"}


def test_run_24h_preflight_refuses_data_on_a_network_drive(tmp_path, monkeypatch):
    """tools-1: the preflight measures and checks where the data goes (the data root, the token
    shards, the library, relative to the code), not only the run directory: on a TPU a data path
    on the Drive mount stops the run before anything is written there (before: the check looked
    at --out only, which sat on the local disk while 72-111 GB went to Drive)."""
    import types

    R = _run24()
    drive = tmp_path / "drive"
    monkeypatch.setattr(SV, "DRIVE_PREFIXES", (str(drive),))
    tpu = {"backend": "tpu", "jax": SV.PINNED_JAX, "devices": [{"kind": "TPU v6 lite"}]}
    monkeypatch.setattr(SV, "device_report", lambda: tpu)
    cfg = load_config(str(ROOT / "configs" / "smoke.py"), [f"run.data_dir={drive / 'data_cache'}"])
    run = R.Run.__new__(R.Run)
    run.args, run.cfg, run.paths, run.sup = types.SimpleNamespace(allow_cpu=False), cfg, SV.RunPaths(tmp_path / "run"), None
    SV.write_manifest(run.paths.out, {"budget": {"t0": time.time()}})
    assert SV.on_drive(drive / "data_cache" / "tok") and not SV.on_drive(tmp_path / "run")
    with pytest.raises(SV.PreflightError, match="network drive"):
        run.stage_preflight(None)
    monkeypatch.setattr(SV, "device_report", lambda: {**tpu, "backend": "cpu", "devices": [{"kind": "cpu"}]})
    run.args.allow_cpu = True
    out = run.stage_preflight(None)                   # a CPU rehearsal: a warning, recorded
    assert any("network drive" in w for w in out["warnings"])
    man = SV.read_manifest(run.paths.out)
    assert set(man["host"]["data_free_gb"]) == {"run", "data", "tok"} and man["disk_need_gb"] > 0


def test_run_24h_resumed_training_waits_for_the_library_its_journal_used(tmp_path):
    """The Drive mirror holds no data (tools-1), so on a recycled machine the library is built
    again; a training whose loader journal planned blocks with the library waits for it instead of
    failing (a loader cannot reopen such a journal without the same library), and a ready marker
    whose library is not on this disk is 'not ready', not an error."""
    import types

    R = _run24()
    journal = tmp_path / "loader_journal.jsonl"
    assert R._journal_used_library(journal) is False
    journal.write_text(json.dumps({"type": "header", "settings": {}}) + "\n"
                       + json.dumps({"type": "block", "id": 0, "library": None}) + "\n")
    assert R._journal_used_library(journal) is False
    with open(journal, "a") as f:
        f.write(json.dumps({"type": "block", "id": 1, "library": {"n_passages": 3}}) + "\n" + '{"type": "blo')
    assert R._journal_used_library(journal) is True
    run = R.Run.__new__(R.Run)
    run.cfg, run.paths, run.sup = load_config(str(ROOT / "configs" / "smoke.py")), SV.RunPaths(tmp_path / "run"), None
    SV.write_marker(run.paths, "library", path=str(tmp_path / "gone"))       # the old machine's library
    assert run.library_path() == "" and run.library_ready() is None
    calls = {"n": 0}
    lib = object()

    def ready():
        calls["n"] += 1
        return lib if calls["n"] >= 3 else None

    run.library_ready = ready
    run.bg = types.SimpleNamespace(status=lambda: {"library": {"state": "running"}})
    ctx = types.SimpleNamespace(check=lambda: None, now=time.time, hard_deadline=time.time() + 600)
    assert run._wait_for_library(ctx, poll_s=0.01) is lib and calls["n"] == 3
    run.library_ready = lambda: None
    run.bg = types.SimpleNamespace(status=lambda: {"library": {"state": "failed", "error": "exit code 1"}})
    with pytest.raises(RuntimeError, match="library task failed"):
        run._wait_for_library(ctx, poll_s=0.01)


def test_run_24h_helpers(tmp_path):
    R = _run24()
    from bes.config import END, PAD
    from bes.tokenizer import load as load_tokenizer

    tok = load_tokenizer()
    ref = R._reference_response({"reply": "Paris is the capital of France."}, tok)
    assert ref["text"] == "Paris is the capital of France."
    assert END not in ref["tokens"] and PAD not in ref["tokens"]
    calc = R._reference_response({"reply": [{"type": "text", "text": "So "},
                                            {"type": "calc", "expr": "2*3", "result": "6"},
                                            {"type": "text", "text": " apples."}]}, tok)
    assert calc["text"] == "So 2*3 = 6 apples."
    assert R._reference_response({"reply": ""}, tok) is None
    cfg = load_config(str(ROOT / "configs" / "smoke.py"))
    assert R._decode_settings(cfg)[:3] == ["extra.particles=4", "extra.use_library=true",
                                           "max_tokens=64"]
    root = tmp_path / "proxy"
    rep = root / "reports" / "exact" / "k__run-dev.json"
    rep.parent.mkdir(parents=True)
    rep.write_text(json.dumps({"per_prompt": [
        {"prompt_id": "m1", "math_correct": True, "constraints_ok": None,
         "turns": [{"degenerate": {"degenerate": False}}]},
        {"prompt_id": "c1", "math_correct": None, "constraints_ok": False,
         "turns": [{"degenerate": {"degenerate": True}}]}]}))
    assert R._per_prompt_scores(root, "k/run", "dev") == {"m1": 1.0, "c1": 0.0}
