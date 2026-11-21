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
    assert groups == ["sink", "conf", "ngram", "library", "tilt", "registers", "calc"]
    assert SV.ladder_rung(0) == ({}, [], [])
    sw, ov, g = SV.ladder_rung(4)
    assert g == groups[:4] and sw == {"sink": 0.0, "conf": 0.0, "cand_in": 0.0}
    assert ov == ["data.library_frac_base=0", "data.library_frac_chat=0"]
    sw, _, _ = SV.ladder_rung(len(SV.LADDER))
    M = pytest.importorskip("bes.model.model")
    # the whole ladder switches off every path of the safe configuration
    safe = SV.safe_switches()
    assert set(M.SWITCHES) <= set(sw) | {"kind"} and all(v == 0 for v in sw.values())
    assert set(safe) == set(M.SWITCHES) | {"kind"}
    cfg = load_config(str(ROOT / "configs" / "safe.py"))
    assert SV.parse_switches(cfg.run.switches) == safe
    assert cfg.data.library_frac_base == 0 and cfg.data.library_frac_chat == 0
    final = load_config(str(ROOT / "configs" / "final.py"))
    assert cfg.model == final.model and cfg.optim == final.optim      # same graph, same tree


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
                          "--budget-minutes", "12", "--out", str(out)]
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
