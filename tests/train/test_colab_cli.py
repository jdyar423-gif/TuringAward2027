"""The Colab CLI route: ``scripts/colab_tpu.py`` (the program sent to the kernel) and
``scripts/colab_cli.py`` (the wrapper around the ``colab`` command).

Nothing here talks to Colab. The supervisor runs against a stand-in repository (the real
notebooks, a stand-in ``scripts/run_24h.py`` steered through gate files); the wrapper runs
against a stand-in ``colab`` executable that keeps the runtime's files in a temporary directory
and executes ``colab exec -f`` the way the CLI does (the ``--env`` lines in front of the text)."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import textwrap
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TPU = REPO / "scripts" / "colab_tpu.py"
CLI = REPO / "scripts" / "colab_cli.py"
USED_CELLS = ("1", "1b", "2", "3", "4", "6", "8", "12")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tpu = _load(TPU, "bes_colab_tpu")
cli = _load(CLI, "bes_colab_cli_wrapper")

STANDIN_RUN = '''
"""Stand-in for scripts/run_24h.py: the files the supervisor reads, steered by gate files."""
import json, os, signal, sys, time

args = sys.argv[1:]
out = args[args.index("--out") + 1]
runs = os.path.dirname(out)
resume = "--resume" in args
ORDER = ("preflight", "background", "benchmark", "train", "sample", "critic", "prefopt", "select",
         "judge", "export")
manifest = os.path.join(out, "manifest.json")
if "--move-deadline" in args:                     # the edit-only call: recorded, nothing runs
    with open(os.path.join(runs, "moved.jsonl"), "a") as f:
        f.write(json.dumps({"out": out, "when": args[args.index("--move-deadline") + 1],
                            "run": os.path.exists(manifest)}) + "\\n")
    print("stand-in: deadline moved to", args[args.index("--move-deadline") + 1])
    sys.exit(0 if os.path.exists(manifest) else 1)
# the handler first: a test may signal as soon as it sees the attempt recorded below
if os.environ.get("STANDIN_DEAF"):
    signal.signal(signal.SIGTERM, lambda *a: open(os.path.join(runs, "sigterms"), "a").write("x"))
else:
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(3))
if resume and not os.path.exists(manifest):
    sys.exit("--resume: no run")
if not resume and os.path.exists(manifest):
    sys.exit("already holds a run")
os.makedirs(os.path.join(out, "train"), exist_ok=True)
with open(os.path.join(runs, "attempts.jsonl"), "a") as f:
    f.write(json.dumps(args) + "\\n")
with open(os.path.join(runs, "attempts.jsonl")) as f:
    n = sum(1 for _ in f)


def dump(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f)
    os.replace(path + ".tmp", path)


def stages(done, cur=None, failed=()):
    dump(os.path.join(out, "stages.json"), {"stages": {
        name: {"status": "failed" if name in failed else "finished" if name in done else
               "started" if name == cur else "pending",
               "planned_s": 60.0, "used_s": 1.0, "reason": ""} for name in ORDER}})


now = time.time()
if not resume:
    dump(manifest, {"created": now, "config": args[args.index("--config") + 1],
                    "budget": {"t0": now, "deadline": now + 1800.0, "budget_s": 1800.0}})
stages(ORDER[:3], "train")
dump(os.path.join(out, "train", "ledger.json"), {"tokens": 123456 * n, "steps": 10 * n, "seconds": {"train": 5.0, "lost": 0.0},
                                                  "restarts": n - 1, "rollbacks": 0})
with open(os.path.join(out, "train", "train_log.jsonl"), "a") as f:
    f.write(json.dumps({"event": "step", "step": 10 * n, "p": 0.1, "seq_len": 64, "loss": 5.0 - n,
                        "loss_ema": 5.0, "tok_per_s": 1000.0, "lr": 1.0}) + "\\n")
ck = os.path.join(out, "train", "ckpt", f"ckpt-{n:06d}-step-{10 * n:09d}")
os.makedirs(ck, exist_ok=True)
with open(os.path.join(ck, "arrays.npz"), "wb") as f:
    f.write(bytes([n]) * 5000)
dump(os.path.join(ck, "manifest.json"), {"step": 10 * n})
os.makedirs(os.path.join(out, "train", "ckpt", ".tmp-half"), exist_ok=True)
os.makedirs(os.path.join(out, "logs"), exist_ok=True)
with open(os.path.join(out, "logs", "downloads.log"), "a") as f:
    f.write(f"attempt {n}\\n")
print("stand-in run attempt", n, flush=True)
word = "finish"
if not os.environ.get("STANDIN_AUTOFINISH"):
    gate = os.path.join(runs, f"gate{n}")
    deadline = time.time() + float(os.environ.get("STANDIN_MAX_S", "60"))
    while not os.path.exists(gate):
        if time.time() > deadline:
            sys.exit(4)
        time.sleep(0.02)
    word = ""
    while not word:                               # the file can be seen before its word is written
        with open(gate) as f:
            word = f.read().strip()
        time.sleep(0.005)
else:
    time.sleep(float(os.environ.get("STANDIN_AUTOFINISH")))
if word == "crash":
    sys.exit(1)
if word == "train-failed":                        # an essential stage failed: the run stops, with an export
    os.makedirs(os.path.join(out, "final"), exist_ok=True)
    dump(os.path.join(out, "final", "config.json"), {"name": "emergency"})
    stages(ORDER[:3], failed=("train",))
    sys.exit(2)
final = os.path.join(out, "final")
os.makedirs(os.path.join(final, "ckpt"), exist_ok=True)
with open(os.path.join(final, "weights.bin"), "wb") as f:
    f.write(bytes(range(256)) * 800)
dump(os.path.join(final, "config.json"), {"name": "stand-in"})
dump(os.path.join(final, "ckpt", "manifest.json"), {"step": 10 * n})
stages(ORDER, failed=("judge",) if word == "judge-failed" else ())
dump(os.path.join(out, "stage_table.json"), {"complete": True})
if word == "linger":                              # complete, but the process does not exit
    time.sleep(120)
'''

STANDIN_EXPORT = '''
import os, sys
print("verified", sum(len(f) for _, _, f in os.walk(sys.argv[sys.argv.index("--verify") + 1])), "files")
'''


def _standin_repo(root: Path) -> Path:
    """<root>/TuringAward2027 with the real notebooks and the stand-in run."""
    repo = root / "TuringAward2027"
    (repo / "bes").mkdir(parents=True)
    (repo / "bes" / "__init__.py").write_text("")
    shutil.copytree(REPO / "notebooks", repo / "notebooks")
    (repo / "scripts").mkdir()
    (repo / "scripts" / "run_24h.py").write_text(STANDIN_RUN)
    (repo / "scripts" / "export.py").write_text(STANDIN_EXPORT)
    (repo / "requirements-colab.txt").write_text("")
    return repo


def _env(root: Path, **extra: str) -> dict[str, str]:
    env = {"BES_ROOT": str(root), "BES_NOTEBOOK": "colab_v5e_smoke", "BES_CELLS": "3", "BES_ALLOW_CPU": "1",
           "BES_POLL_S": "0.05", "BES_STATUS_S": "0", "BES_HEARTBEAT_S": "0", "BES_RESUME_GAP_S": "30",
           "BES_PART_MB": "0.05", "BES_HOLD_MINUTES": "10", "BES_HOLD_FAILED_MINUTES": "20"}
    env.update(extra)
    return env


def _wait(cond, what: str, timeout: float = 20.0) -> None:
    end = time.time() + timeout
    while not cond():
        if time.time() > end:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.02)


def _attempts(root: Path) -> list[list[str]]:
    path = root / "runs" / "attempts.jsonl"
    return [json.loads(ln) for ln in path.read_text().splitlines()] if path.exists() else []


def _join_parts(export: Path) -> bytes:
    meta = json.loads((export / "parts.json").read_text())
    data = b"".join((export / p["name"]).read_bytes() for p in meta["parts"])
    assert hashlib.sha256(data).hexdigest() == meta["sha256"] and len(data) == meta["bytes"]
    assert all(hashlib.sha256((export / p["name"]).read_bytes()).hexdigest() == p["sha256"] for p in meta["parts"])
    return data


# ============================================================ the program sent to the kernel
def test_the_kernel_program_is_self_contained():
    """`colab exec -f` sends the text of the file, with `--env` lines in front of it, to an
    IPython kernel: no `__future__` import (it could not follow those lines), no `__file__`, no
    import from the repository (it is not on the runtime before the clone)."""
    text = TPU.read_text()
    code = text.split('"""', 2)[2]                      # after the module docstring
    assert "__future__" not in code and "__file__" not in code
    assert not re.search(r"^\s*(from|import) bes\b", code, re.M)
    prelude = "import os\nos.environ['BES_NOTEBOOK'] = 'colab_v5e_smoke'\n"      # what --env puts in front
    compile(prelude + text, "colab exec", "exec")
    imported = set(re.findall(r"^import (\w+)", code, re.M))
    assert imported <= set(sys.stdlib_module_names), imported - set(sys.stdlib_module_names)


def test_the_kernel_program_matches_the_repository():
    """Its stage list is the run's, its default repository and branch are the notebooks', and
    every cell it runs exists in both notebooks as plain Python (it compiles them)."""
    from bes.train import stages

    import inspect

    from bes.train import supervisor

    assert tpu.STAGES == stages.ORDER and tpu.DONE == stages.DONE
    essential = inspect.signature(supervisor.Supervisor.__init__).parameters["essential"].default
    assert tpu.ESSENTIAL == tuple(essential)
    for name in tpu.NOTEBOOKS:
        nb = json.loads((REPO / "notebooks" / f"{name}.ipynb").read_text())
        cells = {}
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                src = "".join(c["source"])
                cells[re.match(r"# (\d+b?)\.", src).group(1)] = src
        for label in USED_CELLS:
            compile(cells[label], f"{name} cell {label}", "exec")
        ns: dict = {}
        exec(compile(cells["1"], "params", "exec"), ns)
        assert (ns["REPO_URL"], ns["BRANCH"]) == (tpu.REPO_URL, tpu.BRANCH)
        exec(compile(cells["1b"], "helpers", "exec"), ns)            # definitions only
        mirror: dict = {"__name__": "mirror"}
        exec(compile(ns["MIRROR_CODE"], "mirror", "exec"), mirror)
        assert (tpu.SNAP_ROOT_FILES, tpu.SNAP_FILES, tpu.SNAP_TREES, tpu.SNAP_CKPT_ROOTS) == (
            mirror["ROOT_FILES"], mirror["FILES"], mirror["TREES"], mirror["CKPT_ROOTS"])
        assert all(tpu._temporary(n) == mirror["_skip"](n) for n in
                   (".tmp-x", ".trash-1", ".old-2", ".part-ckpt", "a.part", "x.lock", "__pycache__", "ckpt-1", "log"))
    assert tpu.COMMANDS == cli.COMMANDS and tpu.RESTORE_PREFIX == cli.RESTORE_PREFIX
    assert {p["notebook"] for p in cli.PRESETS.values()} == set(tpu.NOTEBOOKS)


def test_settings_reject_what_they_do_not_know():
    assert tpu.settings({})["cells"] == ["2", "3", "4"] and tpu.settings({"BES_CELLS": ""})["cells"] == []
    assert tpu.settings({"BES_BRANCH": ""})["branch"] == ""          # the default branch of a fork
    for bad in ({"BES_NOTEBOOK": "x"}, {"BES_ACTION": "go"}, {"BES_USE_DRIVE": "maybe"},
                {"BES_BUDGET_MINUTES": "soon"}, {"BES_RUN_DIR": "runs/v9"},
                {"BES_RUN_DIR": "/content/drive/MyDrive/bes_runs/v9"}, {"BES_RUN_DIR": "/content/drive"},
                {"BES_RUN_DIR": "/content"}, {"BES_RUN_DIR": "/content/TuringAward2027/"},
                {"BES_MOVE_DEADLINE": "tomorrow"}):
        with pytest.raises(RuntimeError):
            tpu.settings(bad)
    assert tpu.settings({})["run_dir"] is None                       # the notebook's RUN_DIR
    assert tpu.settings({"BES_RUN_DIR": "/content/runs/v9/"})["run_dir"] == "/content/runs/v9"
    assert tpu.settings({})["move_deadline"] is None and tpu.settings({"BES_MOVE_DEADLINE": " +45 "})["move_deadline"] == "+45"


def test_supervisor_runs_the_notebook_cells_and_packages_the_export(tmp_path):
    """A new run through the real cells (3 and 6; also 2, the runtime check in its own process,
    where the machine has the /content it measures), then the supervisor: the progress report in
    the status file, the export as parts, and it leaves after the hold time with the status
    saying so."""
    root = tmp_path / "content"
    _standin_repo(root)
    cells = "2,3" if os.path.isdir("/content") else "3"
    env = dict(os.environ, **_env(root, BES_CELLS=cells, BES_HOLD_MINUTES="0.005", STANDIN_AUTOFINISH="0.3"))
    p = subprocess.run([sys.executable, str(TPU)], capture_output=True, text=True, env=env, timeout=240,
                       cwd=tmp_path)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    out = p.stdout
    assert all(f"cell {c}: # {c}." in out for c in cells.split(","))
    assert "no Drive mirror: Drive is not mounted" in out and "HF_TOKEN not set" in out
    assert "supervising: status in" in out and "the run is complete" in out and "export packaged" in out
    assert "verified" in out                                  # cell 12 ran the export's verification
    assert "the supervisor leaves" in out
    [argv] = _attempts(root)
    assert argv == ["--config", "configs/smoke.py", "--out", str(root / "runs" / "smoke"),
                    "--budget-minutes", "30", "--allow-cpu"]
    info = json.loads((root / "bes_cli" / "status.json").read_text())
    assert info["state"] == "released" and info["export"]["bytes"] > 200_000 and info["resumes"] == 0
    text = (root / "bes_cli" / "status.txt").read_text()
    assert "supervisor: released" in text and "stage        status" in text and "export       finished" in text
    data = _join_parts(root / "bes_cli" / "export")
    assert len(json.loads((root / "bes_cli" / "export" / "parts.json").read_text())["parts"]) >= 4
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        assert tar.extractfile("final/weights.bin").read() == bytes(range(256)) * 800
        assert "final/ckpt/manifest.json" in tar.getnames()


def _supervisor(root: Path, clock: list[float], **extra: str):
    out = io.StringIO()
    sup = tpu.Supervisor(env=_env(root, **extra), out=out, clock=lambda: clock[0], sleep=lambda s: None)
    return sup, out


def _setup(sup) -> None:
    sup.clone()
    sup.load_cells()
    sup.parameters()
    sup.setup_cells()
    sup.start_or_resume()


def test_supervisor_continues_a_dead_run_and_obeys_the_commands(tmp_path, monkeypatch):
    """The life of a run under the supervisor: a crash is continued once the gap has passed and
    automatic continuations are left; `stop` ends the run and is not undone automatically;
    `release` is refused while the run runs; `resume` continues; a complete run is packaged and
    the runtime kept for the hold time, then the supervisor leaves."""
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "content"
    _standin_repo(root)
    runs = root / "runs"
    clock = [time.time()]
    sup, out = _supervisor(root, clock, BES_AUTO_RESUME="1")
    _setup(sup)
    ctl = root / "bes_cli" / "ctl"
    assert sup.run_alive() and sup.tick() and sup.state == "running"
    _wait(lambda: len(_attempts(root)) == 1, "the first attempt")
    assert "--resume" not in _attempts(root)[0]

    (runs / "gate1").write_text("crash")
    _wait(lambda: not sup.run_alive(), "the crash")
    assert sup.tick() and sup.state == "stopped" and "continuing shortly" in sup.note     # the gap has not passed
    assert len(_attempts(root)) == 1
    clock[0] += 31
    assert sup.tick() and sup.state == "running" and sup.resumes_left == 0
    _wait(lambda: len(_attempts(root)) == 2, "the continued run")
    assert _attempts(root)[1] == ["--out", str(runs / "smoke"), "--resume", "--allow-cpu"]

    ctl.write_text("release\n")                              # refused: the run would die with the runtime
    assert sup.tick() and sup.run_alive() and "not released: the run is running" in out.getvalue()
    ctl.write_text("stop\n")
    assert sup.tick() and not ctl.exists()
    _wait(lambda: not sup.run_alive(), "the stop")
    clock[0] += 31
    assert sup.tick() and sup.state == "stopped" and len(_attempts(root)) == 2             # not continued
    assert sup.hold_until == pytest.approx(clock[0] + 20 * 60)
    info = json.loads((root / "bes_cli" / "status.json").read_text())
    assert info["state"] == "stopped" and info["run_alive"] is False and info["hold_until"] == sup.hold_until

    ctl.write_text("resume\n")
    assert sup.tick() and sup.state == "running" and sup.hold_until is None
    _wait(lambda: len(_attempts(root)) == 3, "the run continued on request")
    ctl.write_text("dance\n")
    assert sup.tick() and "ignoring the command 'dance'" in out.getvalue()
    (runs / "gate3").write_text("finish")
    _wait(lambda: not sup.run_alive(), "the end of the run")
    assert sup.tick() and sup.state == "complete" and sup.export and sup.resumes == 2
    assert sup.hold_until == pytest.approx(clock[0] + 10 * 60, abs=5)
    _join_parts(root / "bes_cli" / "export")
    text = (root / "bes_cli" / "status.txt").read_text()
    assert "supervisor: complete" in text and "export packaged" in text and "the runtime is kept for another" in text
    clock[0] += 9 * 60
    assert sup.tick()
    clock[0] += 2 * 60
    assert not sup.tick() and sup.state == "released"
    assert json.loads((root / "bes_cli" / "status.json").read_text())["state"] == "released"


def test_supervisor_stays_through_failures_and_bounds_what_an_unattended_runtime_costs(tmp_path, monkeypatch):
    """Leaving the kernel hands the runtime back, so a failing round is reported and the loop goes
    on; a failed finish still starts the hold time (else the runtime would be kept for ever); and a
    run still running long after its own deadline is stopped once."""
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "content"
    _standin_repo(root)
    clock = [time.time()]
    sup, out = _supervisor(root, clock, BES_AUTO_RESUME="0", BES_OVERRUN_MINUTES="60")
    answers = [OSError("disk"), ValueError("json"), True, False]

    def tick():
        a = answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a

    monkeypatch.setattr(sup, "tick", tick)
    sup.loop()
    assert answers == [] and out.getvalue().count("a supervisor round failed") == 2
    monkeypatch.undo()
    monkeypatch.chdir(tmp_path)

    _setup(sup)
    _wait(lambda: len(_attempts(root)) == 1, "the first attempt")
    assert sup.tick() and sup.state == "running" and not sup.stop_requested
    clock[0] += 1800 + 59 * 60                                 # the stand-in's deadline is 30 minutes away
    assert sup.tick() and not sup.stop_requested
    clock[0] += 2 * 60
    assert sup.tick() and sup.stop_requested and "after its deadline (or after the start" in out.getvalue()
    _wait(lambda: not sup.run_alive(), "the stop")
    assert sup.tick() and sup.state == "stopped" and len(_attempts(root)) == 1

    (root / "bes_cli" / "ctl").write_text("resume\n")
    assert sup.tick() and not sup.stop_requested
    _wait(lambda: len(_attempts(root)) == 2, "the continued run")
    (root / "runs" / "gate2").write_text("finish")
    _wait(lambda: not sup.run_alive(), "the end of the run")
    monkeypatch.setattr(sup, "finish", lambda: (_ for _ in ()).throw(OSError("no space left on device")))
    assert sup.tick() and sup.state == "complete" and "finishing failed (OSError" in out.getvalue()
    assert sup.hold_until == pytest.approx(clock[0] + 10 * 60)
    clock[0] += 11 * 60
    assert not sup.tick()


def test_a_supervisor_started_again_only_supervises_a_run_that_runs(tmp_path, monkeypatch):
    """After a kernel restart or an interrupt the program is sent again while the run runs. The
    setup cells must not run then: cell 2 opens the TPU, which the run holds, and would fail, and
    a supervisor that leaves on that failure hands the runtime back with the run on it. A setup
    that fails for another reason while the run runs does not make it leave either."""
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "content"
    _standin_repo(root)
    clock = [time.time()]
    first, _ = _supervisor(root, clock)
    _setup(first)
    _wait(lambda: len(_attempts(root)) == 1, "the first attempt")

    again, out = _supervisor(root, clock, BES_CELLS="2,3,4")       # the default cells
    monkeypatch.setattr(again, "loop", lambda: None)
    monkeypatch.setattr(again, "setup_cells", lambda: pytest.fail("the setup cells ran next to a running run"))
    again.main()
    assert "already running on this runtime: supervising it (the setup cells are skipped)" in out.getvalue()
    assert again.state == "running" and Path(os.getcwd()) == root / "TuringAward2027"
    assert len(_attempts(root)) == 1

    third, out = _supervisor(root, clock, BES_USE_DRIVE="1")       # Drive is not mounted: the setup fails
    monkeypatch.setattr(third, "loop", lambda: None)
    third.main()
    assert "the setup failed (RuntimeError: BES_USE_DRIVE=1" in out.getvalue()
    assert "but the run is running: supervising it" in out.getvalue() and third.state == "running"
    (root / "runs" / "gate1").write_text("crash")
    _wait(lambda: not first.run_alive(), "the crash")
    fourth, _ = _supervisor(root, clock, BES_USE_DRIVE="1")        # no run is running: the failure is the answer
    with pytest.raises(RuntimeError, match="Drive is not mounted"):
        fourth.main()
    assert json.loads((root / "bes_cli" / "status.json").read_text())["state"] == "released"


def test_supervisor_reads_completion_the_way_the_run_does(tmp_path, monkeypatch):
    """A failed stage that is not essential is over (the run exits 0 without it and `--resume`
    would do nothing): the supervisor finishes, packages and says which stage failed. A failed
    essential stage is incomplete: it is continued, and once nothing is left the export the run
    wrote on the way out is packaged all the same."""
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "content"
    _standin_repo(root)
    clock = [time.time()]
    sup, out = _supervisor(root, clock, BES_AUTO_RESUME="1")
    _setup(sup)
    _wait(lambda: len(_attempts(root)) == 1, "the first attempt")
    (root / "runs" / "gate1").write_text("train-failed")
    _wait(lambda: not sup.run_alive(), "the failed run")
    assert sup.run_state() == "incomplete" and sup.failed_stages() == ["train"]
    clock[0] += 31
    assert sup.tick() and sup.state == "running"                   # continued: --resume retries the stage
    _wait(lambda: len(_attempts(root)) == 2, "the continued run")
    (root / "runs" / "gate2").write_text("train-failed")
    _wait(lambda: not sup.run_alive(), "the second failure")
    clock[0] += 31
    assert sup.tick() and sup.state == "stopped" and len(_attempts(root)) == 2
    assert "failed: train" in out.getvalue() and "packaging it for fetch" in out.getvalue()
    _join_parts(root / "bes_cli" / "export")                       # the emergency export can be fetched

    (root / "bes_cli" / "ctl").write_text("resume\n")
    assert sup.tick()
    _wait(lambda: len(_attempts(root)) == 3, "the run continued on request")
    (root / "runs" / "gate3").write_text("judge-failed")
    _wait(lambda: not sup.run_alive(), "the end of the run")
    assert sup.run_state() == "complete" and sup.failed_stages() == ["judge"]
    assert sup.tick() and sup.state == "complete" and len(_attempts(root)) == 3
    assert "the run is complete (failed and left out: judge)" in out.getvalue()
    with tarfile.open(fileobj=io.BytesIO(_join_parts(root / "bes_cli" / "export"))) as tar:
        assert "final/weights.bin" in tar.getnames()               # the real export replaced the emergency one


def test_supervisor_gets_rid_of_a_run_that_does_not_go_away(tmp_path, monkeypatch):
    """One SIGTERM is not enough for a run that is stuck: after the grace time a second one (the
    run exits at once on it), then SIGKILL for its process group; a complete run whose process
    lingers is killed after the linger time. Otherwise a live pid would keep the runtime for ever."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STANDIN_DEAF", "1")                        # the stand-in only counts SIGTERMs
    root = tmp_path / "content"
    _standin_repo(root)
    clock = [time.time()]
    sup, out = _supervisor(root, clock, BES_AUTO_RESUME="0", BES_STOP_GRACE_MINUTES="30",
                           BES_KILL_GRACE_MINUTES="5", BES_LINGER_MINUTES="10")
    _setup(sup)
    _wait(lambda: len(_attempts(root)) == 1, "the first attempt")
    terms = root / "runs" / "sigterms"
    (root / "bes_cli" / "ctl").write_text("stop\n")
    assert sup.tick() and sup.state == "stopping" and sup.stop_step == 1
    _wait(lambda: terms.exists() and len(terms.read_text()) == 1, "the first SIGTERM")
    clock[0] += 29 * 60
    assert sup.tick() and sup.stop_step == 1
    clock[0] += 2 * 60
    assert sup.tick() and sup.stop_step == 2
    _wait(lambda: len(terms.read_text()) == 2, "the second SIGTERM")
    clock[0] += 3 * 60                                             # 34 minutes after the first SIGTERM
    assert sup.tick() and sup.stop_step == 2 and sup.run_alive()
    clock[0] += 2 * 60                                             # 36: the grace times (30 + 5) are over
    assert sup.tick() and sup.stop_step == 3
    _wait(lambda: not sup.run_alive(), "the kill")
    assert sup.tick() and sup.state == "stopped" and sup.stop_step == 0 and sup.hold_until is not None

    (root / "bes_cli" / "ctl").write_text("resume\n")
    assert sup.tick() and sup.state == "running"
    _wait(lambda: len(_attempts(root)) == 2, "the continued run")
    (root / "runs" / "gate2").write_text("linger")
    _wait(lambda: sup.run_state() == "complete", "the stages to be complete")
    assert sup.run_alive() and sup.tick() and sup.state == "running"
    clock[0] += 9 * 60
    assert sup.tick() and sup.run_alive()
    clock[0] += 2 * 60
    assert sup.tick() and "complete but its process has not exited" in out.getvalue()
    _wait(lambda: not sup.run_alive(), "the kill of the lingering run")
    assert sup.tick() and sup.state == "complete" and sup.export


def test_supervisor_bounds_a_late_continuation_from_its_own_start_and_leaves_in_the_end(tmp_path, monkeypatch):
    """A run continued long after its deadline is not stopped in its first round (it still has
    its export to write): the overrun time counts from the start of the attempt. A damaged
    command file is dropped. And whatever happens, the supervisor leaves BES_MAX_HOURS after
    the last start, continuation or command."""
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "content"
    _standin_repo(root)
    clock = [time.time()]
    sup, out = _supervisor(root, clock, BES_AUTO_RESUME="0", BES_OVERRUN_MINUTES="60", BES_MAX_HOURS="10")
    _setup(sup)
    _wait(lambda: len(_attempts(root)) == 1, "the first attempt")
    (root / "runs" / "gate1").write_text("crash")
    _wait(lambda: not sup.run_alive(), "the crash")
    clock[0] += 1800 + 4 * 3600                                    # four hours after the run's deadline
    ctl = root / "bes_cli" / "ctl"
    ctl.write_bytes("stop\n".encode("utf-16"))                    # not a command: dropped, the round goes on
    assert sup.tick() and not ctl.exists() and "ignoring the command" in out.getvalue()
    assert "a supervisor round failed" not in out.getvalue()
    ctl.write_text("resume\n")
    assert sup.tick() and sup.state == "running" and not sup.stop_requested
    _wait(lambda: len(_attempts(root)) == 2, "the continued run")
    clock[0] += 59 * 60
    assert sup.tick() and not sup.stop_requested and sup.state == "running"
    clock[0] += 2 * 60
    assert sup.tick() and sup.stop_requested and sup.state == "stopping"
    assert "after its deadline (or after the start of this attempt" in out.getvalue()
    _wait(lambda: not sup.run_alive(), "the stop")
    assert sup.tick() and sup.state == "stopped"

    late, out = _supervisor(root, clock, BES_ACTION="resume", BES_MAX_HOURS="10", BES_OVERRUN_MINUTES="100000")
    _setup(late)
    _wait(lambda: len(_attempts(root)) == 3, "the third attempt")
    clock[0] += 9 * 3600
    assert late.tick() and late.state == "running"
    clock[0] += 2 * 3600
    assert not late.tick() and late.state == "released" and "BES_MAX_HOURS" in out.getvalue()
    (root / "runs" / "gate3").write_text("crash")
    _wait(lambda: not late.run_alive(), "the end of the stand-in")


def test_the_program_does_not_act_on_what_an_earlier_execution_left(tmp_path, monkeypatch):
    """A kernel keeps os.environ between executions, a second `colab exec` waits behind a running
    one, and a command file can outlive the supervisor it was meant for. So: the settings are
    consumed when they are read; a program that starts long after it was sent leaves without
    touching anything (it would continue a run someone stopped, hours later); and a command file
    older than the supervisor is dropped."""
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "content"
    _standin_repo(root)
    for k, v in _env(root, BES_HOLD_MINUTES="0", BES_AUTO_RESUME="0").items():
        monkeypatch.setenv(k, v)
    out = io.StringIO()
    sup = tpu.Supervisor(out=out)                                  # the settings come from os.environ
    assert sup.s["hold_s"] == 0.0 and sup.s["auto_resume"] == 0 and sup.s["root"] == str(root)
    assert not [k for k in os.environ if k.startswith("BES_")]
    assert tpu.settings(os.environ)["hold_s"] == 120 * 60          # a later start gets the defaults again

    late = tpu.Supervisor(env=_env(root, BES_SENT_AT=str(time.time() - 3 * 3600)), out=out)
    late.main()
    assert "waited behind another execution" in out.getvalue() and "leaving without touching anything" in out.getvalue()
    assert not (root / "bes_cli").exists() and not _attempts(root)

    ctl = root / "bes_cli" / "ctl"
    ctl.parent.mkdir()
    ctl.write_text("stop\n")
    os.utime(ctl, (time.time() - 600, time.time() - 600))
    out = io.StringIO()
    monkeypatch.setenv("STANDIN_AUTOFINISH", "2")                  # the stand-in run ends by itself
    sup = tpu.Supervisor(env=_env(root, BES_SENT_AT=str(time.time() - 5)), out=out)
    monkeypatch.setattr(sup, "loop", lambda: None)
    sup.main()
    assert "dropped a command file that was older than this supervisor" in out.getvalue() and not ctl.exists()
    _wait(lambda: len(_attempts(root)) == 1, "the run")            # sent 5 s ago: it did start the run

    ctl.write_text("package 0.02\n")                               # a part size with the command
    assert sup.command() == "package" and sup.s["part_bytes"] == int(0.02 * 2 ** 20)
    ctl.write_text("package huge\n")
    assert sup.command() == "package" and "ignoring the part size 'huge'" in out.getvalue()
    _wait(lambda: not sup.run_alive(), "the end of the stand-in", timeout=30)


def test_supervisor_start_and_resume_actions(tmp_path, monkeypatch):
    """`start` refuses a run directory that holds a run, `resume` one that holds none; `auto`
    supervises a run that is already running instead of starting another."""
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "content"
    _standin_repo(root)
    clock = [time.time()]
    sup, _ = _supervisor(root, clock, BES_ACTION="resume")
    with pytest.raises(RuntimeError, match="no run to continue"):
        _setup(sup)
    sup, _ = _supervisor(root, clock)
    _setup(sup)
    assert sup.run_alive()
    _wait(lambda: len(_attempts(root)) == 1, "the first attempt")
    again, out = _supervisor(root, clock)                      # a second supervisor on the same runtime
    _setup(again)
    assert "already running on this runtime: supervising it" in out.getvalue() and len(_attempts(root)) == 1
    (root / "runs" / "gate1").write_text("crash")
    _wait(lambda: not sup.run_alive(), "the crash")
    sup, _ = _supervisor(root, clock, BES_ACTION="start")
    with pytest.raises(RuntimeError, match="already holds a run"):
        _setup(sup)
    sup, out = _supervisor(root, clock, BES_ACTION="auto")
    _setup(sup)                                                # auto: the run exists, so it is continued
    _wait(lambda: len(_attempts(root)) == 2, "the continued run")
    assert "--resume" in _attempts(root)[1] and sup.resumes == 1
    (root / "runs" / "gate2").write_text("finish")
    _wait(lambda: not sup.run_alive(), "the end of the run")
    done, out = _supervisor(root, clock)
    _setup(done)
    assert "is complete: nothing to start" in out.getvalue() and len(_attempts(root)) == 2


def test_two_runs_on_one_runtime_keep_to_their_own_directories(tmp_path, monkeypatch):
    """BES_RUN_DIR: a comparison of two configurations runs them one after the other on the same
    runtime. Each run has its directory, pid file, log and mirror directory; a supervisor started
    for one stops, starts and continues that one only."""
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "content"
    _standin_repo(root)
    clock = [time.time()]
    a_dir, b_dir = root / "runs" / "a", root / "runs" / "b"
    a, out = _supervisor(root, clock, BES_RUN_DIR=str(a_dir) + "/", BES_CONFIG="configs/v9.py",
                         BES_BUDGET_MINUTES="1400")
    _setup(a)
    _wait(lambda: len(_attempts(root)) == 1, "run a")
    assert _attempts(root)[0] == ["--config", "configs/v9.py", "--out", str(a_dir), "--budget-minutes", "1400",
                                  "--allow-cpu"]
    assert a.run_dir == str(a_dir) and a.ns["PID_FILE"] == f"{a_dir}.pid" and a.ns["LOG"] == f"{a_dir}.log"
    assert f"RUN_DIR {a_dir}," in out.getvalue()
    (root / "bes_cli" / "ctl").write_text("stop\n")              # what `send stop` uploads
    a.tick()
    _wait(lambda: not a.run_alive(), "the stop of run a")
    assert a.run_state() == "incomplete" and a.stop_requested
    b, _ = _supervisor(root, clock, BES_RUN_DIR=str(b_dir), BES_ACTION="start", BES_BUDGET_MINUTES="1275.5")
    _setup(b)                                                    # a new run although runs/a holds one
    _wait(lambda: len(_attempts(root)) == 2, "run b")
    assert _attempts(root)[1] == ["--config", "configs/smoke.py", "--out", str(b_dir), "--budget-minutes",
                                  "1275.5", "--allow-cpu"]
    assert b.run_alive() and not a.run_alive() and (a_dir / "manifest.json").exists()
    (root / "bes_cli" / "ctl").write_text("stop\n")
    b.tick()
    _wait(lambda: not b.run_alive(), "the stop of run b")
    again, _ = _supervisor(root, clock, BES_RUN_DIR=str(a_dir), BES_ACTION="resume")
    _setup(again)                                                # the winner goes on from its checkpoint
    _wait(lambda: len(_attempts(root)) == 3, "the continued run a")
    assert _attempts(root)[2] == ["--out", str(a_dir), "--resume", "--allow-cpu"]
    assert again.run_alive() and not b.run_alive()
    (root / "runs" / "gate3").write_text("finish")
    _wait(lambda: not again.run_alive(), "the end of run a")
    assert again.run_state() == "complete" and b.run_state() == "incomplete"
    assert (a_dir / "final" / "weights.bin").exists() and not (b_dir / "final").exists()
    # the mirror of a run with a directory of its own: next to the notebook's, under the same name
    (root / "drive" / "MyDrive").mkdir(parents=True)
    sup, _ = _supervisor(root, clock, BES_RUN_DIR=str(b_dir))
    sup.clone()
    sup.load_cells()
    sup.parameters()
    assert sup.ns["USE_DRIVE"] is True and sup.ns["DRIVE_DIR"] == str(root / "drive/MyDrive/bes_runs/b")
    sup, out = _supervisor(root, clock, BES_RUN_DIR=str(b_dir), BES_DRIVE_DIR=str(root / "drive/MyDrive/x"))
    sup.clone()
    sup.load_cells()
    sup.parameters()
    assert sup.ns["DRIVE_DIR"] == str(root / "drive/MyDrive/x")
    assert f"(DRIVE_DIR {root / 'drive/MyDrive/x'})" in out.getvalue()


def test_a_continuation_moves_the_deadline_first(tmp_path, monkeypatch):
    """BES_MOVE_DEADLINE: a run that continues on another runtime gets its deadline moved by the
    run's own command before cell 8 continues it, once; a failure of that command stops the
    continuation, and a run that is running is left alone."""
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "content"
    _standin_repo(root)
    clock = [time.time()]
    moved = root / "runs" / "moved.jsonl"
    sup, _ = _supervisor(root, clock)
    _setup(sup)
    _wait(lambda: len(_attempts(root)) == 1, "the first attempt")
    again, out = _supervisor(root, clock, BES_MOVE_DEADLINE="1800000000")    # the run runs: not touched
    _setup(again)
    assert "its deadline is not moved while it runs" in out.getvalue() and not moved.exists()
    (root / "runs" / "gate1").write_text("crash")
    _wait(lambda: not sup.run_alive(), "the crash")
    sup, out = _supervisor(root, clock, BES_ACTION="resume", BES_MOVE_DEADLINE="1800000000")
    _setup(sup)
    _wait(lambda: len(_attempts(root)) == 2, "the continued run")
    [call] = [json.loads(ln) for ln in moved.read_text().splitlines()]
    assert call == {"out": str(root / "runs" / "smoke"), "when": "1800000000", "run": True}
    text = out.getvalue()
    assert text.index("deadline moved to 1800000000") < text.index("continuing the run")
    assert sup.s["move_deadline"] is None and "--resume" in _attempts(root)[1]
    (root / "runs" / "gate2").write_text("crash")
    _wait(lambda: not sup.run_alive(), "the second crash")
    clock[0] += 120.0
    sup.tick()                                                   # an automatic continuation: no second move
    _wait(lambda: len(_attempts(root)) == 3, "the automatic continuation")
    assert len(moved.read_text().splitlines()) == 1
    (root / "runs" / "gate3").write_text("finish")
    _wait(lambda: not sup.run_alive(), "the end")
    # a directory without a run: nothing is moved and nothing is continued
    other, _ = _supervisor(root, clock, BES_RUN_DIR=str(root / "runs" / "none"), BES_ACTION="resume",
                           BES_MOVE_DEADLINE="+30")
    with pytest.raises(RuntimeError, match="no run to continue"):
        _setup(other)
    assert len(moved.read_text().splitlines()) == 1
    # the run's command refuses (here: the stand-in, for a directory whose manifest is gone)
    broken, _ = _supervisor(root, clock, BES_ACTION="resume", BES_MOVE_DEADLINE="+30")
    broken.clone()
    broken.load_cells()
    broken.parameters()
    broken.setup_cells()
    broken.run_dir = str(root / "runs" / "none")
    with pytest.raises(RuntimeError, match="the deadline could not be moved"):
        broken.move_deadline()


def _tar_of(parts_dir: Path) -> tuple[dict, bytes]:
    meta = json.loads((parts_dir / "parts.json").read_text())
    data = b"".join((parts_dir / p["name"]).read_bytes() for p in meta["parts"])
    assert hashlib.sha256(data).hexdigest() == meta["sha256"] and len(data) == meta["bytes"]
    return meta, data


def test_a_snapshot_carries_the_run_to_another_runtime(tmp_path, monkeypatch):
    """`snapshot` packs what the Drive mirror holds while the run goes on; unpacked on another
    runtime (BES_RESTORE) it is the run directory the continuation starts from. A damaged upload
    is refused before anything is unpacked."""
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "content"
    _standin_repo(root)
    clock = [time.time()]
    sup, out = _supervisor(root, clock, BES_PART_MB="0.004")
    _setup(sup)
    _wait(lambda: len(_attempts(root)) == 1 and (root / "runs" / "smoke" / "logs").is_dir(), "the first attempt")
    ctl, snap = root / "bes_cli" / "ctl", root / "bes_cli" / "snapshot"
    ctl.write_text("snapshot\n")
    assert sup.tick() and sup.run_alive() and sup.state == "running"          # the run goes on
    meta, data = _tar_of(snap)
    assert meta["checkpoints"] == {"train/ckpt": "ckpt-000001-step-000000010"} and len(meta["parts"]) >= 3
    assert meta["run_dir"] == str(root / "runs" / "smoke") and meta["tokens"] == 123456
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        names = set(tar.getnames())
        assert tar.extractfile("run/train/ckpt/ckpt-000001-step-000000010/arrays.npz").read() == b"\x01" * 5000
    assert {"run/manifest.json", "run/stages.json", "run/train/ledger.json", "run/train/train_log.jsonl",
            "run/logs/downloads.log", "run/train/ckpt/ckpt-000001-step-000000010/manifest.json", "run.log"} <= names
    assert not any(".tmp-half" in n for n in names)
    info = json.loads((root / "bes_cli" / "status.json").read_text())
    assert info["snapshot"]["parts"] == len(meta["parts"]) and info["snapshot"]["sha256"] == meta["sha256"]
    assert "snapshot:" in (root / "bes_cli" / "status.txt").read_text()
    # a later snapshot replaces the set as a whole (the newer checkpoint, nothing left over)
    (root / "runs" / "gate1").write_text("crash")
    _wait(lambda: not sup.run_alive(), "the crash")
    clock[0] += 120.0
    sup.tick()
    _wait(lambda: len(_attempts(root)) == 2 and (root / "runs" / "smoke" / "train" / "ckpt" /
                                                   "ckpt-000002-step-000000020" / "manifest.json").exists(),
          "the continued run")
    ctl.write_text("snapshot 0.012\n")
    sup.tick()
    meta, data = _tar_of(snap)
    assert meta["checkpoints"] == {"train/ckpt": "ckpt-000002-step-000000020"}
    assert max(p["bytes"] for p in meta["parts"]) == int(0.012 * 2 ** 20)
    assert sorted(os.listdir(root / "bes_cli")) == ["export", "snapshot", "status.json", "status.txt",
                                                     "supervisor.log"] or "snapshot.new" not in os.listdir(root / "bes_cli")
    assert not (root / "bes_cli" / "snapshot.old").exists()
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        assert not any("ckpt-000001" in n for n in tar.getnames())           # only the newest checkpoint
    (root / "runs" / "gate2").write_text("crash")
    _wait(lambda: not sup.run_alive(), "the second crash")

    # another runtime: the parts are uploaded next to /content, the program unpacks and continues
    other = tmp_path / "content2"
    _standin_repo(other)
    for i, part in enumerate(meta["parts"]):
        shutil.copyfile(snap / part["name"], other / f"bes_restore.{i:03d}")
    shutil.copyfile(snap / "parts.json", other / "bes_restore.parts.json")
    new, out = _supervisor(other, clock, BES_RESTORE="1", BES_ACTION="resume", BES_MOVE_DEADLINE="+30")
    new.clone()
    new.load_cells()
    new.parameters()
    new.setup_cells()
    new.restore_snapshot()
    run2 = other / "runs" / "smoke"
    assert (run2 / "train" / "ckpt" / "ckpt-000002-step-000000020" / "arrays.npz").read_bytes() == b"\x02" * 5000
    assert json.loads((run2 / "manifest.json").read_text())["config"] == "configs/smoke.py"
    assert (run2 / "logs" / "downloads.log").read_text() == "attempt 1\nattempt 2\n"
    assert "stand-in run attempt 2" in (other / "runs" / "smoke.log").read_text()
    assert not list(other.glob("bes_restore.*")) and "snapshot restored into" in out.getvalue()
    new.start_or_resume()
    _wait(lambda: len(_attempts(other)) == 1, "the continuation on the other runtime")
    assert _attempts(other)[0] == ["--out", str(run2), "--resume", "--allow-cpu"]
    moved = [json.loads(ln) for ln in (other / "runs" / "moved.jsonl").read_text().splitlines()]
    assert moved == [{"out": str(run2), "when": "+30", "run": True}]         # the deadline first, then the run
    (other / "runs" / "gate1").write_text("finish")
    _wait(lambda: not new.run_alive(), "the end")
    assert new.run_state() == "complete"

    # refusals: nothing uploaded; a damaged part (nothing is unpacked); a run that is already there
    third = tmp_path / "content3"
    _standin_repo(third)
    sup3, _ = _supervisor(third, clock, BES_RESTORE="1")
    sup3.clone()
    sup3.load_cells()
    sup3.parameters()
    with pytest.raises(RuntimeError, match="was not uploaded"):
        sup3.restore_snapshot()
    for i, part in enumerate(meta["parts"]):
        shutil.copyfile(snap / part["name"], third / f"bes_restore.{i:03d}")
    shutil.copyfile(snap / "parts.json", third / "bes_restore.parts.json")
    blob = (third / "bes_restore.001").read_bytes()
    (third / "bes_restore.001").write_bytes(blob[:-1] + bytes([blob[-1] ^ 1]))
    with pytest.raises(RuntimeError, match="did not arrive intact"):
        sup3.restore_snapshot()
    assert not (third / "runs" / "smoke").exists()
    (third / "bes_restore.001").write_bytes(blob)
    (third / "runs" / "smoke").mkdir(parents=True)
    (third / "runs" / "smoke" / "manifest.json").write_text("{}")
    out3 = io.StringIO()
    sup3.out = out3
    sup3.restore_snapshot()
    assert "already holds a run" in out3.getvalue() and not list(third.glob("bes_restore.*"))
    assert (third / "runs" / "smoke" / "manifest.json").read_text() == "{}"


def test_supervisor_reads_the_token_file_and_refuses_an_unmounted_drive(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    root = tmp_path / "content"
    _standin_repo(root)
    (root / "bes_hf_token").write_text("hf_standin\n")
    sup, out = _supervisor(root, [time.time()])
    sup.hf_token()
    assert os.environ["HF_TOKEN"] == "hf_standin" and "hf_standin" not in out.getvalue()
    assert not (root / "bes_hf_token").exists()                  # read once, then gone from the disk
    monkeypatch.delenv("HF_TOKEN")
    sup, _ = _supervisor(root, [time.time()], BES_USE_DRIVE="1")
    sup.clone()
    sup.load_cells()
    with pytest.raises(RuntimeError, match="Drive is not mounted"):
        sup.parameters()
    (root / "drive" / "MyDrive").mkdir(parents=True)             # mounted: the mirror is on, cell 3 does not mount
    sup, _ = _supervisor(root, [time.time()], STANDIN_AUTOFINISH="5")
    sup.clone()
    sup.load_cells()
    sup.parameters()
    assert sup.ns["USE_DRIVE"] is True and sup.ns["DRIVE_DIR"] == str(root / "drive/MyDrive/bes_runs/smoke")
    sup.setup_cells()                                            # would import google.colab if it mounted
    assert sup.ns["USE_DRIVE"] is True


# ============================================================ the wrapper around `colab`
FAKE_COLAB = """#!{python}
\"\"\"Stand-in for the `colab` command (0.8.0): the runtime's files live under $FAKE_COLAB_HOME/content.
Where the wrapper depends on it, it answers like the real one: `status` and `stop` exit 0 for a
session they do not know, `sessions` lists runtimes unknown to this machine as [?], a failed
download or upload prints the error with the runtime's token in the URL and exits 1, `upload`
does not create directories, and `exec` exits 0 whether or not the program raised.\"\"\"
import json, os, shutil, subprocess, sys, time

home = os.environ["FAKE_COLAB_HOME"]
args = sys.argv[1:]
with open(os.path.join(home, "calls.jsonl"), "a") as f:
    f.write(json.dumps(args) + "\\n")
cmd, rest = args[0], args[1:]
URL = "https://rt.invalid/api/contents/content/x?authuser=0&colab-runtime-proxy-token=SECRETTOKEN&content=1"


def opt(name, default=None):
    return rest[rest.index(name) + 1] if name in rest else default


def here(remote):                      # a path on the runtime
    return os.path.join(home, remote.lstrip("/"))


def known():
    d = os.path.join(home, "sessions")
    return sorted(os.listdir(d)) if os.path.isdir(d) else []


session = opt("-s")
marker = os.path.join(home, "sessions", session or "-")
if cmd == "sessions":
    rows = [f"[{{n}}] ep-{{n}} | Hardware: {{open(os.path.join(home, 'sessions', n)).read().upper()}} | Shape: Standard | Variant: TPU"
            for n in known()]
    rows += [f"[?] {{o.split(':')[0]}} | Hardware: {{o.split(':')[1]}} | Shape: Standard | Variant: TPU"
             for o in os.environ.get("FAKE_COLAB_ORPHANS", "").split(",") if o]
    print("\\n".join(rows) if rows else "[colab] No active sessions found on server.")
elif cmd == "status" and os.environ.get("FAKE_COLAB_NO_KEYRING"):
    print("Warning: Could not read the OAuth token from the OS keyring (No recommended backend was available).",
          file=sys.stderr)
    print("To authorize colab-cli, visit this URL in any browser:", file=sys.stderr)
    sys.exit(1)                                   # input() without a terminal: EOFError
elif cmd == "status" and os.environ.get("FAKE_COLAB_STATUS_GARBAGE"):
    print("session table v2 follows")
elif cmd == "status":
    busy = "BUSY (exec(/x/scripts/colab_tpu.py))" if os.environ.get("FAKE_COLAB_BUSY") else "IDLE"
    print(f"[{{session}}] ep-{{session}} | Hardware: V6E1 | Shape: Standard | Variant: TPU | Status: {{busy}}"
          if os.path.exists(marker) else f"[colab] Session '{{session}}' not found.")
elif cmd == "new":
    if os.environ.get("FAKE_COLAB_NO_TPU") and "--tpu" in rest:
        print("[colab] Backend rejected accelerator 'V6E1'.", file=sys.stderr)
        sys.exit(1)
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    with open(marker, "w") as f:
        f.write(opt("--tpu", "cpu"))
    os.makedirs(here("content"), exist_ok=True)
    print("[colab] Session READY.")
elif cmd == "exec":
    mode = os.environ.get("FAKE_COLAB_EXEC", "record")
    os.makedirs(here("content"), exist_ok=True)
    if mode == "run":                  # the program itself, the way the CLI sends it: --env lines in front
        env = [rest[i + 1] for i, a in enumerate(rest) if a == "--env"]
        prelude = "import os\\n" + "".join(
            "os.environ[%r] = %r\\n" % tuple(e.split("=", 1)) for e in env)
        with open(opt("-f")) as f:
            code = prelude + f.read()
        subprocess.call([sys.executable, "-c", code], cwd=here("content"),
                        env=dict(os.environ, BES_ROOT=here("content")))
    elif mode in ("record", "fail"):   # what the supervisor leaves behind when it runs / when its setup failed
        os.makedirs(here("content/bes_cli"), exist_ok=True)
        info = {{"time": time.time(), "supervisor_started": time.time(), "state": "running", "note": ""}}
        if mode == "fail":
            info.update(state="released", note="the setup failed; the kernel is idle now")
            print("RuntimeError: this is not a TPU v6e", file=sys.stderr)
        with open(here("content/bes_cli/status.json"), "w") as f:
            json.dump(info, f)
    sys.exit(0)                        # also when the program raised in the kernel
elif cmd == "upload":
    local, remote = [a for a in rest if a not in ("-s", session)]
    if os.environ.get("FAKE_COLAB_HTTP_ERROR") or not os.path.isdir(os.path.dirname(here(remote))):
        print(f"[colab] Upload failed: 500 Server Error: Internal Server Error for url: {{URL}}")
        sys.exit(1)
    shutil.copyfile(local, here(remote))
    print(f"[colab] Uploaded '{{local}}' to '{{remote}}'")
elif cmd == "download":
    remote, local = [a for a in rest if a not in ("-s", session)]
    if os.environ.get("FAKE_COLAB_HTTP_ERROR"):
        print(f"[colab] Download failed: 502 Server Error: Bad Gateway for url: {{URL}}")
        sys.exit(1)
    if not os.path.isfile(here(remote)):
        print(f"[colab] Download failed: File or directory not found: {{remote}}")
        sys.exit(1)
    shutil.copyfile(here(remote), local)
    print(f"[colab] Downloaded '{{remote}}' to '{{local}}'")
elif cmd == "url":
    print("https://colab.research.google.com/notebooks/empty.ipynb?dbu=%2Ftun%2Fm%2Fep-1")
elif cmd == "stop":
    if os.path.exists(marker):
        os.remove(marker)
        print(f"[colab] Stopping session '{{session}}'...\\n[colab] Session terminated.")
    else:
        print(f"[colab] Session '{{session}}' not found.")
elif cmd == "drivemount":
    if os.environ.get("FAKE_COLAB_NO_DRIVE"):
        sys.exit(1)
    os.makedirs(here("content/drive/MyDrive"), exist_ok=True)
elif cmd == "ls":
    path = here([a for a in rest if a not in ("-s", session)][0])
    if not os.path.isdir(path):
        print("[colab] Error: File or directory not found")
        sys.exit(1)
    print("\\n".join(n + ("/" if os.path.isdir(os.path.join(path, n)) else "") for n in sorted(os.listdir(path))))
else:
    sys.exit(f"fake colab: unknown command {{cmd}}")
"""


@pytest.fixture
def fake_colab(tmp_path, monkeypatch):
    """A `colab` command on PATH and its home; returns (home, calls) where calls() lists the argvs."""
    home = tmp_path / "colab_home"
    home.mkdir()
    binary = tmp_path / "bin" / "colab"
    binary.parent.mkdir()
    binary.write_text(FAKE_COLAB.format(python=sys.executable))
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("COLAB_BIN", str(binary))
    monkeypatch.setenv("FAKE_COLAB_HOME", str(home))
    for name in ("COLAB_CLI_ARGS", "FAKE_COLAB_EXEC", "FAKE_COLAB_ORPHANS", "FAKE_COLAB_HTTP_ERROR",
                 "FAKE_COLAB_BUSY", "FAKE_COLAB_NO_DRIVE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)

    def calls(downloads: bool = False) -> list[list[str]]:
        path = home / "calls.jsonl"
        rows = [json.loads(ln) for ln in path.read_text().splitlines()] if path.exists() else []
        rows = [[re.sub(r"^BES_SENT_AT=\d+$", "BES_SENT_AT=<now>", a) for a in r] for r in rows]
        return rows if downloads else [r for r in rows if r[0] not in ("download", "ls")]

    return home, calls


def _session(home: Path, name: str = "bes24h", tpu: str = "v6e1") -> None:
    """A session this machine's CLI knows, with its runtime's /content."""
    (home / "sessions").mkdir(exist_ok=True)
    (home / "sessions" / name).write_text(tpu)
    (home / "content").mkdir(exist_ok=True)


def _status(home: Path, age: float = 5.0, state: str = "running", note: str = "") -> None:
    path = home / "content" / "bes_cli" / "status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"time": time.time() - age, "state": state, "note": note,
                                "supervisor_started": time.time() - age - 10}))


def test_start_signs_in_creates_the_v6e1_runtime_and_sends_the_program(fake_colab, capfd):
    """The commands of a first start, in order, as the Colab CLI 0.8.0 spells them: `sessions`
    (sign-in, then once more to see the account's runtimes), `status -s`, `new -s NAME --tpu
    v6e1`, `drivemount -s NAME` (the 24-hour run is mirrored to Drive unless --no-drive),
    `exec -s NAME --timeout 0 -f colab_tpu.py`; then the runtime's status says whether a
    supervisor runs (the exit status of `colab exec` does not)."""
    _, calls = fake_colab
    assert cli.main(["start"]) == 0
    assert calls() == [
        ["sessions"],
        ["status", "-s", "bes24h"],
        ["sessions"],
        ["new", "-s", "bes24h", "--tpu", "v6e1"],
        ["drivemount", "-s", "bes24h"],
        ["exec", "-s", "bes24h", "--timeout", "0", "-f", str(TPU),
         "--env", "BES_NOTEBOOK=colab_v6e_24h", "--env", "BES_ACTION=auto", "--env", "BES_SENT_AT=<now>",
         "--env", "BES_USE_DRIVE=1"],
    ]
    assert calls(downloads=True)[-1][:3] == ["download", "-s", "bes24h"]
    out = capfd.readouterr().out
    assert "compute units are charged from now on" in out and "From here on Ctrl-C only detaches" in out
    assert "the supervisor is running: running" in out
    assert cli.main(["start", "-s", "second", "--new-runtime", "--no-drive"]) == 0       # no mirror on request
    assert not any(c[:3] == ["drivemount", "-s", "second"] for c in calls()) and "BES_USE_DRIVE=0" in calls()[-1]


def test_overrides_reach_a_new_run(tmp_path, monkeypatch, fake_colab):
    """``start --override`` (repeatable) travels as BES_OVERRIDES and becomes the ``--override``
    options of a new run, in order; values keep their commas and equals signs; an item with a
    space, a semicolon or without a section is refused before anything is sent."""
    home, calls = fake_colab
    assert cli.main(["start", "--smoke", "--action", "start", "--override", "model.remat_keep=kernel",
                     "--override", "run.post_minutes=critic=0,judge=15,buffer=20"]) == 0
    env = [a for a in calls()[-1] if "=" in a]
    assert "BES_OVERRIDES=model.remat_keep=kernel;run.post_minutes=critic=0,judge=15,buffer=20" in env
    n = len(calls())
    for bad in ("model.remat_keep=a b", "remat_keep=kernel", "model.x=1;run.y=2"):
        assert cli.main(["start", "--smoke", "--override", bad]) != 0
    assert not any(c and c[0] == "exec" for c in calls()[n:])
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "content"
    _standin_repo(root)
    sup, out = _supervisor(root, [time.time()], BES_ACTION="start", BES_BUDGET_MINUTES="30",
                           BES_OVERRIDES="model.remat_keep=kernel;run.post_minutes=critic=0,judge=15,buffer=20")
    _setup(sup)
    _wait(lambda: len(_attempts(root)) == 1, "the run")
    argv = _attempts(root)[0]
    i = argv.index("--override")
    assert argv[i:i + 4] == ["--override", "model.remat_keep=kernel", "--override",
                             "run.post_minutes=critic=0,judge=15,buffer=20"]
    assert "OVERRIDES model.remat_keep=kernel run.post_minutes=critic=0,judge=15,buffer=20" in out.getvalue()
    (root / "bes_cli" / "ctl").write_text("stop\n")
    sup.tick()
    _wait(lambda: not sup.run_alive(), "the stop")


def test_start_options_reach_the_runtime(fake_colab, capfd):
    """What cell 1 of the notebook would hold is an option here (the runtime runs the branch on
    GitHub, not the local clone). The smoke run has a short hold of its own and no mirror; the
    CPU rehearsal has its own session and exists only for the smoke run."""
    home, calls = fake_colab
    token = home / "token"
    token.write_text("hf_standin")
    assert cli.main(["start", "--smoke", "--budget-minutes", "20", "--auto-resume", "0", "--action", "start",
                     "--hf-token-file", str(token), "--branch", "main", "--repo-url", "https://github.com/x/y",
                     "--part-mb", "4", "--config", "configs/canary.py"]) == 0
    assert ["new", "-s", "bes-smoke", "--tpu", "v5e1"] in calls()
    assert ["upload", "-s", "bes-smoke", str(token), "/content/bes_hf_token"] in calls()
    assert (home / "content" / "bes_hf_token").read_text() == "hf_standin"
    assert not any(c[0] == "drivemount" for c in calls())
    env = [a for a in calls()[-1] if "=" in a]
    assert env == ["BES_NOTEBOOK=colab_v5e_smoke", "BES_ACTION=start", "BES_SENT_AT=<now>", "BES_HOLD_MINUTES=20",
                   "BES_HOLD_FAILED_MINUTES=30", "BES_REPO_URL=https://github.com/x/y", "BES_BRANCH=main",
                   "BES_CONFIG=configs/canary.py", "BES_BUDGET_MINUTES=20", "BES_AUTO_RESUME=0", "BES_PART_MB=4"]
    assert cli.main(["start", "--smoke", "--cpu", "--drive", "--hold-minutes", "45",
                     "--drive-dir", "/content/drive/MyDrive/x", "--run-dir", "/content/runs/x",
                     "--move-deadline=+45"]) == 0
    assert ["new", "-s", "bes-cpu"] in calls() and ["drivemount", "-s", "bes-cpu"] in calls()
    env = [a for a in calls()[-1] if "=" in a]
    assert "BES_ALLOW_CPU=1" in env and "BES_USE_DRIVE=1" in env and "BES_HOLD_MINUTES=45" in env
    assert "BES_DRIVE_DIR=/content/drive/MyDrive/x" in env and "BES_RUN_DIR=/content/runs/x" in env
    assert "BES_MOVE_DEADLINE=+45" in env
    capfd.readouterr()
    before = len(calls())
    assert cli.main(["start", "--cpu"]) == 1                          # the 24-hour configuration on a CPU: no
    assert "use `start --smoke --cpu`" in capfd.readouterr().err and len(calls()) == before


def test_start_checks_first_and_says_when_a_charged_runtime_is_left_idle(fake_colab, monkeypatch, capfd):
    """A mistyped token path is found before the runtime is created. When a step fails after
    that (the Drive mount here), nothing was sent although the runtime exists: `start` says so."""
    _, calls = fake_colab
    assert cli.main(["start", "--hf-token-file", "no-such-file"]) == 1
    assert "does not exist" in capfd.readouterr().err and calls() == []
    monkeypatch.setenv("FAKE_COLAB_NO_DRIVE", "1")
    assert cli.main(["start"]) == 1
    err = capfd.readouterr().err
    assert "Drive was not mounted" in err and "--no-drive" in err
    assert "its runtime exists and is charged while it idles" in err and "down --yes" in err
    assert ["new", "-s", "bes24h", "--tpu", "v6e1"] in calls() and not any(c[0] == "exec" for c in calls())


def test_start_stops_when_the_runtime_is_refused(fake_colab, monkeypatch, capfd):
    _, calls = fake_colab
    monkeypatch.setenv("FAKE_COLAB_NO_TPU", "1")
    assert cli.main(["start"]) == 1
    assert "the runtime was not created" in capfd.readouterr().err
    assert not any(c[0] == "exec" for c in calls())


def test_start_reports_what_the_runtime_says_not_the_exit_status_of_exec(fake_colab, monkeypatch, capfd):
    """`colab exec` exits 0 for a program that raised in the kernel. The supervisor's status file
    knows: a failed setup makes `start` fail and say that the idle runtime is still charged."""
    home, _ = fake_colab
    monkeypatch.setenv("FAKE_COLAB_EXEC", "fail")
    assert cli.main(["start"]) == 1
    err = capfd.readouterr().err
    assert "no supervisor is running on 'bes24h'" in err and "the setup failed" in err and "still charged" in err
    monkeypatch.setenv("FAKE_COLAB_EXEC", "nothing")                # not even a status file
    (home / "content" / "bes_cli" / "status.json").unlink()
    assert cli.main(["start"]) == 1 and "no supervisor status on 'bes24h'" in capfd.readouterr().err
    assert cli.main(["start", "--detach", "--wait", "5", "--log", "bg.log"]) == 1   # the background exec ended
    monkeypatch.setenv("FAKE_COLAB_EXEC", "record")
    assert cli.main(["start", "--detach", "--wait", "20", "--log", "bg.log"]) == 0
    assert "the supervisor is running: running" in capfd.readouterr().out


def test_a_machine_without_a_keyring_is_told_how_to_keep_the_sign_in(fake_colab, monkeypatch, capfd):
    """Each `colab` command is its own process: without a keyring the CLI asks to sign in again at
    the next one, which the wrapper runs without a terminal. Checked on this machine: with
    keyrings.alt installed next to the CLI, keyring chooses its file backend."""
    _, calls = fake_colab
    monkeypatch.setenv("FAKE_COLAB_NO_KEYRING", "1")
    assert cli.main(["start"]) == 1
    err = capfd.readouterr().err
    assert "pip install keyrings.alt" in err and "COLAB_CLI_ARGS=--auth=adc" in err
    assert [c[0] for c in calls()] == ["sessions", "status"]          # nothing was created


def test_failures_of_the_cli_are_explained_by_their_cause():
    def said(text):
        return cli.explain("`colab status`", subprocess.CompletedProcess([], 1, "", text))

    locked = said("Warning: Keyring 'login' is locked; waiting on unlock prompt for 30s (use --auth=adc ...)")
    assert "keyring is locked" in locked and "PYTHON_KEYRING_BACKEND=keyrings.alt.file.PlaintextKeyring" in locked
    assert "no keyring it can use" in said("Could not read the OAuth token from the OS keyring (No recommended ...)")
    adc = said("DefaultCredentialsError: Your default credentials were not found.")
    assert "gcloud auth application-default login" in adc and "keyring" not in adc
    assert "wants you to sign in" in said("To authorize colab-cli, visit this URL in any browser:")
    other = said("502 Server Error for url: https://rt/api/contents/x?colab-runtime-proxy-token=abc.DEF-123&content=1")
    assert "502 Server Error" in other and "abc.DEF-123" not in other and "token=REDACTED&content=1" in other


def test_start_does_not_queue_behind_a_running_supervisor(fake_colab, monkeypatch, capfd):
    """A second `colab exec` on a session whose kernel runs the supervisor would wait for it to
    end and then act hours later: `start` sees the fresh status and does not send it. The CLI's
    own BUSY mark counts only while the runtime has no status at all (a start of a moment ago):
    a closed terminal leaves that mark behind for ever, and it must not stop a new supervisor
    when the old one is gone."""
    home, calls = fake_colab
    _session(home)
    _status(home, age=30)
    assert cli.main(["start", "--no-drive", "--hold-minutes", "5"]) == 0
    assert not any(c[0] in ("exec", "new") for c in calls())
    out = capfd.readouterr().out
    assert "a supervisor is already running" in out and "options of this start were not applied" in out
    _status(home, age=3600)                                          # stale: it died
    assert cli.main(["start", "--no-drive"]) == 0 and calls()[-1][0] == "exec"
    _status(home, age=5, state="released")                           # it said it left
    assert cli.main(["start", "--no-drive"]) == 0 and [c[0] for c in calls()].count("exec") == 2
    monkeypatch.setenv("FAKE_COLAB_BUSY", "1")
    _status(home, age=3600)                                          # the mark of a terminal that was closed
    assert cli.main(["start", "--no-drive"]) == 0 and [c[0] for c in calls()].count("exec") == 3
    (home / "content" / "bes_cli" / "status.json").unlink()          # no status yet: a start of a moment ago
    assert cli.main(["start", "--no-drive"]) == 0 and [c[0] for c in calls()].count("exec") == 3
    assert "repeat with --ignore-busy" in capfd.readouterr().out
    assert cli.main(["start", "--no-drive", "--ignore-busy"]) == 0 and [c[0] for c in calls()].count("exec") == 4


def test_a_failed_download_is_not_taken_for_a_missing_file(fake_colab, monkeypatch, capfd):
    """A 502 from the runtime's file interface is not "no supervisor" (which would send a second
    one) and not "not packaged yet": the wrapper stops and shows the CLI's message, without the
    runtime's token that the CLI prints in the URL."""
    home, calls = fake_colab
    _session(home)
    _status(home, age=30)
    monkeypatch.setenv("FAKE_COLAB_HTTP_ERROR", "1")
    for argv in (["start"], ["status"], ["send", "stop"], ["fetch", "--to", "got"]):
        assert cli.main(argv) == 1, argv
        err = capfd.readouterr().err
        assert "502 Server Error: Bad Gateway" in err and "SECRETTOKEN" not in err and "REDACTED" in err, argv
    assert not any(c[0] in ("exec", "new", "upload") for c in calls())
    monkeypatch.delenv("FAKE_COLAB_HTTP_ERROR")
    export = home / "content" / "bes_cli" / "export"
    export.mkdir()
    (export / "parts.json").write_text(json.dumps({"parts": [{"name": "final.tar.000", "bytes": 1, "sha256": "0"}],
                                                   "bytes": 1, "sha256": "0"}))
    assert cli.main(["fetch", "--to", "got"]) == 1                    # the index is there, the part is not
    assert "is not on the runtime any more" in capfd.readouterr().err


def test_start_does_not_create_a_second_runtime_next_to_one_it_does_not_know(fake_colab, monkeypatch, capfd):
    """The CLI keeps its sessions in a local file. From another machine the session of a running
    job looks missing; creating it there would charge a second runtime and, with Drive, continue
    the same run twice. `colab sessions` lists such a runtime as [?]: start refuses, status and
    down say what is the matter instead of "reclaimed" and "stopped"."""
    _, calls = fake_colab
    monkeypatch.setenv("FAKE_COLAB_ORPHANS", "m-s-1abc:V6E1")
    assert cli.main(["start"]) == 1
    err = capfd.readouterr().err
    assert "not in this machine's Colab CLI state" in err and "m-s-1abc (V6E1)" in err and "--new-runtime" in err
    assert not any(c[0] in ("new", "exec") for c in calls())
    assert cli.main(["status"]) == 1 and "started from another machine" in capfd.readouterr().err
    assert cli.main(["down", "--yes"]) == 1
    assert "nothing was stopped" in capfd.readouterr().err
    assert cli.main(["start", "--new-runtime"]) == 0 and ["new", "-s", "bes24h", "--tpu", "v6e1"] in calls()
    monkeypatch.setenv("FAKE_COLAB_STATUS_GARBAGE", "1")              # an answer the wrapper cannot read
    before = len(calls())
    assert cli.main(["start", "-s", "other"]) == 1
    assert "does not understand" in capfd.readouterr().err
    assert not any(c[0] in ("new", "exec") for c in calls()[before:])


def test_send_needs_a_supervisor_and_carries_the_part_size(fake_colab, capfd):
    home, calls = fake_colab
    _session(home)
    assert cli.main(["send", "stop"]) == 1                             # no status at all
    assert "no supervisor is running" in capfd.readouterr().err
    _status(home, age=5, state="released")
    assert cli.main(["send", "stop"]) == 1 and not any(c[0] == "upload" for c in calls())
    _status(home, age=5)
    assert cli.main(["send", "package", "--part-mb", "4"]) == 0
    assert (home / "content" / "bes_cli" / "ctl").read_text() == "package 4\n"
    _status(home, age=5, state="released")
    assert cli.main(["send", "--force", "release"]) == 0


def test_snapshot_downloads_a_copy_of_the_run_and_restore_uploads_it(fake_colab, monkeypatch, capfd, tmp_path):
    """`snapshot` asks a running supervisor (here: the real one, ticking in a thread on the
    stand-in runtime), waits for the new set and downloads it checked; `start --restore` uploads
    that set to another runtime before the program is sent, and checks it before anything is
    created."""
    home, calls = fake_colab
    _session(home)
    root = home / "content"
    _standin_repo(root)
    clock = [time.time()]
    sup, _ = _supervisor(root, clock, BES_PART_MB="0.004")
    _setup(sup)
    _wait(lambda: len(_attempts(root)) == 1 and (root / "runs" / "smoke" / "logs").is_dir(), "the run")
    stop = threading.Event()

    def supervise():
        while not stop.is_set():
            clock[0] = time.time()
            sup.tick()
            time.sleep(0.05)

    thread = threading.Thread(target=supervise, daemon=True)
    thread.start()
    try:
        _wait(lambda: (root / "bes_cli" / "status.json").exists(), "the first status")
        assert cli.main(["snapshot", "--to", "snap", "--wait", "30"]) == 0
        out = capfd.readouterr().out
        assert "asked the supervisor of 'bes24h' for a snapshot" in out and "SHA-256 verified" in out
        assert "start --action resume --restore" in out and f"--run-dir {root / 'runs' / 'smoke'}" in out
        meta, data = _tar_of(Path("snap"))
        assert meta["checkpoints"] == {"train/ckpt": "ckpt-000001-step-000000010"} and len(meta["parts"]) >= 3
        assert not (Path("snap") / ".incoming").exists()
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            assert "run/manifest.json" in tar.getnames() and "run.log" in tar.getnames()
        first = meta["created"]
        assert cli.main(["snapshot", "--to", "snap", "--wait", "30", "--part-mb", "0.016"]) == 0     # a newer one
        meta, _ = _tar_of(Path("snap"))
        assert meta["created"] > first and max(p["bytes"] for p in meta["parts"]) == int(0.016 * 2 ** 20)
        assert sorted(f.name for f in Path("snap").iterdir()) == sorted([p["name"] for p in meta["parts"]] + ["parts.json"])
        assert cli.main(["snapshot", "--to", "again", "--no-request"]) == 0                          # the last one
        assert _tar_of(Path("again"))[0]["sha256"] == meta["sha256"]
        capfd.readouterr()
    finally:
        stop.set()
        thread.join(timeout=10)
    (root / "runs" / "gate1").write_text("crash")
    _status(home, age=5, state="released")                       # no supervisor (and its status knows no snapshot)
    assert cli.main(["snapshot", "--to", "snap"]) == 1
    assert "only a running supervisor takes a snapshot" in capfd.readouterr().err
    assert cli.main(["snapshot", "--to", "snap", "--no-request"]) == 1
    assert "has not taken a snapshot yet" in capfd.readouterr().err
    assert _tar_of(Path("snap"))[0]["sha256"] == meta["sha256"]                                   # the copy is kept

    # another runtime: checked first, uploaded before the program is sent, parts.json last
    home2 = tmp_path / "colab_home2"
    home2.mkdir()
    monkeypatch.setenv("FAKE_COLAB_HOME", str(home2))
    Path("empty").mkdir()
    assert cli.main(["start", "--restore", "empty", "--no-drive"]) == 1
    assert "holds no snapshot" in capfd.readouterr().err and not (home2 / "calls.jsonl").exists()
    part = Path("again") / meta["parts"][1]["name"]
    part.write_bytes(part.read_bytes()[:-3])
    assert cli.main(["start", "--restore", "again", "--no-drive"]) == 1
    assert "missing or damaged" in capfd.readouterr().err and not (home2 / "calls.jsonl").exists()
    assert cli.main(["start", "--restore", "snap", "--action", "resume", "--no-drive", "--run-dir", "/content/runs/smoke",
                     "--move-deadline=+30"]) == 0
    rows = [json.loads(ln) for ln in (home2 / "calls.jsonl").read_text().splitlines()]
    ups = [r[-1] for r in rows if r[0] == "upload"]
    assert ups == [f"/content/bes_restore.{i:03d}" for i in range(len(meta["parts"]))] + ["/content/bes_restore.parts.json"]
    kinds = [r[0] for r in rows]
    assert kinds.index("new") < kinds.index("upload") and max(i for i, k in enumerate(kinds) if k == "upload") < kinds.index("exec")
    env = [a for a in rows[kinds.index("exec")] if "=" in a]
    assert "BES_RESTORE=1" in env and "BES_ACTION=resume" in env and "BES_MOVE_DEADLINE=+30" in env
    for i, p_ in enumerate(meta["parts"]):
        assert hashlib.sha256((home2 / "content" / f"bes_restore.{i:03d}").read_bytes()).hexdigest() == p_["sha256"]


def test_the_route_end_to_end_with_stand_ins(fake_colab, monkeypatch, capfd):
    """start (the program really runs, the way `colab exec` runs it, in a stand-in runtime),
    status, send, fetch, url and down."""
    home, calls = fake_colab
    _standin_repo(home / "content")
    monkeypatch.setenv("FAKE_COLAB_EXEC", "run")
    for k, v in _env(home / "content", BES_HOLD_MINUTES="0.005", STANDIN_AUTOFINISH="0.3").items():
        if k not in ("BES_ROOT", "BES_NOTEBOOK", "BES_ALLOW_CPU"):
            monkeypatch.setenv(k, v)
    assert cli.main(["start", "--smoke", "--cpu", "--hold-minutes", "0.005"]) == 0
    out = capfd.readouterr().out
    assert "the run is complete" in out and "export packaged" in out
    assert "the supervisor of 'bes-cpu' has left (the hold time is over" in out and "`down --yes`" in out
    where = ["--smoke", "-s", "bes-cpu"]
    assert cli.main(["status", *where]) == 0
    out = capfd.readouterr().out
    assert "supervisor: released" in out and "export       finished" in out and "the supervisor has left" in out
    assert cli.main(["fetch", *where, "--to", "got"]) == 0
    assert "charged until it is given back" in capfd.readouterr().out
    assert (Path("got") / "final" / "weights.bin").read_bytes() == bytes(range(256)) * 800
    assert json.loads((Path("got") / "final" / "ckpt" / "manifest.json").read_text()) == {"step": 10}
    assert not (Path("got") / ".parts").exists()
    assert cli.main(["url", *where]) == 0 and "colab.research.google.com" in capfd.readouterr().out
    assert cli.main(["down", *where]) == 1 and (home / "sessions" / "bes-cpu").exists()
    assert cli.main(["down", *where, "--yes"]) == 0 and not (home / "sessions" / "bes-cpu").exists()
    assert cli.main(["status", *where]) == 1 and "does not exist" in capfd.readouterr().err


def test_fetch_rejects_a_damaged_part_and_an_archive_that_is_not_the_export(fake_colab, monkeypatch, capfd, tmp_path):
    home, _ = fake_colab
    _session(home)
    export = home / "content" / "bes_cli" / "export"
    export.mkdir(parents=True)
    (export / "final.tar.000").write_bytes(b"damaged")
    (export / "parts.json").write_text(json.dumps({"parts": [
        {"name": "final.tar.000", "bytes": 7, "sha256": hashlib.sha256(b"intact!").hexdigest()}],
        "bytes": 7, "sha256": hashlib.sha256(b"intact!").hexdigest()}))
    assert cli.main(["fetch", "--to", "got"]) == 1
    err = capfd.readouterr().err
    assert "did not arrive intact" in err and "send package --part-mb 4" in err
    assert not (Path("got") / "final").exists()

    def archive(build) -> Path:
        path = tmp_path / "a.tar"
        with tarfile.open(path, "w") as tar:
            build(tar)
        return path

    def member(tar, name, data=b"x", kind=None):
        info = tarfile.TarInfo(name)
        info.size = len(data)
        if kind:
            info.type, info.linkname, info.size = kind, "/etc/passwd", 0
        tar.addfile(info, io.BytesIO(data) if not kind else None)

    for bad in (lambda tar: member(tar, "final/../../escape"), lambda tar: member(tar, "/abs/x"),
                lambda tar: member(tar, "other/x"), lambda tar: member(tar, "final/link", kind=tarfile.SYMTYPE)):
        with pytest.raises(cli.Failed, match="something unexpected"):
            cli._extract(archive(bad), tmp_path / "out")
    assert not (tmp_path / "out").exists() and not (tmp_path / "escape").exists()
    good = archive(lambda tar: (member(tar, "final/a.txt", b"one"), member(tar, "final/d/b.txt", b"two")))
    monkeypatch.delattr(tarfile, "data_filter", raising=False)        # a Python without extraction filters
    cli._extract(good, tmp_path / "out")
    assert (tmp_path / "out" / "final" / "d" / "b.txt").read_bytes() == b"two"


def test_without_the_colab_command_the_wrapper_says_how_to_install_it(monkeypatch, capfd):
    monkeypatch.setenv("COLAB_BIN", "colab-that-does-not-exist")
    assert cli.main(["status"]) == 1
    assert "pip install google-colab-cli" in capfd.readouterr().err


def test_documents_give_the_commands_of_the_cli_route():
    """RUNBOOK and README name the two scripts and give the commands the wrapper issues."""
    runbook = (REPO / "RUNBOOK.md").read_text()
    for needle in ("scripts/colab_cli.py", "scripts/colab_tpu.py", "pip install google-colab-cli",
                   "colab new -s bes24h --tpu v6e1", "colab exec -s bes24h --timeout 0 -f scripts/colab_tpu.py",
                   "https://github.com/googlecolab/google-colab-cli", "colab restart-kernel"):
        assert needle in runbook, needle
    readme = (REPO / "README.md").read_text()
    assert "scripts/colab_cli.py" in readme and "https://github.com/googlecolab/google-colab-cli" in readme
    assert textwrap.dedent(cli.__doc__).count("colab_cli.py") >= 6
