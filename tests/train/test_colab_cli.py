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


def stages(done, cur=None):
    dump(os.path.join(out, "stages.json"), {"stages": {
        name: {"status": "finished" if name in done else "started" if name == cur else "pending",
               "planned_s": 60.0, "used_s": 1.0, "reason": ""} for name in ORDER}})


now = time.time()
if not resume:
    dump(manifest, {"created": now, "config": args[args.index("--config") + 1],
                    "budget": {"t0": now, "deadline": now + 1800.0, "budget_s": 1800.0}})
signal.signal(signal.SIGTERM, lambda *a: sys.exit(3))
stages(ORDER[:3], "train")
dump(os.path.join(out, "train", "ledger.json"), {"tokens": 123456 * n, "steps": 10 * n, "seconds": {"train": 5.0, "lost": 0.0},
                                                  "restarts": n - 1, "rollbacks": 0})
with open(os.path.join(out, "train", "train_log.jsonl"), "a") as f:
    f.write(json.dumps({"event": "step", "step": 10 * n, "p": 0.1, "seq_len": 64, "loss": 5.0 - n,
                        "loss_ema": 5.0, "tok_per_s": 1000.0, "lr": 1.0}) + "\\n")
print("stand-in run attempt", n, flush=True)
word = "finish"
if not os.environ.get("STANDIN_AUTOFINISH"):
    gate = os.path.join(runs, f"gate{n}")
    deadline = time.time() + float(os.environ.get("STANDIN_MAX_S", "60"))
    while not os.path.exists(gate):
        if time.time() > deadline:
            sys.exit(4)
        time.sleep(0.02)
    with open(gate) as f:
        word = f.read().strip()
else:
    time.sleep(float(os.environ.get("STANDIN_AUTOFINISH")))
if word == "crash":
    sys.exit(1)
final = os.path.join(out, "final")
os.makedirs(os.path.join(final, "ckpt"), exist_ok=True)
with open(os.path.join(final, "weights.bin"), "wb") as f:
    f.write(bytes(range(256)) * 800)
dump(os.path.join(final, "config.json"), {"name": "stand-in"})
dump(os.path.join(final, "ckpt", "manifest.json"), {"step": 10 * n})
stages(ORDER)
dump(os.path.join(out, "stage_table.json"), {"complete": True})
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

    assert tpu.STAGES == stages.ORDER and tpu.DONE == stages.DONE
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
    assert tpu.COMMANDS == cli.COMMANDS
    assert {p["notebook"] for p in cli.PRESETS.values()} == set(tpu.NOTEBOOKS)


def test_settings_reject_what_they_do_not_know():
    assert tpu.settings({})["cells"] == ["2", "3", "4"] and tpu.settings({"BES_CELLS": ""})["cells"] == []
    assert tpu.settings({"BES_BRANCH": ""})["branch"] == ""          # the default branch of a fork
    for bad in ({"BES_NOTEBOOK": "x"}, {"BES_ACTION": "go"}, {"BES_USE_DRIVE": "maybe"},
                {"BES_BUDGET_MINUTES": "soon"}):
        with pytest.raises(RuntimeError):
            tpu.settings(bad)


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
    assert sup.tick() and sup.stop_requested and "past its deadline and still running" in out.getvalue()
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
FAKE_COLAB = '''#!{python}
"""Stand-in for the `colab` command: the runtime's files live under $FAKE_COLAB_HOME/content."""
import json, os, shutil, subprocess, sys

home = os.environ["FAKE_COLAB_HOME"]
args = sys.argv[1:]
with open(os.path.join(home, "calls.jsonl"), "a") as f:
    f.write(json.dumps(args) + "\\n")
cmd, rest = args[0], args[1:]


def opt(name, default=None):
    return rest[rest.index(name) + 1] if name in rest else default


def here(remote):                      # a path on the runtime
    return os.path.join(home, remote.lstrip("/"))


session = opt("-s")
marker = os.path.join(home, "sessions", session or "-")
if cmd == "sessions":
    print("[colab] No active sessions found on server.")
elif cmd == "status" and os.environ.get("FAKE_COLAB_NO_KEYRING"):
    print("Warning: Could not read the OAuth token from the OS keyring (No recommended backend was available).",
          file=sys.stderr)
    print("To authorize colab-cli, visit this URL in any browser:", file=sys.stderr)
    sys.exit(1)                                   # input() without a terminal: EOFError
elif cmd == "status":
    print(f"[{{session}}] ep-1 | Hardware: V6E1 | Shape: Standard | Variant: TPU | Status: IDLE"
          if os.path.exists(marker) else f"[colab] Session '{{session}}' not found.")
elif cmd == "new":
    if os.environ.get("FAKE_COLAB_NO_TPU") and "--tpu" in rest:
        print("[colab] Backend rejected accelerator 'V6E1'.", file=sys.stderr)
        sys.exit(1)
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    with open(marker, "w") as f:
        f.write(opt("--tpu", "cpu"))
    print("[colab] Session READY.")
elif cmd == "exec":
    if os.environ.get("FAKE_COLAB_EXEC") != "run":
        print("[fake] exec recorded")
        sys.exit(0)
    env = [rest[i + 1] for i, a in enumerate(rest) if a == "--env"]
    prelude = "import os\\n" + "".join(
        "os.environ[%r] = %r\\n" % tuple(e.split("=", 1)) for e in env)
    with open(opt("-f")) as f:
        code = prelude + f.read()
    os.makedirs(here("content"), exist_ok=True)
    child = dict(os.environ, BES_ROOT=here("content"))
    sys.exit(subprocess.call([sys.executable, "-c", code], cwd=here("content"), env=child))
elif cmd == "upload":
    local, remote = [a for a in rest if a not in ("-s", session)]
    os.makedirs(os.path.dirname(here(remote)), exist_ok=True)
    shutil.copyfile(local, here(remote))
    print(f"[colab] Uploaded '{{local}}' to '{{remote}}'")
elif cmd == "download":
    remote, local = [a for a in rest if a not in ("-s", session)]
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
    print("[colab] Session terminated.")
elif cmd == "drivemount":
    os.makedirs(here("content/drive/MyDrive"), exist_ok=True)
else:
    sys.exit(f"fake colab: unknown command {{cmd}}")
'''


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
    monkeypatch.delenv("COLAB_CLI_ARGS", raising=False)
    monkeypatch.chdir(tmp_path)

    def calls() -> list[list[str]]:
        path = home / "calls.jsonl"
        return [json.loads(ln) for ln in path.read_text().splitlines()] if path.exists() else []

    return home, calls


def test_start_signs_in_creates_the_v6e1_runtime_and_sends_the_program(fake_colab, capfd):
    """The commands of a first start, in order, as the Colab CLI 0.8.0 spells them: `sessions`
    (sign-in), `status -s`, `new -s NAME --tpu v6e1`, `exec -s NAME --timeout 0 -f colab_tpu.py`."""
    _, calls = fake_colab
    assert cli.main(["start"]) == 0
    assert calls() == [
        ["sessions"],
        ["status", "-s", "bes24h"],
        ["new", "-s", "bes24h", "--tpu", "v6e1"],
        ["exec", "-s", "bes24h", "--timeout", "0", "-f", str(TPU),
         "--env", "BES_NOTEBOOK=colab_v6e_24h", "--env", "BES_ACTION=auto"],
    ]
    out = capfd.readouterr().out
    assert "compute units are charged from now on" in out and "Ctrl-C only detaches" in out


def test_start_options_reach_the_runtime(fake_colab):
    home, calls = fake_colab
    token = home / "token"
    token.write_text("hf_standin")
    assert cli.main(["start", "--smoke", "--budget-minutes", "20", "--no-drive", "--auto-resume", "0",
                     "--hold-minutes", "30", "--action", "start", "--hf-token-file", str(token),
                     "--branch", "main", "--repo-url", "https://github.com/x/y"]) == 0
    assert ["new", "-s", "bes-smoke", "--tpu", "v5e1"] in calls()
    assert ["upload", "-s", "bes-smoke", str(token), "/content/bes_hf_token"] in calls()
    assert (home / "content" / "bes_hf_token").read_text() == "hf_standin"
    env = [a for a in calls()[-1] if "=" in a]
    assert env == ["BES_NOTEBOOK=colab_v5e_smoke", "BES_ACTION=start", "BES_REPO_URL=https://github.com/x/y",
                   "BES_BRANCH=main", "BES_BUDGET_MINUTES=20", "BES_USE_DRIVE=0", "BES_AUTO_RESUME=0",
                   "BES_HOLD_MINUTES=30"]
    assert cli.main(["start", "--cpu", "-s", "rehearsal", "--drive"]) == 0
    assert ["new", "-s", "rehearsal"] in calls() and ["drivemount", "-s", "rehearsal"] in calls()
    assert "BES_ALLOW_CPU=1" in calls()[-1] and "BES_USE_DRIVE=1" in calls()[-1]


def test_start_stops_when_the_runtime_is_refused(fake_colab, monkeypatch, capfd):
    _, calls = fake_colab
    monkeypatch.setenv("FAKE_COLAB_NO_TPU", "1")
    assert cli.main(["start"]) == 1
    assert "the runtime was not created" in capfd.readouterr().err
    assert not any(c[0] == "exec" for c in calls())


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


def test_start_does_not_queue_behind_a_running_supervisor(fake_colab, capfd):
    """A second `colab exec` on a session whose kernel runs the supervisor would wait for it to
    end and then hold the runtime again: `start` sees the fresh status and does not send it."""
    home, calls = fake_colab
    (home / "sessions").mkdir()
    (home / "sessions" / "bes24h").write_text("v6e1")
    status = home / "content" / "bes_cli" / "status.json"
    status.parent.mkdir(parents=True)
    status.write_text(json.dumps({"time": time.time() - 30, "state": "running"}))
    assert cli.main(["start"]) == 0
    assert not any(c[0] in ("exec", "new") for c in calls())
    assert "a supervisor is already running" in capfd.readouterr().out
    status.write_text(json.dumps({"time": time.time() - 3600, "state": "running"}))     # stale: it died
    assert cli.main(["start"]) == 0 and calls()[-1][0] == "exec"
    status.write_text(json.dumps({"time": time.time() - 5, "state": "released"}))       # it said it left
    assert cli.main(["start"]) == 0 and [c[0] for c in calls()].count("exec") == 2


def test_the_route_end_to_end_with_stand_ins(fake_colab, monkeypatch, capfd):
    """start (the program really runs, the way `colab exec` runs it, in a stand-in runtime),
    status, send, fetch, url and down."""
    home, calls = fake_colab
    _standin_repo(home / "content")
    monkeypatch.setenv("FAKE_COLAB_EXEC", "run")
    for k, v in _env(home / "content", BES_HOLD_MINUTES="0.005", STANDIN_AUTOFINISH="0.3").items():
        if k not in ("BES_ROOT", "BES_NOTEBOOK", "BES_ALLOW_CPU"):
            monkeypatch.setenv(k, v)
    assert cli.main(["start", "--smoke", "--cpu"]) == 0
    out = capfd.readouterr().out
    assert "the run is complete" in out and "export packaged" in out and "detached from 'bes-smoke'" in out
    assert cli.main(["status", "--smoke"]) == 0
    out = capfd.readouterr().out
    assert "supervisor: released" in out and "export       finished" in out and "the supervisor has left" in out
    assert cli.main(["send", "--smoke", "package"]) == 0
    assert (home / "content" / "bes_cli" / "ctl").read_text().strip() == "package"
    assert cli.main(["fetch", "--smoke", "--to", "got"]) == 0
    assert (Path("got") / "final" / "weights.bin").read_bytes() == bytes(range(256)) * 800
    assert json.loads((Path("got") / "final" / "ckpt" / "manifest.json").read_text()) == {"step": 10}
    assert not (Path("got") / ".parts").exists()
    assert cli.main(["url", "--smoke"]) == 0 and "colab.research.google.com" in capfd.readouterr().out
    assert cli.main(["down", "--smoke"]) == 1 and (home / "sessions" / "bes-smoke").exists()
    assert cli.main(["down", "--smoke", "--yes"]) == 0 and not (home / "sessions" / "bes-smoke").exists()
    assert cli.main(["status", "--smoke"]) == 1 and "does not exist any more" in capfd.readouterr().err


def test_fetch_rejects_a_damaged_part(fake_colab, capfd):
    home, _ = fake_colab
    export = home / "content" / "bes_cli" / "export"
    export.mkdir(parents=True)
    (export / "final.tar.000").write_bytes(b"damaged")
    (export / "parts.json").write_text(json.dumps({"parts": [
        {"name": "final.tar.000", "bytes": 7, "sha256": hashlib.sha256(b"intact!").hexdigest()}],
        "bytes": 7, "sha256": hashlib.sha256(b"intact!").hexdigest()}))
    assert cli.main(["fetch", "--to", "got"]) == 1
    assert "did not arrive intact" in capfd.readouterr().err and not (Path("got") / "final").exists()


def test_without_the_colab_command_the_wrapper_says_how_to_install_it(monkeypatch, capfd):
    monkeypatch.setenv("COLAB_BIN", "colab-that-does-not-exist")
    assert cli.main(["status"]) == 1
    assert "pip install google-colab-cli" in capfd.readouterr().err


def test_documents_give_the_commands_of_the_cli_route():
    """RUNBOOK and README name the two scripts and give the commands the wrapper issues."""
    runbook = (REPO / "RUNBOOK.md").read_text()
    for needle in ("scripts/colab_cli.py", "scripts/colab_tpu.py", "pip install google-colab-cli",
                   "colab new -s bes24h --tpu v6e1", "colab exec -s bes24h --timeout 0 -f scripts/colab_tpu.py",
                   "https://github.com/googlecolab/google-colab-cli"):
        assert needle in runbook, needle
    readme = (REPO / "README.md").read_text()
    assert "scripts/colab_cli.py" in readme and "https://github.com/googlecolab/google-colab-cli" in readme
    assert textwrap.dedent(cli.__doc__).count("colab_cli.py") >= 6
