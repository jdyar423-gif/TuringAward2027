"""The run on a Colab TPU, started from a terminal with the Colab CLI (github.com/googlecolab/google-colab-cli).

    colab new -s bes24h --tpu v6e1
    colab exec -s bes24h --timeout 0 -f scripts/colab_tpu.py

or, with the wrapper that also checks, reports and fetches: ``python scripts/colab_cli.py start``.

``colab exec -f`` reads this file on your machine and sends its text to the kernel of the Colab
runtime: it runs THERE, alone. The repository is not on the runtime yet, so this file imports
nothing from it, has no ``__file__``, and takes its settings from environment variables
(``colab exec --env KEY=VALUE``, which puts lines in front of this text: no ``__future__`` import
may follow them).

What it does, in the kernel:

1. clones the repository (``BES_REPO_URL``, ``BES_BRANCH``) to ``/content/TuringAward2027``;
2. runs the cells of the notebook ``notebooks/<BES_NOTEBOOK>.ipynb`` from that clone, the same code
   a person runs in the browser: 1 (parameters), 1b (helpers), 2 (runtime check), 3 (the code on
   the local disk), 4 (dependencies with jax held), then 6 (a new run) or 8 (continue one). When
   the run is already running on the runtime (the program was sent again after a kernel restart
   or an interrupt), only 1 and 1b run: the runtime check would need the TPU the run holds;
3. stays in the kernel as the run's supervisor until the run has ended and the hold time is over.

Why it stays. A runtime created by the Colab CLI lives while its kernel executes something: the CLI
has no keep-alive, and Colab reclaims the runtime of an idle kernel (its documentation gives no
time; an earlier version reports minutes). The run itself is a separate process, so without this
loop the runtime, its disk and the run would be gone soon after the start. While this loop runs,
every other ``colab exec`` on the session waits behind it. Talk to it through files instead (the
Jupyter contents API of ``colab upload`` / ``colab download`` does not use the kernel):

    /content/bes_cli/status.txt     the progress report of notebook cell 7, rewritten every minute
    /content/bes_cli/status.json    the same moment for programs (time, state, run_alive, ...)
    /content/bes_cli/ctl            a one-word command you upload: stop | resume | package | release
    /content/bes_cli/export/        the export as numbered parts of a tar file, with parts.json

``stop`` sends SIGTERM to the run (it writes a checkpoint and exits; the supervisor keeps the
runtime), ``resume`` continues it (cell 8), ``package`` rebuilds the export parts (``package 4``:
in parts of 4 MB), ``release`` ends the supervisor once the run is not running: the kernel is
idle from then on and Colab takes the runtime back.

What the supervisor decides by itself, so that an unattended runtime neither loses a run nor is
kept for ever:

- the run is complete by the run's own rule: every stage finished or skipped, or failed and not
  essential (preflight, background, benchmark and train are essential). Then cell 12 runs (the
  export's verification, its copy to Drive), the export is packaged, and the runtime is kept for
  BES_HOLD_MINUTES;
- a run that ended incomplete is continued (cell 8) up to BES_AUTO_RESUME times, not after
  ``stop``; when it is not continued, the runtime is kept for BES_HOLD_FAILED_MINUTES and an
  export the run wrote on its way out is packaged all the same;
- a run that was asked to stop and does not exit gets a second SIGTERM and then SIGKILL; a run
  still running long after its deadline is asked to stop; a complete run whose process lingers
  is killed;
- a round that fails is reported and the loop goes on; BES_MAX_HOURS after the last start,
  continuation or command the supervisor leaves whatever the state.

Killing the local ``colab exec`` (Ctrl-C, a closed laptop) does not stop any of this: the CLI
only closes its connection and the kernel keeps executing.

Settings (all optional):

    BES_NOTEBOOK          colab_v6e_24h (default) or colab_v5e_smoke
    BES_REPO_URL, BES_BRANCH   what to clone (defaults: this repository and its branch)
    BES_ACTION            auto (default: continue a run that exists, else start one) | start | resume
    BES_CELLS             the numbered setup cells to run, default "2,3,4"
    BES_USE_DRIVE         auto (default: mirror to Drive when it is mounted) | 0 | 1
    BES_DRIVE_DIR, BES_CONFIG, BES_BUDGET_MINUTES, BES_ALLOW_CPU   replace the notebook's parameters
    BES_AUTO_RESUME       how often a run that died is continued without you, default 2
    BES_HOLD_MINUTES      how long the runtime is kept after the run is complete, default 120
    BES_HOLD_FAILED_MINUTES   ... and after a run that ended incomplete, default 240
    BES_OVERRUN_MINUTES   a run still running this long after its own deadline (or after the start
                          of this attempt, when that was later) is stopped, default 180
    BES_STOP_GRACE_MINUTES    a run that was asked to stop gets a second SIGTERM after this long
                          (it exits at once on it), default 30; BES_KILL_GRACE_MINUTES later (5)
                          its process group is killed
    BES_LINGER_MINUTES    a complete run whose process has not exited is killed after this, default 10
    BES_MAX_HOURS         the supervisor leaves this long after its start or the last start,
                          continuation or command, whatever the state, default 40
    BES_PART_MB           the size of the export's parts, default 16
    BES_SENT_AT           when the program was sent (seconds since the epoch, the sender's clock):
                          a program that starts more than BES_QUEUE_MINUTES (30) later has waited
                          behind another execution and leaves without touching anything
    BES_ROOT              the runtime's working directory, default /content (tests change it)
    HF_TOKEN              a Hugging Face token; better: upload it as /content/bes_hf_token (the CLI
                          writes every --env value into its local history file); the file is
                          removed from the runtime once it is read

Mounting Drive is interactive: run ``colab drivemount -s <session>`` in your terminal before the
start if you want the mirror of ``RUNBOOK.md`` section 4. Nothing here was run on Colab.
"""
import contextlib
import hashlib
import io
import json
import os
import re
import signal
import subprocess
import sys
import tarfile
import threading
import time

REPO_URL = "https://github.com/jdyar423-gif/TuringAward2027"
BRANCH = "claude/modest-carson-yi2d9g"
NOTEBOOKS = ("colab_v6e_24h", "colab_v5e_smoke")
#: bes.train.stages.ORDER (a test keeps the two equal)
STAGES = ("preflight", "background", "benchmark", "train", "sample", "critic", "prefopt", "select",
          "judge", "export")
DONE = ("finished", "skipped")
#: the stages whose failure stops the run (the default of bes.train.supervisor.Supervisor; a test
#: keeps the two equal). A failed stage outside this set is over: the run goes on without it.
ESSENTIAL = ("preflight", "background", "benchmark", "train")
COMMANDS = ("stop", "resume", "package", "release")


def _flag(value, default):
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _number(env, name, default):
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise RuntimeError(f"{name}={raw!r} is not a number") from None


def settings(env):
    """The settings of this program from the environment (the table in the module docstring)."""
    s = {
        "notebook": env.get("BES_NOTEBOOK") or NOTEBOOKS[0],
        "repo_url": env.get("BES_REPO_URL") or REPO_URL,
        "branch": env.get("BES_BRANCH", BRANCH),
        "root": (env.get("BES_ROOT") or "/content").rstrip("/") or "/",
        "action": (env.get("BES_ACTION") or "auto").strip().lower(),
        "cells": [c.strip() for c in (env.get("BES_CELLS") if env.get("BES_CELLS") is not None else "2,3,4").split(",")
                  if c.strip()],
        "use_drive": (env.get("BES_USE_DRIVE") or "auto").strip().lower(),
        "drive_dir": env.get("BES_DRIVE_DIR") or None,
        "config": env.get("BES_CONFIG") or None,
        "budget_minutes": _number(env, "BES_BUDGET_MINUTES", None),
        "allow_cpu": None if not env.get("BES_ALLOW_CPU") else _flag(env.get("BES_ALLOW_CPU"), False),
        "auto_resume": int(_number(env, "BES_AUTO_RESUME", 2)),
        "hold_s": 60.0 * _number(env, "BES_HOLD_MINUTES", 120),
        "hold_failed_s": 60.0 * _number(env, "BES_HOLD_FAILED_MINUTES", 240),
        "overrun_s": 60.0 * _number(env, "BES_OVERRUN_MINUTES", 180),
        "stop_grace_s": 60.0 * _number(env, "BES_STOP_GRACE_MINUTES", 30),
        "kill_grace_s": 60.0 * _number(env, "BES_KILL_GRACE_MINUTES", 5),
        "linger_s": 60.0 * _number(env, "BES_LINGER_MINUTES", 10),
        "max_s": 3600.0 * _number(env, "BES_MAX_HOURS", 40),
        "poll_s": _number(env, "BES_POLL_S", 15),
        "status_s": _number(env, "BES_STATUS_S", 60),
        "heartbeat_s": _number(env, "BES_HEARTBEAT_S", 600),
        "resume_gap_s": _number(env, "BES_RESUME_GAP_S", 60),
        "part_bytes": int(_number(env, "BES_PART_MB", 16) * 2 ** 20),
        "sent_at": _number(env, "BES_SENT_AT", None),
        "queue_s": 60.0 * _number(env, "BES_QUEUE_MINUTES", 30),
    }
    if s["notebook"] not in NOTEBOOKS:
        raise RuntimeError(f"BES_NOTEBOOK={s['notebook']!r}: choose one of {', '.join(NOTEBOOKS)}")
    if s["action"] not in ("auto", "start", "resume"):
        raise RuntimeError(f"BES_ACTION={s['action']!r}: choose auto, start or resume")
    if s["use_drive"] not in ("auto", "0", "1", "false", "true", "no", "yes", "off", "on"):
        raise RuntimeError(f"BES_USE_DRIVE={s['use_drive']!r}: choose auto, 0 or 1")
    s["repo_dir"] = os.path.join(s["root"], "TuringAward2027")
    s["cli_dir"] = os.path.join(s["root"], "bes_cli")
    return s


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write(path, text):
    """Replace ``path`` with ``text`` in one step (a reader never sees half a file)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _secs(x):
    if not isinstance(x, (int, float)):
        return "-"
    return f"{x / 3600:.2f} h" if x >= 3600 else f"{x / 60:.1f} min" if x >= 60 else f"{x:.0f} s"


class Supervisor:
    """Sets the run up with the notebook's cells, then keeps the kernel busy until it is over."""

    def __init__(self, env=None, out=None, clock=time.time, sleep=time.sleep):
        self.s = settings(os.environ if env is None else env)
        if env is None:
            # the kernel keeps os.environ between executions: without this, a later start would
            # inherit the options of an earlier one
            for key in [k for k in os.environ if k.startswith("BES_")]:
                del os.environ[key]
        self.out = out or sys.stdout
        self.clock, self.sleep = clock, sleep
        self.ns = {"__name__": "bes_colab_cli"}
        self.cells = {}
        self.state = "setting up"
        self.note = ""
        self.resumes_left = self.s["auto_resume"]
        self.resumes = 0
        self.stop_requested = False
        self.stop_sent_at = None          # when the run was first sent SIGTERM, and how far it has escalated
        self.stop_step = 0
        self.complete_at = None           # when a complete run was first seen still running
        self.act_failures = 0
        self.hold_until = None
        self.export = None
        self.finished_handled = False
        self.last_start = 0.0
        self.t0 = self.last_activity = self.clock()
        self.log_path = os.path.join(self.s["cli_dir"], "supervisor.log")

    # ------------------------------------------------------------------ plumbing
    def say(self, text):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.clock()))}] {text}"
        print(line, file=self.out, flush=True)
        try:
            os.makedirs(self.s["cli_dir"], exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def path(self, name):
        return os.path.join(self.s["cli_dir"], name)

    def run_cell(self, label):
        """Run one numbered code cell of the notebook in the shared namespace."""
        if label not in self.cells:
            raise RuntimeError(f"notebook {self.s['notebook']} has no code cell '# {label}.'")
        exec(compile(self.cells[label], f"<{self.s['notebook']} cell {label}>", "exec"), self.ns)

    # ------------------------------------------------------------------ setup
    def clone(self):
        s = self.s
        if os.path.isdir(os.path.join(s["repo_dir"], "bes")):
            self.say(f"using the code in {s['repo_dir']}")
            return
        os.makedirs(s["root"], exist_ok=True)
        cmd = ["git", "clone", "--depth", "1"] + (["--branch", s["branch"]] if s["branch"] else [])
        cmd += [s["repo_url"], s["repo_dir"]]
        self.say("cloning: " + " ".join(cmd))
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"git clone failed with exit status {p.returncode}:\n{(p.stdout + p.stderr)[-3000:]}")

    def load_cells(self):
        path = os.path.join(self.s["repo_dir"], "notebooks", self.s["notebook"] + ".ipynb")
        with open(path, encoding="utf-8") as f:
            nb = json.load(f)
        for c in nb["cells"]:
            if c["cell_type"] != "code":
                continue
            src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            m = re.match(r"# (\d+b?)\.", src)
            if m:
                self.cells[m.group(1)] = src

    def parameters(self):
        """Cell 1, then the values this program sets: where things are on this runtime, the Drive
        mirror only when Drive is mounted, and what the environment asks for."""
        s, ns = self.s, self.ns
        self.run_cell("1")
        root = s["root"]

        def moved(path):          # the notebook's /content paths under another root (tests)
            return path if root == "/content" or not path.startswith("/content") else root + path[len("/content"):]

        ns.update(REPO_URL=s["repo_url"], BRANCH=s["branch"], CODE_COPY="", REPO_DIR=s["repo_dir"],
                  RUN_DIR=moved(ns["RUN_DIR"]), DRIVE_DIR=s["drive_dir"] or moved(ns["DRIVE_DIR"]))
        if s["config"]:
            ns["CONFIG"] = s["config"]
        if s["budget_minutes"] is not None:
            b = s["budget_minutes"]
            ns["BUDGET_MINUTES"] = int(b) if float(b).is_integer() else b
        if s["allow_cpu"] is not None:
            ns["ALLOW_CPU"] = s["allow_cpu"]
        # the helpers first (they only need the paths): from here on the run can be asked whether
        # it is running, whatever goes wrong below
        self.run_cell("1b")
        self.run_dir = ns["RUN_DIR"]
        mounted = os.path.isdir(os.path.join(root, "drive", "MyDrive"))
        want = s["use_drive"]
        if want == "auto":
            ns["USE_DRIVE"] = mounted
            if not mounted:
                self.say("no Drive mirror: Drive is not mounted on this runtime (run `colab drivemount -s "
                         "<session>` in your terminal before the start to have one)")
        else:
            ns["USE_DRIVE"] = _flag(want, False)
            if ns["USE_DRIVE"] and not mounted:
                raise RuntimeError("BES_USE_DRIVE=1 but Drive is not mounted on this runtime: run `colab drivemount "
                                   "-s <session>` in your terminal first (mounting needs you), or start without it")
        self.say(f"notebook {s['notebook']}: CONFIG {ns['CONFIG']}, RUN_DIR {ns['RUN_DIR']}, BUDGET_MINUTES "
                 f"{ns['BUDGET_MINUTES']}, USE_DRIVE {ns['USE_DRIVE']}, ALLOW_CPU {ns['ALLOW_CPU']}")

    def setup_cells(self):
        ns = self.ns
        for label in self.s["cells"]:
            self.state = f"setting up (cell {label})"
            self.write_status(report=False)
            self.say(f"cell {label}: " + self.cells.get(label, "?").splitlines()[0][:90])
            if label == "3":
                # cell 3 mounts Drive when USE_DRIVE is set, and mounting waits for a person. Here
                # Drive is mounted already or not used, so the cell runs with the mirror off.
                keep, ns["USE_DRIVE"] = ns["USE_DRIVE"], False
                try:
                    self.run_cell(label)
                finally:
                    ns["USE_DRIVE"] = keep
            else:
                self.run_cell(label)
        if "3" not in self.s["cells"]:
            os.chdir(self.s["repo_dir"])

    def hf_token(self):
        path = os.path.join(self.s["root"], "bes_hf_token")
        token = ""
        try:
            with open(path, encoding="utf-8") as f:
                token = f.read().strip()
            os.remove(path)
        except OSError:
            pass
        if token:
            os.environ["HF_TOKEN"] = token
        self.say("HF_TOKEN " + ("set" if os.environ.get("HF_TOKEN") else
                                "not set: public files only, lower rate limits"))

    # ------------------------------------------------------------------ the run
    def run_alive(self):
        """The notebook's pid check, and that the pid still is the run (a pid can be reused)."""
        ns = self.ns
        if not ns["pid_alive"]():
            return False
        try:
            pid = int(open(ns["PID_FILE"]).read().strip())
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode("utf-8", "replace")
        except (OSError, ValueError):
            return True
        return "run_24h.py" in cmd

    def stages(self):
        return (_read_json(os.path.join(self.run_dir, "stages.json")) or {}).get("stages") or {}

    def run_state(self):
        """'none' (no run directory), 'complete' or 'incomplete', by the run's own rule: a stage is
        over when it is finished or skipped, or failed and not essential."""
        if not os.path.exists(os.path.join(self.run_dir, "manifest.json")):
            return "none"
        st = self.stages()

        def over(name):
            status = (st.get(name) or {}).get("status")
            return status in DONE or (status == "failed" and name not in ESSENTIAL)

        return "complete" if all(over(n) for n in STAGES) else "incomplete"

    def failed_stages(self):
        st = self.stages()
        return [n for n in STAGES if (st.get(n) or {}).get("status") == "failed"]

    def start_or_resume(self):
        ns, action = self.ns, self.s["action"]
        on_drive = bool(ns["USE_DRIVE"]) and os.path.exists(os.path.join(ns["DRIVE_DIR"], "manifest.json"))
        state = self.run_state()
        if self.run_alive():
            self.say("the run is already running on this runtime: supervising it")
        elif state == "complete":
            self.say(f"the run in {self.run_dir} is complete: nothing to start")
        elif state == "incomplete" or on_drive:
            if action == "start":
                raise RuntimeError(f"{self.run_dir} (or its Drive mirror) already holds a run: use BES_ACTION=resume "
                                   "or auto to continue it, or another RUN_DIR for a new one")
            self.resume("a run exists" + (" on Drive" if state == "none" else ""))
        else:
            if action == "resume":
                raise RuntimeError(f"no run to continue in {self.run_dir}")
            self.run_cell("6")
            self.last_start = self.last_activity = self.clock()

    def resume(self, why):
        self.say(f"continuing the run ({why}): cell 8")
        self.stop_sent_at, self.stop_step, self.complete_at = None, 0, None
        self.run_cell("8")
        self.last_start = self.last_activity = self.clock()
        self.resumes += 1

    def overrun(self, now):
        """How late the run is, in seconds, or None. Late: still running BES_OVERRUN_MINUTES after
        its own deadline (manifest.json), or after the start of this attempt when that was later
        (a run continued after its deadline still has its export to write)."""
        b = (_read_json(os.path.join(self.run_dir, "manifest.json")) or {}).get("budget") or {}
        deadline = b.get("deadline")
        if not isinstance(deadline, (int, float)):
            return None
        base = max(float(deadline), self.last_start or self.t0)
        return now - base if now > base + self.s["overrun_s"] else None

    def signal_run(self, sig, group=False):
        """Send ``sig`` to the run (to its whole process group: the cells start it in a session of
        its own, so its pid is the group's). False when it could not be sent."""
        try:
            pid = int(open(self.ns["PID_FILE"]).read().strip())
            (os.killpg if group else os.kill)(pid, sig)
            return True
        except (OSError, ValueError) as e:
            self.say(f"could not signal the run: {e}")
            return False

    def stop_run(self, why):
        """The first SIGTERM: the run stops at its next check and writes a checkpoint.
        ``escalate`` follows up when it does not go away."""
        if self.signal_run(signal.SIGTERM):
            self.stop_sent_at, self.stop_step = self.clock(), 1
            self.say(f"SIGTERM sent to the run ({why}): it stops at its next check and writes a checkpoint")

    def escalate(self, now):
        """For a run that is running although it should not be. Asked to stop: a second SIGTERM
        after the grace time (the run exits at once on it), then SIGKILL for its process group.
        Complete but not exited: SIGKILL after the linger time (its work is on the disk)."""
        s = self.s
        if self.stop_sent_at is not None:
            waited = now - self.stop_sent_at
            if self.stop_step == 1 and waited >= s["stop_grace_s"]:
                self.stop_step = 2
                self.say(f"the run is still running {_secs(waited)} after SIGTERM: sending it again "
                         "(the run exits at once on the second one)")
                self.signal_run(signal.SIGTERM)
            elif self.stop_step == 2 and waited >= s["stop_grace_s"] + s["kill_grace_s"]:
                self.stop_step = 3
                self.say("the run did not exit: SIGKILL to its process group")
                self.signal_run(signal.SIGKILL, group=True)
        elif self.run_state() == "complete":
            if self.complete_at is None:
                self.complete_at = now
            elif now - self.complete_at >= s["linger_s"]:
                self.complete_at = now
                self.say(f"the run is complete but its process has not exited for {_secs(s['linger_s'])}: "
                         "SIGKILL to its process group")
                self.signal_run(signal.SIGKILL, group=True)
        else:
            self.complete_at = None

    # ------------------------------------------------------------------ reporting
    def brief(self):
        """One line about the run: stage, tokens and loss, budget."""
        run = self.run_dir
        st = (_read_json(os.path.join(run, "stages.json")) or {}).get("stages") or {}
        cur = next((n for n in STAGES if (st.get(n) or {}).get("status") == "started"), None)
        done = sum((st.get(n) or {}).get("status") in DONE for n in STAGES)
        parts = [f"stage {cur or '-'} ({done}/{len(STAGES)} done)"]
        led = _read_json(os.path.join(run, "train", "ledger.json")) or {}
        if led:
            parts.append(f"{led.get('tokens', 0):.3e} tokens, {led.get('steps', 0):,} steps")
        last = self.last_step()
        if last:
            parts.append(f"loss {last.get('loss', float('nan')):.4f}, {last.get('tok_per_s', 0):,.0f} tok/s")
        b = (_read_json(os.path.join(run, "manifest.json")) or {}).get("budget") or {}
        if b:
            now = self.clock()
            parts.append(f"elapsed {_secs(now - b['t0'])}, left {_secs(max(0.0, b['deadline'] - now))}")
        return " | ".join(parts)

    def last_step(self):
        path = os.path.join(self.run_dir, "train", "train_log.jsonl")
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - 65536))
                lines = f.read().decode("utf-8", "replace").splitlines()
        except OSError:
            return None
        for line in reversed(lines):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict) and r.get("event") == "step":
                return r
        return None

    def write_status(self, alive=None, report=True):
        """status.json and status.txt (the report of notebook cell 7 under a header; without it
        while the setup cells run, whose output the report's capture would swallow)."""
        now = self.clock()
        info = {"time": now, "state": self.state, "note": self.note, "run_alive": alive,
                "run_dir": getattr(self, "run_dir", None), "notebook": self.s["notebook"],
                "resumes": self.resumes, "auto_resumes_left": self.resumes_left,
                "hold_until": self.hold_until, "export": self.export, "supervisor_started": self.t0,
                "stop_step": self.stop_step, "pid": os.getpid()}
        head = (f"supervisor: {self.state}" + (f" ({self.note})" if self.note else "") + "\n"
                f"written {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))} (runtime clock), "
                f"supervising for {_secs(now - self.t0)}, continued {self.resumes} time(s), "
                f"automatic continuations left {self.resumes_left}\n")
        if self.hold_until:
            head += f"the runtime is kept for another {_secs(max(0.0, self.hold_until - now))}\n"
        if self.export:
            head += (f"export packaged: {len(self.export['parts'])} parts, {self.export['bytes'] / 1e6:.1f} MB "
                     f"in {self.path('export')}\n")
        body = ""
        if report and "show_progress" in self.ns:
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    self.ns["show_progress"]()
            except Exception as e:  # noqa: BLE001 - a report never stops the supervisor
                buf.write(f"\n(progress report failed: {type(e).__name__}: {e})\n")
            body = buf.getvalue()
        try:
            _write(self.path("status.txt"), head + "\n" + body)
            _write(self.path("status.json"), json.dumps(info))
        except OSError as e:
            print(f"status not written: {e}", file=self.out, flush=True)

    def command(self):
        """The one-word command of the control file, consumed whatever it holds; None when there
        is none (or nothing usable in it)."""
        path = self.path("ctl")
        try:
            with open(path, "rb") as f:
                raw = f.read(4096)
        except OSError:
            return None
        try:
            os.remove(path)
        except OSError:
            pass
        text = raw.decode("utf-8", "replace").strip().lower()
        word, _, arg = text.partition(" ")
        if word not in COMMANDS:
            self.say(f"ignoring the command {text[:40]!r}: expected one of {', '.join(COMMANDS)}")
            return None
        self.say(f"command: {text[:40]}")
        if word == "package" and arg.strip():
            try:
                mb = float(arg)
                if not 0.01 <= mb <= 256:
                    raise ValueError
                self.s["part_bytes"] = int(mb * 2 ** 20)
            except ValueError:
                self.say(f"ignoring the part size {arg.strip()[:20]!r} (megabytes, 0.01 to 256)")
        return word

    # ------------------------------------------------------------------ the export
    def package(self):
        """The export directory as numbered parts of one tar file (each fits one contents request)."""
        final = os.path.join(self.run_dir, "final")
        if not os.path.isdir(final):
            self.say(f"no export to package: {final} does not exist")
            return None
        out = self.path("export")
        os.makedirs(out, exist_ok=True)
        for name in os.listdir(out):
            os.remove(os.path.join(out, name))
        tar_path = os.path.join(out, "final.tar")
        with tarfile.open(tar_path, "w") as tar:
            tar.add(final, arcname="final")
        whole, parts = hashlib.sha256(), []
        with open(tar_path, "rb") as f:
            while True:
                chunk = f.read(self.s["part_bytes"])
                if not chunk:
                    break
                whole.update(chunk)
                name = f"final.tar.{len(parts):03d}"
                with open(os.path.join(out, name), "wb") as g:
                    g.write(chunk)
                parts.append({"name": name, "bytes": len(chunk), "sha256": hashlib.sha256(chunk).hexdigest()})
        os.remove(tar_path)
        self.export = {"parts": parts, "bytes": sum(p["bytes"] for p in parts), "sha256": whole.hexdigest(),
                       "created": self.clock()}
        _write(os.path.join(out, "parts.json"), json.dumps(self.export))
        self.say(f"export packaged: {len(parts)} parts, {self.export['bytes'] / 1e6:.1f} MB, sha256 "
                 f"{self.export['sha256'][:16]}... in {out}")
        return self.export

    def finish(self):
        """Once, when the run is complete: cell 12 (the export to Drive, its verification), the parts."""
        failed = self.failed_stages()
        self.say("the run is complete" + (f" (failed and left out: {', '.join(failed)})" if failed else "")
                 + ": " + self.brief())
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                self.run_cell("12")
        except Exception as e:  # noqa: BLE001 - the export stays where it is; say why the cell failed
            buf.write(f"\ncell 12 failed: {type(e).__name__}: {e}\n")
        for line in buf.getvalue().strip().splitlines()[-12:]:
            self.say("  " + line)
        try:
            self.package()
        except OSError as e:
            self.say(f"the export could not be packaged: {e}")

    # ------------------------------------------------------------------ the loop
    def act(self, now):
        """Commands and the run: (leave, command, alive)."""
        s = self.s
        alive = self.run_alive()
        word = self.command()
        if word:
            self.last_activity = now
        if word == "release" and alive:
            self.say("not released: the run is running, and it would die with the runtime. Send stop first "
                     "(or `colab stop -s <session>` to give up the runtime and the run at once)")
        elif word == "release":
            return True, word, alive
        if word == "stop":
            self.stop_requested = True
            if alive:
                self.stop_run("you asked")
            else:
                self.say("the run is not running")
        elif word == "resume":
            self.stop_requested = False
            if alive:
                self.say("the run is running: nothing to continue")
            elif self.run_state() == "complete":
                self.say("the run is complete: nothing to continue")
            else:
                self.resume("you asked")
                alive, self.hold_until = self.run_alive(), None
        elif word == "package":
            try:
                self.package()
            except OSError as e:
                self.say(f"the export could not be packaged: {e}")
        if alive and not self.stop_requested and self.stop_sent_at is None:
            late = self.overrun(now)
            if late is not None:
                self.stop_requested = True
                self.say(f"the run is still running {_secs(late)} after its deadline (or after the start of this "
                         "attempt, when that was later): stopping it. Send resume if that is wrong")
                self.stop_run("past its deadline")
        if alive:
            self.escalate(now)
            self.state = "stopping" if self.stop_sent_at is not None else "running"
            self.note, self.hold_until = "", None
            return False, word, alive
        self.stop_sent_at, self.stop_step, self.complete_at = None, 0, None
        state = self.run_state()
        if state == "complete":
            if not self.finished_handled:
                self.finished_handled = True
                try:
                    self.finish()
                except Exception as e:  # noqa: BLE001 - the hold time below must start all the same
                    self.say(f"finishing failed ({type(e).__name__}: {e}); the export stays in the run directory")
            if self.hold_until is None:
                self.hold_until = self.clock() + s["hold_s"]
            self.state, self.note = "complete", "fetch the export, then send release (or wait)"
        elif (state == "incomplete" and not self.stop_requested and self.resumes_left > 0
              and now - self.last_start >= s["resume_gap_s"]):
            self.resumes_left -= 1
            try:
                self.resume(f"the run ended incomplete; {self.resumes_left} automatic continuation(s) left")
            except Exception as e:  # noqa: BLE001 - keep the runtime and say what happened
                self.say(f"continuing failed: {type(e).__name__}: {e}")
            alive = self.run_alive()
            self.state = "running" if alive else "stopped"
        elif state == "incomplete" and not self.stop_requested and self.resumes_left > 0:
            self.state, self.note = "stopped", "continuing shortly"
        else:
            if self.hold_until is None:
                self.hold_until = now + s["hold_failed_s"]
                failed = self.failed_stages()
                self.say("the run is not running and is not continued automatically ("
                         + ("you stopped it" if self.stop_requested else
                            "no run" if state == "none" else "no automatic continuation left")
                         + (f"; failed: {', '.join(failed)}" if failed else "")
                         + f"): the runtime is kept for {_secs(s['hold_failed_s'])}; "
                         "send resume, or release to let it go")
                if os.path.isdir(os.path.join(self.run_dir, "final")):
                    self.say("the run directory holds an export (the run writes one when an essential stage "
                             "fails or it is interrupted near its deadline): packaging it for fetch")
                    try:
                        self.package()
                    except Exception as e:  # noqa: BLE001
                        self.say(f"the export could not be packaged: {type(e).__name__}: {e}")
            self.state = "stopped"
            self.note = "send resume to continue, release to give the runtime back"
        return False, word, alive

    def tick(self):
        """One round of the supervisor. Returns False when it is time to leave. Whatever goes
        wrong with the commands or the run, the hold time, the absolute limit and the status are
        still looked after: they are what ends a runtime nobody uses."""
        s, now = self.s, self.clock()
        leave, word, alive = False, None, None
        try:
            leave, word, alive = self.act(now)
            self.act_failures = 0
        except Exception as e:  # noqa: BLE001 - leaving would hand the runtime back with the run on it
            self.act_failures += 1
            if self.act_failures <= 3 or self.act_failures % 100 == 0:
                self.say(f"a supervisor round failed ({type(e).__name__}: {e}); staying "
                         f"(failure {self.act_failures} in a row)")
        if leave:
            self.state, self.note = "released", "the kernel is idle now"
        elif self.hold_until is not None and now >= self.hold_until:
            leave, self.state, self.note = True, "released", "the hold time is over; the kernel is idle now"
            self.say("the hold time is over")
        elif now - self.last_activity >= s["max_s"]:
            leave, self.state = True, "released"
            self.note = f"nothing was started, continued or commanded for {_secs(s['max_s'])}"
            self.say(f"leaving: {self.note} (BES_MAX_HOURS); a run that is still running does not keep the runtime")
        if leave:
            self.write_status(alive)
            return False
        if now - getattr(self, "_status_at", 0.0) >= s["status_s"] or word:
            self._status_at = now
            self.write_status(alive)
        if now - getattr(self, "_beat_at", 0.0) >= s["heartbeat_s"]:
            self._beat_at = now
            extra = f" | runtime kept for {_secs(max(0.0, self.hold_until - now))}" if self.hold_until else ""
            self.say(f"run {'RUNNING' if alive else self.state.upper()} | {self.brief()}{extra}")
        return True

    def alive_or_false(self):
        try:
            return self.run_alive()
        except Exception:  # noqa: BLE001 - before cell 1b there is nothing to ask
            return False

    def _setup_beat(self, done):
        """While the setup runs (the dependencies can take 15 silent minutes), keep the status
        fresh: it is how `colab_cli.py start` sees that a supervisor already holds this kernel."""
        while not done.wait(30.0):
            self.write_status(report=False)

    def loop(self):
        """Rounds until one says to leave. A round that fails is reported and the loop goes on:
        leaving on an error would hand the runtime back with the run on it."""
        failures = 0
        while True:
            try:
                if not self.tick():
                    return
                failures = 0
            except Exception as e:  # noqa: BLE001
                failures += 1
                if failures <= 3 or failures % 100 == 0:
                    self.say(f"a supervisor round failed ({type(e).__name__}: {e}); staying "
                             f"(failure {failures} in a row)")
            self.sleep(self.s["poll_s"])

    def main(self):
        s = self.s
        if s["sent_at"] is not None and self.t0 - s["sent_at"] > s["queue_s"]:
            # a second `colab exec` on a busy kernel waits until the first has ended: hours later,
            # after a stop or a release, it must not continue the run or hold the runtime again
            print(f"this program was sent {_secs(self.t0 - s['sent_at'])} ago and waited behind another execution "
                  "(or the clocks of your machine and the runtime differ by more than "
                  f"{_secs(s['queue_s'])}): leaving without touching anything. Start again if you want a "
                  "supervisor now.", file=self.out, flush=True)
            return
        self.say(f"supervisor of the BES run on this runtime (notebook {s['notebook']}, action {s['action']})")
        try:                                  # a command sent when no supervisor was there is not for this one
            if os.path.getmtime(self.path("ctl")) < self.t0:
                os.remove(self.path("ctl"))
                self.say("dropped a command file that was older than this supervisor")
        except OSError:
            pass
        self.write_status(report=False)
        done = threading.Event()
        beat = threading.Thread(target=self._setup_beat, args=(done,), daemon=True)
        beat.start()
        try:
            self.clone()
            self.load_cells()
            self.parameters()
            if self.run_alive():
                # nothing to set up, and the runtime check of cell 2 must not run: it opens the TPU,
                # which the run holds
                self.say("the run is already running on this runtime: supervising it (the setup cells are skipped)")
                os.chdir(s["repo_dir"])
            else:
                self.setup_cells()
                self.hf_token()
                self.start_or_resume()
        except BaseException as e:
            done.set()
            beat.join(timeout=5.0)
            if isinstance(e, Exception) and self.alive_or_false():
                self.say(f"the setup failed ({type(e).__name__}: {e}), but the run is running: supervising it")
            else:
                self.state, self.note = "released", "the setup failed; the kernel is idle now"
                self.write_status(report=False)
                raise
        done.set()
        beat.join(timeout=5.0)
        self.state = "running"
        self.say(f"supervising: status in {self.path('status.txt')}, commands ({', '.join(COMMANDS)}) as one word in "
                 f"{self.path('ctl')}. This cell runs until the run is over; ending the local `colab exec` "
                 "does not end it.")
        try:
            self.loop()
        except KeyboardInterrupt:
            self.state, self.note = "released", "the kernel was interrupted"
            self.write_status()
            self.say("interrupted: the supervisor leaves; a running run keeps running, but nothing keeps the "
                     "runtime alive any more. Start this program again to supervise it.")
            return
        self.say("the supervisor leaves: the kernel is idle from now on and Colab takes the runtime back "
                 "(`colab stop -s <session>` releases it at once). A run that is still running does not keep it.")


if __name__ == "__main__":
    Supervisor().main()
