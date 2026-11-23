"""The run on a Colab TPU from your terminal, through the Colab CLI.

    pip install google-colab-cli          # https://github.com/googlecolab/google-colab-cli (Linux, macOS)
    python scripts/colab_cli.py start --smoke      # 30 minutes on a v5e-1 TPU: do this first ...
    python scripts/colab_cli.py down --smoke --yes # ... and give its runtime back
    python scripts/colab_cli.py start              # the 24-hour run on a v6e-1 TPU (mounts Drive for the mirror)
    python scripts/colab_cli.py status             # the progress report
    python scripts/colab_cli.py url                # a browser link that attaches Colab to the same runtime
    python scripts/colab_cli.py send stop          # stop | resume | package | release | snapshot
    python scripts/colab_cli.py snapshot --to snap # a copy of the run's records and newest checkpoint, here
    python scripts/colab_cli.py fetch --to export  # the exported model, checked by SHA-256
    python scripts/colab_cli.py down --yes         # give the runtime back: this is what ends the charges

``start`` signs you in when needed (the Colab CLI prints an authorization link and asks for the
code Google shows: paste it in that terminal, nowhere else), creates the session
(``colab new -s bes24h --tpu v6e1``) and sends ``scripts/colab_tpu.py`` to its kernel
(``colab exec -s bes24h --timeout 0 -f scripts/colab_tpu.py``). That program clones this
repository on the runtime, runs the notebook's own cells and stays in the kernel as the run's
supervisor, because a runtime made by the CLI lives only while its kernel executes something.
Read its docstring for the files it keeps under ``/content/bes_cli`` and what the commands do.

The 24-hour run mirrors its records, newest checkpoint and export to Google Drive
(``RUNBOOK.md`` section 4), as the notebook does: ``start`` mounts Drive first, which asks you
to authorise in a browser (``--no-drive`` runs without the mirror; then a runtime that Colab
takes back takes the run with it). The smoke run has no mirror unless you pass ``--drive``.

Once it says "== starting", ``start`` stays attached and prints the supervisor's lines; from
then on Ctrl-C, a closed terminal or a lost connection only detach it: the run and its
supervisor go on. ``--detach`` starts it in the background. Before that line nothing runs yet,
although the runtime may already exist and be charged. While the supervisor runs, every other
``colab exec`` on the session waits behind it (and ``colab restart-kernel`` would kill it), so
this wrapper talks to the runtime through files (``colab upload``/``download``).

``release`` and the end of the supervisor's hold time only stop keeping the runtime alive: it
is charged until Colab reclaims it. ``down --yes`` (``colab stop``) gives it back at once.

The Colab CLI keeps its sessions in a file on the machine that created them: use this wrapper
from that machine. From another one the session looks missing, and ``start`` refuses to create a
second runtime while the account has one this machine does not know (``--new-runtime`` overrides).

The runtime runs the branch as it is on GitHub, not your local clone: what you would set in
the notebook's cell 1 is an option here (``--budget-minutes``, ``--config``, ``--drive-dir``,
``--run-dir`` for a run next to another one on the same runtime: ``RUNBOOK.md`` section 2.2).
The commands can be typed by hand as well; ``RUNBOOK.md`` section 2.1 says what is lost then.
``COLAB_BIN`` names another ``colab`` executable, ``COLAB_CLI_ARGS`` adds global options to
every call (for example ``--auth=adc``). Nothing here was run against Colab: the wrapper is
tested with a stand-in for the ``colab`` command (``tests/train/test_colab_cli.py``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

PROGRAM = Path(__file__).resolve().with_name("colab_tpu.py")
REMOTE = "/content/bes_cli"
#: drive: the mirror to Google Drive unless --no-drive (the 24-hour run), or only with --drive.
#: env: what the smoke run sets unless you do: a short hold, so its runtime is not kept for hours.
PRESETS = {
    "final": {"session": "bes24h", "notebook": "colab_v6e_24h", "tpu": "v6e1", "drive": True, "env": {}},
    "smoke": {"session": "bes-smoke", "notebook": "colab_v5e_smoke", "tpu": "v5e1", "drive": None,
              "env": {"BES_HOLD_MINUTES": "20", "BES_HOLD_FAILED_MINUTES": "30"}},
}
CPU_SESSION = "bes-cpu"             # start --smoke --cpu: its own session, never the TPU one
COMMANDS = ("stop", "resume", "package", "release", "snapshot")
RESTORE_PREFIX = "bes_restore."     # scripts/colab_tpu.py reads <root>/bes_restore.parts.json and .NNN
STALE_S = 240.0                     # a status older than this: no supervisor is writing it
INSTALL = ("the `colab` command was not found. Install the Colab CLI (Linux or macOS, Python 3.12+):\n"
           "    pip install google-colab-cli        (or: uv tool install google-colab-cli)\n"
           "    https://github.com/googlecolab/google-colab-cli")
ADC = ("    gcloud auth application-default login --scopes=openid,"
       "https://www.googleapis.com/auth/cloud-platform,"
       "https://www.googleapis.com/auth/userinfo.email,"
       "https://www.googleapis.com/auth/colaboratory\n"
       "then set COLAB_CLI_ARGS=--auth=adc for this wrapper.")
NO_KEYRING = ("The Colab CLI could not keep its sign-in: this machine has no keyring it can use (a server, WSL,\n"
              "a container), so every command asks to sign in again. Either give it a file store with\n"
              "    pip install keyrings.alt        (same environment as google-colab-cli; plaintext, owner-only)\n"
              "or use Application Default Credentials: sign in once with\n" + ADC)
LOCKED_KEYRING = ("The Colab CLI could not read its sign-in: this machine's keyring is locked (usual over SSH).\n"
                  "Unlock it, or make the CLI use a file store: `pip install keyrings.alt` next to the CLI and\n"
                  "set PYTHON_KEYRING_BACKEND=keyrings.alt.file.PlaintextKeyring, or use Application Default\n"
                  "Credentials: sign in once with\n" + ADC)
_TOKEN = re.compile(r"(colab-runtime-proxy-token=)[^&\s'\"]+")
_SESSION = re.compile(r"^\[(.+?)\] (\S+) \| Hardware: (\S+)")


class Failed(Exception):
    """A step did not work; the message says what to do."""


def text_of(p: subprocess.CompletedProcess) -> str:
    """What a `colab` command printed, without the runtime's token (the CLI puts it in error URLs)."""
    return _TOKEN.sub(r"\1REDACTED", (p.stdout + p.stderr).strip())


def explain(what: str, p: subprocess.CompletedProcess) -> str:
    """The message for a `colab` command that failed, with the usual causes spelled out."""
    out = text_of(p)
    low = out.lower()
    if "is locked" in low and "keyring" in low:
        return f"{what} failed.\n{LOCKED_KEYRING}"
    if "keyring" in low:
        return f"{what} failed.\n{NO_KEYRING}"
    if "default credentials" in low:
        return f"{what} failed: no usable Application Default Credentials. Sign in with\n{ADC}"
    if "authorization code" in out or "To authorize colab-cli" in out or "EOFError" in out:
        return (f"{what} failed because the Colab CLI wants you to sign in: run `colab sessions` in this terminal "
                "(it prints a link and asks for the code Google shows), then repeat the command.")
    return f"{what} failed:\n{out[-1500:]}"


class Colab:
    """The ``colab`` command for one session."""

    def __init__(self, session: str):
        self.session = session
        self.binary = os.environ.get("COLAB_BIN") or "colab"
        self.extra = shlex.split(os.environ.get("COLAB_CLI_ARGS", ""))
        if shutil.which(self.binary) is None:
            raise Failed(INSTALL)

    def argv(self, *args: str) -> list[str]:
        return [self.binary, *self.extra, *args]

    def quiet(self, *args: str, timeout: float = 300.0) -> subprocess.CompletedProcess:
        """Run and capture. No terminal is passed on, so a sign-in prompt fails instead of waiting."""
        try:
            return subprocess.run(self.argv(*args), capture_output=True, text=True, stdin=subprocess.DEVNULL,
                                  timeout=timeout)
        except subprocess.TimeoutExpired:
            raise Failed(f"`{' '.join(self.argv(*args))}` did not answer within {timeout:.0f} s") from None

    def shown(self, *args: str) -> int:
        """Run with this terminal (prompts and output are yours)."""
        return subprocess.call(self.argv(*args))

    def status_line(self) -> str | None:
        """The line of `colab status -s NAME` for a session this machine's CLI knows; None when it
        says the session is not found. Anything else is an error: guessing "not found" would
        create a second runtime."""
        p = self.quiet("status", "-s", self.session)
        if p.returncode != 0:
            raise Failed(explain("`colab status`", p))
        for line in p.stdout.splitlines():
            if line.startswith(f"[{self.session}] "):
                return line
        if f"Session '{self.session}' not found" in p.stdout:
            return None
        raise Failed(f"`colab status -s {self.session}` answered something this wrapper does not understand "
                     f"(another version of the Colab CLI?):\n{text_of(p)[-800:]}")

    def exists(self) -> bool:
        return self.status_line() is not None

    def sessions(self) -> list[dict]:
        """The runtimes of the account as `colab sessions` lists them: name ('?' for one this
        machine's CLI does not know), endpoint, hardware."""
        p = self.quiet("sessions")
        if p.returncode != 0:
            raise Failed(explain("`colab sessions`", p))
        rows = [_SESSION.match(line.strip()) for line in p.stdout.splitlines()]
        return [{"name": m.group(1), "endpoint": m.group(2), "hardware": m.group(3)} for m in rows if m]

    def download(self, remote: str, local: Path) -> bool:
        """True when the file arrived, False when it does not exist on the runtime. Every other
        failure is raised with the CLI's own message: it is not the same as "not there"."""
        p = self.quiet("download", "-s", self.session, remote, str(local), timeout=1800.0)
        if p.returncode == 0 and local.is_file():
            return True
        if "File or directory not found" in p.stdout + p.stderr:
            return False
        raise Failed(explain(f"the download of {remote}", p))

    def drive_mounted(self) -> bool:
        """True when the runtime has Drive mounted (asked through the file interface, not the kernel)."""
        p = self.quiet("ls", "-s", self.session, "/content/drive")
        return p.returncode == 0 and "MyDrive" in p.stdout

    def upload(self, local: Path, remote: str) -> None:
        p = self.quiet("upload", "-s", self.session, str(local), remote, timeout=1800.0)
        if p.returncode != 0:
            raise Failed(explain(f"the upload to {remote}", p))


def unknown_runtimes(colab: Colab) -> list[dict]:
    return [r for r in colab.sessions() if r["name"] == "?"]


def missing(colab: Colab) -> str:
    """Why this machine cannot see the session, for a message."""
    others = unknown_runtimes(colab)
    if not others:
        return (f"session '{colab.session}' does not exist: Colab reclaimed its runtime, or it was never created "
                "(`colab sessions` lists what the account has)")
    listed = ", ".join(f"{r['endpoint']} ({r['hardware']})" for r in others)
    return (f"session '{colab.session}' is not in this machine's Colab CLI state, but the account has "
            f"{len(others)} runtime(s) this machine does not know: {listed}. If one of them is the run, it was "
            "started from another machine (or with another --config): use the wrapper there. A runtime nobody "
            "needs can be stopped in the Colab web page (Manage sessions).")


def remote_status(colab: Colab) -> tuple[dict | None, str]:
    """(status.json, status.txt) of the supervisor on the runtime; (None, '') when there is none."""
    with tempfile.TemporaryDirectory() as tmp:
        js, txt = Path(tmp) / "status.json", Path(tmp) / "status.txt"
        if not colab.download(f"{REMOTE}/status.json", js):
            return None, ""
        try:
            info = json.loads(js.read_text(encoding="utf-8"))
        except ValueError:
            return None, ""
        text = txt.read_text(encoding="utf-8", errors="replace") if colab.download(f"{REMOTE}/status.txt", txt) else ""
        return info, text


def supervisor_age(info: dict | None) -> float | None:
    """Seconds since the supervisor last wrote its status (this machine's clock against the
    runtime's); None when it is not running (no status, or it said that it left)."""
    if not info or info.get("state") == "released" or not isinstance(info.get("time"), (int, float)):
        return None
    return time.time() - float(info["time"])


def running(info: dict | None) -> bool:
    age = supervisor_age(info)
    return age is not None and age < STALE_S


def environment(args: argparse.Namespace, preset: dict, drive: bool | None) -> list[str]:
    """The ``--env KEY=VALUE`` arguments of ``colab exec`` for these options. BES_SENT_AT lets the
    program see that it waited behind another execution (it then leaves without acting)."""
    env = {"BES_NOTEBOOK": preset["notebook"], "BES_ACTION": args.action, "BES_SENT_AT": f"{time.time():.0f}"}
    env.update(preset["env"])
    if args.repo_url:
        env["BES_REPO_URL"] = args.repo_url
    if args.branch is not None:
        env["BES_BRANCH"] = args.branch
    if args.config:
        env["BES_CONFIG"] = args.config
    if getattr(args, "override", None):
        env["BES_OVERRIDES"] = ";".join(checked_overrides(args.override))
    if args.budget_minutes is not None:
        env["BES_BUDGET_MINUTES"] = f"{args.budget_minutes:g}"
    if args.cpu:
        env["BES_ALLOW_CPU"] = "1"
    if drive is not None:
        env["BES_USE_DRIVE"] = "1" if drive else "0"
    if args.drive_dir:
        env["BES_DRIVE_DIR"] = args.drive_dir
    if args.run_dir:
        env["BES_RUN_DIR"] = args.run_dir
    if args.move_deadline:
        env["BES_MOVE_DEADLINE"] = args.move_deadline
    if args.restore:
        env["BES_RESTORE"] = "1"
    if args.auto_resume is not None:
        env["BES_AUTO_RESUME"] = str(args.auto_resume)
    if args.hold_minutes is not None:
        env["BES_HOLD_MINUTES"] = f"{args.hold_minutes:g}"
    if args.part_mb is not None:
        env["BES_PART_MB"] = f"{args.part_mb:g}"
    out: list[str] = []
    for k, v in env.items():
        out += ["--env", f"{k}={v}"]
    return out


def after_start(colab: Colab, hints: str) -> int:
    """What the runtime says once `colab exec` has returned (its exit status does not: it is 0 for
    a program that failed in the kernel and 1 for a connection that merely dropped)."""
    info, _ = remote_status(colab)
    if info is None:
        print(f"== no supervisor status on '{colab.session}': the program did not get as far as writing one. "
              "The runtime is idle and still charged until Colab reclaims it: fix the cause and start again, "
              "or give it back (down --yes).", file=sys.stderr)
        return 1
    note = f" ({info.get('note')})" if info.get("note") else ""
    if running(info):
        print(f"== detached from '{colab.session}'; the supervisor is running: {info.get('state')}{note}\n{hints}")
        return 0
    if info.get("state") == "released" and "failed" not in str(info.get("note")):
        print(f"== the supervisor of '{colab.session}' has left{note}. Nothing keeps the runtime alive now, but it "
              "is charged until Colab reclaims it: `down --yes` gives it back at once.")
        return 0
    print(f"== no supervisor is running on '{colab.session}': last state {info.get('state')}{note}. The runtime is "
          "idle and still charged until Colab reclaims it: fix the cause and start again, or give it back "
          "(down --yes).", file=sys.stderr)
    return 1


def checked_overrides(items) -> list[str]:
    """The ``--override`` items of ``start`` (``section.field=value``; no space and no ``;`` in
    one, because they travel in one environment variable), or :class:`Failed`."""
    for item in items:
        if not re.fullmatch(r"[A-Za-z_]+\.[A-Za-z_0-9]+=[^\s;]+", item):
            raise Failed(f"--override {item!r}: write section.field=value without spaces or ';'")
    return list(items)


def cmd_start(args: argparse.Namespace) -> int:
    preset = PRESETS["smoke" if args.smoke else "final"]
    checked_overrides(args.override)                     # before anything is created and charged
    if args.cpu and not args.smoke:
        raise Failed("--cpu is for a rehearsal of this route with the smoke run: use `start --smoke --cpu`. "
                     "(Without --smoke it would run the 24-hour configuration on a CPU runtime.)")
    session = args.session or (CPU_SESSION if args.cpu else preset["session"])
    colab = Colab(session)
    if not PROGRAM.is_file():
        raise Failed(f"{PROGRAM} is missing")
    token = Path(args.hf_token_file).expanduser() if args.hf_token_file else None
    if token is not None and not token.is_file():        # before anything is created and charged
        raise Failed(f"{token} does not exist")
    restore = local_snapshot(Path(args.restore)) if args.restore else None    # before anything is charged
    drive = args.drive if args.drive is not None else preset["drive"]
    flag = (" --smoke" if args.smoke else "") + (f" -s {session}" if session != preset["session"] else "")
    print(f"== signing in if needed, and the sessions of this account (`{' '.join(colab.argv('sessions'))}`)")
    if colab.shown("sessions") != 0:
        raise Failed("`colab sessions` failed (see above): sign in first; it prints a link and asks for the code "
                     "Google shows")
    line = colab.status_line()
    if line is not None:
        print(f"== session '{session}' exists: using it")
        info = remote_status(colab)[0]
        age = supervisor_age(info)
        if age is not None and age < STALE_S:
            print(f"a supervisor is already running on '{session}' (status written {age:.0f} s ago): not sending "
                  "another one, it would only wait behind it. The options of this start were not applied.\n"
                  f"progress: python scripts/colab_cli.py status{flag}")
            return 0
        if info is None and "BUSY" in line and PROGRAM.name in line and not args.ignore_busy:
            # the CLI's local mark of a `colab exec` that has not written a status yet: a start of a
            # moment ago. (With a status that is stale or says "left", the mark is itself stale: a
            # closed terminal leaves it behind, and the program is sent.)
            print(f"this machine's Colab CLI says a `colab exec` of the supervisor is attached to '{session}', and "
                  "the runtime has no supervisor status yet: a start of a moment ago? Not sending another one. "
                  "If no start is in progress (a closed terminal leaves that mark behind), repeat with "
                  "--ignore-busy.")
            return 0
    else:
        others = unknown_runtimes(colab)
        if others and not args.new_runtime:
            raise Failed(missing(colab) + "\nNot creating another runtime next to it. If you do want a second one, "
                         "repeat with --new-runtime.")
        new = ["new", "-s", session] + ([] if args.cpu else ["--tpu", args.tpu or preset["tpu"]])
        print(f"== creating the runtime (`{' '.join(colab.argv(*new))}`): compute units are charged from now on")
        if colab.shown(*new) != 0:
            raise Failed("the runtime was not created: the Colab CLI's message above says why")
    try:
        _prepare(colab, token, drive, restore)
    except (Failed, KeyboardInterrupt) as e:
        print(f"\n== nothing was sent to '{session}', but its runtime exists and is charged while it idles: "
              f"`python scripts/colab_cli.py start{flag}` again, or give it back with "
              f"`python scripts/colab_cli.py down{flag} --yes`.", file=sys.stderr)
        if isinstance(e, KeyboardInterrupt):
            return 130
        raise
    return _send_program(args, preset, colab, drive, flag)


def local_snapshot(src: Path) -> tuple[Path, dict]:
    """``(directory, parts.json)`` of a snapshot that `snapshot --to` downloaded, every part
    present and intact."""
    src = src.expanduser().resolve()
    try:
        meta = json.loads((src / "parts.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise Failed(f"{src} holds no snapshot (no readable parts.json): `snapshot --to {src}` downloads one") from None
    for part in meta.get("parts") or []:
        local = src / part["name"]
        if not local.is_file() or _sha256(local) != part["sha256"]:
            raise Failed(f"{local} is missing or damaged: download the snapshot again")
    if not meta.get("parts"):
        raise Failed(f"{src / 'parts.json'} lists no parts")
    return src, meta


def _prepare(colab: Colab, token: Path | None, drive: bool | None,
             restore: tuple[Path, dict] | None = None) -> None:
    """What `start` does on the runtime before the program is sent: the token file, a snapshot
    to continue from, and Drive."""
    session = colab.session
    if token is not None:
        colab.upload(token, "/content/bes_hf_token")
        print("== Hugging Face token uploaded (the supervisor reads it into the run's environment and removes the file)")
    if restore is not None:
        src, meta = restore
        print(f"== uploading the snapshot of {meta.get('run_dir')} ({len(meta['parts'])} parts, "
              f"{meta['bytes'] / 1e6:.1f} MB, checkpoints {meta.get('checkpoints') or 'none'})")
        for i, part in enumerate(meta["parts"]):
            colab.upload(src / part["name"], f"/content/{RESTORE_PREFIX}{i:03d}")
        colab.upload(src / "parts.json", f"/content/{RESTORE_PREFIX}parts.json")        # last: the set is complete
    if drive and not colab.drive_mounted():
        print(f"== mounting Google Drive for the mirror (`{' '.join(colab.argv('drivemount', '-s', session))}`): "
              "follow its prompts")
        if colab.shown("drivemount", "-s", session) != 0 or not colab.drive_mounted():
            raise Failed("Drive was not mounted. Mount it yourself (`colab drivemount -s " + session + "`) and "
                         "start again to have the mirror, or start with --no-drive: then nothing of the run is "
                         "outside the runtime until you fetch the export.")
    elif drive:
        print("== Google Drive is mounted on the runtime: the run is mirrored")


def _send_program(args: argparse.Namespace, preset: dict, colab: Colab, drive: bool | None, flag: str) -> int:
    """`colab exec` of the program, attached or in the background, and what the runtime says then."""
    session = colab.session
    run = colab.argv("exec", "-s", session, "--timeout", "0", "-f", str(PROGRAM), *environment(args, preset, drive))
    hints = (f"  progress:  python scripts/colab_cli.py status{flag}\n"
             f"  browser:   python scripts/colab_cli.py url{flag}\n"
             f"  export:    python scripts/colab_cli.py fetch{flag}\n"
             f"  give back: python scripts/colab_cli.py down{flag} --yes   (ends the charges; the runtime's disk is lost)")
    if args.detach:
        log = Path(args.log or f"colab_{session}.log").resolve()
        with open(log, "ab") as f:
            proc = subprocess.Popen(run, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True)
        print(f"== sent in the background (local pid {proc.pid}, its output in {log}); waiting for the supervisor's "
              "first status")
        sent, end = time.time(), time.time() + args.wait
        while time.time() < end:
            info, _ = remote_status(colab)
            if running(info) and float(info.get("supervisor_started") or 0) >= sent - STALE_S:
                print(f"== the supervisor is running: {info.get('state')}\n{hints}")
                return 0
            if proc.poll() is not None:
                return after_start(colab, hints)
            time.sleep(min(3.0, max(0.1, args.wait / 20)))
        print(f"== no status from the supervisor within {args.wait:.0f} s (the kernel may still be starting): "
              f"look at {log} and at `status`.\n{hints}")
        return 0
    print(f"== starting (`{' '.join(run)}`)\n   From here on Ctrl-C only detaches this terminal: the run and its "
          "supervisor go on.\n")
    try:
        subprocess.call(run)
    except KeyboardInterrupt:
        pass
    print()
    return after_start(colab, hints)


def _session(args: argparse.Namespace) -> str:
    return args.session or PRESETS["smoke" if args.smoke else "final"]["session"]


def cmd_status(args: argparse.Namespace) -> int:
    colab = Colab(_session(args))
    if not colab.exists():
        raise Failed(missing(colab))
    info, text = remote_status(colab)
    if info is None:
        raise Failed(f"no supervisor status on '{colab.session}' ({REMOTE}/status.json does not exist): the program "
                     "has not started there (a start that is still connecting, or none at all)")
    print(text.rstrip() or json.dumps(info, indent=1))
    age = supervisor_age(info)
    if age is None:
        print("\nWARNING: the supervisor has left: nothing keeps this runtime alive. Start again to supervise, "
              "or give the runtime back.")
    elif age > STALE_S:
        print(f"\nWARNING: this status is {age / 60:.1f} minutes old: the supervisor is not running (or the clocks "
              "differ). Without it Colab reclaims the runtime; `start` again supervises a run that still runs.")
    else:
        print(f"\n(status written {age:.0f} s ago; supervisor state: {info.get('state')})")
    return 0


def cmd_send(args: argparse.Namespace) -> int:
    colab = Colab(_session(args))
    if not colab.exists():
        raise Failed(missing(colab))
    if not running(remote_status(colab)[0]) and not args.force:
        raise Failed(f"no supervisor is running on '{colab.session}' (see `status`): nothing was sent. A command "
                     "left there would only be dropped by the next supervisor. `start` again first.")
    text = args.word + (f" {args.part_mb:g}" if args.word in ("package", "snapshot") and args.part_mb is not None
                        else "")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ctl"
        path.write_text(text + "\n", encoding="utf-8")
        colab.upload(path, f"{REMOTE}/ctl")
    print(f"sent '{text}' to the supervisor of '{colab.session}'; it acts within its next round (15 s) "
          "and says so in the status")
    return 0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract(tar_path: Path, to: Path) -> None:
    """Unpack the export: regular files and directories under ``final/`` only (works on a Python
    without tarfile's extraction filters as well)."""
    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        for m in members:
            parts = Path(m.name).parts
            if not (m.isfile() or m.isdir()) or m.name.startswith("/") or ".." in parts or parts[:1] != ("final",):
                raise Failed(f"the export archive holds something unexpected ({m.name!r}); not unpacked")
        if hasattr(tarfile, "data_filter"):
            tar.extractall(to, members=members, filter="data")
        else:
            tar.extractall(to, members=members)


def cmd_fetch(args: argparse.Namespace) -> int:
    colab = Colab(_session(args))
    if not colab.exists():
        raise Failed(missing(colab))
    to = Path(args.to).resolve()
    parts_dir = to / ".parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    index = parts_dir / "parts.json"
    if not colab.download(f"{REMOTE}/export/parts.json", index):
        raise Failed("the export is not packaged on the runtime: `send package` asks the supervisor for it "
                     "(it packages by itself when the run is complete); then fetch again")
    meta = json.loads(index.read_text(encoding="utf-8"))
    smaller = ("If a part is too large for the runtime's file interface, ask for smaller ones and fetch again:\n"
               "    python scripts/colab_cli.py send package --part-mb 4")
    for i, part in enumerate(meta["parts"]):
        local = parts_dir / part["name"]
        if local.is_file() and _sha256(local) == part["sha256"]:
            continue
        print(f"part {i + 1}/{len(meta['parts'])}: {part['name']} ({part['bytes'] / 1e6:.1f} MB)")
        try:
            arrived = colab.download(f"{REMOTE}/export/{part['name']}", local)
        except Failed as e:
            raise Failed(f"{e}\n{smaller}\nFinished parts are kept in {parts_dir}.") from None
        if not arrived:
            raise Failed(f"{part['name']} is not on the runtime any more (the export was packaged again?): "
                         "run fetch again")
        if _sha256(local) != part["sha256"]:
            raise Failed(f"{part['name']} did not arrive intact; run fetch again (finished parts are kept).\n{smaller}")
    tar_path = parts_dir / "final.tar"
    whole = hashlib.sha256()
    with open(tar_path, "wb") as out:
        for part in meta["parts"]:
            data = (parts_dir / part["name"]).read_bytes()
            whole.update(data)
            out.write(data)
    if whole.hexdigest() != meta["sha256"]:
        raise Failed("the joined export does not match its SHA-256; run fetch again")
    _extract(tar_path, to)
    shutil.rmtree(parts_dir)
    print(f"export in {to / 'final'} ({meta['bytes'] / 1e6:.1f} MB, SHA-256 of the archive verified).\n"
          f"check its files:  python scripts/export.py --verify {to / 'final'}\n"
          "The runtime is charged until it is given back: `down --yes` once you have what you need.")
    return 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    """Ask the supervisor for a snapshot (unless --no-request), wait for it and download it."""
    colab = Colab(_session(args))
    if not colab.exists():
        raise Failed(missing(colab))
    info, _ = remote_status(colab)
    last = float(((info or {}).get("snapshot") or {}).get("created") or 0.0)
    if not args.no_request:
        if not running(info):
            raise Failed(f"no supervisor is running on '{colab.session}' (see `status`): only a running supervisor "
                         "takes a snapshot. --no-request downloads the one it took last.")
        text = "snapshot" + (f" {args.part_mb:g}" if args.part_mb is not None else "")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ctl"
            path.write_text(text + "\n", encoding="utf-8")
            colab.upload(path, f"{REMOTE}/ctl")
        print(f"asked the supervisor of '{colab.session}' for a snapshot; waiting for it (up to {args.wait:.0f} s)")
        end = time.time() + args.wait
        while True:
            time.sleep(min(5.0, max(0.1, args.wait / 40)))
            info, _ = remote_status(colab)
            if float(((info or {}).get("snapshot") or {}).get("created") or 0.0) > last:
                break
            if time.time() > end:
                raise Failed(f"no new snapshot within {args.wait:.0f} s (`status` says what the supervisor does); "
                             "nothing was downloaded")
    elif not last:
        raise Failed(f"the supervisor of '{colab.session}' has not taken a snapshot yet")
    to = Path(args.to).resolve()
    inc = to / ".incoming"
    inc.mkdir(parents=True, exist_ok=True)
    index = inc / "parts.json"
    if not colab.download(f"{REMOTE}/snapshot/parts.json", index):
        raise Failed("the snapshot is not on the runtime (snapshot/parts.json does not exist)")
    meta = json.loads(index.read_text(encoding="utf-8"))
    keep = {p["name"] for p in meta["parts"]} | {"parts.json"}
    for stale in inc.iterdir():
        if stale.name not in keep:
            stale.unlink()
    whole = hashlib.sha256()
    for i, part in enumerate(meta["parts"]):
        local = inc / part["name"]
        if not (local.is_file() and _sha256(local) == part["sha256"]):
            if not colab.download(f"{REMOTE}/snapshot/{part['name']}", local) or _sha256(local) != part["sha256"]:
                raise Failed(f"{part['name']} did not arrive intact (a newer snapshot replaced it?): run snapshot "
                             f"again; finished parts are kept in {inc}")
        whole.update(local.read_bytes())
        if (i + 1) % 10 == 0 or i + 1 == len(meta["parts"]):
            print(f"  {i + 1}/{len(meta['parts'])} parts")
    if whole.hexdigest() != meta["sha256"]:
        raise Failed("the joined snapshot does not match its SHA-256; run snapshot again")
    for old in to.iterdir():                          # the complete new set replaces the previous one
        if old.is_file():
            old.unlink()
    for part in meta["parts"]:
        os.replace(inc / part["name"], to / part["name"])
    os.replace(index, to / "parts.json")
    inc.rmdir()
    print(f"snapshot in {to}: {len(meta['parts'])} parts, {meta['bytes'] / 1e6:.1f} MB, checkpoints "
          f"{meta.get('checkpoints') or 'none'}, {meta.get('tokens')} training tokens (SHA-256 verified).\n"
          f"continue it on another runtime:  python scripts/colab_cli.py start --action resume --restore {to}"
          + (f" --run-dir {meta['run_dir']}" if meta.get("run_dir") else ""))
    return 0


def cmd_url(args: argparse.Namespace) -> int:
    colab = Colab(_session(args))
    return colab.shown("url", "-s", colab.session, *(["--open"] if args.open else []))


def cmd_down(args: argparse.Namespace) -> int:
    colab = Colab(_session(args))
    if not args.yes:
        raise Failed(f"this stops the runtime of '{colab.session}': its disk, the run directory and anything not "
                     "on Drive or fetched are lost. Repeat with --yes if that is what you want.")
    p = colab.quiet("stop", "-s", colab.session)
    out = text_of(p)
    if p.returncode != 0:
        raise Failed(explain("`colab stop`", p))
    if "not found" in out:               # the CLI exits 0 for a session it does not know: nothing was stopped
        raise Failed("nothing was stopped: " + missing(colab))
    print(out)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--smoke", action="store_true", help="the 30-minute smoke run (session bes-smoke) instead "
                                                            "of the 24-hour run (session bes24h)")
        p.add_argument("-s", "--session", default=None, help="another session name")
        return p

    p = common(sub.add_parser("start", help="create the TPU runtime if needed and start or continue the run"))
    p.add_argument("--tpu", default=None, help="v6e1 or v5e1 (default: v6e1, with --smoke v5e1)")
    p.add_argument("--cpu", action="store_true", help="with --smoke: a CPU runtime and ALLOW_CPU, session "
                                                      f"{CPU_SESSION} (a rehearsal of this route without a TPU)")
    p.add_argument("--action", default="auto", choices=["auto", "start", "resume"],
                   help="auto: continue a run that exists on the runtime or on Drive, else start one")
    p.add_argument("--budget-minutes", type=float, default=None, help="replace the notebook's BUDGET_MINUTES")
    p.add_argument("--config", default=None, help="replace the notebook's CONFIG (a configs/*.py of the branch)")
    p.add_argument("--override", action="append", default=[], metavar="SECTION.FIELD=VALUE",
                   help="an --override of scripts/run_24h.py for a new run (repeat it; no spaces in a "
                        "value: lists take commas, as run.post_minutes=critic=0,judge=15)")
    p.add_argument("--drive", action=argparse.BooleanOptionalAction, default=None,
                   help="--drive mounts Google Drive first (it asks you) and mirrors the run to it: the default "
                        "of the 24-hour run; --no-drive never mirrors: the default of the smoke run unless "
                        "Drive is already mounted")
    p.add_argument("--drive-dir", default=None, help="replace the notebook's DRIVE_DIR")
    p.add_argument("--run-dir", default=None,
                   help="replace the notebook's RUN_DIR (an absolute path on the runtime's local disk): a run "
                        "next to another one on the same runtime; without --drive-dir its mirror goes to a "
                        "directory of the same name next to the notebook's DRIVE_DIR")
    p.add_argument("--move-deadline", default=None, metavar="WHEN",
                   help="before a run is continued, move its deadline: seconds since the epoch, or +MINUTES / "
                        "-MINUTES (a run continued on another runtime: RUNBOOK.md section 5). Write a "
                        "relative value as --move-deadline=+45")
    p.add_argument("--restore", default=None, metavar="DIR",
                   help="upload the snapshot in DIR (`snapshot --to DIR`) and unpack it as the run directory when "
                        "the runtime does not hold the run: a continuation on another runtime without Drive")
    p.add_argument("--hf-token-file", default=None, help="a file with a Hugging Face token, uploaded to the runtime")
    p.add_argument("--repo-url", default=None, help="clone another repository (default: this one)")
    p.add_argument("--branch", default=None, help="clone another branch")
    p.add_argument("--auto-resume", type=int, default=None, help="automatic continuations of a run that died (2)")
    p.add_argument("--hold-minutes", type=float, default=None, help="keep the runtime this long after the run (120)")
    p.add_argument("--part-mb", type=float, default=None, help="size of the export's parts for fetch (16)")
    p.add_argument("--new-runtime", action="store_true",
                   help="create the runtime although the account has one this machine does not know")
    p.add_argument("--ignore-busy", action="store_true",
                   help="send the program although this machine's CLI marks a `colab exec` as attached")
    p.add_argument("--detach", action="store_true", help="send in the background and return once the supervisor "
                                                         "has written its first status")
    p.add_argument("--wait", type=float, default=90.0, help="with --detach: seconds to wait for that status")
    p.add_argument("--log", default=None, help="with --detach: the local file for the output")
    p.set_defaults(fn=cmd_start)
    common(sub.add_parser("status", help="print the supervisor's progress report")).set_defaults(fn=cmd_status)
    p = common(sub.add_parser("send", help="send a command to the supervisor"))
    p.add_argument("word", choices=COMMANDS)
    p.add_argument("--part-mb", type=float, default=None, help="with package or snapshot: megabytes per part")
    p.add_argument("--force", action="store_true", help="send although no supervisor seems to be running")
    p.set_defaults(fn=cmd_send)
    p = common(sub.add_parser("snapshot", help="have the supervisor pack the run's records and newest checkpoint, "
                                               "and download them (a copy of the run outside Colab)"))
    p.add_argument("--to", default="colab_snapshot", help="local directory (default colab_snapshot)")
    p.add_argument("--part-mb", type=float, default=None, help="megabytes per part (default: the supervisor's)")
    p.add_argument("--wait", type=float, default=900.0, help="seconds to wait for the supervisor (default 900)")
    p.add_argument("--no-request", action="store_true", help="download the snapshot the supervisor took last")
    p.set_defaults(fn=cmd_snapshot)
    p = common(sub.add_parser("fetch", help="download the packaged export and verify it"))
    p.add_argument("--to", default="colab_export", help="local directory (default colab_export)")
    p.set_defaults(fn=cmd_fetch)
    p = common(sub.add_parser("url", help="print a browser link that attaches Colab to the runtime"))
    p.add_argument("--open", action="store_true", help="also open it in the browser")
    p.set_defaults(fn=cmd_url)
    p = common(sub.add_parser("down", help="stop the runtime (colab stop)"))
    p.add_argument("--yes", action="store_true")
    p.set_defaults(fn=cmd_down)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return int(args.fn(args))
    except Failed as e:
        print(f"colab_cli: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
