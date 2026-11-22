"""The run on a Colab TPU from your terminal, through the Colab CLI.

    pip install google-colab-cli          # https://github.com/googlecolab/google-colab-cli (Linux, macOS)
    python scripts/colab_cli.py start --smoke      # 30 minutes on a v5e-1 TPU: do this first
    python scripts/colab_cli.py start              # the 24-hour run on a v6e-1 TPU
    python scripts/colab_cli.py status             # the progress report, from any terminal
    python scripts/colab_cli.py url                # a browser link that attaches Colab to the same runtime
    python scripts/colab_cli.py send stop          # stop | resume | package | release
    python scripts/colab_cli.py fetch --to export  # the exported model, checked by SHA-256
    python scripts/colab_cli.py down --yes         # give the runtime back (its disk is lost)

``start`` signs you in when needed (the Colab CLI prints an authorization link and asks for the
code Google shows: paste it in that terminal, nowhere else), creates the session
(``colab new -s bes24h --tpu v6e1``) and sends ``scripts/colab_tpu.py`` to its kernel
(``colab exec -s bes24h --timeout 0 -f scripts/colab_tpu.py``). That program clones this
repository on the runtime, runs the notebook's own cells and stays in the kernel as the run's
supervisor, because a runtime made by the CLI lives only while its kernel executes something.
Read its docstring for the files it keeps under ``/content/bes_cli`` and what the commands do.

``start`` stays attached and prints the supervisor's lines. Ctrl-C, a closed terminal or a lost
connection only detach it: the run and its supervisor go on. ``--detach`` starts it in the
background at once. While the supervisor runs, every other ``colab exec`` on the session waits
behind it, so this wrapper talks to the runtime through files (``colab upload``/``download``).

Without this wrapper the same thing is two commands; see ``RUNBOOK.md`` section 2.1.
``COLAB_BIN`` names another ``colab`` executable, ``COLAB_CLI_ARGS`` adds global options to
every call (for example ``--auth=adc``). Nothing here was run against Colab: the wrapper is
tested with a stand-in for the ``colab`` command (``tests/train/test_colab_cli.py``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
PRESETS = {
    "final": {"session": "bes24h", "notebook": "colab_v6e_24h", "tpu": "v6e1"},
    "smoke": {"session": "bes-smoke", "notebook": "colab_v5e_smoke", "tpu": "v5e1"},
}
COMMANDS = ("stop", "resume", "package", "release")
STALE_S = 240.0                     # a status older than this: no supervisor is writing it
INSTALL = ("the `colab` command was not found. Install the Colab CLI (Linux or macOS, Python 3.12+):\n"
           "    pip install google-colab-cli        (or: uv tool install google-colab-cli)\n"
           "    https://github.com/googlecolab/google-colab-cli")


NO_KEYRING = ("The Colab CLI could not keep its sign-in: this machine has no keyring it can use (a server, WSL,\n"
              "a container), so every command asks to sign in again. Either give it a file store with\n"
              "    pip install keyrings.alt        (same environment as google-colab-cli; plaintext, owner-only)\n"
              "or use Application Default Credentials: sign in once with\n"
              "    gcloud auth application-default login --scopes=openid,"
              "https://www.googleapis.com/auth/cloud-platform,"
              "https://www.googleapis.com/auth/userinfo.email,"
              "https://www.googleapis.com/auth/colaboratory\n"
              "then set COLAB_CLI_ARGS=--auth=adc for this wrapper.")


class Failed(Exception):
    """A step did not work; the message says what to do."""


def explain(what: str, p: subprocess.CompletedProcess) -> str:
    """The message for a `colab` command that failed, with the usual causes spelled out."""
    out = (p.stdout + p.stderr).strip()
    if "keyring" in out.lower():
        return f"{what} failed.\n{NO_KEYRING}"
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

    def exists(self) -> bool:
        p = self.quiet("status", "-s", self.session)
        if p.returncode != 0:
            raise Failed(explain("`colab status`", p))
        return f"[{self.session}] " in p.stdout

    def download(self, remote: str, local: Path) -> bool:
        p = self.quiet("download", "-s", self.session, remote, str(local), timeout=1800.0)
        return p.returncode == 0 and local.is_file()

    def upload(self, local: Path, remote: str) -> None:
        p = self.quiet("upload", "-s", self.session, str(local), remote)
        if p.returncode != 0:
            raise Failed(explain(f"the upload to {remote}", p))


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


def environment(args: argparse.Namespace, preset: dict) -> list[str]:
    """The ``--env KEY=VALUE`` arguments of ``colab exec`` for these options."""
    env = {"BES_NOTEBOOK": preset["notebook"], "BES_ACTION": args.action}
    if args.repo_url:
        env["BES_REPO_URL"] = args.repo_url
    if args.branch is not None:
        env["BES_BRANCH"] = args.branch
    if args.budget_minutes is not None:
        env["BES_BUDGET_MINUTES"] = f"{args.budget_minutes:g}"
    if args.cpu:
        env["BES_ALLOW_CPU"] = "1"
    if args.drive is not None:
        env["BES_USE_DRIVE"] = "1" if args.drive else "0"
    if args.auto_resume is not None:
        env["BES_AUTO_RESUME"] = str(args.auto_resume)
    if args.hold_minutes is not None:
        env["BES_HOLD_MINUTES"] = f"{args.hold_minutes:g}"
    out: list[str] = []
    for k, v in env.items():
        out += ["--env", f"{k}={v}"]
    return out


def cmd_start(args: argparse.Namespace) -> int:
    preset = PRESETS["smoke" if args.smoke else "final"]
    session = args.session or preset["session"]
    colab = Colab(session)
    if not PROGRAM.is_file():
        raise Failed(f"{PROGRAM} is missing")
    print(f"== signing in if needed, and the sessions of this account (`{' '.join(colab.argv('sessions'))}`)")
    if colab.shown("sessions") != 0:
        raise Failed("`colab sessions` failed: sign in first (it prints a link and asks for the code Google shows)")
    if colab.exists():
        print(f"== session '{session}' exists: using it")
        age = supervisor_age(remote_status(colab)[0])
        if age is not None and age < STALE_S:
            print(f"a supervisor is already running on '{session}' (status written {age:.0f} s ago): not starting "
                  f"another one, it would only wait behind it.\nprogress: python {Path(__file__).name} status"
                  + (" --smoke" if args.smoke else ""))
            return 0
    else:
        new = ["new", "-s", session] + ([] if args.cpu else ["--tpu", args.tpu or preset["tpu"]])
        print(f"== creating the runtime (`{' '.join(colab.argv(*new))}`): compute units are charged from now on")
        if colab.shown(*new) != 0:
            raise Failed("the runtime was not created (the message above says why: no entitlement for this TPU, "
                         "too many sessions, or no capacity right now)")
    if args.hf_token_file:
        token = Path(args.hf_token_file).expanduser()
        if not token.is_file():
            raise Failed(f"{token} does not exist")
        colab.upload(token, "/content/bes_hf_token")
        print("== Hugging Face token uploaded (the supervisor reads it into the run's environment and removes the file)")
    if args.drive:
        print(f"== mounting Drive (`{' '.join(colab.argv('drivemount', '-s', session))}`): follow its prompts")
        if colab.shown("drivemount", "-s", session) != 0:
            raise Failed("Drive was not mounted; start again without --drive to run without the mirror")
    run = colab.argv("exec", "-s", session, "--timeout", "0", "-f", str(PROGRAM), *environment(args, preset))
    flag = " --smoke" if args.smoke else ""
    hints = (f"  progress:  python scripts/colab_cli.py status{flag}\n"
             f"  browser:   python scripts/colab_cli.py url{flag}\n"
             f"  give back: python scripts/colab_cli.py down{flag} --yes   (the runtime's disk is lost)")
    if args.detach:
        log = Path(args.log or f"colab_{session}.log").resolve()
        with open(log, "ab") as f:
            proc = subprocess.Popen(run, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True)
        print(f"== started in the background (local pid {proc.pid}, its output in {log})\n{hints}")
        return 0
    print(f"== starting (`{' '.join(run)}`)\n   Ctrl-C only detaches this terminal: the run and its supervisor go on.\n")
    try:
        code = subprocess.call(run)
    except KeyboardInterrupt:
        code = 130
    print(f"\n== detached from '{session}' (exit status {code}). If the supervisor was running, it still is:\n{hints}\n"
          "   If the start failed before the run began, the runtime is idle and still charged until Colab\n"
          "   reclaims it: fix the cause and start again, or give it back.")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    colab = Colab(args.session or PRESETS["smoke" if args.smoke else "final"]["session"])
    if not colab.exists():
        raise Failed(f"session '{colab.session}' does not exist any more (Colab reclaimed it, or it was never "
                     "created): `colab sessions` lists what this account has")
    info, text = remote_status(colab)
    if info is None:
        raise Failed(f"no supervisor status on '{colab.session}' yet ({REMOTE}/status.json): the start may still "
                     "be cloning, or it was never started here")
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
    colab = Colab(args.session or PRESETS["smoke" if args.smoke else "final"]["session"])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ctl"
        path.write_text(args.word + "\n", encoding="utf-8")
        colab.upload(path, f"{REMOTE}/ctl")
    print(f"sent '{args.word}' to the supervisor of '{colab.session}'; it acts within its next round (15 s) "
          "and says so in the status")
    return 0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_fetch(args: argparse.Namespace) -> int:
    colab = Colab(args.session or PRESETS["smoke" if args.smoke else "final"]["session"])
    to = Path(args.to).resolve()
    parts_dir = to / ".parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    index = parts_dir / "parts.json"
    if not colab.download(f"{REMOTE}/export/parts.json", index):
        raise Failed("the export is not packaged on the runtime yet: `send package` asks the supervisor for it "
                     "(it packages by itself when the run is complete); then fetch again")
    meta = json.loads(index.read_text(encoding="utf-8"))
    for i, part in enumerate(meta["parts"]):
        local = parts_dir / part["name"]
        if local.is_file() and _sha256(local) == part["sha256"]:
            continue
        print(f"part {i + 1}/{len(meta['parts'])}: {part['name']} ({part['bytes'] / 1e6:.1f} MB)")
        if not colab.download(f"{REMOTE}/export/{part['name']}", local) or _sha256(local) != part["sha256"]:
            raise Failed(f"{part['name']} did not arrive intact; run fetch again (finished parts are kept)")
    tar_path = parts_dir / "final.tar"
    whole = hashlib.sha256()
    with open(tar_path, "wb") as out:
        for part in meta["parts"]:
            data = (parts_dir / part["name"]).read_bytes()
            whole.update(data)
            out.write(data)
    if whole.hexdigest() != meta["sha256"]:
        raise Failed("the joined export does not match its SHA-256; run fetch again")
    with tarfile.open(tar_path) as tar:
        tar.extractall(to, filter="data")
    shutil.rmtree(parts_dir)
    print(f"export in {to / 'final'} ({meta['bytes'] / 1e6:.1f} MB, SHA-256 of the archive verified).\n"
          f"check its files:  python scripts/export.py --verify {to / 'final'}")
    return 0


def cmd_url(args: argparse.Namespace) -> int:
    colab = Colab(args.session or PRESETS["smoke" if args.smoke else "final"]["session"])
    return colab.shown("url", "-s", colab.session, *(["--open"] if args.open else []))


def cmd_down(args: argparse.Namespace) -> int:
    colab = Colab(args.session or PRESETS["smoke" if args.smoke else "final"]["session"])
    if not args.yes:
        raise Failed(f"this stops the runtime of '{colab.session}': its disk, the run directory and anything not "
                     "on Drive or fetched are lost. Repeat with --yes if that is what you want.")
    return colab.shown("stop", "-s", colab.session)


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
    p.add_argument("--cpu", action="store_true", help="a CPU runtime and ALLOW_CPU (a rehearsal of this route)")
    p.add_argument("--action", default="auto", choices=["auto", "start", "resume"],
                   help="auto: continue a run that exists on the runtime or on Drive, else start one")
    p.add_argument("--budget-minutes", type=float, default=None, help="replace the notebook's BUDGET_MINUTES")
    p.add_argument("--drive", action=argparse.BooleanOptionalAction, default=None,
                   help="--drive mounts Drive first (interactive) for the mirror; --no-drive never mirrors; "
                        "default: mirror when Drive is already mounted")
    p.add_argument("--hf-token-file", default=None, help="a file with a Hugging Face token, uploaded to the runtime")
    p.add_argument("--repo-url", default=None, help="clone another repository (default: this one)")
    p.add_argument("--branch", default=None, help="clone another branch")
    p.add_argument("--auto-resume", type=int, default=None, help="automatic continuations of a run that died (2)")
    p.add_argument("--hold-minutes", type=float, default=None, help="keep the runtime this long after the run (120)")
    p.add_argument("--detach", action="store_true", help="start in the background and return")
    p.add_argument("--log", default=None, help="with --detach: the local file for the output")
    p.set_defaults(fn=cmd_start)
    common(sub.add_parser("status", help="print the supervisor's progress report")).set_defaults(fn=cmd_status)
    p = common(sub.add_parser("send", help="send a command to the supervisor"))
    p.add_argument("word", choices=COMMANDS)
    p.set_defaults(fn=cmd_send)
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
