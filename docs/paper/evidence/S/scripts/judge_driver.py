"""Runtime side: the pre-registered automatic judge (bes.eval.proxy judge, a GGUF model on CPU,
both orders) on blinded pair sets, on the host's idle cores.

    python judge_driver.py TAG JUDGE THREADS STREAMS SET [SET ...]

/content/bes_cli/judge/TAG/pairs.tar.gz (pairs/<SET>/{meta.json,pairs.jsonl}; never the keys) is
unpacked into /content/bes_cli/judge/TAG/root; STREAMS sets are judged at a time, each pinned to
THREADS cores (logical cores 0, 1 and their siblings stay free), niced. Progress in
/content/bes_cli/judge/TAG/progress.log; at the end ratings.tar.gz (ratings/<SET>/*.jsonl, logs).
"""
import os
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

tag, judge, threads, streams = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
sets = sys.argv[5:]
base = Path("/content/bes_cli/judge") / tag
root = base / "root"
root.mkdir(parents=True, exist_ok=True)
REPO = "/content/TuringAward2027"
lock = threading.Lock()


def say(msg):
    line = f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}"
    with lock:
        with open(base / "progress.log", "a") as f:
            f.write(line + "\n")
    print(line, flush=True)


with tarfile.open(base / "pairs.tar.gz") as tf:
    tf.extractall(root)
# the judge's model file: checked (size, SHA-256) and downloaded from the Hub when missing
t0 = time.time()
pre = subprocess.run([sys.executable, "-c", f"from bes.eval.proxy import opponents; print(opponents.fetch({judge!r}))"],
                     cwd=REPO, capture_output=True, text=True, env=dict(os.environ, JAX_PLATFORMS="cpu"))
say(f"judge file after {time.time() - t0:.0f} s: exit {pre.returncode} {(pre.stdout + pre.stderr).strip()[-200:]}")
if pre.returncode != 0:
    raise SystemExit(1)
n = os.cpu_count() or 44
half = n // 2
if n < 40:                     # a small CPU runtime: every core, one stream decides
    usable = list(range(n))
# physical cores first (2..half-1), then their siblings: a stream gets whole physical cores when it can
if n >= 40:
    usable = list(range(2, half)) + list(range(half + 2, n))
queue = list(sets)


def stream(i):
    cores = usable[i * threads:(i + 1) * threads]
    while True:
        with lock:
            if not queue:
                return
            name = queue.pop(0)
        cmd = ["taskset", "-c", ",".join(map(str, cores)), "nice", "-n", "10", sys.executable, "-m", "bes.eval.proxy",
               "--root", str(root), "judge", "--pairs", name, "--judge", judge, "--threads", str(len(cores)), "--no-download"]
        t0 = time.time()
        say(f"start {name} with {judge} on cores {cores[0]}-{cores[-1]} ({len(cores)})")
        with open(base / f"{name}.{judge}.log", "w") as log:
            rc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, JAX_PLATFORMS="cpu", PYTHONUNBUFFERED="1")).returncode
        tail = ""
        try:
            tail = [l for l in open(base / f"{name}.{judge}.log", errors="replace").read().splitlines() if l.strip()][-1][:200]
        except Exception:  # noqa: BLE001
            pass
        say(f"end {name}: exit {rc} after {time.time() - t0:.0f} s | {tail}")


ts = [threading.Thread(target=stream, args=(i,)) for i in range(streams)]
for t in ts:
    t.start()
for t in ts:
    t.join()
out = base / "ratings.tar.gz"
with tarfile.open(out, "w:gz") as tf:
    if (root / "ratings").exists():
        tf.add(root / "ratings", arcname="ratings")
    for f in base.glob("*.log"):
        tf.add(f, arcname=f"logs/{f.name}")
say(f"all done: {out} ({out.stat().st_size} bytes)")
