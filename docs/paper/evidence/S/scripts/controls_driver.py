"""CPU runtime side: the registered control arms of the opponents on the test split
(docs/EVAL_PROTOCOL.md section 3): each opponent with the same K as our decoder (--samples K),
kept by the opponent's own mean log-probability (K-matched control) and by our decoder's
selector (selector-matched control). The two rules share the K candidates (generation cache).

    python controls_driver.py K WORKERS THREADS [MODEL ...]      (default models: falcon mobilellm)

Root /content/bes_cli/ctl_root; progress /content/bes_cli/ctl_progress.log; after every finished
arm /content/bes_cli/ctl_outputs.tar.gz is rewritten (outputs/ of the root: replies and meta, no
model files)."""
import os, subprocess, sys, tarfile, time
from pathlib import Path

k, workers, threads = sys.argv[1], sys.argv[2], sys.argv[3]
models = sys.argv[4:] or ["falcon", "mobilellm"]
REPO = "/content/TuringAward2027"
root = Path("/content/bes_cli/ctl_root")
root.mkdir(parents=True, exist_ok=True)
log_path = Path("/content/bes_cli/ctl_progress.log")


def say(msg):
    line = f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}"
    with open(log_path, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def pack():
    tmp = "/content/bes_cli/ctl_outputs.tar.gz.tmp"
    with tarfile.open(tmp, "w:gz") as tf:
        if (root / "outputs").exists():
            tf.add(root / "outputs", arcname="outputs")
        tf.add(log_path, arcname="ctl_progress.log")
    os.replace(tmp, "/content/bes_cli/ctl_outputs.tar.gz")


env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONUNBUFFERED="1")
for model in models:
    for select in ("logprob_mean", "decoder_select"):
        cmd = [sys.executable, "-m", "bes.eval.proxy", "--root", str(root), "run", "--model", model, "--preset", "recommended",
               "--split", "test", "--samples", k, "--select", select, "--workers", workers, "--threads", threads]
        t0 = time.time()
        say(f"start {model} samples {k} select {select}")
        with open(f"/content/bes_cli/ctl_{model}_{select}.log", "w") as log:
            rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        tail = ""
        try:
            tail = [l for l in open(f"/content/bes_cli/ctl_{model}_{select}.log", errors="replace").read().splitlines() if l.strip()][-1][:240]
        except Exception:  # noqa: BLE001
            pass
        say(f"end {model} {select}: exit {rc} after {time.time() - t0:.0f} s | {tail}")
        pack()
say("all done: /content/bes_cli/ctl_outputs.tar.gz")
