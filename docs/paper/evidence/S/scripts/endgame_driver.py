"""Runtime side of endgame_testgen.sh: the 200 test replies from the run's newest weights on the
host's idle cores (CPU only), leaving logical cores 0, 1 and their siblings to the run.

    python endgame_driver.py TAG MINUTES [extra gen_test_replies options]
"""
import os
import subprocess
import sys

tag, minutes, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
n = os.cpu_count() or 44
half = n // 2
cores = list(range(2, half)) + list(range(half + 2, n))
repo = "/content/TuringAward2027"
cmd = ["taskset", "-c", ",".join(map(str, cores)), sys.executable, f"{repo}/scripts/gen_test_replies.py",
       "/content/runs/main16", "--make-export", "--no-download", "--out", f"/content/bes_cli/testgen_{tag}",
       "--split", "test", "--workers", str(len(cores)), "--cores-per-worker", "1", "--nice", "5",
       "--max-rss-gb", "8", "--label", f"final-{tag}", "--minutes", minutes, "--grace-minutes", "3"] + extra
print("endgame_driver:", " ".join(cmd), flush=True)
rc = subprocess.run(cmd, cwd=repo, env=dict(os.environ, JAX_PLATFORMS="cpu", PYTHONUNBUFFERED="1")).returncode
print("endgame_driver: exit", rc, flush=True)
sys.exit(rc)
