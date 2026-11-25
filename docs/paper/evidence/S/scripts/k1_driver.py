"""CPU runtime side: the registered K = 1 arm of our model on the test split: rebuild the official
export from the parts committed in the repository (release/main16-v9), verify it, and answer the
200 test prompts with one particle (`extra.particles=1`, as the run's own `k1` variant), every
other setting the export's.

    python k1_driver.py [WORKERS=8]
Output /content/bes_cli/testgen_k1 (testgen.log, testgen.tar.gz at the end); log /content/bes_cli/k1_driver.log."""
import os, subprocess, sys
R = "/content/TuringAward2027"
workers = sys.argv[1] if len(sys.argv) > 1 else "8"
env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONUNBUFFERED="1")
def sh(c):
    p = subprocess.run(c, shell=True, capture_output=True, text=True, cwd=R, env=env)
    print("$", c[:160], "->", p.returncode, (p.stdout + p.stderr)[-500:], flush=True)
    return p.returncode
sh("git pull -q --ff-only; git log -1 --format='%h %s'")
os.makedirs(R + "/runs/main16", exist_ok=True)
if sh("cat release/main16-v9/official_export.tar.* | tar -x -C runs/main16") != 0:
    raise SystemExit("no export parts")
if sh(f"{sys.executable} scripts/export.py --verify runs/main16/final | tail -n 1") != 0:
    raise SystemExit("the export does not verify")
cmd = [sys.executable, "scripts/gen_test_replies.py", "runs/main16/final", "--out", "/content/bes_cli/testgen_k1", "--split", "test",
       "--workers", workers, "--cores-per-worker", "1", "--set", "extra.particles=1", "--label", "final-k1", "--max-rss-gb", "5"]
print("k1_driver:", " ".join(cmd), flush=True)
sys.exit(subprocess.run(cmd, cwd=R, env=env).returncode)
