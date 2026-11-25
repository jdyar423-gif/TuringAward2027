"""This container's share of an opponent's registered control arm (docs/EVAL_PROTOCOL.md section 3):
the same `run` as controls_driver.py on the CPU runtimes, on a slice of the test split's prompt
order, into a root of its own. The generation cache is keyed by (backend, messages, settings,
seed), so caches of several machines are concatenated afterwards.

    python ctl_local.py K WORKERS THREADS MODEL SELECT SLICE      SLICE like "::-1" (from the end) or "100:150"
Root N/ctl_local_root; progress N/ctl_local_root/progress.log; a file N/ctl_local_root/stop ends it
(finished prompts are in the cache; generations in flight are lost)."""
import os
import sys
import threading
import time
from pathlib import Path

REPO = "/home/user/turingaward2027"
N = Path("/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad")
os.chdir(REPO)
sys.path.insert(0, REPO)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
k, workers, threads, model, select, sl = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
root = N / "ctl_local_root"
root.mkdir(parents=True, exist_ok=True)
log_path = root / "progress.log"
stop_path = root / "stop"


def say(msg):
    line = f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}"
    with open(log_path, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def main():
    from bes.eval.proxy import prompts as prompt_set, runner
    from bes.eval.proxy.__main__ import build_spec

    spec = build_spec(model, "recommended", [], None, 0, k, select, threads, None, None, {}, None, N / "models", False)
    ps = prompt_set.load()
    test = prompt_set.select(ps, "test", None, None)
    if "," in sl:                      # explicit indices of the test order
        todo = [test[int(i)] for i in sl.split(",") if i != ""]
    else:
        todo = test[slice(*[int(x) if x else None for x in sl.split(":")])]
    say(f"start {model} samples {k} select {select}, slice [{sl}] = {len(todo)} prompts, run {spec.run_id}, "
        f"cache {runner.cache_path(root, spec.key).name}, {workers} workers x {threads} threads")

    def keeper():
        while True:
            time.sleep(30)
            if stop_path.exists():
                say("stop file seen: leaving")
                os._exit(0)

    threading.Thread(target=keeper, daemon=True).start()
    res = runner.run(todo, spec, root, workers=workers, n_threads=threads, all_prompts=ps, log=say)
    say(f"all done: {res.done} complete, {res.new_generations} new generations, {len(res.failed)} failed")


if __name__ == "__main__":
    main()
