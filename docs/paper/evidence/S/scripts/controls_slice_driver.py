"""CPU runtime side: a SLICE of an opponent's registered control arms (docs/EVAL_PROTOCOL.md section
3; the same `run` as controls_driver.py), so that several runtimes share the 200 test prompts. The
generation cache is keyed by (backend, messages, settings, seed): which machine made a reply does
not enter the key; the caches are concatenated afterwards and the outputs assembled from the merged
cache.

    python controls_slice_driver.py K WORKERS THREADS MODEL SELECT SLICE [ROOT=/content/bes_cli/ctlrev_root]

SELECT: logprob_mean | decoder_select | both (logprob_mean first; decoder_select then reuses the
first-turn candidates and generates only the second turns that follow another first reply).
SLICE: a Python slice of the test split's prompt order (":100", "100:136", "::-1", ...); several
slices joined by "+" are interleaved one prompt each in turn ("135:117:-1+171:153:-1"); a list of
indices joined by "," names prompts one by one ("5,12,32").
Progress /content/bes_cli/ctlslice_progress.log; /content/bes_cli/ctlslice_cache.tar.gz (cache/ of
the root) is rewritten every 4 minutes and at the end. A file /content/bes_cli/ctlslice_stop ends
the run (finished prompts are in the cache; generations in flight are lost)."""
import os
import sys
import tarfile
import threading
import time
from pathlib import Path

REPO = "/content/TuringAward2027"
os.chdir(REPO)
sys.path.insert(0, REPO)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

k, workers, threads, model, select, sl = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
root = Path(sys.argv[7] if len(sys.argv) > 7 else "/content/bes_cli/ctlrev_root")
root.mkdir(parents=True, exist_ok=True)
log_path = Path("/content/bes_cli/ctlslice_progress.log")
stop_path = Path("/content/bes_cli/ctlslice_stop")
lock = threading.Lock()


def say(msg):
    line = f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}"
    with lock:
        with open(log_path, "a") as f:
            f.write(line + "\n")
    print(line, flush=True)


def pack():
    tmp = "/content/bes_cli/ctlslice_cache.tar.gz.tmp"
    with lock:
        with tarfile.open(tmp, "w:gz") as tf:
            if (root / "cache").exists():
                tf.add(root / "cache", arcname="cache")
        os.replace(tmp, "/content/bes_cli/ctlslice_cache.tar.gz")


def main():
    from bes.eval.proxy import prompts as prompt_set, runner
    from bes.eval.proxy.__main__ import build_spec
    from bes.eval.proxy.common import models_dir

    ps = prompt_set.load()
    test = prompt_set.select(ps, "test", None, None)
    if "," in sl:
        todo = [test[int(i)] for i in sl.split(",") if i != ""]
    else:
        parts = [test[slice(*[int(x) if x else None for x in one.split(":")])] for one in sl.split("+")]
        todo, seen = [], set()
        for i in range(max(len(q) for q in parts)):
            for q in parts:
                if i < len(q) and q[i].id not in seen:
                    seen.add(q[i].id)
                    todo.append(q[i])
    done = threading.Event()

    def keeper():
        while not done.wait(240):
            try:
                pack()
            except Exception as err:  # noqa: BLE001
                say(f"pack failed: {err}")
            if stop_path.exists():
                say("stop file seen: leaving")
                pack()
                os._exit(0)

    threading.Thread(target=keeper, daemon=True).start()
    for rule in (["logprob_mean", "decoder_select"] if select == "both" else [select]):
        spec = build_spec(model, "recommended", [], None, 0, k, rule, threads, None, None, {}, None, models_dir(), True)
        say(f"start {model} samples {k} select {rule}, slice [{sl}] = {len(todo)} prompts, run {spec.run_id}, "
            f"root {root}, cache {runner.cache_path(root, spec.key).name}, {workers} workers x {threads} threads")
        res = runner.run(todo, spec, root, workers=workers, n_threads=threads, all_prompts=ps, log=say)
        pack()
        say(f"end {rule}: {res.done} complete ({res.reused} from cache), {res.new_generations} new generations, "
            f"{len(res.failed)} failed, {res.seconds / 60:.1f} min")
    done.set()
    pack()
    say("all done")


if __name__ == "__main__":
    main()
