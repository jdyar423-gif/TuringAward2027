"""CPU runtime side: the registered control arm of an opponent (docs/EVAL_PROTOCOL.md section 3),
worked from the END of the test split's prompt order backwards, so that a second runtime helps the
one that runs controls_driver.py forwards. The generation cache is keyed by (backend, messages,
settings, seed): which runtime made a reply does not enter the key, the caches of the runtimes are
concatenated afterwards and the outputs are assembled from the merged cache.

    python controls_rev_driver.py K WORKERS THREADS MODEL [SELECT=logprob_mean] [SLICE=::-1]

SLICE is a Python slice of the test split's prompt order: "::-1" from the end backwards,
"100:150" a middle part forwards, "149:99:-1" the same part backwards.

Root /content/bes_cli/ctlrev_root; progress /content/bes_cli/ctlrev_progress.log;
/content/bes_cli/ctlrev_cache.tar.gz (cache/ of the root) is rewritten every 4 minutes and at the
end. A file /content/bes_cli/ctlrev_stop ends the run: generations in flight are lost, everything
finished is in the cache."""
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

k, workers, threads, model = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
select = sys.argv[5] if len(sys.argv) > 5 else "logprob_mean"
sl = sys.argv[6] if len(sys.argv) > 6 else "::-1"
root = Path("/content/bes_cli/ctlrev_root")
root.mkdir(parents=True, exist_ok=True)
log_path = Path("/content/bes_cli/ctlrev_progress.log")
stop_path = Path("/content/bes_cli/ctlrev_stop")
lock = threading.Lock()


def say(msg):
    line = f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}"
    with lock:
        with open(log_path, "a") as f:
            f.write(line + "\n")
    print(line, flush=True)


def pack():
    tmp = "/content/bes_cli/ctlrev_cache.tar.gz.tmp"
    with lock:
        with tarfile.open(tmp, "w:gz") as tf:
            if (root / "cache").exists():
                tf.add(root / "cache", arcname="cache")
        os.replace(tmp, "/content/bes_cli/ctlrev_cache.tar.gz")


def main():
    from bes.eval.proxy import prompts as prompt_set, runner
    from bes.eval.proxy.__main__ import build_spec
    from bes.eval.proxy.common import models_dir

    spec = build_spec(model, "recommended", [], None, 0, k, select, threads, None, None, {}, None,
                      models_dir(), True)
    ps = prompt_set.load()
    todo = prompt_set.select(ps, "test", None, None)[slice(*[int(x) if x else None for x in sl.split(":")])]
    say(f"start {model} samples {k} select {select}, slice [{sl}] = {len(todo)} prompts, run {spec.run_id}, "
        f"cache {runner.cache_path(root, spec.key).name}")

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
    res = runner.run(todo, spec, root, workers=workers, n_threads=threads, all_prompts=ps, log=say)
    done.set()
    pack()
    say(f"all done: {res.done} complete, {res.new_generations} new generations, {len(res.failed)} failed")


if __name__ == "__main__":
    main()
