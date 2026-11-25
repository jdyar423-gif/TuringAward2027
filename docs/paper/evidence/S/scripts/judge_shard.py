"""Runtime side: one stream of the pre-registered automatic judge on a RANGE of a pair set, so that
several streams (and runtimes) share one set. The judging is the registered one unchanged
(bes.eval.proxy.judge.judge_pair with the judge built by bes.eval.proxy.__main__.make_judge: same
model file, prompt, both orders, strict agreement); only the scheduling differs from
`python -m bes.eval.proxy judge`, which takes the pending pairs in file order in one process.

    python judge_shard.py TAG JUDGE THREADS CORES SET LO HI [WAITPATH::NEEDLE]

Root /content/bes_cli/judge/TAG/root (pairs/<SET>/..., ratings/<SET>/judge-<key>.jsonl when a pass was
begun elsewhere: those pairs are skipped). Pairs LO <= index < HI of pairs.jsonl that have no
rating yet are judged in file order; records (the same fields as run_judge writes, with the pair
set's commitment and the pair's digest) are appended to
/content/bes_cli/judge/TAG/shard_<SET>_<LO>_<HI>.jsonl; progress in .../shard_<SET>_<LO>_<HI>.log.
CORES like 0-3 or 4,5,6,7. Resumable."""
import os
import sys
import tarfile
import time
from pathlib import Path

REPO = "/content/TuringAward2027"
os.chdir(REPO)
sys.path.insert(0, REPO)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

tag, judge_name, threads, cores_s, set_, lo, hi = sys.argv[1:8]
threads, lo, hi = int(threads), int(lo), int(hi)
# optional 8th argument PATH::NEEDLE: start only when the file PATH contains NEEDLE (the stream that
# holds these cores now has ended); this process sleeps until then
if len(sys.argv) > 8 and sys.argv[8]:
    wpath, _, needle = sys.argv[8].partition("::")
    while True:
        try:
            if needle in open(wpath, errors="replace").read():
                break
        except OSError:
            pass
        time.sleep(5)
cores = []
for part in cores_s.split(","):
    a, _, b = part.partition("-")
    cores += list(range(int(a), int(b or a) + 1))
os.sched_setaffinity(0, cores)
try:
    os.nice(10)
except OSError:
    pass
base = Path("/content/bes_cli/judge") / tag
root = base / "root"
out = base / f"shard_{set_}_{lo}_{hi}.jsonl"
logp = base / f"shard_{set_}_{lo}_{hi}.log"


def say(msg):
    line = f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}"
    with open(logp, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


if not (root / "pairs" / set_ / "pairs.jsonl").exists():
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(base / "pairs.tar.gz") as tf:
        tf.extractall(root)

from bes.eval.proxy import make_ab  # noqa: E402
from bes.eval.proxy.__main__ import make_judge  # noqa: E402
from bes.eval.proxy.common import append_jsonl, models_dir, read_jsonl  # noqa: E402
from bes.eval.proxy.judge import judge_pair, ratings_path  # noqa: E402

pairs, _meta = make_ab.load_pairs(root, set_)
pub = make_ab.published(root, set_)
# the judge's model file: checked (size, SHA-256) and downloaded when missing, one stream at a time
import fcntl  # noqa: E402
from bes.eval.proxy import opponents  # noqa: E402
with open("/content/bes_cli/judge/fetch.lock", "w") as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    t_f = time.time()
    path_f = opponents.fetch(judge_name)
    say(f"judge file after {time.time() - t_f:.0f} s: {path_f}")
judge = make_judge(judge_name, threads, None, models_dir(), False)
source = f"judge-{judge.key}"
done = set()
for p in (ratings_path(root, set_, source), out):
    if Path(p).exists():
        recs = read_jsonl(p)
        bad = make_ab.foreign_records(recs, pub)
        if bad:
            raise SystemExit(f"{p}: {len(bad)} records do not belong to the published pair set")
        done |= {r["pair_id"] for r in recs if r.get("verdict") in make_ab.VERDICTS}
todo = [p for p in pairs[lo:hi] if p["pair_id"] not in done]
for p in todo:
    if pub["digests"].get(p["pair_id"]) != make_ab.pair_digest(p):
        raise SystemExit(f"pair {p['pair_id']} is not a published pair of {set_!r}")
say(f"start {set_} [{lo}, {hi}) with {source} on cores {cores_s} ({threads} threads): {len(todo)} to judge, "
    f"{hi - lo - len(todo)} of the range already rated")
t0 = time.time()
for n, pair in enumerate(todo, 1):
    rec = judge_pair(judge, pair)
    rec.update(commitment=pub["commitment"], pair_digest=pub["digests"][pair["pair_id"]])
    append_jsonl(out, [rec])
    if n % 5 == 0 or n == len(todo):
        say(f"{n}/{len(todo)} pairs, {(time.time() - t0) / 60:.1f} min")
say(f"all done: {len(todo)} pairs in {(time.time() - t0) / 60:.1f} min")
