"""Assemble MobileLLM's control arms from the merged generation caches of all machines, WITHOUT
generating anything (bes.eval.proxy.runner.assemble writes only conversations that are complete in
the cache).

    python ctl_assemble.py MERGED_ROOT CACHE.jsonl [CACHE.jsonl ...]

The cache files are concatenated in the order given (for a key present twice the later line wins, as
in runner.load_cache; the slices are disjoint, so this matters only where two machines overlapped)
into MERGED_ROOT/cache/<backend>.jsonl; then both select rules are assembled and the number of
complete test prompts is printed, with the indices that are missing."""
import json
import sys
from pathlib import Path

R = "/home/user/turingaward2027"
N = Path("/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad")
sys.path.insert(0, R)
from bes.eval.proxy import prompts as prompt_set, runner  # noqa: E402
from bes.eval.proxy.__main__ import build_spec  # noqa: E402

root = Path(sys.argv[1])
files = [Path(f) for f in sys.argv[2:]]
(root / "cache").mkdir(parents=True, exist_ok=True)
ps = prompt_set.load()
test = prompt_set.select(ps, "test", None, None)
out = {}
for rule in ("logprob_mean", "decoder_select"):
    spec = build_spec("mobilellm", "recommended", [], None, 0, 6, rule, 1, None, None, {}, None, N / "models", False)
    cpath = runner.cache_path(root, spec.key)
    if rule == "logprob_mean":
        n = 0
        with open(cpath, "w") as f:
            for src in files:
                for line in open(src):
                    if line.strip():
                        json.loads(line)
                        f.write(line if line.endswith("\n") else line + "\n")
                        n += 1
        print(f"merged {n} cache lines from {len(files)} files into {cpath}")
    path = runner.assemble(spec, root, ps)
    recs = {json.loads(line)["prompt_id"] for line in open(path)}
    missing = [i for i, p in enumerate(test) if p.id not in recs]
    out[rule] = (spec.run_id, len(test) - len(missing), missing)
    print(f"{rule}: run {spec.run_id}: {len(test) - len(missing)} of {len(test)} test prompts complete; "
          f"missing indices: {missing[:40]}{' ...' if len(missing) > 40 else ''}")
