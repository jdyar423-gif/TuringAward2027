"""Merge shard files of the registered judge (judge_shard.py) into the pair set's main ratings file.
    python judge_shard_merge.py ROOT SET SHARD.jsonl [SHARD.jsonl ...]
Every record is checked against the published pair set (commitment, pair digest); one record per
pair (the main file's first, then the shards in the order given); the main file is rewritten in
the order of pairs.jsonl. Prints the count."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/user/turingaward2027")
from bes.eval.proxy import make_ab  # noqa: E402
from bes.eval.proxy.common import read_jsonl  # noqa: E402

root, set_, shards = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
pairs, _ = make_ab.load_pairs(root, set_)
pub = make_ab.published(root, set_)
order = {p["pair_id"]: i for i, p in enumerate(pairs)}
mains = sorted((root / "ratings" / set_).glob("judge-qwen3-4b-2507-*.jsonl"))
assert len(mains) <= 1, mains
key = None
recs = {}
dup = 0
for path in [*mains, *map(Path, shards)]:
    rs = read_jsonl(path)
    bad = make_ab.foreign_records(rs, pub)
    if bad:
        raise SystemExit(f"{path}: {len(bad)} records do not belong to the published pair set")
    for r in rs:
        if r.get("verdict") not in make_ab.VERDICTS:
            continue
        key = key or r["judge"]
        assert r["judge"] == key, (r["judge"], key)
        if r["pair_id"] in recs:
            dup += 1
            continue
        recs[r["pair_id"]] = r
main = mains[0] if mains else root / "ratings" / set_ / f"judge-{key}.jsonl"
out = sorted(recs.values(), key=lambda r: order[r["pair_id"]])
main.parent.mkdir(parents=True, exist_ok=True)
tmp = main.with_suffix(".jsonl.tmp")
with open(tmp, "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
tmp.replace(main)
missing = [i for i, p in enumerate(pairs) if p["pair_id"] not in recs]
print(f"{set_}: {len(out)} of {len(pairs)} pairs rated in {main.name} ({dup} duplicate records dropped); "
      f"missing indices: {missing[:12]}{' ...' if len(missing) > 12 else ''}")
