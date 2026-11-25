"""Facts of the registered judge's pass on MobileLLM-R1.5's two K = 6 control pair sets, from the
shard files and logs fetched from the three CPU runtimes (N/judge_remote/testq5 = k6lp,
testq6 = k6ds) and the merged rating files: streams, first start and last end (the logs' clock
is the runtimes', UTC), seconds per pair of the kept records, pairs judged by two streams.
    python judge_pass_facts.py   -> N/peek/judge_pass_facts.json and a printed summary"""
import glob
import json
import os
import re

N = "/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad"
SRC = "judge-qwen3-4b-2507-3605803b"
SETS = (("testq5", "final-prefopt-vs-mobilellm-k6lp"), ("testq6", "final-prefopt-vs-mobilellm-k6ds"))
out = {"sets": {}}
starts, ends, kept_seconds, twice_all, streams = [], [], [], 0, 0
dup_total = {"pairs": 0, "same_verdict": 0, "same_probs": 0, "kept_forward": 0, "kept_helper": 0}
for tag, name in SETS:
    d = f"{N}/judge_remote/{tag}"
    per_shard, seen = {}, {}
    for f in sorted(glob.glob(f"{d}/shards/shard_{name}_*.jsonl")):
        recs = [json.loads(x) for x in open(f) if x.strip()]
        per_shard[os.path.basename(f)] = len(recs)
        for r in recs:
            seen.setdefault(r["pair_id"], []).append(os.path.basename(f))
    logs = {}
    for f in sorted(glob.glob(f"{d}/shard_{name}_*.log")):
        txt = open(f, errors="replace").read()
        st = re.search(r"\[(\d\d:\d\d:\d\d)\] start ", txt)
        dn = [m.group(1) for m in re.finditer(r"\[(\d\d:\d\d:\d\d)\][^\n]*done:", txt)]
        logs[os.path.basename(f)] = {"start": st.group(1) if st else None, "done": dn[-1] if dn else None,
                                     "last_line": txt.strip().splitlines()[-1][:160] if txt.strip() else ""}
        if st:
            starts.append(st.group(1))
        if dn:
            ends.append(dn[-1])
    kept = [json.loads(x) for x in open(f"{N}/study/proxy/ratings/{name}/{SRC}.jsonl") if x.strip()]
    pairs = [json.loads(x) for x in open(f"{N}/study/proxy/pairs/{name}/pairs.jsonl") if x.strip()]
    ids = [p["pair_id"] for p in pairs]
    kept_ids = [r["pair_id"] for r in kept]
    twice = sorted(k for k, v in seen.items() if len(v) > 1)
    # the two records of a pair judged by two streams: the same verdict? which one was kept?
    recs_of = {}
    for f in sorted(glob.glob(f"{d}/shards/shard_{name}_*.jsonl")):
        for x in open(f):
            if x.strip():
                r = json.loads(x)
                recs_of.setdefault(r["pair_id"], []).append((os.path.basename(f), r))
    kept_by_id = {r["pair_id"]: r for r in kept}
    for k in twice:
        v = recs_of[k]
        dup_total["pairs"] += 1
        dup_total["same_verdict"] += len({r["verdict"] for _, r in v}) == 1
        dup_total["same_probs"] += all(r["order1"]["probs"] == v[0][1]["order1"]["probs"] and r["order2_swapped"]["probs"] == v[0][1]["order2_swapped"]["probs"] for _, r in v)
        src = [b for b, r in v if r["seconds"] == kept_by_id[k]["seconds"] and r["verdict"] == kept_by_id[k]["verdict"]]
        assert len(src) == 1, (k, src)
        dup_total["kept_helper" if src[0].endswith("r.jsonl") else "kept_forward"] += 1
    assert all(k in recs_of for k in kept_by_id), "a kept record is in no shard file"
    out["sets"][name] = {"pairs": len(ids), "kept_records": len(kept), "kept_unique": len(set(kept_ids)), "all_pairs_rated": set(kept_ids) == set(ids),
                         "shard_records": per_shard, "pairs_in_two_shards": len(twice), "logs": logs,
                         "seconds_mean": sum(r["seconds"] for r in kept) / max(1, len(kept)), "seconds_sum": sum(r["seconds"] for r in kept),
                         "judge_names": sorted({r.get("judge") for r in kept}), "commitments": sorted({r.get("commitment") for r in kept})}
    kept_seconds += [r["seconds"] for r in kept]
    twice_all += len(twice)
    streams += len(per_shard)
out["streams"] = streams
out["twice"] = twice_all
out["twice_detail"] = dup_total
out["meta_commitments_match"] = all(s["commitments"] == [json.load(open(f"{N}/study/proxy/pairs/{n}/meta.json"))["commitment"]] for n, s in out["sets"].items())
out["seconds_mean"] = sum(kept_seconds) / max(1, len(kept_seconds))
out["first_start"] = min(starts) if starts else None
out["last_done"] = max(ends) if ends else None
out["all_done_logged"] = all(v["done"] for s in out["sets"].values() for v in s["logs"].values())
json.dump(out, open(f"{N}/peek/judge_pass_facts.json", "w"), indent=1)
for name, s in out["sets"].items():
    print(name, "| kept", s["kept_records"], "unique", s["kept_unique"], "complete", s["all_pairs_rated"], "| shards", s["shard_records"],
          "| in two shards", s["pairs_in_two_shards"], "| s/pair %.1f" % s["seconds_mean"], "| judge", s["judge_names"], "| commitments", len(s["commitments"]))
    for k, v in s["logs"].items():
        print("   ", k, v["start"], v["done"], "|", v["last_line"])
print("twice:", out["twice_detail"], "| commitments match the pair sets:", out["meta_commitments_match"])
print("streams", out["streams"], "| twice", out["twice"], "| s/pair %.1f" % out["seconds_mean"], "| first start", out["first_start"], "| last done", out["last_done"], "| all logs done:", out["all_done_logged"])
