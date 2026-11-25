#!/usr/bin/env bash
# The final blind comparison on the 200 test prompts: merge a fetched test generation into the
# evaluation root, publish the pair sets against both opponents and write both readings' chunk
# files (r1: as published; r2: replies A and B exchanged) and the judges' job lists.
# usage: final_pairs.sh TAG [WORKFLOWS=3]   (after endgame_testgen.sh --fetch TAG; TAG: trained | prefopt | official)
set -u
tag="$1"; nwf="${2:-3}"
S=/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad
N=/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad
PY="taskset -c 0,1 $S/venv/bin/python"; ROOT=$N/study/proxy; D=$N/testgen/$tag; J=$N/final_eval/$tag
mkdir -p "$J"; cd /home/user/turingaward2027
rid=$($PY scripts/gen_test_replies.py "$D/out" --merge-into "$ROOT" 2>&1 | tee "$D/merge.log" | grep -o "ours-export-[A-Za-z0-9_./-]*" | tail -n 1)
grep -E "merged|complete prompts" "$D/merge.log" | cut -c1-240
[ -n "$rid" ] || { echo "no run id: see $D/merge.log"; exit 1; }
echo "$rid" > "$J/run_id.txt"; echo "run id: $rid"
for opp in falcon:falcon-e60034381f4a/recommended-5ccec0259b-s0 mobilellm:mobilellm-f680f0533368-qwen3/recommended-d9f149b0d3-s0; do
  name=final-$tag-vs-${opp%%:*}; y=${opp#*:}
  if [ ! -d "$ROOT/pairs/$name" ]; then $PY -m bes.eval.proxy --root "$ROOT" pairs --x "$rid" --y "$y" --name "$name" --split test 2>&1 | grep -E '"pairs"|"n"|multi_turn|rror' | head -n 4; fi
  mkdir -p "$J/$name/r1" "$J/$name/r2"
  $PY "$S/scripts/blind_eval.py" export --root "$ROOT" --pairs "$name" --out "$J/$name/r1" --chunk 15 | tail -n 1 | cut -c1-80
  $PY "$S/scripts/blind_eval.py" export --root "$ROOT" --pairs "$name" --out "$J/$name/r2" --chunk 15 --swap | tail -n 1 | cut -c1-80
done
python3 - "$J" "$tag" "$nwf" <<'PY'
import json, sys, glob, os
J, tag, nwf = sys.argv[1], sys.argv[2], int(sys.argv[3])
jobs = []
for name in sorted(os.listdir(J)):
    if not name.startswith("final-"): continue
    for r in ("r1", "r2"):
        chunks = sorted(glob.glob(f"{J}/{name}/{r}/chunk_*.md"))
        chunks = [c for c in chunks if not os.path.exists(c[:-3] + ".verdicts.json")]
        for i in range(0, len(chunks), 2):
            jobs.append({"set": f"{name} {r}", "chunks": chunks[i:i + 2]})
# interleave so that every workflow holds both opponents and both readings
parts = [jobs[i::nwf] for i in range(nwf)]
for i, p in enumerate(parts):
    json.dump(p, open(f"{J}/jobs_{i}.json", "w"))
print(f"{len(jobs)} judge jobs in {nwf} lists: " + ", ".join(f"{J}/jobs_{i}.json ({len(p)})" for i, p in enumerate(parts)))
PY
