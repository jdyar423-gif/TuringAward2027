#!/usr/bin/env bash
# The neutral-title replication of the two official sets: import both readings (source
# judge-opus-neutral), score, and compare prompt by prompt with the first pair of readings
# (judge-opus) and with the registered judge.
set -u
S=/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad
N=/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad
PY="taskset -c 0 $S/venv/bin/python"; ROOT=$N/study/proxy; cd /home/user/turingaward2027
for spec in "x1:final-prefopt-vs-falcon" "x2:final-prefopt-vs-mobilellm"; do nm=${spec%%:*}; set_=${spec#*:}; D=$N/final_eval/neutral/$nm
  n1=$(ls $D/a/chunk_*.verdicts.json 2>/dev/null | wc -l); n2=$(ls $D/b/chunk_*.verdicts.json 2>/dev/null | wc -l)
  echo "== $set_ (neutral $nm): readings $n1 and $n2 of 14 chunk files"
  [ "$n1" -eq 14 ] && [ "$n2" -eq 14 ] || continue
  $PY scripts/blind_eval.py import2 --root $ROOT --pairs $set_ --source judge-opus-neutral --order1 $D/a/chunk_*.verdicts.json --order2 $D/b/chunk_*.verdicts.json | tail -n 1 | cut -c1-120
  $PY -m bes.eval.proxy --root $ROOT score --pairs $set_ --source judge-opus-neutral 2>&1 | grep -E "^\| (overall|chat|code|instruct|math|writing|single_turn|multi_turn) |Judge diagnostics" | cut -c1-200
  $PY - "$set_" <<'PY'
import json, sys, collections
N = "/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad"
s = sys.argv[1]
def pps(src):
    try:
        return json.load(open(f"{N}/study/proxy/reports/ab/{s}--{src}.json"))["per_prompt_score"]
    except FileNotFoundError:
        return None
a, b, q = pps("judge-opus"), pps("judge-opus-neutral"), pps("judge-qwen3-4b-2507-3605803b")
def tab(x, y, nx, ny):
    c = collections.Counter((x[k], y[k]) for k in x if k in y)
    print(f"rows {nx}, columns {ny} (win / tie / loss):", " | ".join(f"{nm} " + "/".join(str(c[(v, w)]) for w in (1.0, 0.5, 0.0)) for v, nm in ((1.0, "win"), (0.5, "tie"), (0.0, "loss"))))
    both = [(x[k], y[k]) for k in x if k in y and x[k] != 0.5 and y[k] != 0.5]
    same = sum(x[k] == y[k] for k in x if k in y)
    print(f"  same outcome on {same} of {len([k for k in x if k in y])}; both decisive on {len(both)}, agreeing on {sum(u == v for u, v in both)}")
tab(a, b, "first readings", "neutral readings")
if q: tab(b, q, "neutral readings", "registered judge")
# paired sign test of the per-prompt scores (named titles minus neutral titles)
import math
d = [a[k] - b[k] for k in a if k in b]
pos, neg = sum(v > 0 for v in d), sum(v < 0 for v in d)
m, kk = pos + neg, min(pos, neg)
p = min(1.0, 2 * sum(math.comb(m, i) for i in range(kk + 1)) / 2 ** m) if m else 1.0
print(f"  per-prompt score higher with named titles on {pos}, lower on {neg}, mean difference {sum(d)/len(d):+.4f}, sign test p = {p:.3f}")
PY
done
