#!/usr/bin/env bash
# Import both readings of the final pair sets (source judge-opus) and score them; then the same
# counts on the prompts never looked at during the run.  usage: final_score.sh TAG
set -u
tag="$1"
S=/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad
N=/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad
PY="taskset -c 0,1 $S/venv/bin/python"; ROOT=$N/study/proxy; J=$N/final_eval/$tag
cd /home/user/turingaward2027
for name in final-$tag-vs-falcon final-$tag-vs-mobilellm; do
  n1=$(ls "$J/$name/r1"/chunk_*.verdicts.json 2>/dev/null | wc -l); n2=$(ls "$J/$name/r2"/chunk_*.verdicts.json 2>/dev/null | wc -l); m=$(ls "$J/$name/r1"/chunk_*.md | wc -l)
  echo "== $name: readings $n1 and $n2 of $m chunk files"
  [ "$n1" -gt 0 ] && [ "$n2" -gt 0 ] || continue
  $PY "$S/scripts/blind_eval.py" import2 --root "$ROOT" --pairs "$name" --source judge-opus --order1 "$J/$name/r1"/chunk_*.verdicts.json --order2 "$J/$name/r2"/chunk_*.verdicts.json | tail -n 1 | cut -c1-240
  $PY -m bes.eval.proxy --root "$ROOT" score --pairs "$name" --source judge-opus 2>&1 | tail -n 30 | cut -c1-260
done
