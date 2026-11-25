#!/usr/bin/env bash
# Import the judges' verdict files of a study's pair sets (one reading, source opus-1) and score.
# usage: study_score.sh TAG SET...     (SET: e.g. s2-default-vs-falcon)
set -u
tag="$1"; shift
S=/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad
N=/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad
PY="taskset -c 0,1 $S/venv/bin/python"; ROOT=$N/study/proxy
cd /home/user/turingaward2027
for set in "$@"; do
  D=$N/study/$tag/judge/$set
  n=$(ls "$D"/chunk_*.verdicts.json 2>/dev/null | wc -l); m=$(ls "$D"/chunk_*.md | wc -l)
  if [ "$n" != "$m" ]; then echo "$set: $n of $m verdict files"; continue; fi
  if [ ! -e "$D/imported" ]; then
    $PY "$S/scripts/blind_eval.py" import --root "$ROOT" --pairs "$set" --source opus-1 --verdicts "$D"/chunk_*.verdicts.json 2>&1 | tail -n 1 | cut -c1-200 && touch "$D/imported"
  fi
  echo "== $set"; $PY -m bes.eval.proxy --root "$ROOT" score --pairs "$set" --source opus-1 2>&1 | tail -n 14 | cut -c1-260
done
