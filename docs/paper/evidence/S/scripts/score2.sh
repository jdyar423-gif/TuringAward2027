#!/usr/bin/env bash
# Two readings of one pair set (the second on exchanged sides) -> rating source judge-opus -> score.
# usage: score2.sh SET R1_DIR R2_DIR [LINES=14]
set -u
set_="$1"; r1="$2"; r2="$3"; lines="${4:-14}"
S=/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad
N=/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad
PY="taskset -c 0,1 $S/venv/bin/python"; ROOT=$N/study/proxy
cd /home/user/turingaward2027
n1=$(ls "$r1"/chunk_*.verdicts.json 2>/dev/null | wc -l); n2=$(ls "$r2"/chunk_*.verdicts.json 2>/dev/null | wc -l); m=$(ls "$r1"/chunk_*.md | wc -l)
echo "== $set_: readings $n1 and $n2 of $m chunk files"
[ "$n1" -gt 0 ] && [ "$n2" -gt 0 ] || exit 0
$PY "$S/scripts/blind_eval.py" import2 --root "$ROOT" --pairs "$set_" --source judge-opus --order1 "$r1"/chunk_*.verdicts.json --order2 "$r2"/chunk_*.verdicts.json | tail -n 1 | cut -c1-200
$PY -m bes.eval.proxy --root "$ROOT" score --pairs "$set_" --source judge-opus > "$r2/score.out" 2>&1
grep -n "overall\|agree\|first position\|position" "$r2/score.out" | head -n 12 | cut -c1-240
sed -n 1,${lines}p "$ROOT/reports/ab/$set_--judge-opus.md" 2>/dev/null | sed -n 9,${lines}p | cut -c1-200
