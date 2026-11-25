#!/usr/bin/env bash
# Streams of the registered judge on ranges of one pair set (judge_shard.py on a CPU runtime).
#   judge_shard.sh start SESSION TAG JUDGE SET CORES:LO:HI[@WAITPATH::NEEDLE] [...]   (a stream may wait for another's end)
#   judge_shard.sh fetch SESSION TAG SET        (downloads the shard files of that session, prints progress)
#   judge_shard.sh merge TAG SET                (all fetched shards + the main file -> the main ratings file)
set -u
export PATH=/root/.local/bin:$PATH GOOGLE_APPLICATION_CREDENTIALS=/root/.config/colab-cli-auth/adc_user.json COLAB_CLI_ARGS=--auth=adc
S=/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad
N=/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad
ROOT=$N/study/proxy; mode="$1"
if [ "$mode" = start ]; then
  SES="$2"; tag="$3"; judge="$4"; set_="$5"; shift 5; D=$N/judge_remote/$tag; mkdir -p "$D"
  args=("pairs/$set_/meta.json" "pairs/$set_/pairs.jsonl")
  for f in "$ROOT/ratings/$set_"/judge-"$judge"*.jsonl; do [ -e "$f" ] && args+=("ratings/$set_/$(basename "$f")"); done
  tar -czf "$D/pairs_$SES.tar.gz" -C "$ROOT" "${args[@]}" || exit 1
  date +%s > "$S/presence_pause.txt"; timeout 60 colab --auth=adc upload -s "$SES" "$S/presence_pause.txt" /content/bes_cli/presence_pause </dev/null >/dev/null 2>&1
  echo "import os; os.makedirs('/content/bes_cli/judge/$tag', exist_ok=True); print('dir ok')" > "$D/mk.py"
  timeout 60 colab --auth=adc upload -s "$SES" "$D/mk.py" /content/bes_cli/judge_mk.py </dev/null >/dev/null 2>&1; sleep 3
  TAG=peek bash "$S/scripts/exec_detached.sh" "$SES" "$S/peek_nonce.txt" 3 --env "BES_CMD=/content/bes_cli/judge_mk.py" | tail -n 1
  TAG=peek timeout 60 bash "$S/scripts/exec_wait.sh" "$SES" "$S/peek_nonce.txt" "$D/mk.out" 40 | tail -n 1
  timeout 120 colab --auth=adc upload -s "$SES" "$D/pairs_$SES.tar.gz" /content/bes_cli/judge/$tag/pairs.tar.gz </dev/null 2>&1 | tail -n 1 | cut -c1-120
  timeout 60 colab --auth=adc upload -s "$SES" "$S/scripts/judge_shard.py" /content/bes_cli/judge_shard.py </dev/null >/dev/null 2>&1; sleep 2
  for spec in "$@"; do
    cores=${spec%%:*}; rest=${spec#*:}; lo=${rest%%:*}; rest2=${rest#*:}; hi=${rest2%%@*}; wait_=""; case "$rest2" in *@*) wait_=${rest2#*@};; esac
    echo "$SES $set_ $lo $hi $cores" >> "$D/ranges.txt"
    date +%s > "$S/presence_pause.txt"; timeout 60 colab --auth=adc upload -s "$SES" "$S/presence_pause.txt" /content/bes_cli/presence_pause </dev/null >/dev/null 2>&1
    TAG=js${tag}_${lo} bash "$S/scripts/exec_detached.sh" "$SES" "$S/js${tag}_${lo}_nonce.txt" 3 --env "BES_CMD=/content/bes_cli/judge_shard.py $tag $judge 4 $cores $set_ $lo $hi $wait_" | tail -n 1
    sleep 4
  done
elif [ "$mode" = fetch ]; then
  SES="$2"; tag="$3"; set_="$4"; D=$N/judge_remote/$tag; mkdir -p "$D/shards"
  grep "^$SES $set_ " "$D/ranges.txt" | sort -u | while read -r _s _set lo hi cores; do
    b=shard_${set_}_${lo}_${hi}
    rm -f "$D/shards/$b.jsonl.tmp" "$D/$b.log"
    timeout 60 colab --auth=adc download -s "$SES" /content/bes_cli/judge/$tag/$b.jsonl "$D/shards/$b.jsonl.tmp" </dev/null >/dev/null 2>&1
    if [ -s "$D/shards/$b.jsonl.tmp" ] && [ "$(wc -l < "$D/shards/$b.jsonl.tmp")" -ge "$( [ -e "$D/shards/$b.jsonl" ] && wc -l < "$D/shards/$b.jsonl" || echo 0)" ]; then mv "$D/shards/$b.jsonl.tmp" "$D/shards/$b.jsonl"; fi
    timeout 40 colab --auth=adc download -s "$SES" /content/bes_cli/judge/$tag/$b.log "$D/$b.log" </dev/null >/dev/null 2>&1
    echo "$SES [$lo,$hi) cores $cores: $( [ -e "$D/shards/$b.jsonl" ] && wc -l < "$D/shards/$b.jsonl" || echo 0) records | $(tail -n 1 "$D/$b.log" 2>/dev/null | cut -c1-110)"
  done
else
  tag="$2"; set_="$3"; D=$N/judge_remote/$tag
  cd /home/user/turingaward2027 && taskset -c 0,1 "$S/venv/bin/python" "$S/scripts/judge_shard_merge.py" "$ROOT" "$set_" "$D"/shards/shard_${set_}_*.jsonl
fi
