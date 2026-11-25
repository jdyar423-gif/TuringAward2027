#!/usr/bin/env bash
# The pre-registered automatic judge on the runtime's idle host cores.
#   judge_remote.sh start TAG JUDGE THREADS STREAMS SET...   (uploads the blinded pairs, starts)
#   judge_remote.sh fetch TAG                                 (progress; ratings into $N/study/proxy when done)
set -u
export PATH=/root/.local/bin:$PATH GOOGLE_APPLICATION_CREDENTIALS=/root/.config/colab-cli-auth/adc_user.json COLAB_CLI_ARGS=--auth=adc
S=/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad
N=/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad
SES=${SESSION:-bes24h}; ROOT=$N/study/proxy; mode="$1"; tag="$2"; D=$N/judge_remote/$tag; mkdir -p "$D"
if [ "$mode" = start ]; then
  judge="$3"; threads="$4"; streams="$5"; shift 5
  args=(); for s in "$@"; do args+=("pairs/$s/meta.json" "pairs/$s/pairs.jsonl")
    # ratings this judge already wrote for the set (a pass begun elsewhere): the judge skips them
    for f in "$ROOT/ratings/$s"/judge-"$judge"*.jsonl; do [ -e "$f" ] && args+=("ratings/$s/$(basename "$f")"); done
  done
  tar -czf "$D/pairs.tar.gz" -C "$ROOT" "${args[@]}" || exit 1
  date +%s > "$S/presence_pause.txt"; timeout 60 colab --auth=adc upload -s "$SES" "$S/presence_pause.txt" /content/bes_cli/presence_pause </dev/null >/dev/null 2>&1
  echo "import os; os.makedirs('/content/bes_cli/judge/$tag', exist_ok=True); print('dir ok')" > "$D/mk.py"
  timeout 60 colab --auth=adc upload -s "$SES" "$D/mk.py" /content/bes_cli/judge_mk.py </dev/null >/dev/null 2>&1; sleep 3
  TAG=peek bash "$S/scripts/exec_detached.sh" "$SES" "$S/peek_nonce.txt" 3 --env "BES_CMD=/content/bes_cli/judge_mk.py" | tail -n 1
  TAG=peek timeout 60 bash "$S/scripts/exec_wait.sh" "$SES" "$S/peek_nonce.txt" "$D/mk.out" 40 | tail -n 1
  timeout 120 colab --auth=adc upload -s "$SES" "$D/pairs.tar.gz" /content/bes_cli/judge/$tag/pairs.tar.gz </dev/null 2>&1 | tail -n 1
  timeout 60 colab --auth=adc upload -s "$SES" "$S/scripts/judge_driver.py" /content/bes_cli/judge_driver.py </dev/null >/dev/null 2>&1; sleep 3
  TAG=jd$tag bash "$S/scripts/exec_detached.sh" "$SES" "$S/jd${tag}_nonce.txt" 3 --env "BES_CMD=/content/bes_cli/judge_driver.py $tag $judge $threads $streams $*"
else
  rm -f "$D/progress.log"; timeout 40 colab --auth=adc download -s "$SES" /content/bes_cli/judge/$tag/progress.log "$D/progress.log" </dev/null >/dev/null 2>&1; tail -n 6 "$D/progress.log" 2>/dev/null | cut -c1-260
  if grep -q "all done" "$D/progress.log" 2>/dev/null; then
    rm -f "$D/ratings.tar.gz"; timeout 120 colab --auth=adc download -s "$SES" /content/bes_cli/judge/$tag/ratings.tar.gz "$D/ratings.tar.gz" </dev/null >/dev/null 2>&1
    mkdir -p "$D/x"; tar -xzf "$D/ratings.tar.gz" -C "$D/x" && find "$D/x/ratings" -name "*.jsonl" | while read f; do rel=${f#$D/x/}; mkdir -p "$ROOT/$(dirname "$rel")"; if [ -e "$ROOT/$rel" ] && [ "$(wc -l < "$ROOT/$rel")" -ge "$(wc -l < "$f")" ]; then echo "exists with as many records, kept: $rel"; else cp "$f" "$ROOT/$rel"; echo "copied $rel ($(wc -l < "$f") records)"; fi; done
  fi
fi
