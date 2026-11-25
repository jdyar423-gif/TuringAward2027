#!/usr/bin/env bash
# Keep-awake for the time after the TPU: a tracked background task that only sleeps, and
# restarts a presence loop of a CPU runtime when it died (never one whose stop file exists).
# usage: anchor3.sh MAX_S "SESSION:NAME ..."
set -u
max_s="$1"; pairs="${2:-bes-cpu:C bes-cpu2:E bes-cpu2:F bes-cpu3:G bes-cpu3:H}"
S=/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad
while [ $SECONDS -lt "$max_s" ]; do
  sleep 30
  for p in $pairs; do
    ses=${p%%:*}; name=${p#*:}
    [ -e "$S/presence_stop_$ses" ] && continue
    if [ "$(pgrep -fc "scripts/presence.sh $ses $name ")" -lt 1 ]; then
      setsid nohup bash "$S/scripts/presence.sh" "$ses" "$name" 200 > /dev/null 2>&1 < /dev/null &
      echo "[$(date -u +%H:%M:%S)] presence $ses $name restarted"
    fi
  done
done
echo "anchor3: end after ${SECONDS}s"
