#!/usr/bin/env bash
# Run experiments from a file (one "name|args" per line), N at a time.
# Usage: scripts/queue.sh jobs.txt 2 [extra args applied to all]
set -u
cd "$(dirname "$0")/.."
FILE=$1; N=${2:-2}; shift 2; EXTRA="$*"
THREADS=${THREADS:-$((4 / N))}
grep -v '^\s*#' "$FILE" | grep -v '^\s*$' | while IFS='|' read -r name args; do
  echo "python3 train.py --name $name --threads $THREADS $EXTRA $args > logs/$name.log 2>&1"
done | xargs -P "$N" -I{} bash -c "{}"
