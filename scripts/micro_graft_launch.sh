#!/usr/bin/env bash
# Launch the graft experiment of the micro bench (docs/MICRO_GRAFT.md) on a many-core machine:
# build the pools once, then one process per arm, each with its own --out sub-directory and its
# own CPUs (no two processes write one results file), wait for all of them, write the report.
#
#   TOKENS=1000000 bash scripts/micro_graft_launch.sh            # 15 runs on CPUs 0-29
#   DRY_RUN=1 TOKENS=2000000 bash scripts/micro_graft_launch.sh   # print the commands only
#
# Variables (defaults): TOKENS (1000000), OUT (runs/graft_<TOKENS>), FIRST_CPU (0), THREADS (2,
# cores per run; run i gets CPUs FIRST_CPU + THREADS*i ...), ARMS (the 15 arms of
# micro_ablate.GRAFT_EXPERIMENT), PYTHON (python), EXTRA (options added to every run, e.g.
# "--samples-dir /data/samples"), DRY_RUN (print, do not run).
set -euo pipefail
cd "$(dirname "$0")/.."
TOKENS=${TOKENS:-1000000}
OUT=${OUT:-runs/graft_${TOKENS}}
FIRST_CPU=${FIRST_CPU:-0}
THREADS=${THREADS:-2}
PYTHON=${PYTHON:-python}
EXTRA=${EXTRA:-}
FRACS=0.25,0.5,0.75,1.0
ARMS=${ARMS:-"ref ref_s1 ref_s2 v9_pages v9_pages_s1 v9_pages_s2 graft_25 graft_25_s1 graft_25_s2 graft_50 graft_50_s1 graft_50_s2 graft_75 graft_75_s1 graft_75_s2"}
export JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1

cpus_of() {  # the CPU list of run number $1
  local first=$((FIRST_CPU + THREADS * $1))
  seq -s, "$first" $((first + THREADS - 1))
}
show() { printf '+'; printf ' %s' "$@"; printf '\n'; }

mkdir -p "$OUT"
prep=("$PYTHON" scripts/micro_ablate.py --prepare --arms ref --tokens "$TOKENS" --eval-fracs "$FRACS"
      --out "$OUT" --threads "$THREADS" --cpus "$(cpus_of 0)")
# shellcheck disable=SC2206
prep+=($EXTRA)
show "${prep[@]}"
[ -n "${DRY_RUN:-}" ] || "${prep[@]}"

pids=()
# an interrupted launcher takes its runs with it (they resume from their last checkpoint when the
# same command is given again); without this they would go on, and a second launch would put two
# processes on one --out
stop_runs() {
  echo "interrupted: stopping ${#pids[@]} run(s)" >&2
  for p in ${pids[@]+"${pids[@]}"}; do kill "$p" 2>/dev/null; done
  wait
  exit 130
}
trap stop_runs INT TERM HUP
i=0
for arm in $ARMS; do
  if [ -z "${DRY_RUN:-}" ] && pgrep -f "micro_ablate.py .*--out $OUT/$arm( |\$)" > /dev/null; then
    echo "a run with --out $OUT/$arm is alive: stop it first (pkill -f 'micro_ablate.py .*--out $OUT/')" >&2
    for p in ${pids[@]+"${pids[@]}"}; do kill "$p" 2>/dev/null; done
    exit 2
  fi
  cmd=("$PYTHON" scripts/micro_ablate.py --arms "$arm" --tokens "$TOKENS" --eval-fracs "$FRACS"
       --out "$OUT/$arm" --threads "$THREADS" --cpus "$(cpus_of "$i")" --no-probe)
  # shellcheck disable=SC2206
  cmd+=($EXTRA)
  show "${cmd[@]}"
  echo "  > $OUT/$arm.log"
  if [ -z "${DRY_RUN:-}" ]; then
    nohup "${cmd[@]}" > "$OUT/$arm.log" 2>&1 &
    pids+=($!)
  fi
  i=$((i + 1))
done
failed=0
for p in ${pids[@]+"${pids[@]}"}; do
  wait "$p" || failed=$((failed + 1))
done
[ "$failed" -eq 0 ] || echo "$failed run(s) failed: see $OUT/<arm>.log" >&2
rep=("$PYTHON" scripts/micro_graft_report.py "$OUT" --json "$OUT/graft_report.json"
     --merge "$OUT/results_all.jsonl")
show "${rep[@]}"
echo "  > $OUT/graft_report.txt"
[ -n "${DRY_RUN:-}" ] || "${rep[@]}" > "$OUT/graft_report.txt"
exit "$failed"
