#!/usr/bin/env bash
# Rebuild every CSV in paper/data/ and every figure in paper/figures/ from the sources.
#
#   bash make_all.sh                         # re-read the newest snapshot, then all CSVs, then all figures,
#                                            # with the paper's cut-off (the newest step of numbers.json)
#   bash make_all.sh --cutoff newest         # ... with every record of the snapshot (or --cutoff STEP)
#   bash make_all.sh --blind final-TAG-vs-falcon,final-TAG-vs-mobilellm
#                                            # ... and add the final test-split blind comparisons (report
#                                            #     names in study/proxy/reports/ab/, without .json)
#   bash make_all.sh --no-extract            # keep the records extracted last time (no snapshot read)
#   bash make_all.sh --tables                # also rewrite paper/tables.md (make_tables.py)
#
# Inputs: see FIGURES.md ("How to rebuild") and paths.py. Writes only the data/ and figures/ next to
# this script (and tables.md with --tables), never those of another copy. Exit codes: 0 done, 2 bad
# argument or missing report, 3 snapshot still being copied, 4 cut-off newer than the snapshot (nothing
# written in each case), 1 a figure script failed. Python runs on CPU cores 2 and 3 at low priority.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/venv/bin/python
RUN="taskset -c 2,3 nice -n 5"
export JAX_PLATFORMS=cpu

EXTRACT=1
TABLES=0
BLIND=""
CUTOFF="paper"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-extract) EXTRACT=0 ;;
    --tables) TABLES=1 ;;
    --blind) BLIND="${2:?--blind needs a comma-separated list of report names}"; shift ;;
    --blind=*) BLIND="${1#--blind=}" ;;
    --cutoff) CUTOFF="${2:?--cutoff needs paper, newest or a step number}"; shift ;;
    --cutoff=*) CUTOFF="${1#--cutoff=}" ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "make_all.sh: unknown argument $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

cd "$HERE"
LOG="$HERE/_work/make_all.log"
mkdir -p "$HERE/_work"
echo "== make_data.py (cut-off: $CUTOFF)" > "$LOG"
set +e
$RUN "$PY" -I -B make_data.py --cutoff "$CUTOFF" ${BLIND:+--blind "$BLIND"} \
  $([[ $EXTRACT == 1 ]] || echo --no-extract) >> "$LOG" 2>&1
rc=$?
set -e
grep -E "^extracted|^cut-off|^MISMATCH|mismatch\(es\)|NOT added|largest gap|^  final step records|^make_data.py:" "$LOG" || true
if [[ $rc != 0 ]]; then
  echo "make_all.sh: make_data.py failed (exit $rc); no figure was redrawn. Last lines of $LOG:" >&2
  tail -4 "$LOG" >&2
  exit $rc
fi
for f in fig1_loss fig2_throughput fig3_kernels fig4_v9cost fig5_openings fig6_graft fig7_model fig8_heldout \
         fig9_waterfall fig10_costmodel fig11_blind; do
  echo "== $f.py" >> "$LOG"
  $RUN "$PY" -I -B "$f.py" >> "$LOG" 2>&1 || { echo "make_all.sh: $f.py failed; see $LOG" >&2; exit 1; }
done
if [[ $TABLES == 1 ]]; then
  $RUN "$PY" -I -B make_tables.py >> "$LOG" 2>&1
fi
OUT="${PAPER_FIGURES_OUT:-$HERE}"
ls -1 "$OUT"/F*.svg "$OUT"/F*.png | sed 's|.*/||' | paste -sd' ' -
echo "done; the printed checks of every script are in $LOG"
