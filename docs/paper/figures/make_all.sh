#!/usr/bin/env bash
# Rebuild every CSV, figure and table of the paper from the sources (run again when the logs grow).
#
#   bash make_all.sh            # re-extract the newest live snapshot, then data, figures, tables
#   bash make_all.sh --no-extract
#
# Heavy steps run on CPU cores 2 and 3 at low priority. Nothing outside paper/ is written.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/venv/bin/python
N=/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad
DATA="$HERE/../data"
RUN="taskset -c 2,3 nice -n 10"
export JAX_PLATFORMS=cpu

if [[ "${1:-}" != "--no-extract" ]]; then
  # the stopped final run is fixed; the live (v9) run's newest snapshot replaces the old extraction
  [[ -d "$DATA/_src/final/run" ]] || $RUN "$PY" -I -B "$HERE/extract_records.py" "$N/snap_final_main16" "$DATA/_src/final"
  rm -rf "$DATA/_src/live.new"
  $RUN "$PY" -I -B "$HERE/extract_records.py" "$N/snap_main" "$DATA/_src/live.new" > /dev/null
  rm -rf "$DATA/_src/live" && mv "$DATA/_src/live.new" "$DATA/_src/live"
fi
cd "$HERE"
$RUN "$PY" -I -B make_data.py
for f in fig1_loss fig2_throughput fig3_kernels fig4_v9cost fig5_openings fig6_graft fig7_model fig8_heldout; do
  $RUN "$PY" -I -B "$f.py"
done
$RUN "$PY" -I -B make_tables.py
