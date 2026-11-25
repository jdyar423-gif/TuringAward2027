#!/bin/bash
# Queue of held-out evaluations on CPU cores 2 and 3 (second-pass lead author, 24 Nov 2026).
N=/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad
W=$N/paper/lead2_work
PY=/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/venv/bin/python
OUT=$W/heldout
mkdir -p $OUT
V9MAN=$N/paper/data/_src/live/run/manifest.json
FMAN=$N/paper/lead_work/final_snap/run/manifest.json
run() { echo "=== $(date -u +%H:%M:%S) $*"; taskset -c 2,3 nice -n 10 $PY -I $W/heldout_eval.py "$@" 2>&1 | grep -v "^WARNING\|^I0000\|^W0000\|^E0000"; }
# v9: the newest snapshot (read once into memory; four switch settings)
run --manifest $V9MAN --ckpt $N/snap_main --name v9_newest --out $OUT/v9_newest \
    --variant on: --variant pagesoff:page=0,gain=0 --variant pageoff:page=0 --variant gainoff:gain=0
# v9: the previous snapshot, if it is still there
if [ -d $N/snap_main.prev ]; then
  run --manifest $V9MAN --ckpt $N/snap_main.prev --name v9_prev --out $OUT/v9_prev --variant on: --variant pagesoff:page=0,gain=0
fi
# final: the stopped run at the graft, and earlier checkpoints for the trend
run --manifest $FMAN --ckpt $N/final_look/run/train/ckpt/ckpt-000026-step-000029722 --name final_29722 --out $OUT/final_29722
run --manifest $FMAN --ckpt $N/snap_look/run/train/ckpt/ckpt-000024-step-000026890 --name final_26890 --out $OUT/final_26890
run --manifest $FMAN --ckpt $N/snap_view/cur/run/train/ckpt/ckpt-000021-step-000022613 --name final_22613 --out $OUT/final_22613
run --manifest $FMAN --ckpt $N/snap_keep_step5075 --name final_5075 --out $OUT/final_5075
echo "=== $(date -u +%H:%M:%S) all done"
