#!/bin/bash
# Second queue: the v9 snapshot rotated at about 12:24 (snap_main is now step 35,294, snap_main.prev
# step 34,166, already evaluated), so the first queue's 'v9_prev' entry would repeat 34,166. Wait for the
# first queue's last v9 variant, stop it, then evaluate the newest v9 snapshot and the final checkpoints.
N=/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad
W=$N/paper/lead2_work
PY=/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/venv/bin/python
OUT=$W/heldout
V9MAN=$N/paper/data/_src/live/run/manifest.json
FMAN=$N/paper/lead_work/final_snap/run/manifest.json
until [ -f $OUT/v9_newest_gainoff.npz ]; do sleep 10; done
pkill -f run_heldout.sh; sleep 1; pkill -f "heldout_eval.py --manifest $V9MAN --ckpt $N/snap_main.prev"; sleep 2
run() { echo "=== $(date -u +%H:%M:%S) $*"; taskset -c 2,3 nice -n 10 $PY -I $W/heldout_eval.py "$@" 2>&1 | grep --line-buffered -v "^WARNING\|^I0000\|^W0000\|^E0000"; }
run --manifest $V9MAN --ckpt $N/snap_main --name v9_35294 --out $OUT/v9_35294 --variant on: --variant pagesoff:page=0,gain=0
run --manifest $FMAN --ckpt $N/final_look/run/train/ckpt/ckpt-000026-step-000029722 --name final_29722 --out $OUT/final_29722
run --manifest $FMAN --ckpt $N/snap_look/run/train/ckpt/ckpt-000024-step-000026890 --name final_26890 --out $OUT/final_26890
run --manifest $FMAN --ckpt $N/snap_view/cur/run/train/ckpt/ckpt-000021-step-000022613 --name final_22613 --out $OUT/final_22613
run --manifest $FMAN --ckpt $N/snap_keep_step5075 --name final_5075 --out $OUT/final_5075
echo "=== $(date -u +%H:%M:%S) all done" | tee -a $W/run_heldout.log
