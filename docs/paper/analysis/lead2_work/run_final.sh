#!/bin/bash
N=/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad
W=$N/paper/lead2_work
PY=/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/venv/bin/python
OPP=$(pgrep -f "opp_bpb.py" | head -1)
[ -n "$OPP" ] && kill -STOP $OPP && echo "paused opp_bpb $OPP"
echo "=== $(date -u +%H:%M:%S) final_29722"
taskset -c 2,3 nice -n 10 $PY -I $W/heldout_eval.py --manifest $N/paper/lead_work/final_snap/run/manifest.json --ckpt $N/snap_final_main16 --name final_29722 --out $W/heldout/final_29722 2>&1 | grep --line-buffered -v "^WARNING\|^I0000\|^W0000\|^E0000"
echo "=== $(date -u +%H:%M:%S) graft_grad"
taskset -c 2,3 nice -n 10 $PY -I $W/graft_grad.py 2>&1 | grep --line-buffered -v "^WARNING\|^I0000\|^W0000\|^E0000"
[ -n "$OPP" ] && kill -CONT $OPP && echo "resumed opp_bpb $OPP"
echo "=== $(date -u +%H:%M:%S) final done"
