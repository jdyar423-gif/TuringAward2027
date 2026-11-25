#!/bin/bash
W=/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/paper/lead2_work
PY=/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/venv/bin/python
until grep -q "all done" $W/run_heldout.log; do sleep 20; done
echo "=== $(date -u +%H:%M:%S) graft_grad"
taskset -c 2,3 nice -n 10 $PY -I $W/graft_grad.py 2>&1 | grep --line-buffered -v "^WARNING\|^I0000\|^W0000\|^E0000"
echo "=== $(date -u +%H:%M:%S) opp_bpb"
cd $W && taskset -c 2,3 nice -n 10 $PY -I $W/opp_bpb.py $W/heldout/v9_newest_on.npz $W/opp_bpb_web.json 2>&1 | tail -5
echo "=== $(date -u +%H:%M:%S) after done"
