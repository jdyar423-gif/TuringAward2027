#!/bin/bash
# extract the newest v9 train log and run log from the current snapshot (read-only on the snapshot)
W=/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/paper/lead2_work
PY=/tmp/claude-0/-home-user/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/venv/bin/python
OUT=$W/v9_snap_latest
cd $W && $PY -I -c "
import sys, os, json; sys.path.insert(0,'.')
import snaptar
mp,buf,tf=snaptar.open_snapshot('/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/snap_main')
os.makedirs('$OUT/run/train',exist_ok=True)
for name in ['run/train/train_log.jsonl','run/train/ledger.json','run.log']:
    open('$OUT/'+name,'wb').write(snaptar.read_member_bytes(tf,name))
tf.close(); mp.close()
print('ok')
"
