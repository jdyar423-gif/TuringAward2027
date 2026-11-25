"""Fifth pass: lead3_work/heldout_eval.py unchanged, run on one of the two checkpoints of the last
snapshot N/snap_final: CKPT_KIND=train (default; train/ckpt/ckpt-000022-step-000051812, the weights at
the end of training) or CKPT_KIND=post (post/ckpt/ckpt-000001-step-000051812, the weights after
preference optimisation, which the export shipped). The only change is which arrays.npz member
snaptar.find_checkpoint picks (the original takes the newest train/ckpt member).

    CKPT_KIND=post taskset -c 2,3 nice -n 5 S/venv/bin/python -I -B heldout_eval5.py \
        --manifest lead5_work/snap_final_x/run/manifest.json --ckpt N/snap_final --name NAME \
        --out lead5_work/heldout/NAME [--variant on: --variant pagesoff:page=0,gain=0]
"""
import json
import os
import sys
from pathlib import Path

LEAD3 = Path(__file__).resolve().parents[2] / "lead3_work"
sys.path.insert(0, str(LEAD3))
import snaptar  # noqa: E402  (lead3_work/snaptar.py, the module heldout_eval imports)

KIND = os.environ.get("CKPT_KIND", "train")
assert KIND in ("train", "post"), KIND


def find_checkpoint(tf):
    arrays = [m for m in tf.getmembers() if m.name.endswith("/arrays.npz") and f"/{KIND}/ckpt/" in m.name]
    if not arrays:
        raise FileNotFoundError(f"no {KIND}/ckpt/*/arrays.npz in the snapshot")
    arrays.sort(key=lambda m: m.name)
    am = arrays[-1]
    mm = tf.getmember(am.name.rsplit("/", 1)[0] + "/manifest.json")
    print(f"checkpoint member: {am.name}", flush=True)
    return json.loads(tf.extractfile(mm).read()), am


snaptar.find_checkpoint = find_checkpoint
import heldout_eval  # noqa: E402

heldout_eval.main()
