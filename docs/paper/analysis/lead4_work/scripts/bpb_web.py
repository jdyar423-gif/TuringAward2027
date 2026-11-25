"""Fourth pass (REVIEW_2 item 8, Table M6): our bits per UTF-8 byte on the 64 held-out web rows, from any
npz that lead3_work/heldout_eval.py writes (keys web/ce, web/w, web/tokens), as lead3_work/heldout46.py
computes it (byte lengths of the predicted tokens from bes.tokenizer, positions with weight > 0).
Usage: bpb_web.py NPZ [NPZ ...]; prints one line per file and the byte count. The repository is
imported read-only from /home/user/turingaward2027.
"""
import json
import math
import sys

import numpy as np

sys.path.insert(0, "/home/user/turingaward2027")
from bes.tokenizer import load as load_tok  # noqa: E402

Lb = load_tok().token_byte_lengths()
out = {}
for p in sys.argv[1:]:
    z = np.load(p)
    y = z["web/tokens"][:, 1:]
    w = z["web/w"].astype(np.float64)
    nbytes = float((Lb[y] * (w > 0)).sum())
    bpb = float((z["web/ce"].astype(np.float64) * w).sum()) / math.log(2) / nbytes
    out[p] = {"bpb_web": round(bpb, 4), "bytes": nbytes}
    print(f"{p}: {bpb:.4f} bits per byte over {nbytes:.0f} bytes")
print(json.dumps(out))
