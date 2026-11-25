"""Print the replies of one or more pairs (for checking labels when the notes are unclear).

    python show.py falcon 3 17 [--cut 1200]      # pair-file indices
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
args = sys.argv[1:]
cut = 1200
if "--cut" in args:
    i = args.index("--cut")
    cut = int(args[i + 1])
    del args[i:i + 2]
opp, idx = args[0], [int(x) for x in args[1:]]
rows = json.loads((HERE / "records.json").read_text())[opp]["rows"]
for k in idx:
    r = rows[k]
    print(f"===== #{k:03d} {r['prompt_id']} [{r['category']}] first={r['outcome_first_round']}")
    for t, (u, o, th) in enumerate(zip(r["user_turns"], r["ours_replies"], r["theirs_replies"]), 1):
        print(f"--- turn {t} USER: {u[:400]}")
        print(f"--- OURS ({len(o)} chars):\n{o[:cut]}")
        print(f"--- THEIRS ({len(th)} chars):\n{th[:cut]}")
