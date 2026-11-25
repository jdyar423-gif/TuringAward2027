"""Fifth pass: lead3_work/rate_during_micro.py on the final train log (lead5_work/snap_final_x), so that the
whole of the micro graft runs (14:12:51-16:41:01 UTC) and the restart arms (16:42:47-17:29:56) are covered:
the live run's logged rate at 2,048 tokens (phase long) before, during each, and after."""
import datetime as dt
import json
import statistics as st
from pathlib import Path
HERE = Path(__file__).resolve().parent
T0 = 1795455991.9461858
S = [d for d in map(json.loads, open(HERE.parent / "snap_final_x/run/train/train_log.jsonl"))
     if d.get("event") == "step" and d.get("seq_len") == 2048 and d.get("phase") == "long"]
ts = lambda h, m, s: dt.datetime(2026, 11, 24, h, m, s, tzinfo=dt.timezone.utc).timestamp() - T0  # noqa: E731
a, b, c, d_ = ts(14, 12, 51), ts(16, 41, 1), ts(16, 42, 47), ts(17, 29, 56)
win = {"before": [x for x in S if x["time"] < a], "during_6M_runs": [x for x in S if a <= x["time"] <= b],
       "during_restart_arms": [x for x in S if c <= x["time"] <= d_], "after": [x for x in S if x["time"] > d_]}
out = {k: {"median": round(st.median([x["tok_per_s"] for x in v])), "n": len(v)} for k, v in win.items() if v}
print(json.dumps(out))
(HERE.parent / "rate_during_micro5.json").write_text(json.dumps(out, indent=1))
