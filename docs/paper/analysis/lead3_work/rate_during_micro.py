"""The live run's logged rate at 2,048 tokens before and during the micro graft experiment
(14:12:51-16:41:01 UTC on 24 November), from the v9 train log of the 16:28 snapshot."""
import json, statistics as st, datetime as dt
from pathlib import Path
HERE = Path(__file__).resolve().parent
T0 = 1795455991.9461858
S = [d for d in map(json.loads, open(HERE / "v9_snap_1628/run/train/train_log.jsonl"))
     if d.get("event") == "step" and d.get("seq_len") == 2048]
ts = lambda h, m, s: dt.datetime(2026, 11, 24, h, m, s, tzinfo=dt.timezone.utc).timestamp() - T0
a, b = ts(14, 12, 51), ts(16, 41, 1)
before = [d["tok_per_s"] for d in S if d["time"] < a]
during = [d["tok_per_s"] for d in S if a <= d["time"] <= b]
out = {"before": {"median": round(st.median(before)), "n": len(before)},
       "during": {"median": round(st.median(during)), "n": len(during), "last_record_utc": "16:27:21 (cut-off)"}}
print(json.dumps(out)); (HERE / "rate_during_micro.json").write_text(json.dumps(out, indent=1))
