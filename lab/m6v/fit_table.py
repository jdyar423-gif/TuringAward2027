"""Table P9 of the paper from the saved records: the reader's test against the judge.

    python lab/m6v/fit_table.py POOL_ROWS.json VERDICTS.jsonl DEFAULT_DIR

POOL_ROWS: rows of ``lab/m6v/fit_pool.py --cut`` (fit, window, overlap per candidate); VERDICTS: the
judge's records of the stored answers (``pair_id``, ``verdict``: A = the stored answer); DEFAULT_DIR:
the judged default replies of the same model (``judge*.jsonl`` / ``lane_*.jsonl``), for the model's own
reply to the same prompt. Candidates the library's own rule takes are left out."""
import glob
import json
import sys
from pathlib import Path

import numpy as np

VAL = {"A": 1.0, "B": 0.0, "tie": 0.5}


def main(argv):
    rows = json.loads(Path(argv[0]).read_text())
    verd = {}
    for line in open(argv[1]):
        if line.strip():
            r = json.loads(line)
            verd.setdefault(r["pair_id"], r)
    own = {}
    for pat in ("judge*.jsonl", "lane_*.jsonl"):
        for f in sorted(glob.glob(str(Path(argv[2]) / pat))):
            for line in open(f):
                if line.strip():
                    r = json.loads(line)
                    own.setdefault(r["pair_id"], r)
    xs = [dict(r, v=VAL[verd[r["pair_id"]]["verdict"]], o=VAL[own[r["prompt_id"]]["verdict"]])
          for r in rows if not r["own_rule"] and r["pair_id"] in verd and r["fit"] is not None and r["prompt_id"] in own]
    print(f"{len(xs)} judged candidates of {len({x['prompt_id'] for x in xs})} prompts")

    def line(name, g):
        if g:
            print(f"| {name} | {len(g)} | {np.mean([x['v'] for x in g]):.3f} | {np.mean([x['o'] for x in g]):.3f} |")

    for lo, hi, name in ((-0.05, 9, "fit ≥ −0.05"), (-0.10, -0.05, "−0.10 ≤ fit < −0.05"), (-0.20, -0.10, "−0.20 ≤ fit < −0.10"),
                         (-9, -0.20, "fit < −0.20")):
        line(name, [x for x in xs if lo <= x["fit"] < hi])
    for lo, hi, name in ((-9, 6, "window ≤ 6 nats"), (6, 8, "6 < window ≤ 8"), (8, 99, "window > 8")):
        line(name, [x for x in xs if lo < x["window"] <= hi])
    for v, name in ((1.0, "won"), (0.5, "tied"), (0.0, "lost")):
        w = [x["window"] for x in xs if x["v"] == v]
        print(f"stored answer {name}: n {len(w)}, median window {np.median(w):.1f}, quartiles {np.percentile(w, 25):.1f}-{np.percentile(w, 75):.1f}")

    def auc(score, label):
        s, y = np.asarray(score, float), np.asarray(label, bool)
        d = s[y][:, None] - s[~y][None, :]
        return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)

    y = [x["v"] >= x["o"] for x in xs]
    print("AUC of 'the stored answer is at least as good as the model's own reply': "
          f"fit {auc([x['fit'] for x in xs], y):.3f}, -window {auc([-x['window'] for x in xs], y):.3f}, "
          f"overlap {auc([x['own'] for x in xs], y):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
