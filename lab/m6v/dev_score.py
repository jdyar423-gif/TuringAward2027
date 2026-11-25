"""Scores of judged pairs (``pairjudge.py`` records: verdict A = ours, B = the opponent, tie).

    python lab/m6v/dev_score.py OUT_DIR RECORDS.jsonl [RECORDS.jsonl ...] [--json FILE]

OUT_DIR holds ``rows.json`` of ``dev_pairs.py``. Prints wins / losses / ties, the score
``(W + T / 2) / N`` with a Wilson-style interval from the paired sign statistics (bootstrap over
pairs, 4,000 resamples, seed 0), the decisive win rate, the sign-test p-value, and the same by
category, by the number of turns and by the library tier of the first reply."""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def sign_p(w: int, l: int) -> float:
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def line(name: str, vs: list[str], rng=None) -> dict:
    w, l, t = vs.count("A"), vs.count("B"), len(vs) - vs.count("A") - vs.count("B")
    n = len(vs)
    x = np.array([1.0 if v == "A" else 0.0 if v == "B" else 0.5 for v in vs])
    lo = hi = float("nan")
    if n and rng is not None:
        bs = x[rng.integers(0, n, (4000, n))].mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"group": name, "n": n, "win": w, "loss": l, "tie": t, "score": float(x.mean()) if n else None,
            "ci": [float(lo), float(hi)], "decisive": w / (w + l) if w + l else None, "p": sign_p(w, l)}


def show(r: dict) -> str:
    if not r["n"]:
        return f"{r['group']:28s} -"
    ci = "" if math.isnan(r["ci"][0]) else f" [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]"
    dec = "-" if r["decisive"] is None else f"{r['decisive']:.3f}"
    return (f"{r['group']:28s} {r['win']:3d} / {r['loss']:3d} / {r['tie']:3d}   score {r['score']:.3f}{ci}   "
            f"decisive {dec}   p {r['p']:.4f}   n {r['n']}")


def main(argv: list[str]) -> int:
    out_json = None
    if "--json" in argv:
        i = argv.index("--json")
        out_json = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    rows = {r["pair_id"]: r for r in json.loads((Path(argv[0]) / "rows.json").read_text())}
    recs = {}
    for f in argv[1:]:
        for ln in Path(f).read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                recs[r["pair_id"]] = r
    ids = [p for p in rows if p in recs]
    rng = np.random.default_rng(0)
    res = [line("all", [recs[p]["verdict"] for p in ids], rng)]
    groups = defaultdict(list)
    for p in ids:
        r = rows[p]
        groups[f"category {r['category']}"].append(recs[p]["verdict"])
        groups[f"turns {r['turns']}"].append(recs[p]["verdict"])
        groups[f"tier {(r.get('tiers') or ['none'])[0]}"].append(recs[p]["verdict"])
    res += [line(k, v, rng) for k, v in sorted(groups.items())]
    agree = float(np.mean([bool(recs[p].get("agree")) for p in ids])) if ids else float("nan")
    pos = float(np.mean([recs[p].get("first_position_pref", 0.0) for p in ids])) if ids else float("nan")
    print(f"{len(ids)} of {len(rows)} pairs judged; the two orders agree on {agree:.3f}; first-position preference {pos:+.3f}")
    for r in res:
        print(show(r))
    if out_json:
        Path(out_json).write_text(json.dumps({"judged": len(ids), "pairs": len(rows), "agree": agree,
                                              "first_position_pref": pos, "groups": res}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
