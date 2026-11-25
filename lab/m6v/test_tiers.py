"""A registered pair set's score by the way each reply of ours was made.

    python lab/m6v/test_tiers.py REPORT.json REPLIES.jsonl [--json OUT]

REPORT: ``reports/ab/<set>--<source>.json`` of ``python -m bes.eval.proxy score`` (``per_prompt_score``:
1 a win of ours, 0 a loss, 0.5 a tie). REPLIES: ``replies.jsonl`` of ``scripts/gen_test_replies.py``
for the same run. Groups: the library tier of the first reply (verbatim / exemplar / none), its match
tag when exemplars carry one, the category, the number of turns."""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def sign_p(w, l):
    n = w + l
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n)


def tier(fin):
    fin = fin or {}
    if fin.get("rule") == "verbatim":
        return "verbatim"
    return "exemplar" if fin.get("library") else "none"


def main(argv):
    out = None
    if "--json" in argv:
        i = argv.index("--json")
        out = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    rep = json.loads(Path(argv[0]).read_text())
    per = rep["per_prompt_score"]
    if isinstance(per, list):
        per = {r["prompt_id"]: r["score"] for r in per}
    replies = {}
    for ln in Path(argv[1]).read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            replies[r["id"]] = r
    groups = defaultdict(list)
    for pid, s in per.items():
        r = replies.get(pid)
        if r is None:
            groups["(no reply record)"].append(s)
            continue
        groups["all"].append(s)
        groups["tier " + tier(r["finish"][0])].append(s)
        groups["category " + r["category"]].append(s)
        groups[f"turns {len(r['turns'])}"].append(s)
    res = []
    for k in sorted(groups):
        v = groups[k]
        w, l = sum(x == 1 for x in v), sum(x == 0 for x in v)
        res.append({"group": k, "n": len(v), "win": w, "loss": l, "tie": len(v) - w - l, "score": sum(v) / len(v),
                    "decisive": w / (w + l) if w + l else None, "p": sign_p(w, l)})
        print(f"{k:22s} {w:3d} / {l:3d} / {len(v) - w - l:3d}   score {sum(v) / len(v):.3f}   p {sign_p(w, l):.4f}   n {len(v)}")
    if out:
        Path(out).write_text(json.dumps({"report": argv[0], "overall": rep.get("overall"), "groups": res}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
