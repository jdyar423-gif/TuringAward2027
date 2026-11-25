"""Fourth pass (REVIEW_2 items 8 and 33; Tables M4 and M11): exact scores of any runs on one split of the
proxy prompts, as lead2_work/exact_dev.py computes them for `dev` (bes.eval.exact_score on every reply:
maths answers, constraints, degenerate replies, stock refusals and deflections, words), optionally
without some prompt ids (Table M11: the 11 test prompts seen by mid-run samples).

  exact_split.py --split test --run ours=<run id> --run falcon=<run id> ... [--exclude ID,ID,...]
                 [--ids-from ours] --out OUT.json
  exact_split.py --count-only --split test        # prompts, turns, maths and constrained prompts of the split

Run ids are directories/files under study/proxy/outputs (`<model key>/<preset>-<hash>-s0`); ours on `test`
is the x of the R1 report. By default every prompt of the split is scored and a run missing one is an
error; --ids-from NAME restricts to the prompts that run answered (as exact_dev.py does for its first 60).
The repository is imported read-only from /home/user/turingaward2027.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path("/home/user/turingaward2027")
sys.path.insert(0, str(REPO))
N = Path("/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad")
ROOT = N / "study/proxy"

from bes.eval import exact_score as ES  # noqa: E402
from bes.eval.proxy import prompts as prompt_set  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True, choices=("dev", "test"))
ap.add_argument("--run", action="append", default=[], help="NAME=RUN_ID")
ap.add_argument("--exclude", default="", help="comma-separated prompt ids to leave out")
ap.add_argument("--ids-from", default=None, help="score only the prompts this run answered")
ap.add_argument("--count-only", action="store_true")
ap.add_argument("--out", default=None)
args = ap.parse_args()

prompts = {p.id: p for p in prompt_set.load()}
split_ids = sorted(i for i, p in prompts.items() if p.split == args.split)
excl = {x for x in args.exclude.split(",") if x}
if args.count_only:
    ids = [i for i in split_ids if i not in excl]
    c = {"split": args.split, "prompts": len(ids), "turns": sum(len(prompts[i].turns) for i in ids),
         "two_turn": sum(len(prompts[i].turns) == 2 for i in ids),
         "by_category": dict(Counter(prompts[i].category for i in ids)),
         "constrained_prompts": sum(bool(prompts[i].constraints) for i in ids),
         "maths_prompts": sum(prompts[i].category == "math" for i in ids), "excluded": sorted(excl)}
    print(json.dumps(c))
    if args.out:
        Path(args.out).write_text(json.dumps(c, indent=1))
    sys.exit(0)

from bes.eval.proxy import runner  # noqa: E402

runs = dict(r.split("=", 1) for r in args.run)
recs = {k: runner.load_outputs(ROOT, v)[0] for k, v in runs.items()}
ids = [i for i in split_ids if i not in excl]
if args.ids_from:
    ids = [i for i in ids if i in recs[args.ids_from]]
for k, r in recs.items():
    missing = [i for i in ids if i not in r]
    if missing:
        raise SystemExit(f"{k}: {len(missing)} of {len(ids)} prompts missing")


def stock(reply):
    return {"refusal": bool(ES.canned_refusal(reply)),
            "deflection": bool(ES.stock_deflection(reply)) if hasattr(ES, "stock_deflection") else None}


out = {"split": args.split, "prompts": ids, "excluded": sorted(excl), "runs": runs, "summary": {}}
for k, r in recs.items():
    scores, st = [], Counter()
    for i in ids:
        rec = r[i]
        replies = [t["reply"] for t in rec["turns"]]
        fins = [t.get("finish_reason") for t in rec["turns"]]
        scores.append(ES.score_conversation(prompts[i], replies, fins))
        for rep in replies:
            s = stock(rep)
            st["turns"] += 1
            st["refusal"] += bool(s["refusal"])
            st["deflection"] += bool(s["deflection"])
    summ = ES.summarize(scores)
    out["summary"][k] = {"math": summ["math"], "constraints": {kk: summ["constraints"][kk] for kk in
                                                                ("n_prompts", "prompts_all_ok", "n_constraints", "constraints_ok")},
                         "degenerate": summ["degenerate"], "rep4_mean": summ["rep4_mean"], "words": summ["words"],
                         "stock": dict(st), "n_prompts": summ["n_prompts"], "n_turns": summ["n_turns"]}
    s = out["summary"][k]
    print(f"{k:24s} maths {s['math']['correct']} of {s['math']['n']}; constraints all ok {s['constraints']['prompts_all_ok']} "
          f"of {s['constraints']['n_prompts']}; degenerate {s['degenerate']['degenerate']} (loop {s['degenerate']['loop']}, "
          f"truncated {s['degenerate']['truncated']}, empty {s['degenerate']['empty']}); stock refusal {st['refusal']}, "
          f"deflection {st['deflection']} of {st['turns']} turns; words median {s['words']['median']}")
if args.out:
    Path(args.out).write_text(json.dumps(out, indent=1, default=str))
