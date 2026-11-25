import json, sys, statistics
from pathlib import Path
from bes.eval import exact_score
from bes.eval.proxy import prompts as prompt_set, runner
N = Path(sys.argv[1])
ids = json.load(open(N / "paper/lead2_work/exact_dev.json"))["prompts"]
prompts = {p.id: p for p in prompt_set.load()}
root = N / "records/unpacked/proxy"
for rid in ["ours-step51812-c04f3f6b3fca71f1/select-default-9a89782942-s0", "ours-step51812-c04f3f6b3fca71f1/select-k1-f93654208f-s0", "ours-step51812-c04f3f6b3fca71f1/select-noband-9680da6210-s0"]:
    try:
        recs, _ = runner.load_outputs(root, rid)
    except Exception as e:
        print(rid, "ERR", e); continue
    for sub, name in [(ids, "first60"), ([p for p in recs if prompts[p].split == "dev"], "all dev")]:
        sc = [exact_score.score_conversation(prompts[pid], [t["reply"] for t in recs[pid]["turns"]], [t.get("finish_reason") for t in recs[pid]["turns"]]) for pid in sub if pid in recs]
        turns = [t for s in sc for t in s["turns"]]
        print(rid.split("/")[1], name, len(sc), "math", sum(bool(s["math_correct"]) for s in sc if s["math_correct"] is not None), "of", sum(1 for s in sc if s["math_correct"] is not None),
              "cons", sum(bool(s["constraints_ok"]) for s in sc if s["constraints_ok"] is not None), "of", sum(1 for s in sc if s["constraints_ok"] is not None),
              "trunc", sum(t["degenerate"]["truncated"] for t in turns), "deg", sum(t["degenerate"]["degenerate"] for t in turns), "of", len(turns),
              "median words(upper)", sorted(t["degenerate"]["words"] for t in turns)[len(turns)//2])
