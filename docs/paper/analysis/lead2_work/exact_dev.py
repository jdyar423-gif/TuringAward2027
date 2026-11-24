"""Review item 34: exact scores (maths answers, constraints, degenerate replies, stock refusals and
deflections) of the replies that exist on the first 60 dev prompts: ours (final's export of step
25,280, default decoding and the five other settings of the decode study), Falcon-H1-Tiny
(recommended preset), MobileLLM-R1.5 (recommended and neutral arms). Same 60 prompts for every
run. Uses bes.eval.exact_score as the protocol does; writes only exact_dev.json here."""
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
from bes.eval.proxy import runner  # noqa: E402

prompts = {p.id: p for p in prompt_set.load()}
runs = {
    "ours_default": "ours-export-step25280-2efdca545a4e2f2c/dev-s1-default-9a89782942-s0",
    "ours_k1": "ours-export-step25280-2efdca545a4e2f2c/dev-s1-k1-f93654208f-s0",
    "ours_ngram3": "ours-export-step25280-2efdca545a4e2f2c/dev-s1-ngram3-908483c4c4-s0",
    "ours_t04": "ours-export-step25280-2efdca545a4e2f2c/dev-s1-t04-eb17090d64-s0",
    "ours_t08": "ours-export-step25280-2efdca545a4e2f2c/dev-s1-t08-abbdeb7de0-s0",
    "ours_short": "ours-export-step25280-2efdca545a4e2f2c/dev-s1-short-37e897ff55-s0",
    "v9_31912_default": "ours-export-step31912-5e12a59158a3b192/dev-s2-default-9a89782942-s0",
    "v9_31912_ngram3": "ours-export-step31912-5e12a59158a3b192/dev-s2-ngram3-908483c4c4-s0",
    "falcon": "falcon-e60034381f4a/recommended-5ccec0259b-s0",
    "mobilellm_recommended": "mobilellm-f680f0533368-qwen3/recommended-d9f149b0d3-s0",
    "mobilellm_neutral": "mobilellm-f680f0533368-qwen3/neutral-48fefc857c-s0",
}
recs = {k: runner.load_outputs(ROOT, v)[0] for k, v in runs.items()}
ids = sorted(set(recs["ours_default"]))
ids = [i for i in ids if i in prompts and prompts[i].split == "dev"]
for k, r in recs.items():
    missing = [i for i in ids if i not in r]
    if missing:
        print(k, "missing", len(missing))
print("prompts", len(ids), Counter(prompts[i].category for i in ids))


def stock(reply):
    return {"refusal": bool(ES.canned_refusal(reply)),
            "deflection": bool(ES.stock_deflection(reply)) if hasattr(ES, "stock_deflection") else None}


out = {"prompts": ids, "runs": runs, "summary": {}}
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
(Path(__file__).resolve().parent / "exact_dev.json").write_text(json.dumps(out, indent=1, default=str))
