"""Fifth pass (Table 8, the trajectory on dev): exact scores of the run's own select-stage replies at the end
(prefopt weights of step 51,812; variants default, k1, noband on the 200 dev prompts;
N/records/unpacked/proxy/outputs/ours-step51812-c04f3f6b3fca71f1/), computed as lead4_work/scripts/exact_split.py
computes them, (a) on the first 60 dev prompts of Table 8 (the prompts the v9 31,912 run answered,
N/study/proxy/outputs/ours-export-step31912-5e12a59158a3b192/dev-s2-default-9a89782942-s0) and (b) on all 200,
checked against the run's own exact reports in N/records/unpacked/proxy/reports/exact/ (200 dev prompts).

    taskset -c 2,3 nice -n 5 S/venv/bin/python -I -B exact_dev_final.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path("/home/user/turingaward2027")
sys.path.insert(0, str(REPO))
N = Path("/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad")
HERE = Path(__file__).resolve().parent
from bes.eval import exact_score as ES  # noqa: E402
from bes.eval.proxy import prompts as prompt_set  # noqa: E402
from bes.eval.proxy import runner  # noqa: E402

REC = N / "records/unpacked/proxy"
ROOT = N / "study/proxy"
prompts = {p.id: p for p in prompt_set.load()}
runs = {k: runner.load_outputs(REC, f"ours-step51812-c04f3f6b3fca71f1/select-{k}-{h}-s0")[0]
        for k, h in (("default", "9a89782942"), ("k1", "f93654208f"), ("noband", "9680da6210"))}
s2 = runner.load_outputs(ROOT, "ours-export-step31912-5e12a59158a3b192/dev-s2-default-9a89782942-s0")[0]
first60 = sorted(i for i in s2 if prompts[i].split == "dev")
dev200 = sorted(i for i, p in prompts.items() if p.split == "dev")
out = {"first60_ids": len(first60), "sets": {}}
for setname, ids in (("first60", first60), ("dev200", dev200)):
    out["sets"][setname] = {}
    for k, r in runs.items():
        missing = [i for i in ids if i not in r]
        assert not missing, (k, setname, len(missing))
        scores, st = [], Counter()
        for i in ids:
            reps = [t["reply"] for t in r[i]["turns"]]
            fins = [t.get("finish_reason") for t in r[i]["turns"]]
            scores.append(ES.score_conversation(prompts[i], reps, fins))
            for rep in reps:
                st["turns"] += 1
                st["refusal"] += bool(ES.canned_refusal(rep))
                st["deflection"] += bool(ES.stock_deflection(rep))
        s = ES.summarize(scores)
        nt = s["n_turns"]
        row = {"maths": [round(s["math"]["correct"] * s["math"]["n"]), s["math"]["n"]],
               "constraints_all_ok": [round(s["constraints"]["prompts_all_ok"] * s["constraints"]["n_prompts"]),
                                      s["constraints"]["n_prompts"]],
               "cut_at_limit": [round(s["degenerate"]["truncated"] * nt), nt],
               "degenerate": [round(s["degenerate"]["degenerate"] * nt), nt],
               "words_median": s["words"]["median"], "words_mean": s["words"].get("mean"), "stock": dict(st)}
        if setname == "dev200":
            rp = REC / "reports/exact" / f"ours-step51812-c04f3f6b3fca71f1__select-{k}-" \
                 f"{ {'default': '9a89782942', 'k1': 'f93654208f', 'noband': '9680da6210'}[k]}-s0-dev.json"
            rs = json.loads(rp.read_text())["summary"]
            row["matches_report"] = (rs["math"]["correct"] == s["math"]["correct"]
                                     and rs["constraints"]["prompts_all_ok"] == s["constraints"]["prompts_all_ok"]
                                     and rs["degenerate"] == s["degenerate"])
        out["sets"][setname][k] = row
        print(setname, k, json.dumps(row))
(HERE.parent / "exact_dev_final.json").write_text(json.dumps(out, indent=1))
