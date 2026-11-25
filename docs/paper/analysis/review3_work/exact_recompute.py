"""Exact scores on the test split recomputed from the run outputs (read-only; nothing written under the proxy root)."""
import json
import statistics
import sys
from pathlib import Path

from bes.eval import exact_score
from bes.eval.proxy import prompts as prompt_set, runner

ROOT = Path(sys.argv[1])
RUNS = {
    "ours": "ours-export-step51812-47fd99c151a50a58/final-prefopt-9a89782942-s0",
    "ours_k1": "ours-export-step51812-47fd99c151a50a58/final-k1-f93654208f-s0",
    "trained": "ours-export-step51812-ef5599caa3974edc/final-trained-9a89782942-s0",
    "falcon": "falcon-e60034381f4a/recommended-5ccec0259b-s0",
    "falcon_k6lp": "falcon-e60034381f4a/recommended-2dcb1c92a6-s0",
    "falcon_k6ds": "falcon-e60034381f4a/recommended-d1f2a63f82-s0",
    "mob_rec": "mobilellm-f680f0533368-qwen3/recommended-d9f149b0d3-s0",
    "mob_neu": "mobilellm-f680f0533368-qwen3/neutral-48fefc857c-s0",
}
SEEN = {"chat-002", "chat-017", "math-034", "instruct-034", "instruct-062", "writing-100", "code-026",
        "math-001", "code-002", "writing-001", "instruct-001"}
prompts = {p.id: p for p in prompt_set.load()}
out = {}
for name, rid in RUNS.items():
    records, _ = runner.load_outputs(ROOT, rid)
    sc = [exact_score.score_conversation(prompts[pid], [t["reply"] for t in r["turns"]],
                                         [t.get("finish_reason") for t in r["turns"]])
          for pid, r in records.items() if pid in prompts and prompts[pid].split == "test"]
    turns = [t for s in sc for t in s["turns"]]
    deg = [t["degenerate"] for t in turns]
    words = [d["words"] for d in deg]
    res = {"prompts": len(sc), "turns": len(turns),
           "math": sum(bool(s["math_correct"]) for s in sc if s["math_correct"] is not None),
           "math_n": sum(1 for s in sc if s["math_correct"] is not None),
           "cons": sum(bool(s["constraints_ok"]) for s in sc if s["constraints_ok"] is not None),
           "cons_n": sum(1 for s in sc if s["constraints_ok"] is not None),
           "single": sum(s["n_constraints_ok"] for s in sc), "single_n": sum(s["n_constraints"] for s in sc),
           "deg": sum(bool(d["degenerate"]) for d in deg), "trunc": sum(bool(d["truncated"]) for d in deg),
           "loop": sum(bool(d["loop"]) for d in deg), "replines": sum(bool(d["repeated_lines"]) for d in deg),
           "empty": sum(bool(d["empty"]) for d in deg), "fence": sum(bool(d["unclosed_fence"]) for d in deg),
           "role": sum(bool(d["role_leak"]) for d in deg),
           "refusal": sum(bool(d.get("refusal")) for d in deg), "deflection": sum(bool(d.get("deflection")) for d in deg),
           "prompts_with_deg": sum(1 for s in sc if any(t["degenerate"]["degenerate"] for t in s["turns"])),
           "words_median_true": statistics.median(words), "words_upper_median": sorted(words)[len(words) // 2],
           "words_mean": sum(words) / len(words), "rep4_mean_full": sum(d["rep4"] for d in deg) / len(deg)}
    u = [s for s in sc if s["prompt_id"] not in SEEN]
    res["u189"] = {"prompts": len(u), "turns": sum(len(s["turns"]) for s in u),
                   "math": sum(bool(s["math_correct"]) for s in u if s["math_correct"] is not None),
                   "math_n": sum(1 for s in u if s["math_correct"] is not None),
                   "cons": sum(bool(s["constraints_ok"]) for s in u if s["constraints_ok"] is not None),
                   "cons_n": sum(1 for s in u if s["constraints_ok"] is not None),
                   "deg": sum(bool(t["degenerate"]["degenerate"]) for s in u for t in s["turns"]),
                   "trunc": sum(bool(t["degenerate"]["truncated"]) for s in u for t in s["turns"])}
    res["mean_words_per_prompt_reply"] = None
    out[name] = res
    print(name, json.dumps(res))
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
