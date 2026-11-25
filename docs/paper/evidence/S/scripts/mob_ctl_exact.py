"""Exact scores of MobileLLM-R1.5's arms on the test split, recounted from the per-turn records of
the exact reports (N/study/proxy/reports/exact/<run>-test.json, written by `python -m bes.eval.proxy
exact`): the released preset (`recommended-d9f149b0d3`, as a check against Tables 8 and M4) and the
two K = 6 control arms (`recommended-32f0c6d864`: kept by its own log-probability;
`recommended-aca173fa2c`: kept by our selector), on all 200 prompts and on the 189 no mid-run
sample touched.
    python mob_ctl_exact.py   -> N/peek/mob_ctl_exact.json and a printed summary"""
import json

N = "/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad"
SEEN = set("chat-002,chat-017,math-034,instruct-034,instruct-062,writing-100,code-026,math-001,code-002,writing-001,instruct-001".split(","))
ARMS = (("k6lp", "32f0c6d864"), ("k6ds", "aca173fa2c"), ("released", "d9f149b0d3"))
KINDS = ("degenerate", "empty", "truncated", "loop", "repeated_lines", "unclosed_fence", "role_leak", "refusal", "deflection")
# the published row of the released preset (Tables 8 and M4), which this recount must reproduce
PUBLISHED = {"degenerate": 158, "truncated": 150, "loop": 14, "repeated_lines": 54, "empty": 2, "unclosed_fence": 7,
             "role_leak": 0, "math_correct": 12, "con_all_ok": 8, "single_ok": 10, "words_upper_median": 395,
             "prompts_with_degenerate": 139}


def stats(pp):
    turns = [t["degenerate"] for p in pp for t in p["turns"]]
    n = len(turns)
    r = {"turns": n, "prompts": len(pp)}
    for k in KINDS:
        r[k] = sum(bool(t[k]) for t in turns)
    r["degenerate_and_truncated"] = sum(bool(t["degenerate"]) and bool(t["truncated"]) for t in turns)
    r["rep4_mean"] = sum(t["rep4"] for t in turns) / n
    w = sorted(t["words"] for t in turns)
    r["words_mean"] = sum(w) / n
    r["words_upper_median"] = w[n // 2]
    r["prompts_with_degenerate"] = sum(any(t["degenerate"]["degenerate"] for t in p["turns"]) for p in pp)
    m = [p for p in pp if p["math_correct"] is not None]
    r["math_n"], r["math_correct"] = len(m), sum(bool(p["math_correct"]) for p in m)
    c = [p for p in pp if p["n_constraints"] > 0]
    r["con_n"], r["con_all_ok"] = len(c), sum(bool(p["constraints_ok"]) for p in c)
    r["single_n"], r["single_ok"] = sum(p["n_constraints"] for p in c), sum(p["n_constraints_ok"] for p in c)
    return r


out = {}
for name, tag in ARMS:
    rep = json.load(open(f"{N}/study/proxy/reports/exact/mobilellm-f680f0533368-qwen3__recommended-{tag}-s0-test.json"))
    pp = rep["per_prompt"]
    out[name] = {"run": rep["run_id"], "all": stats(pp), "189": stats([p for p in pp if p["prompt_id"] not in SEEN])}
    a, s = out[name]["all"], rep["summary"]
    # the recount agrees with the report's own summary (rates rounded to four decimals there)
    assert abs(a["degenerate"] / a["turns"] - s["degenerate"]["degenerate"]) < 6e-5
    assert abs(a["rep4_mean"] - s["rep4_mean"]) < 6e-5 and abs(a["words_mean"] - s["words"]["mean"]) < 0.06
    assert a["words_upper_median"] == s["words"]["median"]
    assert abs(a["math_correct"] / a["math_n"] - s["math"]["correct"]) < 6e-5
    assert abs(a["con_all_ok"] / a["con_n"] - s["constraints"]["prompts_all_ok"]) < 6e-5
    assert abs(a["single_ok"] / a["single_n"] - s["constraints"]["constraints_ok"]) < 6e-5
for k, v in PUBLISHED.items():
    assert out["released"]["all"][k] == v, (k, out["released"]["all"][k], v)
json.dump(out, open(f"{N}/peek/mob_ctl_exact.json", "w"), indent=1)
for name, _ in ARMS:
    for sub in ("all", "189"):
        r = out[name][sub]
        print(f"{name:9s} {sub:3s} | maths {r['math_correct']}/{r['math_n']} | all constraints {r['con_all_ok']}/{r['con_n']} | single {r['single_ok']}/{r['single_n']}"
              f" | degenerate {r['degenerate']}/{r['turns']} (cut {r['truncated']}, loop {r['loop']}, lines {r['repeated_lines']}, empty {r['empty']}, fence {r['unclosed_fence']}, role {r['role_leak']})"
              f" | prompts with one {r['prompts_with_degenerate']} | words {r['words_upper_median']}; {r['words_mean']:.1f} | rep4 {r['rep4_mean']:.4f}")
