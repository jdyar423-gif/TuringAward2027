"""Fifth pass: the exact scores on the test split as counts, from lead5_work/exact_test.json and
exact_test189.json (written by lead4_work/scripts/exact_split.py from the replies), checked field by field
against the run's own exact reports (study/proxy/reports/exact/<run>-test.json, written by
`python -m bes.eval.proxy exact RUN --split test`). The neutral arm of MobileLLM-R1.5 has no test report
there; its values come from exact_split.py only. Writes exact_counts.json.

    taskset -c 2,3 nice -n 5 S/venv/bin/python -I -B exact_check.py
"""
import json
from pathlib import Path

N = Path("/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad")
HERE = Path(__file__).resolve().parent
W = HERE.parent
REP = N / "study/proxy/reports/exact"
out = {}
for tag, fname in (("200", "exact_test.json"), ("189", "exact_test189.json")):
    d = json.loads((W / fname).read_text())
    out[tag] = {}
    for k, s in d["summary"].items():
        m, c, g = s["math"], s["constraints"], s["degenerate"]
        nt = s["n_turns"]
        row = {"run": d["runs"][k], "prompts": s["n_prompts"], "turns": nt,
               "maths_correct": round(m["correct"] * m["n"]), "maths_n": m["n"], "maths_share": m["correct"],
               "constraints_all_ok": round(c["prompts_all_ok"] * c["n_prompts"]), "constrained_n": c["n_prompts"],
               "constraints_share": c["prompts_all_ok"],
               "single_ok": round(c["constraints_ok"] * c["n_constraints"]), "single_n": c["n_constraints"],
               "single_share": c["constraints_ok"],
               "degenerate": round(g["degenerate"] * nt), "degenerate_share": g["degenerate"],
               "truncated": round(g["truncated"] * nt), "loop": round(g["loop"] * nt),
               "repeated_lines": round(g["repeated_lines"] * nt), "empty": round(g["empty"] * nt),
               "unclosed_fence": round(g.get("unclosed_fence", 0) * nt), "role_leak": round(g.get("role_leak", 0) * nt),
               "degenerate_detail": g, "stock": s["stock"], "words": s["words"], "rep4_mean": s["rep4_mean"]}
        if tag == "200":
            rp = REP / (d["runs"][k].replace("/", "__") + "-test.json")
            if rp.exists():
                rs = json.loads(rp.read_text())["summary"]
                row["report"] = rp.name
                row["matches_report"] = {
                    "math": rs["math"]["correct"] == m["correct"] and rs["math"]["n"] == m["n"],
                    "constraints": all(rs["constraints"][kk] == c[kk] for kk in
                                       ("n_prompts", "prompts_all_ok", "n_constraints", "constraints_ok")),
                    "degenerate": rs["degenerate"] == g, "words": rs["words"] == s["words"]}
            else:
                row["report"] = None
        out[tag][k] = row
(W / "exact_counts.json").write_text(json.dumps(out, indent=1))
for tag, rows in out.items():
    print("== prompts", tag)
    for k, r in rows.items():
        print(f"{k:22s} maths {r['maths_correct']}/{r['maths_n']} ({r['maths_share']}); all constraints "
              f"{r['constraints_all_ok']}/{r['constrained_n']} ({r['constraints_share']}); single {r['single_ok']}/{r['single_n']} "
              f"({r['single_share']}); degenerate {r['degenerate']}/{r['turns']} ({r['degenerate_share']}): truncated "
              f"{r['truncated']}, loop {r['loop']}, repeated lines {r['repeated_lines']}, empty {r['empty']}, fence "
              f"{r['unclosed_fence']}, role {r['role_leak']}; words mean {r['words'].get('mean')} median {r['words'].get('median')}; "
              f"stock {r['stock']}; report {r.get('report')} {r.get('matches_report')}")
