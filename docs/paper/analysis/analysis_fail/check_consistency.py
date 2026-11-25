"""Step 4: labels.json, numbers.json, tables.md and failure_modes.md agree with each other.

    taskset -c 0 $S/venv/bin/python -I $N/paper/analysis_fail/check_consistency.py

1. Recomputes from labels.json ALONE (outcomes and labels per prompt) the outcome counts and every
   label table of the report, and compares them with numbers.json.
2. Parses tables.md and compares each table with numbers.json.
3. Re-renders failure_modes.md from the template and numbers.json and compares it with the file.
4. The codebook embedded in labels.json equals codebook.json; amendment F covers exactly the MATH
   labels on right maths answers (numbers.json, computed from the exact scorer).
5. Regression checks for the review of 25 Nov (seven items): the corrected statements are in the
   report and the wrong ones are not.
6. The numbers that the paper's Appendix O quotes still hold (values copied from appendix_O.md,
   which this script does not open).
Prints "all consistency checks passed" or the first failure.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CATS = ("chat", "code", "instruct", "math", "writing")
ORDER = ["INCOH", "LOOP", "FACT", "MATH", "CODE", "INSTR", "TRUNC", "THIN", "WEAK", "OK"]


def fail(msg: str) -> None:
    print("CONSISTENCY CHECK FAILED:", msg)
    sys.exit(1)


L = json.loads((HERE / "labels.json").read_text())
n = json.loads((HERE / "numbers.json").read_text())
cb = json.loads((HERE / "codebook.json").read_text())
P = L["prompts"]
if len(P) != 200:
    fail(f"labels.json has {len(P)} prompts")


def table(rows, side):
    by = {c: Counter() for c in CATS}
    for e, v in rows:
        by[e["category"]][v[f"reason_{side}"]["code"]] += 1
    out = []
    for code in ORDER:
        vals = [by[c][code] for c in CATS]
        if sum(vals):
            out.append([code] + vals + [sum(vals)])
    tot = [sum(by[c].values()) for c in CATS]
    return out + [["total"] + tot + [sum(tot)]]


def sel(opp, pred):
    return [(e, e[f"vs_{opp}"]) for e in P if pred(e, e[f"vs_{opp}"])]


# 1. labels.json -> numbers.json
for opp, rnd, key in (("falcon", "outcome_first_round", "falcon_first_round"),
                      ("falcon", "outcome_neutral_round", "falcon_neutral_round"),
                      ("mobilellm", "outcome_first_round", "mobilellm_first_round"),
                      ("mobilellm", "outcome_neutral_round", "mobilellm_neutral_round")):
    cnt = Counter(e[f"vs_{opp}"][rnd] for e in P)
    if [cnt["win"], cnt["loss"], cnt["tie"]] != n[key]:
        fail(f"{key}: labels.json gives {[cnt['win'], cnt['loss'], cnt['tie']]}, numbers.json {n[key]}")
if n["falcon_first_round"] != [35, 108, 57] or n["mobilellm_first_round"] != [116, 15, 69]:
    fail("first-round totals are not 35/108/57 and 116/15/69")
bb = lambda e, v: v["readings"]["r1"]["verdict"] == v["readings"]["r2"]["verdict"] == "both_bad"   # noqa: E731
first = lambda o: (lambda e, v: v["outcome_first_round"] == o)                                     # noqa: E731
neutral = lambda o: (lambda e, v: v["outcome_neutral_round"] == o)                                 # noqa: E731
expect = {
    "loss_falcon_ours": table(sel("falcon", first("loss")), "ours"),
    "loss_falcon_theirs": table(sel("falcon", first("loss")), "theirs"),
    "loss_falcon_ours_neutral": table(sel("falcon", neutral("loss")), "ours"),
    "bb_falcon_ours": table(sel("falcon", bb), "ours"),
    "bb_falcon_theirs": table(sel("falcon", bb), "theirs"),
    "bb_mobilellm_ours": table(sel("mobilellm", bb), "ours"),
    "bb_mobilellm_theirs": table(sel("mobilellm", bb), "theirs"),
    "win_mobilellm_theirs": table(sel("mobilellm", first("win")), "theirs"),
    "win_mobilellm_ours": table(sel("mobilellm", first("win")), "ours"),
    "win_falcon_theirs": table(sel("falcon", first("win")), "theirs"),
    "win_falcon_ours": table(sel("falcon", first("win")), "ours"),
    "loss_mobilellm_ours": table(sel("mobilellm", first("loss")), "ours"),
    "loss_mobilellm_theirs": table(sel("mobilellm", first("loss")), "theirs"),
}
for name, t in expect.items():
    if n["tables"][name] != t:
        fail(f"table {name}: labels.json gives {t}, numbers.json {n['tables'][name]}")
    if t[-1][-1] != sum(r[-1] for r in t[:-1]):
        fail(f"table {name} does not sum to its n")
# the maths-loss split: labels.json labels against numbers.json's exact maths rows
right = {m["prompt_id"] for m in n["math_rows"] if m["ours"]}
for opp in ("falcon", "mobilellm"):
    ls = [(e, v) for e, v in sel(opp, first("loss")) if e["category"] == "math"]
    split = {"n": len(ls),
             "right_faulty_working": sum(1 for e, v in ls if e["prompt_id"] in right),
             "missing": sum(1 for e, v in ls if e["prompt_id"] not in right and v["reason_ours"]["code"] != "MATH")}
    split["wrong"] = split["n"] - split["right_faulty_working"] - split["missing"]
    for k, v in split.items():
        if n[f"math_losses_{opp}"][k] != v:
            fail(f"math_losses_{opp}[{k}]: labels.json gives {v}, numbers.json {n[f'math_losses_{opp}'][k]}")

# 2. tables.md -> numbers.json
md = (HERE / "tables.md").read_text()
for block in md.split("### ")[1:]:
    name, body = block.split("\n", 1)
    rows = [ln.strip("|").split("|") for ln in body.strip().splitlines()[2:]]
    parsed = [[r[0].strip()] + [int(x) for x in r[1:]] for r in rows]
    if parsed != n["tables"][name.strip()]:
        fail(f"tables.md {name.strip()} differs from numbers.json")
if set(re.findall(r"^### (\S+)", md, flags=re.M)) != set(n["tables"]):
    fail("tables.md and numbers.json hold different tables")

# 3. the report on disk is the rendering of the current template and numbers
sys.path.insert(0, str(HERE))
import build_report  # noqa: E402  (reads numbers.json, codebook.json, labels.json at import)
rendered = build_report.render((HERE / "failure_modes.template.md").read_text())
report = (HERE / "failure_modes.md").read_text()
if rendered != report:
    fail("failure_modes.md is not the rendering of the current template and numbers.json (rebuild it)")

# 4. codebook
if L["codebook"] != cb:
    fail("the codebook embedded in labels.json differs from codebook.json (rerun analyze.py)")
F = [a for a in cb["amendments_after_labelling"] if a["id"] == "F"]
if len(F) != 1 or sorted([c["set"], c["side"], c["prompt_id"]] for c in F[0]["covers"]) != sorted(n["math_label_on_right_answer"]):
    fail("amendment F does not list exactly the MATH labels on right maths answers")
if "wrong logic that gives a wrong result" in cb["labels"]["MATH"]["definition"]:
    fail("the MATH definition still excludes misstated working with a right answer")

# 5. regression checks for the review items
must = [
    ("item 1", "Maths (18): 16 are wrong (15) or missing (1) answers, and the other 2 are right answers with misstated working"),
    ("item 1", "maths losses wrong or missing answers in 16 of 18 (the other 2 are right answers with misstated working)"),
    ("item 1", "it now also covers misstated working when the final answer is right"),
    ("item 2", "MobileLLM's own reply was faulted in 7 of the 15 and OK in 8"),
    ("item 3", "degenerate on 82 of the 116 wins"),
    ("item 4", "Ours is OK in 16 of these 26 and in 24 of all 35 wins"),
    ("item 5", "only 19 notes had been read"),
    ("item 5", "A, B and C were added together after the first 20 Falcon pairs"),
    ("item 6", "1 numbered line where exactly 5 numbered steps were asked (instruct-070"),
    ("item 7", "Code and chat give the most wins (10 each of the 35)"),
    ("item 7", "(INCOH, LOOP, TRUNC: 43; no loss is TRUNC)"),
    ("item 7", "Grouping the check types include, ends_with, ends_with_question and postscript (9 checks), ours misses 6 and Falcon 0; counting starts_with as well"),
]
must_not = [
    ("item 1", "every loss is a wrong or missing answer"),
    ("item 1", "(18 of 18)"),
    ("item 2", "faulted in 5 of them"),
    ("item 3", "degenerate on 82 of these prompts"),
    ("item 4", "Ours is OK in 24 of them"),
    ("item 5", "only the 15 notes"),
    ("item 5", "(B) The preferred reply"),
    ("item 6", "22 steps for 5"),
    ("item 7", "Code is where we win most"),
    ("item 7", "(INCOH, LOOP: 43)"),
]
for item, s in must:
    if s not in report:
        fail(f"review {item}: expected text missing: {s!r}")
for item, s in must_not:
    if s in report:
        fail(f"review {item}: corrected text still present: {s!r}")

# 6. numbers quoted by the paper's Appendix O (copied from appendix_O.md of 24 Nov, 23:49)
c = lambda t, row, col="all": next((r[1 + CATS.index(col)] if col != "all" else r[-1])  # noqa: E731
                                    for r in n["tables"][t] if r[0] == row) if any(r[0] == row for r in n["tables"][t]) else 0
alt = "loss_falcon_ours_other_tiebreak"
quotes = [
    ("O.3 wrong content 46", n["loss_group::content (FACT, MATH, CODE)"], 46),
    ("O.3 INCOH/LOOP/TRUNC 43", n["loss_group::degenerate or off-task (INCOH, LOOP, TRUNC)"], 43),
    ("O.3 instructions/thin 19", n["loss_group::instruction, thin or weaker (INSTR, THIN, WEAK)"], 19),
    ("O.3 other tie-break 45/33/30", [c(alt, "FACT") + c(alt, "MATH") + c(alt, "CODE"),
                                       c(alt, "INCOH") + c(alt, "LOOP") + c(alt, "TRUNC"),
                                       c(alt, "INSTR") + c(alt, "THIN") + c(alt, "WEAK")], [45, 33, 30]),
    ("O.3 tie-broken labels 19", n["tiebreak_losses_falcon"], 19),
    ("O.3 writing 21 of 32", [c("loss_falcon_ours", "INCOH", "writing") + c("loss_falcon_ours", "LOOP", "writing"),
                               c("loss_falcon_ours", "total", "writing")], [21, 32]),
    ("O.3 chat FACT 14 of 24", [c("loss_falcon_ours", "FACT", "chat"), c("loss_falcon_ours", "total", "chat")], [14, 24]),
    ("O.3 instruct INCOH 9 of 21, INSTR 6", [c("loss_falcon_ours", "INCOH", "instruct"), c("loss_falcon_ours", "total", "instruct"),
                                              c("loss_falcon_ours", "INSTR", "instruct")], [9, 21, 6]),
    ("O.3 code 10 of 13", [c("loss_falcon_ours", "CODE", "code"), c("loss_falcon_ours", "total", "code")], [10, 13]),
    ("O.3 maths MATH 17 of 18, two right answers", [c("loss_falcon_ours", "MATH", "math"), c("loss_falcon_ours", "total", "math"),
                                                     n["math_losses_falcon"]["right_faulty_working"]], [17, 18, 2]),
    ("O.3 Falcon OK in 75", c("loss_falcon_theirs", "OK"), 75),
    ("O.3 neutral 109 losses, 106 shared", [n["n_loss_falcon_neutral"], n["loss_both_rounds"]], [109, 106]),
    ("O.4 both bad 36, same label 22", [n["n_bb_falcon_first"], n["bb_falcon_same_label"]], [36, 22]),
    ("O.4 LOOP 8/0, INCOH 10/6", [c("bb_falcon_ours", "LOOP"), c("bb_falcon_theirs", "LOOP"),
                                   c("bb_falcon_theirs", "INCOH"), c("bb_falcon_ours", "INCOH")], [8, 0, 10, 6]),
    ("O.4 MobileLLM both bad 61, FACT 23", [n["n_bb_mobilellm_first"], c("bb_mobilellm_ours", "FACT")], [61, 23]),
    ("O.4 wins over Falcon: code 10 + facts 10, ours OK 24", [c("win_falcon_theirs", "CODE"), c("win_falcon_theirs", "FACT"),
                                                               c("win_falcon_ours", "OK")], [10, 10, 24]),
    ("O.4 one particle 17 of 35, 99 of 108", [n["k1_on_k6_wins"]["win"], n["k1_on_k6_losses"]["loss"]], [17, 99]),
    ("O.4 MobileLLM wins 93 (56 + 37), 82 of 116", [c("win_mobilellm_theirs", "INCOH"), c("win_mobilellm_theirs", "LOOP"),
                                                     n["win_m_theirs_degenerate_flag"], n["n_win_mobilellm"]], [56, 37, 82, 116]),
    ("O.4 ours defect 60, INSTR 21, OK 56", [n["n_win_mobilellm"] - c("win_mobilellm_ours", "OK"), c("win_mobilellm_ours", "INSTR"),
                                             c("win_mobilellm_ours", "OK")], [60, 21, 56]),
    ("O.4 losses to MobileLLM 12/2/1, its OK 8", [c("loss_mobilellm_ours", "total", "math"), c("loss_mobilellm_ours", "total", "code"),
                                                    c("loss_mobilellm_ours", "total", "chat"), c("loss_mobilellm_theirs", "OK")], [12, 2, 1, 8]),
    ("O.5 maths 6/15/12", [n["math_correct"][k] for k in ("ours", "falcon", "mobilellm")], [6, 15, 12]),
    ("O.5 subcodes 10/6/5/2/1", [n["math_sub"][k] for k in ("SETUP", "SLIP", "NONSENSE", "CONTRADICT", "NOANSWER")], [10, 6, 5, 2, 1]),
    ("O.5 MATH 23, THIN 1", [n["math_reader_label_when_wrong"]["MATH"], n["math_reader_label_when_wrong"]["THIN"]], [23, 1]),
    ("O.5 checks 46/41/10 of 56", [n["constraints_ok"]["ours"], n["constraints_ok"]["falcon"], n["constraints_ok"]["mobilellm"],
                                   n["n_constraints"]], [46, 41, 10, 56]),
    ("O.5 placement 6 of 9, Falcon 0; four structural", [n["placement_checks"]["ours_missed"], n["placement_checks"]["n"],
                                                          n["placement_checks"]["falcon_missed"], len(n["ours_structural_misses"])], [6, 9, 0, 4]),
    ("O.5 19 of the 26", [n["loss_falcon_constraints_met"], n["loss_falcon_constraint_prompts"]], [19, 26]),
    ("O.6 detectors 5/2/158, 25, 3", [n["degenerate"]["ours"]["flags"]["degenerate"], n["degenerate"]["falcon"]["flags"]["degenerate"],
                                      n["degenerate"]["mobilellm"]["flags"]["degenerate"], len(n["ours_loop_labels"]["falcon"]),
                                      len(n["ours_loop_flagged_by_exact"])], [5, 2, 158, 25, 3]),
    ("O.6 rep4 0.065/0.015, 8 and 11", [n["rep4_ours"]["loop_median"], n["rep4_ours"]["other_median"], n["rep4_ours"]["loop_ge_0.1"],
                                         n["rep4_ours"]["other_ge_0.1"]], [0.065, 0.015, 8, 11]),
    ("O.6 turn-2 copies 4/1/1", [len(n["turn2_repeats_turn1"][k]) for k in ("ours", "falcon", "mobilellm")], [4, 1, 1]),
    ("O.7 agreement", [n["agreement"]["falcon"]["four_way_max_agreement"]["4"], n["agreement"]["mobilellm"]["four_way_max_agreement"]["4"],
                       n["agreement"]["falcon"]["fleiss_kappa_four_way"], n["agreement"]["mobilellm"]["fleiss_kappa_four_way"],
                       n["agreement"]["falcon"]["fleiss_kappa_ours_bad"], n["agreement"]["mobilellm"]["fleiss_kappa_ours_bad"],
                       n["loss_all4_theirs"]], [173, 184, 0.864, 0.915, 0.909, 0.916, 106]),
    ("O.7 cross-set 142, 141, 119, 36", [n["cross_set_ours"][k] for k in ("same", "both_defect", "same_when_both_defect", "ok_in_one_only")],
     [142, 141, 119, 36]),
]
for label, got, want in quotes:
    if got != want:
        fail(f"Appendix O quote {label}: numbers.json now gives {got}, the appendix says {want}")
print(f"all consistency checks passed ({len(expect)} tables from labels.json, {md.count('### ')} tables in tables.md, "
      f"{len(must) + len(must_not)} review checks, {len(quotes)} Appendix O quotes)")
