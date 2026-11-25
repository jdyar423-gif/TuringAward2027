"""Step 2 of the failure analysis: labels.json and every count of the report, with sum checks.

    cd /home/user/turingaward2027
    taskset -c 0 $S/venv/bin/python $N/paper/analysis_fail/analyze.py

Inputs (this directory): records.json (build_records.py), codebook.json, labels_manual_falcon.tsv,
labels_manual_mobilellm.tsv (the manual labels, one line per pair, made from the readers' notes
and checked against the replies), and the K = 1 report for one contrast.
Outputs: labels.json, numbers.json (every number and table the report cites), tables.md
(the tables, for reading), examples.md (the 12 pairs chosen by the fixed rule).
Any count that does not sum to its n stops the script.
"""
from __future__ import annotations

import difflib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
N = Path("/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad")
CATS = ("chat", "code", "instruct", "math", "writing")
OPPS = ("falcon", "mobilellm")
OPP_NAME = {"falcon": "Falcon-H1-Tiny-90M-Instruct", "mobilellm": "MobileLLM-R1.5-140M"}
READINGS = ("r1", "r2", "na", "nb")
VERD = ("ours", "theirs", "tie", "both_bad")

# Maths failures of OUR last-turn answer, subcoded by reading each reply (codebook
# maths_failure_subcodes); the 6 prompts not listed are correct by the exact scorer.
MATH_SUB = {
    "math-001": ("SLIP", "47 x 23 = 1176"),
    "math-007": ("SETUP", "multiplies the fractions instead of adding"),
    "math-008": ("SETUP", "turn 2 divides 18, not 126 (turn 1 right)"),
    "math-010": ("SLIP", "14 + 22 + 9 + 35 = 96"),
    "math-014": ("SLIP", "copies 240 as 241, then 36.5"),
    "math-015": ("SLIP", "92 - 80 = 18, then a negative percentage"),
    "math-017": ("SLIP", "8 / 32 = 0.2"),
    "math-020": ("SETUP", "km to m by dividing by 100; laps not computed"),
    "math-024": ("NONSENSE", "turn 2 working incoherent, ends '75 grams for 10 people' (turn 1 right)"),
    "math-026": ("SETUP", "subtracts the raise, ignores the second raise"),
    "math-030": ("SETUP", "takes 1 foot = 2.54 cm"),
    "math-032": ("SLIP", "'$85 - $250 = $15', carried into turn 2"),
    "math-034": ("NOANSWER", "'In a typical week, there are 7 days.'"),
    "math-036": ("SETUP", "turn 2 divides by 12 instead of 84 (turn 1 right)"),
    "math-039": ("SETUP", "no m^2 to cm^2 conversion"),
    "math-042": ("NONSENSE", "'13:15 * 60 = 830 minutes'"),
    "math-044": ("CONTRADICT", "computes 128, then answers 320 + 128 = 448"),
    "math-045": ("SETUP", "adds two sides only"),
    "math-047": ("SETUP", "'3/8 full means 1/8 full'"),
    "math-048": ("NONSENSE", "invented 'sum' formula, 85"),
    "math-050": ("NONSENSE", "treats 18 km as a speed, 5 km/h"),
    "math-057": ("SETUP", "multiplies the bill by 3, ignores the tip"),
    "math-058": ("CONTRADICT", "reaches $40.50, then continues to 28.67"),
    "math-059": ("NONSENSE", "'18 classes + 18 = 36 students'"),
}
# Correct last answers whose working the readers faulted (from the notes, both sets).
MATH_RIGHT_BAD_WORKING = {"math-031": "misstates the conversion factor",
                          "math-049": "35 x 2 = 70, then x 7: right number by a wrong route",
                          "math-055": "'multiply by the perimeter'"}


def fail(msg: str) -> None:
    print("CHECK FAILED:", msg)
    sys.exit(1)


def check_sum(name: str, counts, n: int) -> None:
    s = sum(counts.values()) if isinstance(counts, dict) else sum(counts)
    if s != n:
        fail(f"{name}: sums to {s}, not {n}")


def load_manual(opp: str, codes: set[str]) -> dict[int, dict]:
    out = {}
    for line in (HERE / f"labels_manual_{opp}.tsv").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 7:
            fail(f"{opp}: short line {line!r}")
        idx = int(f[0])
        for lab, sup in ((f[2], f[3]), (f[4], f[5])):
            if lab not in codes:
                fail(f"{opp} {idx}: unknown label {lab}")
            if (lab == "OK") != (sup == "-"):
                fail(f"{opp} {idx}: support {sup!r} does not fit label {lab}")
            if sup != "-" and not 0 <= int(sup) <= 4:
                fail(f"{opp} {idx}: support {sup}")
        if idx in out:
            fail(f"{opp}: index {idx} twice")
        out[idx] = {"prompt_id": f[1], "ours": f[2], "ours_support": None if f[3] == "-" else int(f[3]),
                    "theirs": f[4], "theirs_support": None if f[5] == "-" else int(f[5]), "comment": f[6]}
    return out


def table(rows_by_cat: dict[str, Counter], order: list[str], cats=CATS) -> list[list]:
    """[[code, n_chat, ..., n_all], ...] plus a total row."""
    out = []
    for code in order:
        vals = [rows_by_cat[c][code] for c in cats]
        if sum(vals):
            out.append([code] + vals + [sum(vals)])
    tot = [sum(rows_by_cat[c].values()) for c in cats]
    out.append(["total"] + tot + [sum(tot)])
    return out


def md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(lines)


def fleiss(items: list[list[str]], cats) -> float:
    n = len(items[0])
    N_ = len(items)
    pj = Counter()
    P = 0.0
    for it in items:
        c = Counter(it)
        pj.update(c)
        P += (sum(v * (v - 1) for v in c.values())) / (n * (n - 1))
    P /= N_
    Pe = sum((pj[k] / (N_ * n)) ** 2 for k in cats)
    return (P - Pe) / (1 - Pe)


def main() -> None:
    rec = json.loads((HERE / "records.json").read_text())
    cb = json.loads((HERE / "codebook.json").read_text())
    codes = set(cb["labels"])
    order = ["INCOH", "LOOP", "FACT", "MATH", "CODE", "INSTR", "TRUNC", "THIN", "WEAK", "OK"]
    if set(order) != codes:
        fail("codebook codes changed")
    manual = {o: load_manual(o, codes) for o in OPPS}
    num: dict = {}
    tabs: dict = {}

    # ------------------------------------------------------------- labels.json
    rows = {o: rec[o]["rows"] for o in OPPS}
    for o in OPPS:
        if len(manual[o]) != 200:
            fail(f"{o}: {len(manual[o])} labels")
        for r in rows[o]:
            m = manual[o].get(r["pair_file_index"])
            if m is None or m["prompt_id"] != r["prompt_id"]:
                fail(f"{o}: label of index {r['pair_file_index']} missing or for another prompt")
            r["label"] = m
            if r["outcome_first_round"] == "loss" and m["ours"] == "OK":
                fail(f"{o} {r['prompt_id']}: a loss with ours OK")
            if r["outcome_first_round"] == "win" and m["theirs"] == "OK":
                fail(f"{o} {r['prompt_id']}: a win with theirs OK")
    by_prompt = {o: {r["prompt_id"]: r for r in rows[o]} for o in OPPS}
    pids = sorted(by_prompt["falcon"])
    if pids != sorted(by_prompt["mobilellm"]) or len(pids) != 200:
        fail("the two sets do not cover the same 200 prompts")
    for pid in pids:                                       # our replies are the same in both sets
        if by_prompt["falcon"][pid]["ours_replies"] != by_prompt["mobilellm"][pid]["ours_replies"]:
            fail(f"{pid}: our replies differ between the sets")
    lab_name = {k: v["name"] for k, v in cb["labels"].items()}
    out = []
    for pid in pids:
        fr = by_prompt["falcon"][pid]
        e = {"prompt_id": pid, "category": fr["category"], "multi_turn": fr["multi_turn"],
             "seen_during_run": fr["seen_during_run"], "user_turns": fr["user_turns"]}
        for o in OPPS:
            r = by_prompt[o][pid]
            m = r["label"]
            e[f"vs_{o}"] = {
                "pairset": r["pairset"], "pair_id": r["pair_id"], "pair_file_index": r["pair_file_index"],
                "ours_shown_as_in_pair_file": r["ours_side_in_pair_file"],
                "outcome_first_round": r["outcome_first_round"],
                "outcome_neutral_round": r["outcome_neutral_round"],
                "outcome_registered_judge": r["outcome_registered_judge"],
                "readings": {k: {"round": "first" if k in ("r1", "r2") else "neutral",
                                 "sides_exchanged": r["readings"][k]["exchanged"],
                                 "raw_verdict": r["readings"][k]["raw_verdict"],
                                 "verdict": r["readings"][k]["verdict"],
                                 "note": r["readings"][k]["note"]} for k in READINGS},
                "reason_ours": {"code": m["ours"], "name": lab_name[m["ours"]], "notes_naming_it": m["ours_support"]},
                "reason_theirs": {"code": m["theirs"], "name": lab_name[m["theirs"]], "notes_naming_it": m["theirs_support"]},
                "labeller_comment": m["comment"],
            }
        out.append(e)
    (HERE / "labels.json").write_text(json.dumps({
        "about": ("Failure-analysis labels for the 200 test prompts of the blind comparisons final-prefopt-vs-falcon "
                  "and final-prefopt-vs-mobilellm. Outcomes are for OUR model (win / tie / loss); 'tie' includes "
                  "split readings and both-bad. Readings: r1 and r2 = first round (r2 with sides exchanged), na and "
                  "nb = second round with neutral titles (nb exchanged). 'verdict' is the raw verdict mapped to "
                  "ours / theirs / tie / both_bad. reason_ours / reason_theirs: the labeller's main reason for "
                  "each reply (codebook below; 'notes_naming_it' = how many of the four notes name that defect)."),
        "codebook": cb, "opponents": {f"vs_{o}": {"name": OPP_NAME[o], "run": rec[o]["y"], "ours": rec[o]["x"]}
                                       for o in OPPS},
        "prompts": out}, indent=1, ensure_ascii=False))

    # ------------------------------------------------------------- outcomes
    for o in OPPS:
        for rnd in ("first_round", "neutral_round"):
            c = Counter(r[f"outcome_{rnd}"] for r in rows[o])
            num[f"{o}_{rnd}"] = [c["win"], c["loss"], c["tie"]]
            for cat in CATS:
                cc = Counter(r[f"outcome_{rnd}"] for r in rows[o] if r["category"] == cat)
                num[f"{o}_{rnd}_{cat}"] = [cc["win"], cc["loss"], cc["tie"]]
    if num["falcon_first_round"] != [35, 108, 57] or num["mobilellm_first_round"] != [116, 15, 69]:
        fail("first-round outcome totals")

    def subset(o, pred):
        return [r for r in rows[o] if pred(r)]

    def by_cat(rs, side):
        d = {c: Counter() for c in CATS}
        for r in rs:
            d[r["category"]][r["label"][side]] += 1
        return d

    # ------------------------------------------------------------- (b) losses to Falcon
    loss_f = subset("falcon", lambda r: r["outcome_first_round"] == "loss")
    num["n_loss_falcon"] = len(loss_f)
    t = table(by_cat(loss_f, "ours"), order)
    check_sum("loss falcon ours", [x[-1] for x in t[:-1]], len(loss_f))
    tabs["loss_falcon_ours"] = t
    t = table(by_cat(loss_f, "theirs"), order)
    check_sum("loss falcon theirs", [x[-1] for x in t[:-1]], len(loss_f))
    tabs["loss_falcon_theirs"] = t
    # neutral-round losses as a robustness view
    loss_fn = subset("falcon", lambda r: r["outcome_neutral_round"] == "loss")
    tabs["loss_falcon_ours_neutral"] = table(by_cat(loss_fn, "ours"), order)
    check_sum("loss falcon neutral", [x[-1] for x in tabs["loss_falcon_ours_neutral"][:-1]], len(loss_fn))
    num["n_loss_falcon_neutral"] = len(loss_fn)
    num["loss_both_rounds"] = sum(1 for r in loss_f if r["outcome_neutral_round"] == "loss")
    num["loss_all4_theirs"] = sum(1 for r in loss_f if all(r["readings"][k]["verdict"] == "theirs" for k in READINGS))

    # content errors vs form/degeneracy among the losses
    groups = {"content (FACT, MATH, CODE)": {"FACT", "MATH", "CODE"},
              "degenerate or off-task (INCOH, LOOP, TRUNC)": {"INCOH", "LOOP", "TRUNC"},
              "instruction, thin or weaker (INSTR, THIN, WEAK)": {"INSTR", "THIN", "WEAK"}}
    for gname, gset in groups.items():
        num[f"loss_group::{gname}"] = sum(1 for r in loss_f if r["label"]["ours"] in gset)
    check_sum("loss groups", [num[f"loss_group::{g}"] for g in groups], len(loss_f))

    # ------------------------------------------------------------- (b) both-bad prompts
    def bb(r, ks):
        return all(r["readings"][k]["verdict"] == "both_bad" for k in ks)
    bb_f = subset("falcon", lambda r: bb(r, ("r1", "r2")))
    num["n_bb_falcon_first"] = len(bb_f)
    num["n_bb_falcon_all4"] = sum(1 for r in rows["falcon"] if bb(r, READINGS))
    num["n_bb_falcon_ge3"] = sum(1 for r in rows["falcon"] if sum(r["readings"][k]["verdict"] == "both_bad" for k in READINGS) >= 3)
    num["n_bb_falcon_any"] = sum(1 for r in rows["falcon"] if any(r["readings"][k]["verdict"] == "both_bad" for k in READINGS))
    num["bb_falcon_by_cat"] = dict(Counter(r["category"] for r in bb_f))
    for side in ("ours", "theirs"):
        t = table(by_cat(bb_f, side), order)
        check_sum(f"bb falcon {side}", [x[-1] for x in t[:-1]], len(bb_f))
        tabs[f"bb_falcon_{side}"] = t
    num["bb_falcon_same_label"] = sum(1 for r in bb_f if r["label"]["ours"] == r["label"]["theirs"])
    bb_m = subset("mobilellm", lambda r: bb(r, ("r1", "r2")))
    num["n_bb_mobilellm_first"] = len(bb_m)
    for side in ("ours", "theirs"):
        t = table(by_cat(bb_m, side), order)
        check_sum(f"bb mobilellm {side}", [x[-1] for x in t[:-1]], len(bb_m))
        tabs[f"bb_mobilellm_{side}"] = t
    num["n_bb_both_sets"] = len({r["prompt_id"] for r in bb_f} & {r["prompt_id"] for r in bb_m})

    # ------------------------------------------------------------- (c) wins
    win_m = subset("mobilellm", lambda r: r["outcome_first_round"] == "win")
    win_f = subset("falcon", lambda r: r["outcome_first_round"] == "win")
    loss_m = subset("mobilellm", lambda r: r["outcome_first_round"] == "loss")
    for name, rs, side in (("win_mobilellm_theirs", win_m, "theirs"), ("win_mobilellm_ours", win_m, "ours"),
                           ("win_falcon_theirs", win_f, "theirs"), ("win_falcon_ours", win_f, "ours"),
                           ("loss_mobilellm_ours", loss_m, "ours"), ("loss_mobilellm_theirs", loss_m, "theirs")):
        t = table(by_cat(rs, side), order)
        check_sum(name, [x[-1] for x in t[:-1]], len(rs))
        tabs[name] = t
    num["n_win_mobilellm"], num["n_win_falcon"], num["n_loss_mobilellm"] = len(win_m), len(win_f), len(loss_m)
    num["win_falcon_ids"] = [r["prompt_id"] for r in win_f]
    num["loss_mobilellm_ids"] = [r["prompt_id"] for r in loss_m]
    # exact degenerate flags of MobileLLM in our wins
    num["win_m_theirs_degenerate_flag"] = sum(1 for r in win_m if any(t["degenerate"] for t in r["exact_theirs"]["turns"]))

    # ------------------------------------------------------------- ties that are not both-bad (Falcon)
    tie_f = subset("falcon", lambda r: r["outcome_first_round"] == "tie")
    num["n_tie_falcon"] = len(tie_f)
    num["tie_falcon_split"] = sum(1 for r in tie_f if not bb(r, ("r1", "r2")))

    # ------------------------------------------------------------- (d) exact scorers: maths
    mrows = []
    sub = Counter()
    for pid in [p for p in pids if p.startswith("math")]:
        f = by_prompt["falcon"][pid]
        mo = by_prompt["mobilellm"][pid]
        ok = lambda e: bool(e["math_correct"])                            # noqa: E731
        ext = lambda e: "/".join(str(t["math"]["extracted"]) for t in e["turns"] if "math" in t)  # noqa: E731
        s = MATH_SUB.get(pid)
        if ok(f["exact_ours"]) == (s is not None):
            fail(f"{pid}: maths subcode does not match the exact scorer")
        if s:
            sub[s[0]] += 1
        mrows.append({"prompt_id": pid, "turns": 2 if f["multi_turn"] else 1, "answer": f["answer"],
                      "ours": ok(f["exact_ours"]), "falcon": ok(f["exact_theirs"]), "mobilellm": ok(mo["exact_theirs"]),
                      "ours_extracted": ext(f["exact_ours"]), "falcon_extracted": ext(f["exact_theirs"]),
                      "mobilellm_extracted": ext(mo["exact_theirs"]),
                      "ours_subcode": s[0] if s else "correct",
                      "ours_how": s[1] if s else MATH_RIGHT_BAD_WORKING.get(pid, ""),
                      "outcome_vs_falcon": f["outcome_first_round"], "outcome_vs_mobilellm": mo["outcome_first_round"],
                      "label_vs_falcon": f["label"]["ours"], "label_vs_mobilellm": mo["label"]["ours"]})
    num["math_rows"] = mrows
    num["math_correct"] = {k: sum(m[k] for m in mrows) for k in ("ours", "falcon", "mobilellm")}
    num["math_sub"] = dict(sub)
    check_sum("maths subcodes", sub, 30 - num["math_correct"]["ours"])
    num["math_only_ours"] = [m["prompt_id"] for m in mrows if m["ours"] and not m["falcon"] and not m["mobilellm"]]
    num["math_none"] = [m["prompt_id"] for m in mrows if not (m["ours"] or m["falcon"] or m["mobilellm"])]
    num["math_all3"] = [m["prompt_id"] for m in mrows if m["ours"] and m["falcon"] and m["mobilellm"]]
    num["math_ours_not_falcon"] = [m["prompt_id"] for m in mrows if m["ours"] and not m["falcon"]]
    num["math_falcon_not_ours"] = [m["prompt_id"] for m in mrows if m["falcon"] and not m["ours"]]
    # first turns of the two-turn maths prompts
    tt = []
    for pid in [m["prompt_id"] for m in mrows if m["turns"] == 2]:
        f = by_prompt["falcon"][pid]
        mo = by_prompt["mobilellm"][pid]
        def turns_ok(e):
            return [t["math"]["correct"] for t in e["turns"] if "math" in t]
        tt.append({"prompt_id": pid, "ours": turns_ok(f["exact_ours"]), "falcon": turns_ok(f["exact_theirs"]),
                   "mobilellm": turns_ok(mo["exact_theirs"])})
    num["math_two_turn"] = tt
    num["math_two_turn_first_ok"] = {k: sum(x[k][0] for x in tt) for k in ("ours", "falcon", "mobilellm")}
    num["math_two_turn_last_ok"] = {k: sum(x[k][-1] for x in tt) for k in ("ours", "falcon", "mobilellm")}
    # readers vs exact on maths, ours
    num["math_reader_label_when_wrong"] = dict(Counter(
        (by_prompt["falcon"][m["prompt_id"]]["label"]["ours"]) for m in mrows if not m["ours"]))
    num["math_outcome_when_right_vs_falcon"] = {m["prompt_id"]: m["outcome_vs_falcon"] for m in mrows if m["ours"]}
    # maths losses split by what the exact scorer says of our last-turn answer
    for o in OPPS:
        ls = [r for r in rows[o] if r["category"] == "math" and r["outcome_first_round"] == "loss"]
        right = sorted(r["prompt_id"] for r in ls if r["exact_ours"]["math_correct"])
        if any(by_prompt[o][p]["label"]["ours"] != "MATH" for p in right):
            fail(f"{o}: a maths loss with a right answer is not labelled MATH")
        missing = sorted(r["prompt_id"] for r in ls if not r["exact_ours"]["math_correct"] and r["label"]["ours"] != "MATH")
        wrong = len(ls) - len(right) - len(missing)
        num[f"math_losses_{o}"] = {"n": len(ls), "wrong": wrong, "missing": len(missing), "missing_ids": missing,
                                   "right_faulty_working": len(right), "right_ids": right}
        check_sum(f"maths losses {o}", [wrong, len(missing), len(right)], len(ls))
    # amendment F: MATH labels on replies whose last-turn answer the exact scorer counts as right
    covered = set()
    for o in OPPS:
        for r in rows[o]:
            if r["category"] != "math":
                continue
            for side, ex in (("ours", r["exact_ours"]), ("theirs", r["exact_theirs"])):
                if r["label"][side] == "MATH" and ex["math_correct"]:
                    covered.add((o, side, r["prompt_id"]))
    listed = {(c["set"], c["side"], c["prompt_id"]) for a in cb["amendments_after_labelling"] if a["id"] == "F"
              for c in a["covers"]}
    if covered != listed:
        fail(f"amendment F lists {sorted(listed)} but the labels give {sorted(covered)}")
    num["math_label_on_right_answer"] = [list(x) for x in sorted(covered)]

    # ------------------------------------------------------------- (d) constraints and degeneracy
    ctype = {k: Counter() for k in ("n", "ours", "falcon", "mobilellm")}
    ours_fail = []
    for pid in pids:
        f = by_prompt["falcon"][pid]
        mo = by_prompt["mobilellm"][pid]
        for who, e in (("ours", f["exact_ours"]), ("falcon", f["exact_theirs"]), ("mobilellm", mo["exact_theirs"])):
            for t in e["turns"]:
                for c in t.get("constraints", []):
                    if who == "ours":
                        ctype["n"][c["type"]] += 1
                        if not c["ok"]:
                            ours_fail.append({"prompt_id": pid, "type": c["type"], "measured": c.get("measured"),
                                              "reply_words": t["words"], "cut_at_token_limit": "truncated" in t["flags"],
                                              "label_vs_falcon": f["label"]["ours"],
                                              "outcome_vs_falcon": f["outcome_first_round"]})
                    ctype[who][c["type"]] += int(c["ok"])
    num["constraint_types"] = {k: {w: ctype[w][k] for w in ("n", "ours", "falcon", "mobilellm")}
                               for k in sorted(ctype["n"])}
    num["n_constraints"] = sum(ctype["n"].values())
    num["constraints_ok"] = {w: sum(ctype[w].values()) for w in ("ours", "falcon", "mobilellm")}
    num["ours_constraint_failures"] = ours_fail
    # "putting given content in a given place": the grouping is stated, with and without starts_with
    placement = ("include", "ends_with", "ends_with_question", "postscript")
    ct = num["constraint_types"]
    pc = {"types": list(placement), "n": sum(ct[k]["n"] for k in placement)}
    pc.update({f"{w}_missed": sum(ct[k]["n"] - ct[k][w] for k in placement) for w in ("ours", "falcon", "mobilellm")})
    wsw = placement + ("starts_with",)
    pc["with_starts_with"] = {"n": sum(ct[k]["n"] for k in wsw),
                              **{f"{w}_missed": sum(ct[k]["n"] - ct[k][w] for k in wsw) for w in ("ours", "falcon", "mobilellm")}}
    num["placement_checks"] = pc
    struct = [dict(f) for f in ours_fail if f["type"] not in placement]
    for f in struct:                      # what was asked, from the prompt's own constraint
        spec = [c for c in by_prompt["falcon"][f["prompt_id"]]["constraints"] if c["type"] == f["type"]]
        f["asked"] = spec[0].get("n") if spec else None
        if f["type"] == "numbered":       # how the steps are laid out in the reply
            import re as _re
            text = by_prompt["falcon"][f["prompt_id"]]["ours_replies"][-1]
            f["lines_starting_with_a_number"] = sum(1 for ln in text.split("\n") if _re.match(r"\s*\d+[.)]", ln))
            f["highest_inline_step_number"] = max(int(x) for x in _re.findall(r"(?<![\d.])(\d{1,3})[.)]\s", text))
            if f["lines_starting_with_a_number"] != f["measured"]:
                fail(f"{f['prompt_id']}: numbered lines {f['lines_starting_with_a_number']} != checker's {f['measured']}")
    num["ours_structural_misses"] = struct
    num["win_falcon_ours_code_ids"] = sorted(r["prompt_id"] for r in win_f if r["label"]["ours"] == "CODE")
    check_sum("our constraint misses", [pc["ours_missed"], len(struct)], len(ours_fail))
    cons_prompts = [p for p in pids if by_prompt["falcon"][p]["exact_ours"]["n_constraints"]]
    num["n_constraint_prompts"] = len(cons_prompts)
    num["constraint_prompts_all_ok"] = {
        "ours": sum(1 for p in cons_prompts if by_prompt["falcon"][p]["exact_ours"]["constraints_ok"]),
        "falcon": sum(1 for p in cons_prompts if by_prompt["falcon"][p]["exact_theirs"]["constraints_ok"]),
        "mobilellm": sum(1 for p in cons_prompts if by_prompt["mobilellm"][p]["exact_theirs"]["constraints_ok"])}
    # constraint-checked prompts that we lost to Falcon although ours met every constraint
    lost_ok = [r for r in loss_f if r["exact_ours"]["n_constraints"] and r["exact_ours"]["constraints_ok"]]
    num["loss_falcon_constraints_met"] = len(lost_ok)
    num["loss_falcon_constraints_met_labels"] = dict(Counter(r["label"]["ours"] for r in lost_ok))
    num["loss_falcon_constraint_prompts"] = sum(1 for r in loss_f if r["exact_ours"]["n_constraints"])
    num["instruct_losses_falcon"] = len([r for r in loss_f if r["category"] == "instruct"])
    num["instruct_losses_falcon_constraints_met"] = len([r for r in lost_ok if r["category"] == "instruct"])
    num["instruct_losses_falcon_constraints_met_labels"] = dict(Counter(r["label"]["ours"] for r in lost_ok
                                                                        if r["category"] == "instruct"))
    # degenerate detectors against the LOOP / TRUNC labels
    deg = {}
    for who, e_of in (("ours", lambda p: by_prompt["falcon"][p]["exact_ours"]),
                      ("falcon", lambda p: by_prompt["falcon"][p]["exact_theirs"]),
                      ("mobilellm", lambda p: by_prompt["mobilellm"][p]["exact_theirs"])):
        flags = Counter()
        n_rep = 0
        prompts_flagged = []
        for p in pids:
            e = e_of(p)
            any_deg = False
            for t in e["turns"]:
                n_rep += 1
                if t["degenerate"]:
                    flags["degenerate"] += 1
                    any_deg = True
                for fl in t["flags"]:
                    flags[fl] += 1
            if any_deg:
                prompts_flagged.append(p)
        deg[who] = {"replies": n_rep, "flags": dict(flags), "prompts_flagged": prompts_flagged}
    num["degenerate"] = deg
    num["ours_loop_labels"] = {o: sorted(r["prompt_id"] for r in rows[o] if r["label"]["ours"] == "LOOP") for o in OPPS}
    num["ours_loop_or_trunc_any_set"] = sorted(set(num["ours_loop_labels"]["falcon"]) | set(num["ours_loop_labels"]["mobilellm"]))
    num["ours_loop_flagged_by_exact"] = sorted(set(num["ours_loop_or_trunc_any_set"]) & set(deg["ours"]["prompts_flagged"]))
    num["theirs_loop_labels_mobilellm"] = sum(1 for r in rows["mobilellm"] if r["label"]["theirs"] == "LOOP")
    # repetition of our own turn 1 in turn 2 (two-turn prompts), measured
    rep = {}
    for who, get in (("ours", lambda p: by_prompt["falcon"][p]["ours_replies"]),
                     ("falcon", lambda p: by_prompt["falcon"][p]["theirs_replies"]),
                     ("mobilellm", lambda p: by_prompt["mobilellm"][p]["theirs_replies"])):
        hits = []
        for p in pids:
            if not by_prompt["falcon"][p]["multi_turn"]:
                continue
            a, b = get(p)
            ratio = difflib.SequenceMatcher(None, a[:3000], b[:3000], autojunk=False).ratio() if a and b else 0.0
            if ratio >= 0.9:
                hits.append(p)
        rep[who] = hits
    num["turn2_repeats_turn1"] = rep          # similarity >= 0.9 (threshold set before looking)
    rep8 = {}
    for who, get in (("ours", lambda p: by_prompt["falcon"][p]["ours_replies"]),
                     ("falcon", lambda p: by_prompt["falcon"][p]["theirs_replies"]),
                     ("mobilellm", lambda p: by_prompt["mobilellm"][p]["theirs_replies"])):
        rep8[who] = [p for p in pids if by_prompt["falcon"][p]["multi_turn"] and all(get(p)) and
                     difflib.SequenceMatcher(None, get(p)[0][:3000], get(p)[1][:3000], autojunk=False).ratio() >= 0.8]
    num["turn2_repeats_turn1_08"] = rep8      # similarity >= 0.8 (chosen after seeing the distribution)
    num["turn2_repeat08_outcomes_vs_falcon"] = dict(Counter(by_prompt["falcon"][p]["outcome_first_round"] for p in rep8["ours"]))
    num["turn2_repeat08_labels_vs_falcon"] = dict(Counter(by_prompt["falcon"][p]["label"]["ours"] for p in rep8["ours"]))
    # repeated-4-gram rate (exact scorer) of our replies, by the readers-based label
    ex_ours = json.loads((N / "study/proxy/reports/exact/ours-export-step51812-47fd99c151a50a58__final-prefopt-9a89782942-s0-test.json").read_text())
    rep4 = {q["prompt_id"]: max(t["degenerate"]["rep4"] for t in q["turns"]) for q in ex_ours["per_prompt"]}
    lp = [rep4[r["prompt_id"]] for r in rows["falcon"] if r["label"]["ours"] == "LOOP"]
    ot = [rep4[r["prompt_id"]] for r in rows["falcon"] if r["label"]["ours"] != "LOOP"]
    srt = lambda v: sorted(v)[len(v) // 2]                                 # noqa: E731
    num["rep4_ours"] = {"loop_n": len(lp), "loop_median": round(srt(lp), 3), "other_n": len(ot),
                        "other_median": round(srt(ot), 3), "loop_ge_0.1": sum(x >= 0.1 for x in lp),
                        "other_ge_0.1": sum(x >= 0.1 for x in ot)}
    num["n_two_turn"] = sum(1 for p in pids if by_prompt["falcon"][p]["multi_turn"])
    num["turn2_repeat_outcomes_vs_falcon"] = dict(Counter(by_prompt["falcon"][p]["outcome_first_round"] for p in rep["ours"]))

    # ------------------------------------------------------------- (e) agreement of the four readings
    agree = {}
    for o in OPPS:
        items = [[r["readings"][k]["verdict"] for k in READINGS] for r in rows[o]]
        modal = Counter(max(Counter(it).values()) for it in items)
        binv = [["bad" if v in ("theirs", "both_bad") else "ok" for v in it] for it in items]
        bmodal = Counter(max(Counter(it).values()) for it in binv)
        dec = [[{"ours": "ours", "theirs": "theirs"}.get(v, "neither") for v in it] for it in items]
        dmodal = Counter(max(Counter(it).values()) for it in dec)
        agree[o] = {"four_way_max_agreement": {str(k): modal[k] for k in (4, 3, 2)},
                    "fleiss_kappa_four_way": round(fleiss(items, VERD), 3),
                    "ours_bad_or_not_max_agreement": {str(k): bmodal[k] for k in (4, 3, 2)},
                    "fleiss_kappa_ours_bad": round(fleiss(binv, ("bad", "ok")), 3),
                    "three_way_max_agreement": {str(k): dmodal[k] for k in (4, 3, 2)},
                    "fleiss_kappa_three_way": round(fleiss(dec, ("ours", "theirs", "neither")), 3),
                    "first_vs_neutral_same_outcome": sum(1 for r in rows[o] if r["outcome_first_round"] == r["outcome_neutral_round"])}
        for k in ("four_way_max_agreement", "ours_bad_or_not_max_agreement", "three_way_max_agreement"):
            check_sum(f"agreement {o} {k}", agree[o][k], 200)
    num["agreement"] = agree
    # support (how many notes name the chosen defect)
    sup = {}
    for name, rs, side in (("losses_falcon_ours", loss_f, "ours"), ("wins_mobilellm_theirs", win_m, "theirs"),
                           ("wins_falcon_theirs", win_f, "theirs"), ("bothbad_falcon_ours", bb_f, "ours"),
                           ("bothbad_falcon_theirs", bb_f, "theirs")):
        c = Counter(r["label"][f"{side}_support"] for r in rs if r["label"][side] != "OK")
        sup[name] = {str(k): c[k] for k in (4, 3, 2, 1, 0)}
        check_sum(f"support {name}", sup[name], sum(1 for r in rs if r["label"][side] != "OK"))
    num["support"] = sup
    # rule B: a defect label on the reply the readers preferred needs at least two notes naming it
    for o in OPPS:
        for r in rows[o]:
            side = {"win": "ours", "loss": "theirs"}.get(r["outcome_first_round"])
            if side and r["label"][side] != "OK" and r["label"][f"{side}_support"] < 2:
                fail(f"{o} {r['prompt_id']}: preferred reply labelled {r['label'][side]} on fewer than two notes")
    c_ok_win_f = sum(1 for r in win_f if r["label"]["ours"] == "OK")
    # labels decided by the tie-break of rule 2 (the comment records "tie ..." for these)
    num["tiebreak_losses_falcon"] = sum(1 for r in loss_f if "tie" in r["label"]["comment"].split(";")[0])
    import re
    alt = {c: Counter() for c in CATS}
    for r in loss_f:
        m = re.search(r"\btie(?: with)? ([A-Z]+)", r["label"]["comment"].split(";")[0])
        alt[r["category"]][m.group(1) if m else r["label"]["ours"]] += 1
    tabs["loss_falcon_ours_other_tiebreak"] = table(alt, order)
    check_sum("loss other tie-break", [x[-1] for x in tabs["loss_falcon_ours_other_tiebreak"][:-1]], len(loss_f))
    num["qwen_on_losses_falcon"] = dict(Counter(r["outcome_registered_judge"] for r in loss_f))
    num["qwen_on_wins_mobilellm"] = dict(Counter(r["outcome_registered_judge"] for r in win_m))
    num["tiebreak_losses_falcon_list"] = [[r["prompt_id"], r["label"]["ours"], r["label"]["comment"]] for r in loss_f
                                          if "tie" in r["label"]["comment"].split(";")[0]]
    # same reply of ours, labelled from two different sets of notes
    same = Counter()
    pairs_def = 0
    conf = Counter()
    for p in pids:
        a, b = by_prompt["falcon"][p]["label"]["ours"], by_prompt["mobilellm"][p]["label"]["ours"]
        same["same" if a == b else "different"] += 1
        if a != "OK" and b != "OK":
            pairs_def += 1
            if a == b:
                same["same_both_defect"] += 1
            else:
                conf[tuple(sorted((a, b)))] += 1
        if (a == "OK") != (b == "OK"):
            same["ok_in_one_only"] += 1
    num["cross_set_ours"] = {"same": same["same"], "different": same["different"], "both_defect": pairs_def,
                             "same_when_both_defect": same["same_both_defect"],
                             "ok_in_one_only": same["ok_in_one_only"],
                             "top_confusions": [[f"{a}/{b}", n] for (a, b), n in conf.most_common(6)]}
    num["ours_ok_by_set"] = {o: sum(1 for r in rows[o] if r["label"]["ours"] == "OK") for o in OPPS}

    # ------------------------------------------------------------- K = 1 contrast
    k1 = json.loads((N / "study/proxy/reports/ab/final-k1-vs-falcon--judge-opus.json").read_text())["per_prompt_score"]
    to = {1.0: "win", 0.5: "tie", 0.0: "loss"}
    num["k1_on_k6_losses"] = dict(Counter(to[k1[r["prompt_id"]]] for r in loss_f))
    num["k1_on_k6_wins"] = dict(Counter(to[k1[r["prompt_id"]]] for r in win_f))
    check_sum("k1 on losses", num["k1_on_k6_losses"], len(loss_f))

    # ------------------------------------------------------------- seen prompts
    seen = [p for p in pids if by_prompt["falcon"][p]["seen_during_run"]]
    num["seen"] = {p: {"vs_falcon": by_prompt["falcon"][p]["outcome_first_round"],
                       "vs_mobilellm": by_prompt["mobilellm"][p]["outcome_first_round"],
                       "label_vs_falcon": by_prompt["falcon"][p]["label"]["ours"]} for p in seen}
    loss_189 = [r for r in loss_f if not r["seen_during_run"]]
    num["n_loss_falcon_189"] = len(loss_189)
    num["loss_falcon_189_ours"] = dict(Counter(r["label"]["ours"] for r in loss_189))

    # ------------------------------------------------------------- category totals for table headers
    for name, rs in (("loss_falcon", loss_f), ("bb_falcon", bb_f), ("win_mobilellm", win_m), ("win_falcon", win_f),
                     ("loss_mobilellm", loss_m), ("bb_mobilellm", bb_m)):
        num[f"cat_n_{name}"] = {c: sum(1 for r in rs if r["category"] == c) for c in CATS}

    # ------------------------------------------------------------- examples by the fixed rule
    ex = []
    for cat in CATS:
        ls = [r for r in rows["falcon"] if r["category"] == cat and r["outcome_first_round"] == "loss"][:2]
        ex += ls
    ex += [r for r in rows["falcon"] if r["outcome_first_round"] == "win"][:2]
    ex.sort(key=lambda r: r["pair_file_index"])
    if len(ex) != 12:
        fail("examples")
    num["examples"] = [r["prompt_id"] for r in ex]

    def cut(s, n=600):
        s = s.replace("\r", "")
        flat = " ".join(x.strip() for x in s.split("\n") if x.strip())
        return (flat[:n] + " [...]") if len(flat) > n else flat
    lines = []
    for i, r in enumerate(ex, 1):
        lab = r["label"]
        tag = " (one of the 11 prompts seen during the run)" if r["seen_during_run"] else ""
        lines.append(f"**E{i}. {r['prompt_id']}** ({r['category']}{', two turns' if r['multi_turn'] else ''}; "
                     f"Falcon pair file position {r['pair_file_index'] + 1}){tag}. Outcome for ours: first round "
                     f"**{r['outcome_first_round']}**, neutral round **{r['outcome_neutral_round']}**. "
                     f"In the pair file ours is reply {r['ours_side_in_pair_file']}. My labels: ours "
                     f"{lab['ours']}, Falcon {lab['theirs']}.")
        lines.append("")
        for t, (u, a, b) in enumerate(zip(r["user_turns"], r["ours_replies"], r["theirs_replies"]), 1):
            pre = f"Turn {t}. " if r["multi_turn"] else ""
            lines.append(f"- {pre}User: {cut(u, 400)}")
            lines.append(f"- {pre}Ours: {cut(a)}")
            lines.append(f"- {pre}Falcon: {cut(b)}")
        code = {"ours": "ours", "theirs": "Falcon", "tie": "tie", "both_bad": "both bad"}
        for k, nm in (("r1", "r1"), ("r2", "r2, sides exchanged"), ("na", "neutral a"), ("nb", "neutral b, exchanged")):
            rd = r["readings"][k]
            lines.append(f"- Reading {nm}: {code[rd['verdict']]}. \"{rd['note_ours_theirs'].replace('[O]', 'OURS').replace('[T]', 'FALCON')}\"")
        lines.append("")
    (HERE / "examples.md").write_text("\n".join(lines))

    # ------------------------------------------------------------- length buckets and reply lengths (reports)
    num["ab_by_length"] = {}
    for o, setname in (("falcon", "final-prefopt-vs-falcon"), ("mobilellm", "final-prefopt-vs-mobilellm")):
        rep = json.loads((N / f"study/proxy/reports/ab/{setname}--judge-opus.json").read_text())["by_length"]
        num["ab_by_length"][o] = {b: [v["wins"], v["losses"], v["ties"] + v["both_bad"]] for b, v in rep.items()}
    words = {}
    for who, fn in (("ours", "ours-export-step51812-47fd99c151a50a58__final-prefopt-9a89782942-s0-test.json"),
                    ("falcon", "falcon-e60034381f4a__recommended-5ccec0259b-s0-test.json"),
                    ("mobilellm", "mobilellm-f680f0533368-qwen3__recommended-d9f149b0d3-s0-test.json")):
        s = json.loads((N / "study/proxy/reports/exact" / fn).read_text())["summary"]
        words[who] = {"mean": s["words"]["mean"], "median": s["words"]["median"]}
    num["words"] = words
    # labels of our losses to Falcon by the report's length bucket (words over all turns, ratio (x+1)/(y+1))
    lb = {}
    for r in loss_f:
        x = sum(len(s.split()) for s in r["ours_replies"])
        y = sum(len(s.split()) for s in r["theirs_replies"])
        k = "x_shorter" if (x + 1) / (y + 1) < 2 / 3 else "x_longer" if (x + 1) / (y + 1) > 1.5 else "similar"
        lb.setdefault(k, Counter())[r["label"]["ours"]] += 1
    num["loss_labels_by_length"] = {k: dict(v) for k, v in lb.items()}
    # our label on the wins over Falcon, by the same length buckets
    wb = {}
    for r in win_f:
        x = sum(len(s.split()) for s in r["ours_replies"])
        y = sum(len(s.split()) for s in r["theirs_replies"])
        k = "x_shorter" if (x + 1) / (y + 1) < 2 / 3 else "x_longer" if (x + 1) / (y + 1) > 1.5 else "similar"
        wb.setdefault(k, Counter())[r["label"]["ours"]] += 1
    num["win_falcon_ours_by_length"] = {k: dict(v) for k, v in wb.items()}
    for k, v in wb.items():
        if sum(v.values()) != num["ab_by_length"]["falcon"][k][0]:
            fail(f"length bucket {k}: {sum(v.values())} wins, report says {num['ab_by_length']['falcon'][k][0]}")
    if sum(v.get("OK", 0) for v in wb.values()) != c_ok_win_f:
        fail("ours OK over the wins by length does not add up to the wins table")
    for k, v in lb.items():
        if sum(v.values()) != num["ab_by_length"]["falcon"][k][1]:
            fail(f"length bucket {k}: {sum(v.values())} losses, report says {num['ab_by_length']['falcon'][k][1]}")

    # ------------------------------------------------------------- write
    num["tables"] = tabs
    (HERE / "numbers.json").write_text(json.dumps(num, indent=1, ensure_ascii=False))
    hdr = ["reason"] + list(CATS) + ["all"]
    md = []
    for name, t in tabs.items():
        md += [f"### {name}", "", md_table(hdr, t), ""]
    (HERE / "tables.md").write_text("\n".join(md))
    print("all checks passed")


if __name__ == "__main__":
    main()
