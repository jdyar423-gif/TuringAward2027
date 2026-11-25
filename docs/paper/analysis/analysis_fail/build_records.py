"""Step 1 of the failure analysis: assemble and verify every record per (prompt, opponent).

Run from the repository (to import ``bes``), read-only everywhere except this directory:

    cd /home/user/turingaward2027
    taskset -c 0 $S/venv/bin/python $N/paper/analysis_fail/build_records.py

Reads: the test prompts (bes.eval.proxy.prompts), the two pair sets and their keys
(bes.eval.proxy.make_ab.load_pairs / load_key, which checks the key against the published
commitment chain), the four readings per pair (r1 as published, r2 exchanged, neutral a as
published, neutral b exchanged: verdicts and notes), the per-prompt outcomes of the three
A/B reports (judge-opus, judge-opus-neutral, judge-qwen3-4b-2507-3605803b), and the exact
scores of ours, Falcon and MobileLLM.

Checks (any failure stops the script): every reading covers the 200 pair ids once; every chunk's
verdict and note files hold exactly the ids of that chunk; in every chunk file the replies shown
as A and B are the pair file's A and B (published readings) or B and A (exchanged readings);
the combined outcomes recomputed from the raw verdicts (a side wins only when both readings of a
round choose it, otherwise tie) equal the reports' per_prompt_score on all 200 prompts; the
outcome totals are 35/108/57 (Falcon) and 116/15/69 (MobileLLM) in the first round.

Writes records.json (everything later steps need) and notes_view_<opp>.txt (the four notes per
pair with "A"/"B" rewritten as [O] (ours) / [T] (theirs), for labelling).
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/user/turingaward2027")           # the repository (read only)
from bes.eval.proxy import make_ab                          # noqa: E402
from bes.eval.proxy import prompts as P                     # noqa: E402

N = Path("/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad")
ROOT = N / "study" / "proxy"
OUT = N / "paper" / "analysis_fail"

SETS = {
    "falcon": {"name": "final-prefopt-vs-falcon", "neutral": "x1",
               "exact": "falcon-e60034381f4a__recommended-5ccec0259b-s0-test.json"},
    "mobilellm": {"name": "final-prefopt-vs-mobilellm", "neutral": "x2",
                  "exact": "mobilellm-f680f0533368-qwen3__recommended-d9f149b0d3-s0-test.json"},
}
OURS_EXACT = "ours-export-step51812-47fd99c151a50a58__final-prefopt-9a89782942-s0-test.json"
# The 11 test prompts a mid-run sample touched (paper/lead5_work/scripts/final_results.py SEEN).
SEEN = {"chat-002", "chat-017", "code-002", "code-026", "instruct-001", "instruct-034",
        "instruct-062", "math-001", "math-034", "writing-001", "writing-100"}
EXPECTED_FIRST = {"falcon": (35, 108, 57), "mobilellm": (116, 15, 69)}


def readings(opp: str) -> dict[str, tuple[Path, bool]]:
    """reading name -> (directory, exchanged?)."""
    s = SETS[opp]
    return {"r1": (N / "final_eval" / "prefopt" / s["name"] / "r1", False),
            "r2": (N / "final_eval" / "prefopt" / s["name"] / "r2", True),
            "na": (N / "final_eval" / "neutral" / s["neutral"] / "a", False),
            "nb": (N / "final_eval" / "neutral" / s["neutral"] / "b", True)}


def fail(msg: str) -> None:
    print("CHECK FAILED:", msg)
    sys.exit(1)


def parse_chunk_md(text: str) -> dict[str, list[tuple[str, str]]]:
    """pair id -> [(shown A, shown B) per turn] from a chunk file."""
    out: dict[str, list[tuple[str, str]]] = {}
    parts = re.split(r"^## Pair \d+: id `(p[0-9a-f]+)` \([^)]*\)\n", text, flags=re.M)
    for i in range(1, len(parts), 2):
        pid, body = parts[i], parts[i + 1]
        turns = re.split(r"^### turn \d+\n", body, flags=re.M)
        turns = [t for t in turns if "----- USER -----" in t]
        got = []
        for t in turns:
            m = re.search(r"----- REPLY A -----\n(.*?)\n\n----- REPLY B -----\n(.*?)\n\n?$", t, flags=re.S)
            if not m:
                fail(f"cannot parse pair {pid}")
            got.append((m.group(1), m.group(2)))
        out[pid] = got
    return out


def shown(text: str) -> str:
    return text if str(text).strip() else "(empty reply)"


def load_reading(opp: str, rname: str, pairs_by_id: dict, key: dict) -> dict[str, dict]:
    d, exchanged = readings(opp)[rname]
    chunks = sorted(d.glob("chunk_*.ids.json"))
    if len(chunks) != 14:
        fail(f"{opp} {rname}: {len(chunks)} chunks")
    out: dict[str, dict] = {}
    for c in chunks:
        stem = c.name[: -len(".ids.json")]
        ids = json.loads(c.read_text())
        verd = json.loads((d / f"{stem}.verdicts.json").read_text())
        notes = json.loads((d / f"{stem}.notes.json").read_text())
        if set(verd) != set(ids) or set(notes) != set(ids):
            fail(f"{opp} {rname} {stem}: verdict/note ids differ from the chunk's ids")
        md = parse_chunk_md((d / f"{stem}.md").read_text())
        if list(md) != ids:
            fail(f"{opp} {rname} {stem}: chunk file ids differ from ids.json")
        for pos, pid in enumerate(ids):
            if pid in out:
                fail(f"{opp} {rname}: pair {pid} read twice")
            pair = pairs_by_id[pid]
            for (sa, sb), t in zip(md[pid], pair["turns"]):
                want = (shown(t["B"]), shown(t["A"])) if exchanged else (shown(t["A"]), shown(t["B"]))
                if (sa.strip(), sb.strip()) != (want[0].strip(), want[1].strip()):
                    fail(f"{opp} {rname} {pid}: replies shown are not the expected sides")
            v = verd[pid]
            if v not in make_ab.VERDICTS:
                fail(f"{opp} {rname} {pid}: bad verdict {v!r}")
            ours_in_file = "A" if key["assignment"][pid] == "x" else "B"
            ours_shown = {"A": "B", "B": "A"}[ours_in_file] if exchanged else ours_in_file
            mapped = {"tie": "tie", "both_bad": "both_bad"}.get(v) or ("ours" if v == ours_shown else "theirs")
            out[pid] = {"raw": v, "mapped": mapped, "note": notes[pid], "ours_shown_as": ours_shown,
                        "chunk": stem, "pos_in_chunk": pos}
    if set(out) != set(pairs_by_id):
        fail(f"{opp} {rname}: covers {len(out)} pairs, not the 200 of the set")
    return out


def rewrite_note(note: str, ours_shown: str) -> str:
    """Replace the side letters by [O] (ours) and [T] (theirs)."""
    theirs = "B" if ours_shown == "A" else "A"
    return re.sub(r"\b([AB])\b", lambda m: "[O]" if m.group(1) == ours_shown else "[T]", note)


def combined(m1: str, m2: str) -> str:
    """Outcome of one round from its two readings (mapped verdicts), as blind_eval import2 +
    judge.combine: a side wins only when both readings choose it, otherwise tie."""
    if m1 == m2 == "ours":
        return "win"
    if m1 == m2 == "theirs":
        return "loss"
    return "tie"


def exact_index(fname: str) -> dict[str, dict]:
    d = json.loads((ROOT / "reports" / "exact" / fname).read_text())
    return {p["prompt_id"]: p for p in d["per_prompt"]}


def exact_brief(p: dict) -> dict:
    turns = []
    for t in p["turns"]:
        dg = t["degenerate"]
        e = {"words": dg["words"], "degenerate": dg["degenerate"],
             "flags": [k for k in ("empty", "truncated", "loop", "repeated_lines", "unclosed_fence",
                                   "role_leak", "refusal", "deflection") if dg.get(k)]}
        if "math" in t:
            e["math"] = {k: t["math"].get(k) for k in ("correct", "any_match", "method", "extracted")}
        if "constraints" in t:
            e["constraints"] = t["constraints"]
        turns.append(e)
    return {"math_correct": p["math_correct"], "constraints_ok": p["constraints_ok"],
            "n_constraints": p["n_constraints"], "n_constraints_ok": p["n_constraints_ok"], "turns": turns}


def main() -> None:
    prompts = {p.id: p for p in P.load() if p.split == "test"}
    if len(prompts) != 200:
        fail(f"{len(prompts)} test prompts")
    ex_ours = exact_index(OURS_EXACT)
    records: dict[str, dict] = {}
    for opp, s in SETS.items():
        pairs, meta = make_ab.load_pairs(ROOT, s["name"])
        key = make_ab.load_key(ROOT, s["name"])           # verified against meta's commitment chain
        make_ab.check_pairs(pairs, key, s["name"])
        by_id = {p["pair_id"]: p for p in pairs}
        if len(pairs) != 200 or {p["prompt_id"] for p in pairs} != set(prompts):
            fail(f"{opp}: pairs do not cover the 200 test prompts")
        reps = {r: load_reading(opp, r, by_id, key) for r in ("r1", "r2", "na", "nb")}
        rep_first = json.loads((ROOT / "reports" / "ab" / f"{s['name']}--judge-opus.json").read_text())
        rep_neut = json.loads((ROOT / "reports" / "ab" / f"{s['name']}--judge-opus-neutral.json").read_text())
        rep_qwen = json.loads((ROOT / "reports" / "ab" /
                               f"{s['name']}--judge-qwen3-4b-2507-3605803b.json").read_text())
        score_to = {1.0: "win", 0.5: "tie", 0.0: "loss"}
        ex_opp = exact_index(s["exact"])
        rows = []
        for order, pair in enumerate(pairs):
            pid, prid = pair["pair_id"], pair["prompt_id"]
            pr = prompts[prid]
            ours_letter = "A" if key["assignment"][pid] == "x" else "B"
            theirs_letter = "B" if ours_letter == "A" else "A"
            v = {r: reps[r][pid] for r in reps}
            first = score_to[rep_first["per_prompt_score"][prid]]
            neut = score_to[rep_neut["per_prompt_score"][prid]]
            if combined(v["r1"]["mapped"], v["r2"]["mapped"]) != first:
                fail(f"{opp} {prid}: first-round outcome does not recompute")
            if combined(v["na"]["mapped"], v["nb"]["mapped"]) != neut:
                fail(f"{opp} {prid}: neutral-round outcome does not recompute")
            rows.append({
                "opponent": opp, "pairset": s["name"], "pair_id": pid, "prompt_id": prid,
                "pair_file_index": order, "category": pair["category"], "multi_turn": pair["multi_turn"],
                "seen_during_run": prid in SEEN,
                "user_turns": [t["user"] for t in pair["turns"]],
                "ours_replies": [t[ours_letter] for t in pair["turns"]],
                "theirs_replies": [t[theirs_letter] for t in pair["turns"]],
                "ours_side_in_pair_file": ours_letter,
                "answer": pr.answer, "constraints": list(pr.constraints), "prompt_notes": pr.notes,
                "outcome_first_round": first, "outcome_neutral_round": neut,
                "outcome_registered_judge": score_to[rep_qwen["per_prompt_score"][prid]],
                "readings": {r: {"exchanged": readings(opp)[r][1], "raw_verdict": v[r]["raw"],
                                 "verdict": v[r]["mapped"], "ours_shown_as": v[r]["ours_shown_as"],
                                 "note": v[r]["note"],
                                 "note_ours_theirs": rewrite_note(v[r]["note"], v[r]["ours_shown_as"]),
                                 "chunk": v[r]["chunk"]} for r in ("r1", "r2", "na", "nb")},
                "exact_ours": exact_brief(ex_ours[prid]), "exact_theirs": exact_brief(ex_opp[prid]),
            })
        c = Counter(r["outcome_first_round"] for r in rows)
        got = (c["win"], c["loss"], c["tie"])
        cn = Counter(r["outcome_neutral_round"] for r in rows)
        print(f"{opp}: first round W/L/T {got}, neutral {(cn['win'], cn['loss'], cn['tie'])}")
        if got != EXPECTED_FIRST[opp]:
            fail(f"{opp}: first-round totals {got} != {EXPECTED_FIRST[opp]}")
        for r in ("r1", "r2", "na", "nb"):
            print(f"   {r}: raw {dict(Counter(x['raw'] for x in reps[r].values()))}; mapped "
                  f"{dict(Counter(x['mapped'] for x in reps[r].values()))}")
        records[opp] = {"pairset": s["name"], "x": key["x"], "y": key["y"], "rows": rows}
        # labelling view
        lines = []
        code = {"ours": "O", "theirs": "T", "tie": "tie", "both_bad": "bad"}
        for r in rows:
            u = " || ".join(t.replace("\n", " ")[:260] for t in r["user_turns"])
            eo, et = r["exact_ours"], r["exact_theirs"]

            def ex(e):
                bits = []
                if e["math_correct"] is not None:
                    bits.append(f"math={'Y' if e['math_correct'] else 'N'}:"
                                + "/".join(str(t.get('math', {}).get('extracted')) for t in e['turns'] if 'math' in t))
                if e["n_constraints"]:
                    bad = [c['type'] for t in e['turns'] for c in t.get('constraints', []) if not c['ok']]
                    bits.append(f"cons={e['n_constraints_ok']}/{e['n_constraints']}" + (f"(miss {','.join(bad)})" if bad else ""))
                fl = sorted({f for t in e['turns'] for f in t['flags']})
                if fl:
                    bits.append("flags=" + ",".join(fl))
                bits.append("w=" + "+".join(str(t['words']) for t in e['turns']))
                return " ".join(bits)
            lines.append(f"#{r['pair_file_index']:03d} {r['prompt_id']} [{r['category']}{', 2t' if r['multi_turn'] else ''}]"
                         f" first={r['outcome_first_round']} neut={r['outcome_neutral_round']} qwen={r['outcome_registered_judge']}"
                         f" | v=" + ",".join(code[r['readings'][k]['verdict']] for k in ('r1', 'r2', 'na', 'nb')))
            lines.append(f"   Q: {u}" + (f"  [ans={r['answer']}]" if r['answer'] is not None else ""))
            lines.append(f"   exO: {ex(eo)} | exT: {ex(et)}")
            for k in ("r1", "r2", "na", "nb"):
                lines.append(f"   {k} {code[r['readings'][k]['verdict']]:>3}: {r['readings'][k]['note_ours_theirs']}")
        (OUT / f"notes_view_{opp}.txt").write_text("\n".join(lines) + "\n")
    (OUT / "records.json").write_text(json.dumps(records, indent=1, ensure_ascii=False))
    print("all checks passed; wrote", OUT / "records.json")


if __name__ == "__main__":
    main()
