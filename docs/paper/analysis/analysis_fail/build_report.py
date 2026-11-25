"""Step 3: failure_modes.md from failure_modes.template.md and numbers.json.

    taskset -c 0 $S/venv/bin/python -I $N/paper/analysis_fail/build_report.py

In the template, {{ expr }} is a Python expression over the computed numbers (n = numbers.json,
c(table, row, col) = a table cell, pct(a, b) = a rounded percentage, and the renderers below).
Every number in the report comes from numbers.json; an expression that fails stops the build.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CATS = ("chat", "code", "instruct", "math", "writing")
n = json.loads((HERE / "numbers.json").read_text())
cb = json.loads((HERE / "codebook.json").read_text())
L = json.loads((HERE / "labels.json").read_text())
SHORT = {"OK": "no decisive defect", "FACT": "factual error", "MATH": "wrong arithmetic or reasoning",
         "INSTR": "instruction or format not followed", "INCOH": "incoherent or off-topic",
         "LOOP": "repetition or loop", "TRUNC": "truncated", "THIN": "too thin, empty or deflecting",
         "CODE": "broken or wrong code", "WEAK": "weaker, no hard error"}


def c(tab: str, row: str, col: str = "all") -> int:
    t = n["tables"][tab]
    j = 1 + CATS.index(col) if col != "all" else len(CATS) + 1
    for r in t:
        if r[0] == row:
            return r[j]
    return 0


def pct(a: float, b: float) -> str:
    return f"{100 * a / b:.0f}%"


def T(tab: str, extra: str | None = None, extra_head: str = "") -> str:
    t = n["tables"][tab]
    head = ["reason"] + list(CATS) + ["all"] + ([extra_head] if extra else [])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    rows = [r for r in t if r[0] != "total"]
    codes = [r[0] for r in rows]
    if extra:
        codes += [r[0] for r in n["tables"][extra] if r[0] not in codes and r[0] != "total"]
    for code in codes:
        r = next((x for x in rows if x[0] == code), [code] + [0] * (len(CATS) + 1))
        cells = [f"{code} ({SHORT[code]})"] + [str(v) for v in r[1:]]
        if extra:
            cells.append(str(c(extra, code)))
        lines.append("| " + " | ".join(cells) + " |")
    tot = next(x for x in t if x[0] == "total")
    cells = ["n (total)"] + [str(v) for v in tot[1:]]
    if extra:
        cells.append(str(next(x for x in n["tables"][extra] if x[0] == "total")[-1]))
    lines.append("| " + " | ".join(f"**{x}**" for x in cells) + " |")
    return "\n".join(lines)


def T2(tab_a: str, tab_b: str) -> str:
    """Two label tables over the same prompts in one: every cell is 'a / b'."""
    ta, tb = n["tables"][tab_a], n["tables"][tab_b]
    order = ["INCOH", "LOOP", "FACT", "MATH", "CODE", "INSTR", "TRUNC", "THIN", "WEAK", "OK"]
    present = {r[0] for r in ta + tb if r[0] != "total"}
    head = ["reason"] + list(CATS) + ["all"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for code in [o for o in order if o in present]:
        cells = [f"{code} ({SHORT[code]})"] + [f"{c(tab_a, code, col)} / {c(tab_b, code, col)}" for col in list(CATS) + ["all"]]
        lines.append("| " + " | ".join(cells) + " |")
    tot_a = next(x for x in ta if x[0] == "total")
    tot_b = next(x for x in tb if x[0] == "total")
    if tot_a != tot_b:
        raise SystemExit(f"T2: totals of {tab_a} and {tab_b} differ")
    lines.append("| " + " | ".join(f"**{x}**" for x in ["n (total)"] + [str(v) for v in tot_a[1:]]) + " |")
    return "\n".join(lines)


def structural() -> str:
    """Our structural constraint misses as the checker recorded them (measured against asked)."""
    unit = {"paragraphs": ("paragraph", "paragraphs"), "sections": ("section", "sections"),
            "numbered": ("numbered line", "numbered lines")}
    asked = {"paragraphs": "paragraphs", "sections": "sections", "numbered": "numbered steps"}
    parts = []
    for f in sorted(n["ours_structural_misses"], key=lambda f: f["prompt_id"]):
        where = f"{f['prompt_id']}; {f['reply_words']} words" + (", cut at the token limit" if f["cut_at_token_limit"] else "")
        if f["type"] == "json":
            parts.append(f"no valid JSON where only JSON was asked ({where})")
            continue
        s = f"{f['measured']} {unit[f['type']][f['measured'] != 1]} where exactly {f['asked']} {asked[f['type']]} were asked ({where}"
        if f["type"] == "numbered":
            k = f["lines_starting_with_a_number"]
            s += (f": its steps, numbered up to {f['highest_inline_step_number']}, run together on "
                  f"{'one line' if k == 1 else str(k) + ' lines'}")
        parts.append(s + ")")
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def codebook() -> str:
    lines = ["| code | reason | definition |", "|---|---|---|"]
    for k in ("FACT", "MATH", "INSTR", "INCOH", "LOOP", "TRUNC", "THIN", "CODE", "WEAK", "OK"):
        v = cb["labels"][k]
        lines.append(f"| {k} | {v['name']} | {v['definition']} |")
    return "\n".join(lines)


def math_table() -> str:
    lines = ["| prompt | turns | answer | ours | Falcon | MobileLLM | how ours fails (subcode) |", "|---|---|---|---|---|---|---|"]
    for m in n["math_rows"]:
        a = m["answer"] if not isinstance(m["answer"], list) else " / ".join(m["answer"])
        f = lambda ok, ex: ("yes" if ok else "no") + f" ({ex})"   # noqa: E731
        how = ("correct" + (f"; working faulted: {m['ours_how']}" if m["ours_how"] else "")) if m["ours"] \
            else f"{m['ours_subcode']}: {m['ours_how']}"
        lines.append(f"| {m['prompt_id']} | {m['turns']} | {a} | {f(m['ours'], m['ours_extracted'])} | "
                     f"{f(m['falcon'], m['falcon_extracted'])} | {f(m['mobilellm'], m['mobilellm_extracted'])} | {how} |")
    tot = n["math_correct"]
    lines.append(f"| **solved** | | | **{tot['ours']} of 30** | **{tot['falcon']} of 30** | **{tot['mobilellm']} of 30** | |")
    return "\n".join(lines)


def cons_table() -> str:
    ct = n["constraint_types"]
    keys = sorted(ct, key=lambda k: (-ct[k]["n"], k))
    half = (len(keys) + 1) // 2
    head = "| type | n | ours | Falcon | MobileLLM | type | n | ours | Falcon | MobileLLM |"
    lines = [head, "|" + "---|" * 10]
    for i in range(half):
        cells = []
        for k in (keys[i], keys[i + half] if i + half < len(keys) else None):
            if k is None:
                cells += [""] * 5
            else:
                v = ct[k]
                cells += [k, str(v["n"]), str(v["ours"]), str(v["falcon"]), str(v["mobilellm"])]
        lines.append("| " + " | ".join(cells) + " |")
    ok = n["constraints_ok"]
    lines.append(f"| **all** | **{n['n_constraints']}** | **{ok['ours']}** | **{ok['falcon']}** | **{ok['mobilellm']}** | | | | | |")
    return "\n".join(lines)


def agree_table() -> str:
    a = n["agreement"]
    rows = [("all four readings give the same verdict (ours / theirs / tie / both bad)", "four_way_max_agreement", "4"),
            ("at least three of the four agree on that verdict", "four_way_max_agreement", "ge3"),
            ("Fleiss' kappa, four-way verdict", "fleiss_kappa_four_way", None),
            ("all four agree whether OUR reply was worse or unusable (theirs or both bad)", "ours_bad_or_not_max_agreement", "4"),
            ("Fleiss' kappa, that binary verdict", "fleiss_kappa_ours_bad", None),
            ("same prompt outcome in the first and the neutral round", "first_vs_neutral_same_outcome", None)]
    lines = ["| measure (n = 200 prompts per set) | vs Falcon | vs MobileLLM |", "|---|---|---|"]
    for label, key, sub in rows:
        vals = []
        for o in ("falcon", "mobilellm"):
            v = a[o][key]
            if sub == "4":
                vals.append(f"{v['4']} ({pct(v['4'], 200)})")
            elif sub == "ge3":
                vals.append(f"{v['4'] + v['3']} ({pct(v['4'] + v['3'], 200)})")
            elif isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(f"{v} ({pct(v, 200)})")
        lines.append(f"| {label} | {vals[0]} | {vals[1]} |")
    return "\n".join(lines)


def flat(s: str, k: int = 600) -> str:
    f = " ".join(x.strip() for x in s.replace("\r", "").split("\n") if x.strip())
    f = f.replace("|", "/")
    return (f[:k] + " [...]") if len(f) > k else f


def examples() -> str:
    by = {e["prompt_id"]: e for e in L["prompts"]}
    rec = json.loads((HERE / "records.json").read_text())["falcon"]["rows"]
    rr = {r["prompt_id"]: r for r in rec}
    out = []
    word = {"ours": "OURS", "theirs": "FALCON", "tie": "tie", "both_bad": "both bad"}
    for i, pid in enumerate(n["examples"], 1):
        e, r = by[pid], rr[pid]
        f = e["vs_falcon"]
        seen = "; one of the 11 prompts seen during the run" if e["seen_during_run"] else ""
        out.append(f"**E{i}. {pid}** ({e['category']}{', two turns' if e['multi_turn'] else ''}; position "
                   f"{f['pair_file_index'] + 1} of the pair file; ours shown as {f['ours_shown_as_in_pair_file']}{seen}). "
                   f"Outcome: first round {f['outcome_first_round']}, neutral round {f['outcome_neutral_round']}. "
                   f"Labels: ours {f['reason_ours']['code']}, Falcon {f['reason_theirs']['code']}.")
        for t, (u, a, b) in enumerate(zip(r["user_turns"], r["ours_replies"], r["theirs_replies"]), 1):
            p = f"T{t} " if e["multi_turn"] else ""
            out.append(f"- {p}User: {flat(u, 300)}")
            out.append(f"- {p}Ours: {flat(a)}")
            out.append(f"- {p}Falcon: {flat(b)}")
        rd = r["readings"]
        notes = "; ".join(f"{nm} {word[rd[k]['verdict']]}: \"{rd[k]['note_ours_theirs'].replace('[O]', 'OURS').replace('[T]', 'FALCON')}\""
                          for k, nm in (("r1", "r1"), ("r2", "r2"), ("na", "a"), ("nb", "b")))
        out.append(f"- Readings: {notes}")
        out.append("")
    return "\n".join(out).rstrip()


def render(tpl: str) -> str:
    env = {"n": n, "c": c, "pct": pct, "T": T, "T2": T2, "codebook": codebook, "structural": structural, "math_table": math_table,
           "cons_table": cons_table, "agree_table": agree_table, "examples": examples, "len": len, "min": min, "max": max,
           "sum": sum, "sorted": sorted, "join": lambda xs: ", ".join(xs)}

    def sub(m):
        expr = m.group(1).strip()
        try:
            return str(eval(expr, env))                   # noqa: S307 (our own template)
        except Exception as exc:                          # noqa: BLE001
            print(f"BUILD FAILED at {{{{ {expr} }}}}: {exc!r}")
            sys.exit(1)
    return re.sub(r"\{\{(.+?)\}\}", sub, tpl, flags=re.S)


if __name__ == "__main__":
    out = render((HERE / "failure_modes.template.md").read_text())
    (HERE / "failure_modes.md").write_text(out)
    print(f"wrote failure_modes.md: {out.count(chr(10)) + 1} lines")
