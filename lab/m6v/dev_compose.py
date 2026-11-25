"""The development score of every decode configuration, composed from judged pieces.

    python lab/m6v/dev_compose.py DEFAULT_DIR LOOSE_DIR PLAN.json VERB_DIR [--json OUT]

DEFAULT_DIR: ``rows.json`` and ``judge*.jsonl`` / ``lane_*.jsonl`` of the replies made with the
library's own floors (200 prompts). LOOSE_DIR: the same for the replies made in the loose mode at the
lowest floor (the prompts whose exemplars change). PLAN.json: per prompt, the score and contrast of the
best loose exemplar of every turn (written when the changed prompts were listed). VERB_DIR: the study
of stored answers against the opponent (``rows.json`` with one row per prompt and candidate entry,
``out*.jsonl`` with the verdicts).

A configuration is ``(T, floor, numbers)``: a stored answer is the reply when the library's own rule
holds or the entry's rank reaches ``T`` (``decode.verbatim_rank``); else the loose mode with ``floor``
(``decode.exemplar_floor``), with or without entries that differ in their numbers
(``decode.exemplar_numbers``); else the library's floors. Only single-turn prompts are composed (in a
two-turn prompt the first reply changes what the second one sees); two-turn prompts keep the verdict
of their default reply, and the count of prompts whose path would change is printed."""
import glob
import json
import sys
from pathlib import Path

VAL = {"A": 1.0, "B": 0.0, "tie": 0.5}


def records(d, patterns=("judge*.jsonl", "lane_*.jsonl", "rev_*.jsonl", "out*.jsonl", "all.jsonl")):
    out = {}
    for pat in patterns:
        for f in sorted(glob.glob(str(Path(d) / pat))):
            for ln in open(f):
                if ln.strip():
                    r = json.loads(ln)
                    out.setdefault(r["pair_id"], r)
    return out


def main(argv):
    out_json = None
    if "--json" in argv:
        i = argv.index("--json")
        out_json = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    ddir, ldir, plan_p, vdir = argv[:4]
    rows = {}
    for f in sorted(glob.glob(str(Path(ddir) / "rows*.json"))):
        rows.update({r["pair_id"]: r for r in json.loads(Path(f).read_text())})
    dflt, loose = records(ddir), records(ldir)
    plan = {r["id"]: r for r in json.loads(Path(plan_p).read_text())}
    vrows = {r["pair_id"]: r for r in json.loads((Path(vdir) / "rows.json").read_text())}
    vrec = records(vdir)
    rank = lambda r: min(r["cov_q"], r["cov_e"]) - (0.0 if r["type_match"] else 0.1)  # noqa: E731
    ids = [p for p in rows if p in dflt]
    single = [p for p in ids if rows[p]["turns"] == 1]
    print(f"{len(ids)} prompts with a default verdict ({len(single)} single-turn); loose verdicts {len(loose)}; "
          f"stored-answer verdicts {len(vrec)} for {len({r['prompt_id'] for r in vrows.values()})} prompts")
    res = []
    for T in (0.0, 0.8, 0.75, 0.7, 0.65, 0.6):
        best = {}
        if T:
            for pid, r in vrows.items():
                if pid in vrec and rank(r) >= T - 1e-12 and r["meets_constraints"]:
                    p = r["prompt_id"]
                    if p not in best or rank(r) > rank(vrows[best[p]]):
                        best[p] = pid
        for floor, numbers in ((0.0, False), (0.45, False), (0.4, False), (0.35, False), (0.45, True), (0.4, True), (0.35, True)):
            v, n_verb, n_loose, miss = [], 0, 0, 0
            for p in ids:
                t0 = (plan.get(p) or {"turns": [{}]})["turns"][0]
                if p in single and rows[p]["tiers"][0] != "verbatim" and p in best:
                    v.append(VAL[vrec[best[p]]["verdict"]])
                    n_verb += 1
                    continue
                use_loose = False
                if floor and p in single and rows[p]["tiers"][0] != "verbatim" and t0.get("loose") and not t0.get("default"):
                    if t0.get("contrast"):
                        use_loose = numbers
                    else:
                        use_loose = (t0.get("score") or 0) >= floor - 1e-12
                if use_loose:
                    if p in loose:
                        v.append(VAL[loose[p]["verdict"]])
                        n_loose += 1
                        continue
                    miss += 1
                v.append(VAL[dflt[p]["verdict"]])
            w, l = sum(x == 1 for x in v), sum(x == 0 for x in v)
            res.append({"verbatim_rank": T, "exemplar_floor": floor, "numbers": numbers, "win": w, "loss": l,
                        "tie": len(v) - w - l, "score": sum(v) / len(v), "more_verbatim": n_verb, "loose_used": n_loose,
                        "loose_missing": miss})
            print(f"T {T:<4} floor {floor:<4} numbers {str(numbers):5s}: {w:3d} / {l:3d} / {len(v) - w - l:3d}  score {sum(v) / len(v):.4f}"
                  f"   (+{n_verb} verbatim, {n_loose} loose" + (f", {miss} loose verdicts missing" if miss else "") + ")")
    if out_json:
        Path(out_json).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
