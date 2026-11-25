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
of their default reply, and the count of prompts whose path would change is printed.

    python lab/m6v/dev_compose.py DEFAULT_DIR LOOSE_DIR PLAN.json --pool POOL_DIR [--json OUT]

The same from the pool of ``lab/m6v/fit_pool.py`` (``rows.json``: every candidate the shipped
library's verbatim rule looks at, in its order, with its own overlap and the reader's fit; the
judge's verdicts of the stored answers in ``judge*.jsonl`` / ``lane_*.jsonl``). The reuse rule of a
configuration is then one of: the library's own rule only; the overlap threshold ``T``
(``decode.verbatim_rank``: the first candidate in the index's order that reaches it); the reader's
test ``(tau, r)`` (``decode.verbatim_fit``, ``decode.verbatim_fit_rank``: the candidate with the best
fit among those whose overlap reaches ``r``, when its fit is at least ``-tau``). For every
configuration the prompts it changes are compared with their default verdict (better / worse, sign
test). A chosen stored answer without a verdict keeps the default verdict and is counted."""
import glob
import json
import sys
from pathlib import Path

VAL = {"A": 1.0, "B": 0.0, "tie": 0.5}
WINDOW = 8.0          # decode.verbatim_fit_window of the composed fit rules (fixed before the final verdicts)


def records(d, patterns=("judge*.jsonl", "lane_*.jsonl", "rev_*.jsonl", "out*.jsonl", "all.jsonl")):
    out = {}
    for pat in patterns:
        for f in sorted(glob.glob(str(Path(d) / pat))):
            for ln in open(f):
                if ln.strip():
                    r = json.loads(ln)
                    out.setdefault(r["pair_id"], r)
    return out


def sign_p(b, w):
    from math import comb

    n = b + w
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, w) + 1)) / 2 ** n)


def pool_main(ddir, ldir, plan_p, pdir, out_json):
    rows = {}
    for f in sorted(glob.glob(str(Path(ddir) / "rows*.json"))):
        rows.update({r["pair_id"]: r for r in json.loads(Path(f).read_text())})
    dflt, loose = records(ddir), records(ldir)
    plan = {r["id"]: r for r in json.loads(Path(plan_p).read_text())}
    pool = {}
    for r in json.loads((Path(pdir) / "rows.json").read_text()):
        pool.setdefault(r["prompt_id"], []).append(r)
    verd = records(pdir)
    ids = [p for p in rows if p in dflt]
    single = {p for p in ids if rows[p]["turns"] == 1}
    print(f"{len(ids)} prompts with a default verdict ({len(single)} single-turn); loose verdicts {len(loose)}; "
          f"pool: {sum(len(v) for v in pool.values())} candidates of {len(pool)} prompts, {len(verd)} judged")

    def pick(p, rule):
        """The pool row whose stored answer is the reply under ``rule``, or None."""
        cands = [r for r in pool.get(p, []) if not r["own_rule"]]
        if rule[0] == "rank":
            return next((r for r in cands if r["own"] >= rule[1] - 1e-12), None)
        if rule[0] == "fit":
            asked = [r for r in cands if r["own"] >= rule[2] - 1e-12 and r["fit"] is not None
                     and (len(rule) < 4 or rule[3] is None or (r.get("window") is not None and r["window"] <= rule[3]))]
            best = max(asked, key=lambda r: r["fit"], default=None)
            return best if best is not None and best["fit"] >= -rule[1] else None
        return None

    has_window = any(r.get("window") is not None for rs in pool.values() for r in rs)
    rules = [("own",)] + [("rank", t) for t in (0.8, 0.75, 0.7, 0.65, 0.6)]
    rules += [("fit", tau, r, w) for w in ((None, WINDOW) if has_window else (None,)) for r in (0.5, 0.4, 0.35)
              for tau in (0.05, 0.075, 0.1, 0.15)]
    modes = ((0.0, False), (0.45, False), (0.4, False), (0.35, False), (0.45, True), (0.4, True), (0.35, True))
    res = []
    for rule in rules:
        for floor, numbers in modes:
            v, n_verb, n_loose, miss, better, worse = [], 0, 0, 0, 0, 0
            for p in ids:
                base = VAL[dflt[p]["verdict"]]
                new = base
                movable = p in single and rows[p]["tiers"][0] != "verbatim"
                c = pick(p, rule) if movable else None
                if c is not None:
                    if c["pair_id"] in verd:
                        new = VAL[verd[c["pair_id"]]["verdict"]]
                        n_verb += 1
                    else:
                        miss += 1
                elif floor and movable:
                    t0 = (plan.get(p) or {"turns": [{}]})["turns"][0]
                    if t0.get("loose") and not t0.get("default"):
                        use = numbers if t0.get("contrast") else (t0.get("score") or 0) >= floor - 1e-12
                        if use and p in loose:
                            new = VAL[loose[p]["verdict"]]
                            n_loose += 1
                v.append(new)
                better += new > base
                worse += new < base
            w, l = sum(x == 1 for x in v), sum(x == 0 for x in v)
            name = {"own": "library's rule only", "rank": "overlap >= %s" % (tuple(rule[1:]) or ("",))[0],
                    "fit": "fit >= -%s, overlap >= %s%s" % ((tuple(rule[1:]) + ("", ""))[:2] + (
                        "" if len(rule) < 4 or rule[3] is None else ", window <= %g" % rule[3],))}[rule[0]]
            res.append({"rule": list(rule), "exemplar_floor": floor, "numbers": numbers, "win": w, "loss": l,
                        "tie": len(v) - w - l, "score": sum(v) / len(v), "more_verbatim": n_verb, "loose_used": n_loose,
                        "unjudged": miss, "better": better, "worse": worse, "p": sign_p(better, worse)})
            if floor == 0.0 or rule[0] == "own":
                print(f"{name:46s} floor {floor:<4} numbers {str(numbers):5s}: {w:3d} / {l:3d} / {len(v) - w - l:3d}  "
                      f"score {sum(v) / len(v):.4f}  (+{n_verb} verbatim, {n_loose} loose"
                      + (f", {miss} unjudged" if miss else "") + f")  better {better} worse {worse} p {sign_p(better, worse):.3f}")
    top = max(res, key=lambda r: (r["score"], -r["more_verbatim"] - r["loose_used"]))
    print("highest composed score:", json.dumps({k: top[k] for k in ("rule", "exemplar_floor", "numbers", "score", "better", "worse")}))
    if out_json:
        Path(out_json).write_text(json.dumps(res, indent=1))
    return 0


def main(argv):
    out_json = None
    if "--json" in argv:
        i = argv.index("--json")
        out_json = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    if "--pool" in argv:
        i = argv.index("--pool")
        pdir = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
        return pool_main(argv[0], argv[1], argv[2], pdir, out_json)
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
