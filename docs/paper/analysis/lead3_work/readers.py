"""Every reader's verdicts on the development pair sets (and the opponents' reference set on the test
prompts), recomputed from the rating files with the repository's own key check and tally
(`bes.eval.proxy.make_ab.load_key`, `bes.eval.proxy.judge.load_verdicts`, `bes.eval.proxy.score.tally`),
plus pair-by-pair agreement between readers. Read-only on the study directory; writes readers.json
next to this script.

    python -I -B readers.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

R = Path("/home/user/turingaward2027")
sys.path.insert(0, str(R))
from bes.eval.proxy import make_ab, score  # noqa: E402
from bes.eval.proxy.judge import load_verdicts  # noqa: E402

N = Path("/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad")
ROOT = N / "study/proxy"
HERE = Path(__file__).resolve().parent

SETS = ["s1-default-vs-falcon", "s1-default-vs-mobilellm", "s2-default-vs-falcon",
        "s2-default-vs-mobilellm", "s1-ngram3-vs-default", "s2-ngram3-vs-default",
        "s1-k1-vs-default", "s1-short-vs-default", "s1-t04-vs-default", "s1-t08-vs-default",
        "ref-falcon-vs-mobilellm-test"]
for name in SETS:
    assert not name.startswith("final-"), name


def side(outcome):
    return {"win": "x", "loss": "y"}.get(outcome, "tie")


out = {}
for name in SETS:
    pairs, meta = make_ab.load_pairs(ROOT, name)
    key = make_ab.load_key(ROOT, name)
    by_id = {p["pair_id"]: p for p in pairs}
    rdir = ROOT / "ratings" / name
    entry = {"x": key["x"], "y": key["y"], "n_pairs": len(pairs), "sources": {}}
    outcomes = {}
    for f in sorted(rdir.glob("*.jsonl")):
        src = f.stem
        verdicts = load_verdicts(ROOT, name, src, key=key)
        oc = {pid: make_ab.unblind(v, pid, key) for pid, v in verdicts.items()}
        outcomes[src] = oc
        t = score.tally(oc.values())
        recs = [json.loads(l) for l in open(f)]
        recs = [r for r in recs if r.get("pair_id") in oc]
        diag = {}
        if recs and "agree" in recs[0]:
            diag = {"orders_agree": round(sum(bool(r["agree"]) for r in recs) / len(recs), 4),
                    "first_position_pref_mean": round(sum(r["first_position_pref"] for r in recs) / len(recs), 4),
                    "clipped": sum(bool(r.get("clipped")) for r in recs),
                    "order1_verdicts": dict(Counter(r["order1"]["verdict"] for r in recs)),
                    "order2_verdicts": dict(Counter(r["order2_swapped"]["verdict"] for r in recs))}
            secs = [r["seconds"] for r in recs if "seconds" in r]
            if secs:
                diag["seconds_per_pair_median"] = sorted(secs)[len(secs) // 2]
        raw = dict(Counter(verdicts.values()))
        cats, turns = {}, {"single_turn": [], "multi_turn": []}
        for pid, o in oc.items():
            p = by_id[pid]
            cats.setdefault(p["category"], []).append(o)
            turns["multi_turn" if p.get("multi_turn") else "single_turn"].append(o)
        entry["sources"][src] = {
            "overall": t, "raw_verdicts": raw, "diagnostics": diag,
            "by_category": {c: {k: score.tally(v)[k] for k in ("wins", "losses", "ties", "both_bad")}
                            for c, v in sorted(cats.items())},
            "by_turns": {k: {kk: score.tally(v)[kk] for kk in ("wins", "losses", "ties", "both_bad")}
                         for k, v in turns.items() if v},
        }
    # pair-by-pair agreement between every two sources
    agree = {}
    srcs = sorted(outcomes)
    for i, a in enumerate(srcs):
        for b in srcs[i + 1:]:
            common = sorted(set(outcomes[a]) & set(outcomes[b]))
            c = Counter()
            for pid in common:
                sa, sb = side(outcomes[a][pid]), side(outcomes[b][pid])
                if sa != "tie" and sb != "tie":
                    c["both_side_same" if sa == sb else "both_side_opposite"] += 1
                elif sa != "tie":
                    c[f"{a}_side_{b}_tie"] += 1
                elif sb != "tie":
                    c[f"{b}_side_{a}_tie"] += 1
                else:
                    c["both_tie"] += 1
            # Cohen's kappa on the three outcomes x / y / tie
            labs = ["x", "y", "tie"]
            n = len(common)
            po = sum(side(outcomes[a][p]) == side(outcomes[b][p]) for p in common) / n if n else None
            pa = Counter(side(outcomes[a][p]) for p in common)
            pb = Counter(side(outcomes[b][p]) for p in common)
            pe = sum(pa[l] * pb[l] for l in labs) / n / n if n else None
            kappa = (po - pe) / (1 - pe) if n and pe < 1 else None
            agree[f"{a} | {b}"] = {"n": n, **dict(c), "kappa_3way": None if kappa is None else round(kappa, 3),
                                   "observed_agreement": None if po is None else round(po, 3)}
    entry["agreement"] = agree
    out[name] = entry

(HERE / "readers.json").write_text(json.dumps(out, indent=1))
for name, e in out.items():
    print("==", name, "x =", e["x"], "y =", e["y"])
    for src, s in e["sources"].items():
        t = s["overall"]
        print(f"  {src:34s} W/L/T {t['wins']}/{t['losses']}/{t['ties'] + t['both_bad']} (both_bad {t['both_bad']})"
              f" score {t['score']} {t['score_ci95']} decisive {t['win_rate_decisive']} p {t['sign_test_p']}"
              f" raw {s['raw_verdicts']} {s['diagnostics']}")
        print("      by_category", s["by_category"])
        print("      by_turns", s["by_turns"])
    for k, v in e["agreement"].items():
        print("  agree", k, v)
