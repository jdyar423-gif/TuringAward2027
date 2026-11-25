"""Fifth pass (the final paper): every number of the blind comparisons on the TEST split, recomputed
from the rating files with the repository's own key check and summary (`bes.eval.proxy.judge.load_verdicts`
with the key, `bes.eval.proxy.score.summarize`), checked against the reports in study/proxy/reports/ab/,
plus what the reports do not hold:

- the restriction to the 189 test prompts that no mid-run sample touched (the 11 SEEN ids of Section 7.2),
  from each summary's per_prompt_score, as lead4_work/scripts/restrict_report.py does;
- the raw verdicts of each Opus reading (A / B / tie / both_bad as the reader saw the sides) counted from
  the readers' chunk files N/final_eval/<group>/<set>/r1|r2/chunk_*.verdicts.json, checked against the
  order1 / order2 records of judge-opus.jsonl (tie and both_bad both map to C there), and each reading
  unblinded on its own (r2 saw the sides exchanged);
- pair-by-pair agreement of the Opus readers and the registered judge on the pairs both rated.

Read-only on the study directory; writes final_results.json next to lead5_work/.

    taskset -c 2,3 nice -n 5 S/venv/bin/python -I -B final_results.py
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
OUT = HERE.parent / "final_results.json"
SEEN = ("chat-002,chat-017,math-034,instruct-034,instruct-062,writing-100,code-026,math-001,code-002,"
        "writing-001,instruct-001").split(",")
assert len(SEEN) == 11
SETS = {
    "final-prefopt-vs-falcon": N / "final_eval/prefopt/final-prefopt-vs-falcon",
    "final-prefopt-vs-mobilellm": N / "final_eval/prefopt/final-prefopt-vs-mobilellm",
    "final-prefopt-vs-trained": N / "final_eval/prefopt/final-prefopt-vs-trained",
    "ref-falcon-vs-mobilellm-test": N / "final_eval/ref/ref-falcon-vs-mobilellm-test",
    # Falcon's registered control arms (K = 6, kept by its own log-probability / by our selector)
    "final-prefopt-vs-falcon-k6lp": N / "final_eval/controls/final-prefopt-vs-falcon-k6lp",
    "final-prefopt-vs-falcon-k6ds": N / "final_eval/controls/final-prefopt-vs-falcon-k6ds",
    # our K = 1 arm (one particle, every other setting the export's)
    "final-k1-vs-falcon": N / "final_eval/k1/final-k1-vs-falcon",
    "final-k1-vs-mobilellm": N / "final_eval/k1/final-k1-vs-mobilellm",
}
# paired comparisons of our per-prompt score (bes.eval.proxy.score.paired_sign_test, as the CLI's `compare`)
PAIRED = [("final-prefopt-vs-falcon", "final-prefopt-vs-falcon-k6lp"),
          ("final-prefopt-vs-falcon", "final-prefopt-vs-falcon-k6ds"),
          ("final-prefopt-vs-falcon", "final-k1-vs-falcon"),
          ("final-prefopt-vs-mobilellm", "final-k1-vs-mobilellm")]
# the same set read by two sources, paired by prompt: the first readings (pair-set names in the chunk titles)
# against the neutral-title replication (FACTS_2.md, 22:28 and 22:31 UTC)
PAIRED_SRC = [("final-prefopt-vs-falcon", "judge-opus", "judge-opus-neutral"),
              ("final-prefopt-vs-mobilellm", "judge-opus", "judge-opus-neutral")]
# the neutral replication's chunk directories (x1 = the Falcon set, x2 = the MobileLLM set; a as published,
# b with the sides exchanged)
NEUTRAL = {"final-prefopt-vs-falcon": N / "final_eval/neutral/x1", "final-prefopt-vs-mobilellm": N / "final_eval/neutral/x2"}
EXTRA = sys.argv[1:]          # further pair sets


def side(o):
    return {"win": "x", "loss": "y"}.get(o, "tie")


def restrict(per_prompt, exclude):
    """W / L / T, score, intervals and sign test on the prompts not excluded (1 win, 0.5 tie, 0 loss)."""
    m = {"1.0": "win", "0.5": "tie", "0.0": "loss"}
    outs = [m[repr(float(v))] for k, v in per_prompt.items() if k not in exclude]
    t = score.tally(outs)
    return {k: t[k] for k in ("n", "wins", "losses", "ties", "score", "score_ci95", "decisive",
                              "win_rate_decisive", "win_rate_decisive_ci95", "sign_test_p")}


def reading_counts(d, key, pairs_by_id, swapped):
    """Raw verdicts of one reading (labels as that reader saw them) and the reading unblinded."""
    raw, unbl, n_files, ids = Counter(), Counter(), 0, set()
    for f in sorted(d.glob("chunk_*.verdicts.json")):
        n_files += 1
        for pid, v in json.loads(f.read_text()).items():
            if pid not in pairs_by_id:
                raw["foreign"] += 1
                continue
            ids.add(pid)
            raw[v] += 1
            vb = {"A": "B", "B": "A"}.get(v, v) if swapped else v
            unbl[make_ab.unblind(vb, pid, key)] += 1
    return {"files": n_files, "pairs": len(ids), "raw": dict(raw), "unblinded": dict(unbl),
            "both_bad_share": round(raw["both_bad"] / len(ids), 4) if ids else None}


def both_bad_overlap(d1, d2):
    """Prompts (pairs) called both_bad by both readings and by at least one."""
    v1, v2 = {}, {}
    for d, v in ((d1, v1), (d2, v2)):
        for f in sorted(d.glob("chunk_*.verdicts.json")):
            v.update(json.loads(f.read_text()))
    common = set(v1) & set(v2)
    return {"pairs": len(common), "both": sum(v1[p] == "both_bad" and v2[p] == "both_bad" for p in common),
            "either": sum(v1[p] == "both_bad" or v2[p] == "both_bad" for p in common)}


def notes_counts(d, key, pairs_by_id, swapped):
    """How many notes there are per reading (the keyword study is left to a later pass)."""
    n = 0
    for f in sorted(d.glob("chunk_*.notes.json")):
        n += len(json.loads(f.read_text()))
    return n


out = {"seen": SEEN, "sets": {}}
names = list(SETS) + [e for e in EXTRA if e not in SETS]
for name in names:
    pairs, meta = make_ab.load_pairs(ROOT, name)
    key = make_ab.load_key(ROOT, name)
    by_id = {p["pair_id"]: p for p in pairs}
    entry = {"x": key["x"], "y": key["y"], "n_pairs": len(pairs),
             "multi_turn": sum(bool(p.get("multi_turn")) for p in pairs),
             "categories": dict(Counter(p["category"] for p in pairs)), "sources": {}}
    seen_present = sorted({p["prompt_id"] for p in pairs} & set(SEEN))
    entry["seen_present"] = seen_present
    outcomes = {}
    for f in sorted((ROOT / "ratings" / name).glob("*.jsonl")):
        src = f.stem
        verdicts = load_verdicts(ROOT, name, src, key=key)          # refuses on a commitment mismatch
        recs = [json.loads(line) for line in open(f)]
        summ = score.summarize(pairs, key, verdicts, recs)
        oc = {pid: make_ab.unblind(v, pid, key) for pid, v in verdicts.items()}
        outcomes[src] = oc
        rep_path = ROOT / "reports" / "ab" / f"{name}--{src}.json"
        rep = json.loads(rep_path.read_text()) if rep_path.exists() else None
        match = None
        if rep is not None:
            match = {k: rep.get(k) == summ.get(k) for k in ("overall", "by_category", "by_turns", "by_length", "judge")}
        raw = dict(Counter(verdicts.values()))
        recs_rated = [r for r in recs if r.get("pair_id") in verdicts]
        o1 = Counter(r["order1"]["verdict"] for r in recs_rated if "order1" in r)
        o2 = Counter(r["order2_swapped"]["verdict"] for r in recs_rated if "order2_swapped" in r)
        entry["sources"][src] = {
            "n_rated": summ["n_rated"], "overall": summ["overall"], "by_category": summ["by_category"],
            "by_turns": summ["by_turns"], "by_length": summ["by_length"], "judge": summ.get("judge"),
            "raw_combined_verdicts": raw, "order1_verdicts": dict(o1), "order2_swapped_verdicts": dict(o2),
            "restricted_189": restrict(summ["per_prompt_score"], set(SEEN)),
            "seen_rated": sorted(set(SEEN) & set(summ["per_prompt_score"])),
            "report_file": str(rep_path.relative_to(ROOT)) if rep is not None else None,
            "report_n_rated": None if rep is None else rep.get("n_rated"),
            "matches_report": match,
            "per_prompt_score": summ["per_prompt_score"],
        }
    # the Opus readings, chunk by chunk
    rd = SETS.get(name)
    if rd is not None and rd.exists():
        entry["readings"] = {r: reading_counts(rd / r, key, by_id, swapped=(r == "r2")) for r in ("r1", "r2")}
        entry["notes"] = {r: notes_counts(rd / r, key, by_id, r == "r2") for r in ("r1", "r2")}
        entry["both_bad_overlap"] = both_bad_overlap(rd / "r1", rd / "r2")
    nd = NEUTRAL.get(name)
    if nd is not None and nd.exists():
        entry["neutral_readings"] = {r: reading_counts(nd / r, key, by_id, swapped=(r == "b")) for r in ("a", "b")}
        entry["neutral_both_bad_overlap"] = both_bad_overlap(nd / "a", nd / "b")
        # the chunk files of the replication name no pair set in their first line
        entry["neutral_titles"] = sorted({(f.read_text().splitlines() or [""])[0] for r in ("a", "b")
                                          for f in (nd / r).glob("chunk_*.md")})
        # check: the chunk files give the order1 / order2 records of judge-opus (tie, both_bad -> C)
        if "judge-opus" in entry["sources"]:
            to_c = {"A": "A", "B": "B", "tie": "C", "both_bad": "C"}
            chk = {}
            for r, okey in (("r1", "order1_verdicts"), ("r2", "order2_swapped_verdicts")):
                c = Counter()
                for k, v in entry["readings"][r]["raw"].items():
                    if k in to_c:
                        c[to_c[k]] += v
                chk[r] = dict(c) == entry["sources"]["judge-opus"][okey]
            entry["readings_match_ratings"] = chk
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
            n = len(common)
            labs = ("x", "y", "tie")
            po = sum(side(outcomes[a][p]) == side(outcomes[b][p]) for p in common) / n if n else None
            pa = Counter(side(outcomes[a][p]) for p in common)
            pb = Counter(side(outcomes[b][p]) for p in common)
            pe = sum(pa[lab] * pb[lab] for lab in labs) / n / n if n else None
            kappa = (po - pe) / (1 - pe) if n and pe < 1 else None
            # the Opus tally restricted to the pairs the other source rated (like with like)
            sub = {s: score.tally([outcomes[s][p] for p in common]) for s in (a, b)}
            agree[f"{a} | {b}"] = {"n": n, **dict(c), "kappa_3way": None if kappa is None else round(kappa, 3),
                                   "observed_agreement": None if po is None else round(po, 3),
                                   "tally_on_common": {s: {k: v[k] for k in ("wins", "losses", "ties", "score",
                                                                            "score_ci95", "sign_test_p")}
                                                       for s, v in sub.items()}}
    entry["agreement"] = agree
    out["sets"][name] = entry

out["paired"] = {}
for a, b in PAIRED:
    sa = out["sets"].get(a, {}).get("sources", {}).get("judge-opus")
    sb = out["sets"].get(b, {}).get("sources", {}).get("judge-opus")
    if not sa or not sb or sa["n_rated"] < 200 or sb["n_rated"] < 200:
        out["paired"][f"{a} | {b}"] = {"skipped": "a set is not fully rated by judge-opus"}
        continue
    pa, pb = sa["per_prompt_score"], sb["per_prompt_score"]
    full = score.paired_sign_test(pa, pb)
    r189 = score.paired_sign_test({k: v for k, v in pa.items() if k not in SEEN}, {k: v for k, v in pb.items() if k not in SEEN})
    out["paired"][f"{a} | {b}"] = {"all": full, "189": r189}

out["paired_sources"] = {}
for name, a, b in PAIRED_SRC:
    srcs = out["sets"].get(name, {}).get("sources", {})
    if a not in srcs or b not in srcs or srcs[a]["n_rated"] < 200 or srcs[b]["n_rated"] < 200:
        out["paired_sources"][f"{name}: {a} | {b}"] = {"skipped": "a source is not fully rated"}
        continue
    pa, pb = srcs[a]["per_prompt_score"], srcs[b]["per_prompt_score"]
    same = sum(pa[k] == pb[k] for k in pa)
    out["paired_sources"][f"{name}: {a} | {b}"] = {"all": score.paired_sign_test(pa, pb), "same_outcome": same,
                                                   "n": len(pa)}

import datetime as _dt  # noqa: E402
qf = sorted((ROOT / "ratings").glob("final-prefopt-vs-*/judge-qwen3-4b-2507-3605803b.jsonl"))
out["qwen_snapshot_utc"] = _dt.datetime.now(_dt.timezone.utc).strftime("%H:%M UTC") + " (" + ", ".join(
    f"{q.parent.name}: {sum(1 for _ in open(q))} records, last written "
    f"{_dt.datetime.fromtimestamp(q.stat().st_mtime, _dt.timezone.utc):%H:%M:%S}" for q in qf) + ")"
OUT.write_text(json.dumps(out, indent=1))
for name, e in out["sets"].items():
    print("==", name, "x =", e["x"], "y =", e["y"], "n", e["n_pairs"], "two-turn", e["multi_turn"], "seen", len(e["seen_present"]))
    for src, s in e["sources"].items():
        t, r = s["overall"], s["restricted_189"]
        print(f"  {src:34s} rated {s['n_rated']}: W/L/T {t['wins']}/{t['losses']}/{t['ties'] + t['both_bad']} "
              f"score {t['score']} {t['score_ci95']} decisive {t['win_rate_decisive']} {t['win_rate_decisive_ci95']} "
              f"p {t['sign_test_p']}; judge {s['judge']}; matches report {s['matches_report']}")
        print(f"      189: n {r['n']} W/L/T {r['wins']}/{r['losses']}/{r['ties']} score {r['score']} {r['score_ci95']} "
              f"decisive {r['win_rate_decisive']} {r['win_rate_decisive_ci95']} p {r['sign_test_p']} (seen rated {len(s['seen_rated'])})")
        print("      by_category", {c: f"{v['wins']}/{v['losses']}/{v['ties'] + v['both_bad']} {v['score']} p {v['sign_test_p']}" for c, v in s["by_category"].items()})
        print("      by_turns", {c: f"{v['wins']}/{v['losses']}/{v['ties'] + v['both_bad']} {v['score']}" for c, v in s["by_turns"].items()})
        print("      by_length", {c: (f"n {v['n']} {v['wins']}/{v['losses']}/{v['ties'] + v['both_bad']} {v['score']} p {v['sign_test_p']}" if v.get('n') else 'n 0') for c, v in s["by_length"].items()})
        print("      combined raw", s["raw_combined_verdicts"], "order1", s["order1_verdicts"], "order2", s["order2_swapped_verdicts"])
    if "readings" in e:
        for r, v in e["readings"].items():
            print(f"  reading {r}: files {v['files']} pairs {v['pairs']} raw {v['raw']} unblinded {v['unblinded']} both_bad share {v['both_bad_share']}")
        print("  readings match the rating records:", e.get("readings_match_ratings"), "notes", e.get("notes"))
    for k, v in e["agreement"].items():
        print("  agree", k, v)
    if "both_bad_overlap" in e:
        print("  both_bad overlap of the two readings", e["both_bad_overlap"])
    if "neutral_readings" in e:
        for r, v in e["neutral_readings"].items():
            print(f"  neutral reading {r}: files {v['files']} pairs {v['pairs']} raw {v['raw']} unblinded {v['unblinded']}")
        print("  neutral both_bad overlap", e["neutral_both_bad_overlap"], "titles", e["neutral_titles"][:3])
for k, v in out["paired"].items():
    print("paired", k, json.dumps(v))
for k, v in out["paired_sources"].items():
    print("paired sources", k, json.dumps(v))
