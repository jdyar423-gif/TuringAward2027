"""Blind judging of a published pair set by readers who never see the key.

    blind_eval.py export --root PROXY --pairs NAME --out DIR [--chunk 20] [--swap] [--title-with-set]
        writes DIR/chunk_NN.md (the pairs as the human rating tool shows them, in a fixed
        shuffled order) and DIR/chunk_NN.ids.json (the pair ids of the chunk, in order).
        --swap exchanges replies A and B (the second reading of a pair set).
        The title of a chunk file names no system. (The readings of 24 November 2026 were made
        with titles that carried the pair set's name, e.g. "(set final-prefopt-vs-falcon)", in
        directories of the same name: the readers could not tell which side was which, but they
        could tell which two systems a set compared. --title-with-set reproduces those files;
        give DIR a neutral name as well. docs/CHANGES.md section 19.10.)
    blind_eval.py import2 --root PROXY --pairs NAME --source judge-<who> --order1 FILE... --order2 FILE...
        two readings of every pair, the second made on the --swap export: one record per pair in
        the automatic judge's format (a disagreement of the two readings is a tie)
    blind_eval.py import --root PROXY --pairs NAME --source judge-<who> --verdicts FILE [FILE ...]
        FILE: JSON {"pair_id": "A" | "B" | "tie" | "both_bad", ...}; appends rating records with
        the pair set's commitment and the pair's digest (what `score --source` verifies)

Only the blinded pair file is read (bes.eval.proxy.make_ab.load_pairs); the key stays unread.
"""
import argparse, json, random, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bes.eval.proxy import make_ab                      # noqa: E402
from bes.eval.proxy.common import append_jsonl, stable_hash  # noqa: E402
from bes.eval.proxy.judge import load_verdicts, ratings_path  # noqa: E402
from bes.eval.proxy.rate import GUIDE                     # noqa: E402

VERDICTS = ("A", "B", "tie", "both_bad")


def block(label, text):
    body = text if str(text).strip() else "(empty reply)"
    return f"----- {label} -----\n{body}\n"


def cmd_export(a):
    root = Path(a.root)
    pairs, _ = make_ab.load_pairs(root, a.pairs)
    order = list(pairs)
    random.Random(stable_hash(["blind-eval-order", a.pairs, bool(a.swap)])).shuffle(order)
    if a.swap:
        order = [dict(p, turns=[{"user": t["user"], "A": t["B"], "B": t["A"]} for t in p["turns"]]) for p in order]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for c in range(0, len(order), a.chunk):
        part = order[c:c + a.chunk]
        title = f"# Blind pairs {c + 1}-{c + len(part)} of {len(order)}" + (f" (set {a.pairs})" if a.title_with_set else "")
        lines = [title, "", GUIDE, "",
                 "Answer for every pair with one of: A (reply A is better), B (reply B is better), tie, "
                 "both_bad.", ""]
        for i, p in enumerate(part, c + 1):
            lines.append(f"## Pair {i}: id `{p['pair_id']}` ({p['category']}{', two turns' if p.get('multi_turn') else ''})")
            for t, turn in enumerate(p["turns"], 1):
                if len(p["turns"]) > 1:
                    lines.append(f"### turn {t}")
                lines += [block("USER", turn["user"]), block("REPLY A", turn["A"]), block("REPLY B", turn["B"])]
            lines.append("")
        (out / f"chunk_{n:02d}.md").write_text("\n".join(lines))
        (out / f"chunk_{n:02d}.ids.json").write_text(json.dumps([p["pair_id"] for p in part]))
        n += 1
    print(f"{len(order)} pairs in {n} chunks under {out}" + (" (replies A and B exchanged)" if a.swap else ""))


def cmd_import2(a):
    """Two readings of every pair, the second on exchanged sides (its labels are those it saw):
    the record of the automatic judge (bes.eval.proxy.judge.judge_pair), so that `score` reports
    how often the two orders agree and the preference for the first position."""
    from bes.eval.proxy.judge import combine

    root = Path(a.root)
    pairs, _ = make_ab.load_pairs(root, a.pairs)
    pub = make_ab.published(root, a.pairs)
    ids = {p["pair_id"] for p in pairs}
    to_c = {"A": "A", "B": "B", "tie": "C", "both_bad": "C"}
    first = {k: to_c[v] for f in a.order1 for k, v in json.loads(Path(f).read_text()).items() if k in ids and v in to_c}
    second = {k: to_c[v] for f in a.order2 for k, v in json.loads(Path(f).read_text()).items() if k in ids and v in to_c}
    path = ratings_path(root, a.pairs, a.source)
    done = load_verdicts(root, a.pairs, a.source, pub=pub)
    new = 0
    for pid in sorted(set(first) & set(second)):
        if pid in done:
            continue
        v1, v2 = first[pid], second[pid]
        hard = lambda v: {o: float(o == v) for o in ("A", "B", "C")}          # noqa: E731
        p1, p2 = hard(v1), hard(v2)
        rec = {"pair_id": pid, "verdict": combine(v1, v2),
               "order1": {"verdict": v1, "probs": p1, "mass": 1.0},
               "order2_swapped": {"verdict": v2, "probs": p2, "mass": 1.0},
               "agree": v1 == {"A": "B", "B": "A", "C": "C"}[v2],
               "soft": {"A": (p1["A"] + p2["B"]) / 2, "B": (p1["B"] + p2["A"]) / 2, "C": (p1["C"] + p2["C"]) / 2},
               "first_position_pref": ((p1["A"] - p1["B"]) + (p2["A"] - p2["B"])) / 2,
               "clipped": False, "judge": a.source, "time": round(time.time(), 1),
               "commitment": pub["commitment"], "pair_digest": pub["digests"][pid]}
        append_jsonl(path, [rec])
        new += 1
    print(f"{new} pairs written to {path}; {len(ids) - len(set(first) & set(second)) } pairs lack one of the two readings")


def cmd_import(a):
    root = Path(a.root)
    pairs, _ = make_ab.load_pairs(root, a.pairs)
    pub = make_ab.published(root, a.pairs)
    ids = {p["pair_id"] for p in pairs}
    done = load_verdicts(root, a.pairs, a.source, pub=pub)
    path = ratings_path(root, a.pairs, a.source)
    new = bad = 0
    for f in a.verdicts:
        data = json.loads(Path(f).read_text())
        for pid, v in data.items():
            if pid not in ids or v not in VERDICTS:
                bad += 1
                continue
            if done.get(pid) == v:
                continue
            append_jsonl(path, [{"pair_id": pid, "verdict": v, "rater": a.source, "time": round(time.time(), 1),
                                 "commitment": pub["commitment"], "pair_digest": pub["digests"][pid]}])
            done[pid] = v
            new += 1
    print(f"{new} verdicts written to {path} ({len(done)} of {len(ids)} pairs rated; {bad} entries refused)")


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
e = sub.add_parser("export"); e.add_argument("--root", required=True); e.add_argument("--pairs", required=True)
e.add_argument("--out", required=True); e.add_argument("--chunk", type=int, default=20)
e.add_argument("--swap", action="store_true", help="exchange replies A and B (the second reading)")
e.add_argument("--title-with-set", action="store_true", help="put the pair set's name into each chunk's title (not blind to which systems are compared)")
j = sub.add_parser("import2"); j.add_argument("--root", required=True); j.add_argument("--pairs", required=True)
j.add_argument("--source", required=True); j.add_argument("--order1", nargs="+", required=True)
j.add_argument("--order2", nargs="+", required=True)
i = sub.add_parser("import"); i.add_argument("--root", required=True); i.add_argument("--pairs", required=True)
i.add_argument("--source", required=True); i.add_argument("--verdicts", nargs="+", required=True)
a = ap.parse_args()
{"export": cmd_export, "import": cmd_import, "import2": cmd_import2}[a.cmd](a)
