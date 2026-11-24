"""Report of the graft experiment of the micro bench (``scripts/micro_ablate.py`` ``GRAFT_ARMS``).

Reads the finished records of every arm and seed from one or more run directories
(``results.jsonl`` in each directory and in each of its immediate sub-directories, which is the
layout of one ``--out`` per arm) or ``results.jsonl`` files, and prints

(a) per arm and checkpoint: the held-out loss of every domain (``mix`` = web and chat weighted
    by the training row shares, as in the E2 report), mean and standard deviation over seeds;
(b) paired by seed: ``v9_pages - ref`` at every checkpoint, ``graft_g - ref`` and
    ``graft_g - v9_pages`` at every checkpoint after the graft point (the last one included):
    mean, standard deviation of the paired differences and number of seeds;
(c) the share of the page gain a graft recovers at the last checkpoint,
    ``(ref - graft_g) / (ref - v9_pages)``: per seed, their mean, and the ratio of the mean
    differences (the steadier of the two when ``ref - v9_pages`` is small for one seed);

and writes the same numbers as JSON (``--json``). An arm's family is its name without a
``_s<seed>`` suffix (``ref_s2`` -> ``ref``), its seed is the record's; the newest finished record
per name is used, and all of them must have the same budget (or pass ``--tokens``).

    python scripts/micro_graft_report.py runs/graft --json runs/graft/graft_report.json
    python scripts/micro_graft_report.py runs/graft --merge runs/graft/results_all.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bes.eval import microbench as mb  # noqa: E402

DOMAINS = ("mix", "web", "chat", "assistant")
_SEED_SUFFIX = re.compile(r"^(?P<family>.+?)_s(?P<seed>\d+)$")


def family_of(name: str) -> str:
    """``ref_s2`` -> ``ref``, ``graft_50_s1`` -> ``graft_50``, ``v9_pages`` -> ``v9_pages``."""
    m = _SEED_SUFFIX.match(name)
    return m.group("family") if m else name


def collect(paths: Iterable[str | Path]) -> tuple[list[dict], list[str]]:
    """Records of every ``results.jsonl`` found (a file, a run directory, or the run directories
    directly below a directory) and the files read."""
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            if (p / mb.RESULTS).is_file():
                files.append(p / mb.RESULTS)
            files += sorted(q / mb.RESULTS for q in p.iterdir() if q.is_dir() and (q / mb.RESULTS).is_file())
        else:
            raise SystemExit(f"{p}: no such file or directory")
    records: list[dict] = []
    for f in files:
        records += mb.read_jsonl(f)
    return records, [str(f) for f in files]


def finished(records: Iterable[dict], tokens: int = 0) -> dict[str, dict]:
    """The newest finished arm record per name (``finished`` time), optionally of one budget."""
    out: dict[str, dict] = {}
    for r in records:
        if r.get("type") != "arm" or r.get("status") != "done":
            continue
        if tokens and int(r["tokens"]) != int(tokens):
            continue
        prev = out.get(r["name"])
        if prev is None or float(r.get("finished", 0.0)) >= float(prev.get("finished", 0.0)):
            out[r["name"]] = r
    return out


def losses(rec: dict) -> dict[int, dict[str, float]]:
    """``{tokens: {domain: held-out loss}}`` of a record, ``mix`` included."""
    out = {}
    for c in rec.get("curve", []):
        ev = c["eval"]
        row = {d: float(v["loss"]) for d, v in ev.items()}
        shares = mb.domain_shares(rec, [d for d in ev if d != "assistant"])
        if shares and all(d in row for d in shares):
            row["mix"] = sum(pi * row[d] for d, pi in shares.items())
        out[int(c["tokens"])] = row
    return out


def stats(xs: Sequence[float]) -> dict[str, Any]:
    """Mean, sample standard deviation (``None`` below two values) and count."""
    n = len(xs)
    if n == 0:
        return {"mean": None, "sd": None, "n": 0}
    mean = sum(xs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1)) if n > 1 else None
    return {"mean": mean, "sd": sd, "n": n}


def summarize(records: Iterable[dict], *, ref: str = "ref", target: str = "v9_pages",
              domains: Sequence[str] = DOMAINS, tokens: int = 0) -> dict:
    """The report's numbers (module docstring) as a JSON-ready dict."""
    done = finished(records, tokens)
    if not done:
        raise ValueError("no finished arm records")
    budgets = sorted({int(r["tokens"]) for r in done.values()})
    if len(budgets) > 1:
        raise ValueError(f"records of several budgets {budgets}: choose one with --tokens")
    fam: dict[str, dict[int, dict]] = {}
    for name, r in sorted(done.items()):
        f, s = family_of(name), int(r["arm"]["seed"])
        if s in fam.setdefault(f, {}):
            raise ValueError(f"family {f!r} has two records with seed {s}: {fam[f][s]['name']!r} and {name!r}")
        fam[f][s] = r
    for need in (ref, target):
        if need not in fam:
            raise ValueError(f"no finished {need!r} arm (families: {sorted(fam)})")
    grafts = sorted((f for f, rs in fam.items() if all(r["arm"].get("kind") == "graft" for r in rs.values())),
                    key=lambda f: (min(float(r["arm"]["graft"]["frac"]) for r in fam[f].values()), f))
    order = [ref, target] + grafts + sorted(f for f in fam if f not in {ref, target, *grafts})
    L = {f: {s: losses(r) for s, r in rs.items()} for f, rs in fam.items()}
    points = sorted({t for f in L for s in L[f] for t in L[f][s]})
    last = points[-1]
    graft_tokens = {}
    for f in grafts:
        ts = {int(r["graft"]["tokens"]) for r in fam[f].values()}
        if len(ts) != 1:
            raise ValueError(f"the records of {f!r} graft at different points: {sorted(ts)}")
        graft_tokens[f] = ts.pop()

    def value(f, s, t, d):
        return L[f].get(s, {}).get(t, {}).get(d)

    table = []
    for f in order:
        seeds = sorted(fam[f])
        for t in points:
            cells = {d: stats([v for v in (value(f, s, t, d) for s in seeds) if v is not None]) for d in domains}
            if any(c["n"] for c in cells.values()):
                table.append({"arm": f, "tokens": t, "seeds": seeds, **cells})

    def paired(a: str, b: str, ts: Sequence[int]) -> list[dict]:
        seeds = sorted(set(fam[a]) & set(fam[b]))
        rows = []
        for t in ts:
            row: dict[str, Any] = {"comparison": f"{a} - {b}", "a": a, "b": b, "tokens": t}
            for d in domains:
                per = {s: value(a, s, t, d) - value(b, s, t, d) for s in seeds
                       if value(a, s, t, d) is not None and value(b, s, t, d) is not None}
                rel = [per[s] / value(b, s, t, d) for s in per]
                row[d] = {**stats(list(per.values())), "per_seed": {str(s): v for s, v in per.items()},
                          "rel_mean": (sum(rel) / len(rel)) if rel else None}
            rows.append(row)
        return rows

    pairs = paired(target, ref, points)
    for f in grafts:
        after = [t for t in points if t > graft_tokens[f]]
        pairs += paired(f, ref, after) + paired(f, target, after)

    recovered = []
    for f in grafts:
        seeds = sorted(set(fam[f]) & set(fam[ref]) & set(fam[target]))
        row: dict[str, Any] = {"arm": f, "tokens": last, "seeds": seeds}
        for d in domains:
            num, den, per = [], [], {}
            for s in seeds:
                lr, lg, lv = value(ref, s, last, d), value(f, s, last, d), value(target, s, last, d)
                if None in (lr, lg, lv):
                    continue
                num.append(lr - lg)
                den.append(lr - lv)
                per[str(s)] = (lr - lg) / (lr - lv) if lr != lv else None
            shares = [v for v in per.values() if v is not None]
            ratio = (sum(num) / sum(den)) if den and sum(den) != 0 else None
            row[d] = {"per_seed": per, **stats(shares), "ratio_of_means": ratio,
                      "ref_minus_graft": stats(num), "ref_minus_target": stats(den)}
        recovered.append(row)

    meta = {}
    for f in order:
        rs = fam[f]
        info: dict[str, Any] = {"names": sorted(r["name"] for r in rs.values()), "seeds": sorted(rs),
                                "spec_hashes": {str(s): r["spec_hash"] for s, r in sorted(rs.items())}}
        if f in graft_tokens:
            g = {str(s): r["graft"] for s, r in sorted(rs.items())}
            info["graft"] = {"tokens": graft_tokens[f], "step": next(iter(g.values()))["step"],
                             "frac": next(iter(g.values()))["frac"],
                             "trunk": {s: v["trunk"]["source"] for s, v in g.items()},
                             "check": {s: {k: v["check"][k] for k in ("loss_before", "loss_after", "difference", "ok")}
                                       for s, v in g.items()},
                             "max_abs_check_difference": max(abs(v["check"]["difference"]) for v in g.values())}
        meta[f] = info
    any_rec = next(iter(done.values()))
    return {"budget": budgets[0], "steps": int(any_rec.get("steps", 0)), "checkpoints": points,
            "last": last, "ref": ref, "target": target, "grafts": grafts, "domains": list(domains),
            "families": meta, "table": table, "paired": pairs, "recovered": recovered}


# ---------------------------------------------------------------------------- text
def _md(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def _ms(c: dict, digits: int = 4, signed: bool = False) -> str:
    if not c or c.get("mean") is None:
        return "n/a"
    fmt = f"{{:{'+' if signed else ''}.{digits}f}}"
    sd = "" if c.get("sd") is None else " ± " + f"{{:.{digits}f}}".format(c["sd"])
    return fmt.format(c["mean"]) + sd + f" ({c['n']})"


def render(rep: dict, inputs: Sequence[str] = ()) -> str:
    doms = rep["domains"]
    fams = rep["families"]
    out = [f"Graft experiment of the micro bench: budget {rep['budget']:,} tokens ({rep['steps']} steps), "
           f"checkpoints {', '.join(f'{t:,}' for t in rep['checkpoints'])} tokens.",
           "Seeds per arm: " + ", ".join(f"{f} {fams[f]['seeds']}" for f in fams) + "."]
    if inputs:
        out.append("Records: " + ", ".join(inputs) + ".")
    for f in rep["grafts"]:
        g = fams[f]["graft"]
        out.append(f"{f}: grafted at step {g['step']} ({g['tokens']:,} tokens, {g['frac']:g} of the budget); "
                   "trunk " + ", ".join(f"seed {s} {v}" for s, v in g["trunk"].items())
                   + f"; largest |loss after - loss before| at the graft {g['max_abs_check_difference']:.2e}.")
    out += ["", "(a) Held-out loss (nats per token): mean ± standard deviation over seeds (number of seeds).", "",
            _md(["arm", "tokens"] + doms, [[r["arm"], f"{r['tokens']:,}"] + [_ms(r[d]) for d in doms]
                                           for r in rep["table"]]), "",
            "(b) Paired differences by seed (nats per token; negative: the first arm has the lower loss): mean ± "
            "standard deviation of the per-seed differences (number of seeds); [mean relative difference].", ""]
    rows = []
    for r in rep["paired"]:
        rows.append([r["comparison"], f"{r['tokens']:,}" + (" (last)" if r["tokens"] == rep["last"] else "")]
                    + [_ms(r[d], signed=True) + ("" if r[d].get("rel_mean") is None
                                                 else f" [{100 * r[d]['rel_mean']:+.2f}%]") for d in doms])
    out += [_md(["comparison", "tokens"] + doms, rows), "",
            f"(c) Share of the page gain recovered at {rep['last']:,} tokens: (ref - graft) / (ref - v9_pages); "
            "per seed, mean ± standard deviation (number of seeds), and the ratio of the mean differences.", ""]
    rows = []
    for r in rep["recovered"]:
        for d in doms:
            c = r[d]
            per = ", ".join(f"{s}: {'n/a' if v is None else f'{v:.3f}'}" for s, v in c["per_seed"].items())
            rows.append([r["arm"], d, per or "n/a", _ms(c, 3),
                         "n/a" if c["ratio_of_means"] is None else f"{c['ratio_of_means']:.3f}",
                         _ms(c["ref_minus_graft"], signed=True), _ms(c["ref_minus_target"], signed=True)])
    out.append(_md(["graft", "domain", "per seed", "mean ± sd (n)", "ratio of means", "ref - graft",
                    "ref - v9_pages"], rows))
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------- main
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories (and their sub-directories) or results.jsonl files")
    ap.add_argument("--json", default="", help="write the report's numbers as JSON to this file")
    ap.add_argument("--merge", default="", help="write the finished records used, one per line, to this file")
    ap.add_argument("--ref", default="ref", help="family of the reference arms (default ref)")
    ap.add_argument("--target", default="v9_pages", help="family of the target arms (default v9_pages)")
    ap.add_argument("--tokens", type=int, default=0, help="only records of this budget")
    ap.add_argument("--domains", default=",".join(DOMAINS), help="comma-separated (mix = share-weighted web/chat)")
    args = ap.parse_args(argv)
    records, files = collect(args.runs)
    try:
        rep = summarize(records, ref=args.ref, target=args.target, tokens=args.tokens,
                        domains=[d for d in args.domains.split(",") if d])
    except ValueError as e:
        raise SystemExit(f"micro_graft_report: {e}")
    rep["inputs"] = files
    sys.stdout.write(render(rep, files))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rep, indent=1, sort_keys=True))
        print(f"wrote {args.json}")
    if args.merge:
        done = finished(records, args.tokens)
        Path(args.merge).parent.mkdir(parents=True, exist_ok=True)
        with open(args.merge, "w") as fh:
            for name in sorted(done):
                fh.write(json.dumps(done[name], sort_keys=True) + "\n")
        print(f"wrote {args.merge} ({len(done)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
