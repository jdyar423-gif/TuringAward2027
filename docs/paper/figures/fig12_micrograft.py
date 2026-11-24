"""F12: graft or the page from the first step, at the micro width (R/docs/MICRO_GRAFT.md).

(a) The held-out loss of each arm minus that of ref with the same seed, against training tokens: the mean
of the three seeds (line) and each seed (small marks beside it). A graft arm is ref up to its graft (its
evaluations equal ref's exactly), and the graft leaves the held-out loss unchanged (the record's check),
so its line starts on the zero line at its graft point. (b) The share of the page gain recovered at the
end, (ref - graft) / (ref - v9_pages), against the graft point: the ratio of the mean differences and each
seed's share; at 0 (the page from the first step) and 1 (never) the share is 1 and 0 by definition.

Data: the 15 finished records of the experiment (paths.MICRO_GRAFT_RECORDS: ref, v9_pages, graft_25,
graft_50, graft_75, seeds 0-2, budget 6,000,000 tokens). Every number is recomputed from the records;
"mix" is the held-out loss of web and chat weighted by the record's training row shares (arm.mixture,
normalised over the evaluated domains other than assistant), as scripts/micro_graft_report.py and
bes.eval.microbench.domain_shares define it. The report written next to the records
(paths.MICRO_GRAFT_REPORT) is only compared with: every number it holds is recomputed, and a difference
prints a line starting with MISMATCH. When the restart arms exist (paths.MICRO_GRAFT_3M: v9_pages from
scratch on a budget of 3,000,000 tokens, seeds 0-2), the archive is unpacked into data/_src/micro_graft_3M
and (a) gets one mark per seed and their mean at their own end, minus ref's loss at the end of the 6M
budget; without it the script says that they are left out. Nothing here depends on the live run or the
cut-off.

Writes data/micro_graft_curve.csv and data/micro_graft_summary.csv (the table view; the figure is drawn
from the summary as read back from disk), then F12_micrograft.svg/png.
"""
import csv
import json
import math
import os
import re
import shutil
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P  # noqa: E402
import style as st  # noqa: E402
import common as C  # noqa: E402

st.setup()
plt = st.plt
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402

REF, TARGET = "ref", "v9_pages"
DOMAINS = ("mix", "web", "chat", "assistant")
RESTART_BUDGET = 3_000_000
SEED_SUFFIX = re.compile(r"^(?P<family>.+?)_s(?P<seed>\d+)$")
REF_COL = st.INK                # the zero line (ref): darker than every step of the page ramp, in grey too
MISMATCHES = []


def mismatch(msg):
    MISMATCHES.append(msg)
    print("MISMATCH (F12) " + msg)


def family_of(name):
    m = SEED_SUFFIX.match(name)
    return m.group("family") if m else name


def read_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except ValueError:          # a torn last line of a file still being written
                    continue
    return out


def finished(records, budget=None):
    """The newest finished arm record per name, as scripts/micro_graft_report.py chooses them."""
    out = {}
    for r in records:
        if r.get("type") != "arm" or r.get("status") != "done":
            continue
        if budget is not None and int(r["tokens"]) != int(budget):
            continue
        prev = out.get(r["name"])
        if prev is None or float(r.get("finished", 0.0)) >= float(prev.get("finished", 0.0)):
            out[r["name"]] = r
    return out


def shares(rec, domains):
    """Weights of mix: the record's training row shares (arm.mixture) of the given domains, normalised."""
    mix = {d: s for d, s in rec["arm"]["mixture"]}
    w = {d: float(mix.get(d, 0.0)) for d in domains if d in mix}
    s = sum(w.values())
    return {d: v / s for d, v in w.items()} if s > 0 else {}


def with_mix(rec, row):
    sh = shares(rec, [d for d in row if d != "assistant"])
    if sh and all(d in row for d in sh):
        row["mix"] = sum(pi * row[d] for d, pi in sh.items())
    return row


def losses(rec):
    """{tokens: {domain: held-out loss}} of one record, mix included."""
    return {int(c["tokens"]): with_mix(rec, {d: float(v["loss"]) for d, v in c["eval"].items()})
            for c in rec["curve"]}


def stats(xs):
    """Mean, sample standard deviation (None below two values) and count, as in the report."""
    n = len(xs)
    if n == 0:
        return {"mean": None, "sd": None, "n": 0}
    mean = sum(xs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1)) if n > 1 else None
    return {"mean": mean, "sd": sd, "n": n}


def write_csv(name, rows, cols):
    path = os.path.join(P.DATA, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    print(f"wrote {path} ({len(rows)} rows)")


def f_or_none(x):
    return None if x in ("", None) else float(x)


# ------------------------------------------------------------------------------------ the 6M records
def load():
    recs = read_jsonl(P.MICRO_GRAFT_RECORDS)
    done = finished(recs)
    if len(done) != len(recs):
        print(f"F12: {len(recs) - len(done)} of the {len(recs)} records in {P.MICRO_GRAFT_RECORDS} are not finished "
              "arm records or are older records of a name; they are not used")
    budgets = sorted({int(r["tokens"]) for r in done.values()})
    if len(budgets) != 1:
        sys.exit(f"fig12_micrograft.py: records of several budgets {budgets} in {P.MICRO_GRAFT_RECORDS}")
    fam = {}
    for name, r in sorted(done.items()):
        f, s = family_of(name), int(r["arm"]["seed"])
        if s in fam.setdefault(f, {}):
            sys.exit(f"fig12_micrograft.py: family {f} has two records with seed {s}")
        fam[f][s] = r
    for need in (REF, TARGET):
        if need not in fam:
            sys.exit(f"fig12_micrograft.py: no finished {need} arm in {P.MICRO_GRAFT_RECORDS}")
    grafts = sorted((f for f, rs in fam.items() if all(r["arm"].get("kind") == "graft" for r in rs.values())),
                    key=lambda f: (min(float(r["arm"]["graft"]["frac"]) for r in fam[f].values()), f))
    if len(grafts) + 1 > len(st.PAGE_RAMP):
        sys.exit(f"fig12_micrograft.py: {len(grafts)} graft families; the page ramp has steps for "
                 f"{len(st.PAGE_RAMP) - 1}")
    others = sorted(set(fam) - {REF, TARGET, *grafts})
    if others:
        print(f"F12: families not drawn: {', '.join(others)}")
    L = {f: {s: losses(r) for s, r in rs.items()} for f, rs in fam.items()}
    points = sorted({t for f in L for s in L[f] for t in L[f][s]})
    g_tok, g_frac, g_step = {}, {}, {}
    for f in grafts:
        ts = {int(r["graft"]["tokens"]) for r in fam[f].values()}
        if len(ts) != 1:
            sys.exit(f"fig12_micrograft.py: the records of {f} graft at different points {sorted(ts)}")
        g_tok[f] = ts.pop()
        g_frac[f] = float(next(iter(fam[f].values()))["graft"]["frac"])
        g_step[f] = int(next(iter(fam[f].values()))["graft"]["step"])
    return dict(done=done, fam=fam, grafts=grafts, L=L, points=points, budget=budgets[0], g_tok=g_tok,
                g_frac=g_frac, g_step=g_step)


def same_as_arm_files(done):
    """results_all.jsonl is the merge the report wrote; compare it with each arm's own results.jsonl next to it
    (the newest finished record per name), where those exist."""
    base = os.path.dirname(P.MICRO_GRAFT_RECORDS)
    same, differ, absent = 0, [], []
    for name, r in sorted(done.items()):
        path = os.path.join(base, name, "results.jsonl")
        if not os.path.exists(path):
            absent.append(name)
            continue
        own = finished(read_jsonl(path)).get(name)
        if own == r:
            same += 1
        else:
            differ.append(name)
    print(f"F12: {os.path.basename(P.MICRO_GRAFT_RECORDS)} against each arm's own results.jsonl: {same} records "
          f"equal" + (f", differ: {', '.join(differ)}" if differ else "") +
          (f", no own file: {', '.join(absent)}" if absent else ""))
    for name in differ:
        mismatch(f"the record of {name} in {P.MICRO_GRAFT_RECORDS} differs from the newest finished record in "
                 f"{os.path.join(base, name, 'results.jsonl')}")


def check_grafts(D):
    """A graft arm before its graft is ref: its evaluations (loss and every row) equal ref's of the same seed
    exactly; the graft leaves the held-out loss unchanged (before = ref's loss at the graft, after = before)."""
    fam, L = D["fam"], D["L"]
    n_eq, n_pts = 0, 0
    for f in D["grafts"]:
        for s, r in sorted(fam[f].items()):
            ref = {int(c["tokens"]): c for c in fam[REF][s]["curve"]}
            g = D["g_tok"][f]
            for c in r["curve"]:
                t = int(c["tokens"])
                if t > g:
                    continue
                n_pts += 1
                same = c["eval"].keys() == ref[t]["eval"].keys() and all(
                    c["eval"][d]["loss"] == ref[t]["eval"][d]["loss"] and c["eval"][d]["ce"] == ref[t]["eval"][d]["ce"]
                    for d in c["eval"])
                n_eq += same
                if not same:
                    mismatch(f"{r['name']}: evaluation at {t:,} tokens (before its graft) differs from {fam[REF][s]['name']}'s")
            h = r["graft"]["heldout"]
            for d, v in h.items():
                if v["before"] != ref[g]["eval"][d]["loss"]:
                    mismatch(f"{r['name']}: held-out {d} before the graft {v['before']!r} != ref's at {g:,} "
                             f"{ref[g]['eval'][d]['loss']!r}")
                if v["after"] != v["before"] or v["difference"] != 0 or v["max_row_difference"] != 0:
                    mismatch(f"{r['name']}: held-out {d} changed at the graft ({v['before']!r} -> {v['after']!r}, "
                             f"largest row difference {v['max_row_difference']!r})")
            ck = r["graft"]["check"]
            if not ck["ok"] or ck["difference"] != 0:
                mismatch(f"{r['name']}: function check at the graft: difference {ck['difference']!r}, ok {ck['ok']}")
            lt = L[f][s][g]["mix"] - L[REF][s][g]["mix"]
            if lt != 0:
                mismatch(f"{r['name']}: mix at its graft differs from ref's by {lt!r}")
    print(f"F12: before and at their graft, {n_eq} of {n_pts} evaluations of the graft arms equal ref's of the same "
          "seed exactly (loss and every row); held-out loss after each graft = before, row by row "
          f"({sum(len(D['fam'][f]) for f in D['grafts'])} grafts)")


# ------------------------------------------------------------------------------------ the report's numbers
def recompute_report(D):
    """The numbers of scripts/micro_graft_report.py's JSON (table, paired, recovered), recomputed here."""
    fam, L, grafts, points = D["fam"], D["L"], D["grafts"], D["points"]
    last = points[-1]

    def value(f, s, t, d):
        return L[f].get(s, {}).get(t, {}).get(d)

    table = {}
    for f in [REF, TARGET] + grafts:
        for t in points:
            for d in DOMAINS:
                table[(f, t, d)] = stats([v for v in (value(f, s, t, d) for s in sorted(fam[f])) if v is not None])
    paired = {}

    def pair(a, b, ts):
        seeds = sorted(set(fam[a]) & set(fam[b]))
        for t in ts:
            for d in DOMAINS:
                per = {s: value(a, s, t, d) - value(b, s, t, d) for s in seeds
                       if value(a, s, t, d) is not None and value(b, s, t, d) is not None}
                rel = [per[s] / value(b, s, t, d) for s in per]
                paired[(f"{a} - {b}", t, d)] = {**stats(list(per.values())),
                                                "per_seed": {str(s): v for s, v in per.items()},
                                                "rel_mean": (sum(rel) / len(rel)) if rel else None}

    pair(TARGET, REF, points)
    for f in grafts:
        after = [t for t in points if t > D["g_tok"][f]]
        pair(f, REF, after)
        pair(f, TARGET, after)
    recovered = {}
    for f in grafts:
        seeds = sorted(set(fam[f]) & set(fam[REF]) & set(fam[TARGET]))
        for d in DOMAINS:
            num, den, per = [], [], {}
            for s in seeds:
                lr, lg, lv = value(REF, s, last, d), value(f, s, last, d), value(TARGET, s, last, d)
                if None in (lr, lg, lv):
                    continue
                num.append(lr - lg)
                den.append(lr - lv)
                per[str(s)] = (lr - lg) / (lr - lv) if lr != lv else None
            sh = [v for v in per.values() if v is not None]
            recovered[(f, d)] = {"per_seed": per, **stats(sh),
                                 "ratio_of_means": (sum(num) / sum(den)) if den and sum(den) != 0 else None,
                                 "ref_minus_graft": stats(num), "ref_minus_target": stats(den)}
    return table, paired, recovered


def compare(mine, theirs, where, acc):
    """Every number of `theirs` against `mine` (same structure); counts and the largest difference in acc."""
    if isinstance(theirs, dict):
        if not isinstance(mine, dict) or set(mine) != set(theirs):
            mismatch(f"graft_report.json {where}: keys {sorted(theirs) if isinstance(theirs, dict) else theirs} "
                     f"against {sorted(mine) if isinstance(mine, dict) else mine}")
            return
        for k in theirs:
            compare(mine[k], theirs[k], f"{where}.{k}", acc)
    elif isinstance(theirs, list):
        if not isinstance(mine, list) or len(mine) != len(theirs):
            mismatch(f"graft_report.json {where}: {theirs} against {mine}")
            return
        for i, (a, b) in enumerate(zip(mine, theirs)):
            compare(a, b, f"{where}[{i}]", acc)
    elif isinstance(theirs, bool) or isinstance(theirs, str) or theirs is None:
        acc["other"] += 1
        if mine != theirs:
            mismatch(f"graft_report.json {where}: {theirs!r}, recomputed {mine!r}")
    else:
        acc["numbers"] += 1
        if mine is None:
            mismatch(f"graft_report.json {where}: {theirs!r}, recomputed: none")
            return
        d = abs(float(mine) - float(theirs))
        if d > acc["largest"][0]:
            acc["largest"] = (d, where)
        if d > 1e-12:
            mismatch(f"graft_report.json {where}: {theirs!r}, recomputed {mine!r} (difference {d:.3g})")


def check_report(D, table, paired, recovered):
    """Compare every number of the report written next to the records with the recomputed ones."""
    path = P.MICRO_GRAFT_REPORT
    if not os.path.exists(path):
        print(f"F12: {path} not found; nothing to compare with")
        return {}
    rep = json.load(open(path))
    acc = {"numbers": 0, "other": 0, "largest": (0.0, "")}
    fam = D["fam"]
    head = {"budget": D["budget"], "checkpoints": D["points"], "last": D["points"][-1], "ref": REF, "target": TARGET,
            "grafts": D["grafts"], "domains": list(DOMAINS),
            "steps": int(next(iter(D["done"].values())).get("steps", 0))}
    for k, v in head.items():
        compare(v, rep.get(k), k, acc)
    for i, row in enumerate(rep.get("table", [])):
        mine = {"arm": row["arm"], "tokens": row["tokens"], "seeds": sorted(fam[row["arm"]])}
        mine.update({d: table[(row["arm"], row["tokens"], d)] for d in DOMAINS})
        compare(mine, row, f"table[{i}]", acc)
    if len(rep.get("table", [])) != len(table) // len(DOMAINS):
        mismatch(f"graft_report.json table: {len(rep.get('table', []))} rows, recomputed {len(table) // len(DOMAINS)}")
    for i, row in enumerate(rep.get("paired", [])):
        mine = {"comparison": row["comparison"], "a": row["a"], "b": row["b"], "tokens": row["tokens"]}
        mine.update({d: paired.get((row["comparison"], row["tokens"], d)) for d in DOMAINS})
        compare(mine, row, f"paired[{i}] ({row['comparison']}, {row['tokens']:,})", acc)
    if len(rep.get("paired", [])) != len(paired) // len(DOMAINS):
        mismatch(f"graft_report.json paired: {len(rep.get('paired', []))} rows, recomputed {len(paired) // len(DOMAINS)}")
    for i, row in enumerate(rep.get("recovered", [])):
        mine = {"arm": row["arm"], "tokens": D["points"][-1], "seeds": sorted(fam[row["arm"]])}
        mine.update({d: recovered[(row["arm"], d)] for d in DOMAINS})
        compare(mine, row, f"recovered[{i}] ({row['arm']})", acc)
    for f, info in rep.get("families", {}).items():
        rs = fam[f]
        mine = {"names": sorted(r["name"] for r in rs.values()), "seeds": sorted(rs),
                "spec_hashes": {str(s): r["spec_hash"] for s, r in sorted(rs.items())}}
        if f in D["g_tok"]:
            g = {str(s): r["graft"] for s, r in sorted(rs.items())}
            mine["graft"] = {"tokens": D["g_tok"][f], "step": D["g_step"][f], "frac": D["g_frac"][f],
                             "trunk": {s: v["trunk"]["source"] for s, v in g.items()},
                             "check": {s: {k: v["check"][k] for k in ("loss_before", "loss_after", "difference", "ok")}
                                       for s, v in g.items()},
                             "max_abs_check_difference": max(abs(v["check"]["difference"]) for v in g.values())}
        compare(mine, info, f"families.{f}", acc)
    big, where = acc["largest"]
    print(f"F12: {os.path.basename(path)}: {acc['numbers']} numbers and {acc['other']} names or flags compared with "
          f"the recomputation; largest difference {big:.3g}" + (f" ({where})" if big else " (all identical)") +
          (f"; {len(MISMATCHES)} MISMATCH line(s)" if MISMATCHES else "; no mismatch"))
    return rep


# ------------------------------------------------------------------------------------ restart arms (3M)
def restart(D):
    """The restart arms (v9_pages from scratch, budget RESTART_BUDGET) from the 3M archive, if it exists:
    (records by seed, note)."""
    tar = P.MICRO_GRAFT_3M
    if not os.path.exists(tar):
        return None, f"{tar} does not exist"
    new = P.SRC_MICRO_3M + ".new"
    shutil.rmtree(new, ignore_errors=True)
    os.makedirs(new)
    try:
        with tarfile.open(tar, "r:gz") as tf:
            members = [m for m in tf.getmembers()
                       if m.isfile() and os.path.basename(m.name) in ("results.jsonl", "results_all.jsonl", "progress.log")]
            if sum(m.size for m in members) > 100 * 2 ** 20:
                raise ValueError("more than 100 MB of record files")
            tf.extractall(new, members=members, filter="data")     # no links, devices or paths outside `new`
    except (tarfile.TarError, OSError, ValueError, EOFError) as e:
        shutil.rmtree(new, ignore_errors=True)
        return None, f"{tar} could not be read ({type(e).__name__}: {e}; still being written?)"
    shutil.rmtree(P.SRC_MICRO_3M, ignore_errors=True)
    os.rename(new, P.SRC_MICRO_3M)
    recs = []
    for root, _dirs, names in sorted(os.walk(P.SRC_MICRO_3M)):
        for n in sorted(names):
            if n.endswith(".jsonl"):
                recs += read_jsonl(os.path.join(root, n))
    done = {n: r for n, r in finished(recs, RESTART_BUDGET).items() if family_of(n) == TARGET}
    v9 = next(iter(D["fam"][TARGET].values()))
    by_seed, notes = {}, []
    for n, r in sorted(done.items()):
        if r["arm"].get("overrides") != v9["arm"]["overrides"] or r["arm"].get("mixture") != v9["arm"]["mixture"]:
            notes.append(f"{n}: configuration differs from the 6M {TARGET} arm ({r['arm'].get('overrides')}); not used")
            continue
        by_seed[int(r["arm"]["seed"])] = r
    want = sorted(D["fam"][REF])
    have = sorted(by_seed)
    if have != want:
        return None, (f"{tar}: finished restart records (family {TARGET}, budget {RESTART_BUDGET:,}) for seeds {have}, "
                      f"need {want}" + (f" ({'; '.join(notes)})" if notes else "") + "; left out")
    return by_seed, "; ".join(notes)


# ------------------------------------------------------------------------------------ the CSVs
def build():
    D = load()
    fam, L, grafts, points = D["fam"], D["L"], D["grafts"], D["points"]
    last = points[-1]
    seeds = sorted(fam[REF])
    for f in [TARGET] + grafts:
        if sorted(fam[f]) != seeds:
            sys.exit(f"fig12_micrograft.py: {f} has seeds {sorted(fam[f])}, ref {seeds}")
    print(f"F12: {len(D['done'])} finished records, budget {D['budget']:,} tokens ({next(iter(D['done'].values()))['steps']}"
          f" steps); checkpoints {', '.join(f'{t:,}' for t in points)}; seeds {seeds}; families "
          + ", ".join(f"{f} ({len(fam[f])})" for f in [REF, TARGET] + grafts))
    same_as_arm_files(D["done"])
    check_grafts(D)
    table, paired, recovered = recompute_report(D)
    rep = check_report(D, table, paired, recovered)
    rep_pair = {(r["comparison"], r["tokens"]): r for r in rep.get("paired", [])}
    rep_rec = {r["arm"]: r for r in rep.get("recovered", [])}

    def src(r):
        return f"{P.rel(P.MICRO_GRAFT_RECORDS)} {r['name']}"

    # ---- one row per record and checkpoint (and the held-out loss right after each graft)
    curve = []
    for f in [REF, TARGET] + grafts:
        for s in seeds:
            r = fam[f][s]
            for c in r["curve"]:
                t = int(c["tokens"])
                row = L[f][s][t]
                stage = {REF: "ref", TARGET: "page from the first step"}.get(f, c.get("phase"))
                curve.append({"arm": f, "name": r["name"], "seed": s, "budget": r["tokens"], "tokens": t,
                              "step": c["step"], "stage": stage, **{d: row.get(d) for d in DOMAINS},
                              "ref_tokens": t, "ref_mix": L[REF][s][t]["mix"],
                              "mix_minus_ref": row["mix"] - L[REF][s][t]["mix"], "source": src(r)})
            if f in grafts:
                g = D["g_tok"][f]
                row = with_mix(r, {d: float(v["after"]) for d, v in r["graft"]["heldout"].items()})
                curve.append({"arm": f, "name": r["name"], "seed": s, "budget": r["tokens"], "tokens": g,
                              "step": r["graft"]["step"], "stage": "right after the graft (record's check)",
                              **{d: row.get(d) for d in DOMAINS}, "ref_tokens": g, "ref_mix": L[REF][s][g]["mix"],
                              "mix_minus_ref": row["mix"] - L[REF][s][g]["mix"], "source": src(r) + " graft.heldout"})

    # ---- panel (a): paired differences in mix (the mean, the spread and each seed)
    summ = []

    def diff_row(f, t, per, what, rep_mean=None, note=""):
        st_ = stats(list(per.values()))
        row = {"panel": "a", "arm": f, "x": t, "x_unit": "tokens", "what": what, "n": st_["n"], "value": st_["mean"],
               "sd": st_["sd"], "min": min(per.values()), "max": max(per.values()),
               **{f"seed_{s}": v for s, v in per.items()}, "note": note}
        if rep_mean is not None:
            row["report_value"] = rep_mean
            row["report_minus_value"] = rep_mean - st_["mean"]
        summ.append(row)
        return row

    for t in points:
        per = {s: L[TARGET][s][t]["mix"] - L[REF][s][t]["mix"] for s in seeds}
        rp = rep_pair.get((f"{TARGET} - {REF}", t))
        diff_row(TARGET, t, per, "mean of the paired differences (mix)", rp["mix"]["mean"] if rp else None)
    for f in grafts:
        g = D["g_tok"][f]
        per = {s: next(c["mix_minus_ref"] for c in curve if c["name"] == fam[f][s]["name"] and c["tokens"] == g
                       and c["stage"].startswith("right after")) for s in seeds}
        diff_row(f, g, per, "right after the graft (record's check): loss after = loss before = ref's",
                 note=f"graft at step {D['g_step'][f]}, {D['g_frac'][f]:g} of the budget")
        for t in points:
            if t <= g:
                continue
            per = {s: L[f][s][t]["mix"] - L[REF][s][t]["mix"] for s in seeds}
            rp = rep_pair.get((f"{f} - {REF}", t))
            diff_row(f, t, per, "mean of the paired differences (mix)", rp["mix"]["mean"] if rp else None)

    # ---- panel (b): share of the page gain recovered at the end
    summ.append({"panel": "b", "arm": TARGET, "x": 0.0, "x_unit": "graft point (fraction of the budget)",
                 "what": "by definition (the page from the first step)", "value": 1.0})
    for f in grafts:
        rc = recovered[(f, "mix")]
        per = {int(s): v for s, v in rc["per_seed"].items()}
        rr = rep_rec.get(f, {}).get("mix", {})
        base = {"panel": "b", "arm": f, "x": D["g_frac"][f], "x_unit": "graft point (fraction of the budget)",
                "n": rc["n"], "min": min(per.values()), "max": max(per.values()),
                **{f"seed_{s}": v for s, v in per.items()}}
        summ.append({**base, "what": "ratio of the mean differences (ref - graft) / (ref - v9_pages)",
                     "value": rc["ratio_of_means"], "report_value": rr.get("ratio_of_means"),
                     "report_minus_value": (rr["ratio_of_means"] - rc["ratio_of_means"]) if rr else None,
                     "note": f"ref - graft {rc['ref_minus_graft']['mean']!r}, ref - v9_pages "
                             f"{rc['ref_minus_target']['mean']!r} (means of the paired differences at {last:,} tokens)"})
        summ.append({**base, "what": "mean of the per-seed shares (not drawn)", "value": rc["mean"], "sd": rc["sd"],
                     "report_value": rr.get("mean"),
                     "report_minus_value": (rr["mean"] - rc["mean"]) if rr else None})
    summ.append({"panel": "b", "arm": REF, "x": 1.0, "x_unit": "graft point (fraction of the budget)",
                 "what": "by definition (never grafted)", "value": 0.0})

    # ---- restart arms (3M), when they exist
    rs, note = restart(D)
    if rs is None:
        print(f"F12: restart arms with half the budget: left out of (a): {note}")
    else:
        ref_end = {s: L[REF][s][last]["mix"] for s in seeds}
        per, ends = {}, {}
        for s, r in sorted(rs.items()):
            Lr = losses(r)
            for t, row in sorted(Lr.items()):
                curve.append({"arm": "restart", "name": r["name"], "seed": s, "budget": r["tokens"], "tokens": t,
                              "step": next(c["step"] for c in r["curve"] if int(c["tokens"]) == t),
                              "stage": "page from the first step, half the budget",
                              **{d: row.get(d) for d in DOMAINS},
                              "source": f"{P.rel(P.MICRO_GRAFT_3M)} {r['name']}"})
            ends[s] = max(Lr)
            per[s] = Lr[ends[s]]["mix"] - ref_end[s]
            curve[-1].update({"ref_tokens": last, "ref_mix": ref_end[s], "mix_minus_ref": per[s]})
        if len(set(ends.values())) != 1:
            print(f"F12: the restart records end at different tokens {sorted(set(ends.values()))}; drawn at the first")
        row = diff_row("restart", min(ends.values()), per,
                       f"end of its own budget minus ref's end ({last:,} tokens of the 6M budget)",
                       note=f"v9_pages from scratch, budget {RESTART_BUDGET:,} tokens" + (f"; {note}" if note else ""))
        own = stats([losses(r)[ends[s]]["mix"] for s, r in sorted(rs.items())])
        print(f"F12: restart arms with half the budget: drawn in (a): seeds {sorted(per)}, own end {row['x']:,} tokens; "
              f"loss there {own['mean']:.4f} ± {own['sd']:.4f}; minus ref's at {last:,}: {row['value']:+.4f} ± "
              f"{row['sd']:.4f} (per seed " + ", ".join(f"{v:+.4f}" for _, v in sorted(per.items())) + ")")
        for f in grafts:                # what a graft gave by the end of the full budget, for comparison
            vs = {s: Lr_end - L[f][s][last]["mix"] for s, Lr_end in
                  ((s, losses(r)[ends[s]]["mix"]) for s, r in sorted(rs.items()))}
            c = stats(list(vs.values()))
            print(f"F12: restart (own end) minus {f} (at {last:,}), paired by seed: {c['mean']:+.4f} ± {c['sd']:.4f} "
                  "(per seed " + ", ".join(f"{v:+.4f}" for _, v in sorted(vs.items())) + ")")
    write_csv("micro_graft_curve.csv", curve, ["arm", "name", "seed", "budget", "tokens", "step", "stage", "mix", "web",
                                               "chat", "assistant", "ref_tokens", "ref_mix", "mix_minus_ref", "source"])
    write_csv("micro_graft_summary.csv", summ, ["panel", "arm", "x", "x_unit", "what", "n", "value", "sd", "min", "max"]
              + [f"seed_{s}" for s in seeds] + ["report_value", "report_minus_value", "note"])
    return D, table, paired, recovered


# ------------------------------------------------------------------------------------ the figure
def spread(ys_pt, min_sep):
    """Positions (points, top to bottom = descending) moved apart to at least min_sep, as little as possible."""
    order = sorted(range(len(ys_pt)), key=lambda i: -ys_pt[i])
    pos = [ys_pt[i] for i in order]
    for _ in range(50):
        moved = False
        for k in range(1, len(pos)):
            gap = pos[k - 1] - pos[k]
            if gap < min_sep - 1e-9:
                push = (min_sep - gap) / 2
                pos[k - 1] += push
                pos[k] -= push
                moved = True
        if not moved:
            break
    out = [0.0] * len(ys_pt)
    for k, i in enumerate(order):
        out[i] = pos[k]
    return out


def draw(D):
    rows = C.read("micro_graft_summary.csv")
    for r in rows:
        for k in list(r):
            if k in ("x", "value", "sd", "min", "max", "report_value", "report_minus_value") or k.startswith("seed_"):
                r[k] = f_or_none(r[k])
    seeds = sorted(int(k[5:]) for k in rows[0] if k.startswith("seed_"))
    grafts = D["grafts"]
    col = {TARGET: st.PAGE_RAMP[0], **{f: st.PAGE_RAMP[i + 1] for i, f in enumerate(grafts)}, "restart": st.PAGE_RAMP[0]}
    A = [r for r in rows if r["panel"] == "a"]
    Bp = [r for r in rows if r["panel"] == "b"]
    M = 1e6
    budget = D["budget"]
    last = D["points"][-1]

    def seed_vals(r):
        return [r[f"seed_{s}"] for s in seeds if r.get(f"seed_{s}") is not None]

    main = [r for r in A if r["arm"] != "restart"]
    rst = [r for r in A if r["arm"] == "restart"]
    lo = min(min(seed_vals(r) + [r["value"]]) for r in main)
    ylim = (lo - 0.006, 0.0105)
    # restart marks: in (a) itself when they fit with a small extension, else in a strip above it (broken axis)
    strip = None
    if rst:
        rv = seed_vals(rst[0]) + [rst[0]["value"]]
        if min(rv) >= ylim[0] - 0.01 and max(rv) <= ylim[1] + 0.01:
            ylim = (min(ylim[0], min(rv) - 0.004), max(ylim[1], max(rv) + 0.006))
        else:
            strip = (min(rv) - 0.006, max(rv) + 0.006)

    W = st.FULL_W
    key_h, title_h, bottom, main_h = 0.42, 0.26, 0.50, 1.85
    scale = main_h / (ylim[1] - ylim[0])                                 # inches per nat in (a)
    strip_h = 0.0 if strip is None else (strip[1] - strip[0]) * scale  # the strip at the same scale ...
    if strip is not None and not 0.30 <= strip_h <= 1.2:               # ... unless that is too thin or too tall
        strip_h = min(max(strip_h, 0.30), 1.2)
        print(f"F12: the restart strip is drawn at {strip_h / (strip[1] - strip[0]):.1f} in per nat, the rest of (a) at "
              f"{scale:.1f}")
    gap = 0.0 if strip is None else 0.12
    top_in = bottom + main_h + gap + strip_h                             # top of the panels (inches)
    H = top_in + title_h + key_h
    fig = plt.figure(figsize=(W, H))
    ax_l, ax_w = 0.66, 3.12
    bx_l, bx_w = 5.36, 1.50
    axA = fig.add_axes([ax_l / W, bottom / H, ax_w / W, main_h / H])
    axB = fig.add_axes([bx_l / W, bottom / H, bx_w / W, (top_in - bottom) / H])

    # ---------------- (a)
    xmax = budget / M + 0.2
    axA.set_xlim(0, xmax)
    axA.set_ylim(*ylim)
    axA.set_xticks([0, 1.5, 3, 4.5, 6])
    axA.set_xticklabels(["0", "1.5", "3", "4.5", "6"])
    evals = [f"{t / M:.1f}" for t in D["points"]]
    axA.set_xlabel(f"Training tokens (millions); held-out loss evaluated at {', '.join(evals[:-1])} and {evals[-1]}")
    axA.set_ylabel("Held-out loss − ref's, same seed\n(nats per token; 0.75 web + 0.25 chat)")
    yt = [v / 100 for v in range(0, -20, -2) if ylim[0] <= v / 100 <= ylim[1]]
    axA.set_yticks(yt)
    axA.set_yticklabels(["0" if v == 0 else f"−{abs(v):.2f}" for v in yt])
    st.grid_y(axA)
    axA.axhline(0, color=REF_COL, lw=0.9, zorder=3, solid_capstyle="butt")
    dodge = 0.10                                                          # millions of tokens: seeds beside the mean
    ends = []
    for f in [TARGET] + grafts:
        pts = sorted((r for r in main if r["arm"] == f), key=lambda r: r["x"])
        c = col[f]
        axA.plot([r["x"] / M for r in pts], [r["value"] for r in pts], color=c, lw=1.4, zorder=4)
        ck = [r for r in pts if not r["what"].startswith("right after")]
        axA.plot([r["x"] / M for r in ck], [r["value"] for r in ck], ls="", marker="o", ms=4.4, mfc=c, mec=st.WHITE,
                 mew=0.7, zorder=6)
        dx = -dodge if f == TARGET else dodge
        for r in ck:
            v = seed_vals(r)
            axA.plot([r["x"] / M + dx] * len(v), v, ls="", marker="o", ms=2.5, mfc=c, mec=st.WHITE, mew=0.35, zorder=5)
        for r in (r for r in pts if r["what"].startswith("right after")):
            axA.plot([r["x"] / M], [r["value"]], ls="", marker="D", ms=5.0, mfc=c, mec=st.WHITE, mew=0.8, zorder=7)
            axA.text(r["x"] / M, r["value"] + 0.0033, f"graft at {D['g_frac'][f]:g}", ha="center", va="bottom",
                     fontsize=st.FS - 0.5, color=st.INK, zorder=8)
        ends.append((f, pts[-1]["x"] / M, pts[-1]["value"]))
    ends.append((REF, last / M, 0.0))
    # names at the right ends, moved apart where they would touch, each joined to its line
    fig.canvas.draw()
    to_pt = 72.0 / fig.dpi
    lab = spread([axA.transData.transform((0, y))[1] * to_pt for _, _, y in ends], 8.0)
    inv = axA.transData.inverted()
    xl = xmax + 0.17
    for (f, x, y), yp in zip(ends, lab):
        yl = inv.transform((0, yp / to_pt))[1]
        axA.plot([x + dodge + 0.09, xl - 0.05], [y, yl], color=col.get(f, REF_COL), lw=0.6, zorder=3, clip_on=False)
        axA.text(xl, yl, f, ha="left", va="center", fontsize=st.FS - 0.5, color=st.INK, clip_on=False)
    axA.text(0.06, ylim[0] + 0.0045,
             "Zero line: ref, and each graft arm up to\nits graft (difference exactly 0 there, every seed)",
             ha="left", va="bottom", fontsize=st.FS - 0.5, color=st.INK2, linespacing=1.25)

    # restart arms (v9_pages from scratch on half the budget), when their records exist
    if rst:
        r = rst[0]
        tax = axA
        if strip is not None:
            axS = fig.add_axes([ax_l / W, (bottom + main_h + gap) / H, ax_w / W, strip_h / H], sharex=axA)
            axS.set_ylim(*strip)
            axS.spines["bottom"].set_visible(False)
            axS.tick_params(axis="x", bottom=False, labelbottom=False)
            step = 0.01 if strip[1] - strip[0] <= 0.04 else 0.02
            ticks = [k * step for k in range(math.ceil(strip[0] / step), math.floor(strip[1] / step) + 1)]
            axS.set_yticks(ticks)
            axS.set_yticklabels([("+" if v > 0 else ("−" if v < 0 else "")) + f"{abs(v):.2f}" for v in ticks])
            st.grid_y(axS)
            brk = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=5, linestyle="none", color=st.AXIS, mec=st.AXIS,
                       mew=0.7, clip_on=False)
            axA.plot([0], [1], transform=axA.transAxes, **brk)          # the axis is broken between the two parts
            axS.plot([0], [0], transform=axS.transAxes, **brk)
            tax = axS
        x = r["x"] / M
        v = seed_vals(r)
        tax.plot([x - dodge] * len(v), v, ls="", marker="s", ms=2.6, mfc=col["restart"], mec=st.WHITE, mew=0.35,
                 zorder=5)
        tax.plot([x], [r["value"]], ls="", marker="s", ms=4.8, mfc=st.WHITE, mec=col["restart"], mew=1.1, zorder=6)
        if strip is not None:       # the strip has room for what the marks compare
            tax.text(x + 0.16, r["value"], f"restart with half the budget: loss at its own end\n({x:.1f}M tokens) "
                     f"minus ref's at the end of the {budget / M:.0f}M budget", ha="left", va="center",
                     fontsize=st.FS - 0.5, color=st.INK, linespacing=1.2)
        else:                       # among the lines of (a): a short name on a white ground, on the side with room
            right = x + 0.16 + 1.5 <= xmax
            tax.text(x + 0.16 if right else x - dodge - 0.12, r["value"], "restart with half the budget",
                     ha="left" if right else "right", va="center", fontsize=st.FS - 0.5, color=st.INK, zorder=9,
                     bbox=dict(boxstyle="square,pad=0.15", fc=st.WHITE, ec="none"))
    fig.text(0.11 / W, (top_in + 0.11) / H, "a", fontsize=st.FS_TITLE, fontweight="semibold", ha="left", va="bottom")
    fig.text(0.30 / W, (top_in + 0.11) / H, "Held-out loss minus ref's, against training tokens",
             fontsize=st.FS_TITLE, fontweight="semibold", ha="left", va="bottom")

    # ---------------- (b)
    axB.set_xlim(-0.09, 1.09)
    hi = max(max(seed_vals(r) + [r["value"]]) for r in Bp)
    axB.set_ylim(-0.1, max(1.25, hi) + 0.1)
    axB.set_yticks([0, 0.25, 0.5, 0.75, 1.0, 1.25])
    axB.set_yticklabels(["0", "0.25", "0.5", "0.75", "1", "1.25"])
    axB.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axB.set_xticklabels(["0\nfirst step", "0.25", "0.5", "0.75", "1\nnever"])
    axB.set_xlabel("Graft point (fraction of the budget)")
    axB.set_ylabel(f"(ref − graft) / (ref − v9_pages),\nheld-out loss at the end ({last / M:.1f}M tokens)")
    st.grid_y(axB)
    for y in (0.0, 1.0):
        axB.axhline(y, color=st.MUTED, lw=0.6, zorder=1)
    for r in Bp:
        if r["what"].startswith("by definition"):
            c = col.get(r["arm"], REF_COL)
            axB.plot([r["x"]], [r["value"]], ls="", marker="o", ms=5.0, mfc=st.WHITE, mec=c, mew=1.1, zorder=5)
        elif r["what"].startswith("ratio"):
            c = col[r["arm"]]
            axB.plot([r["x"]], [r["value"]], ls="", marker="o", ms=5.0, mfc=c, mec=st.WHITE, mew=0.8, zorder=6)
            v = seed_vals(r)
            axB.plot([r["x"] + 0.05] * len(v), v, ls="", marker="o", ms=2.6, mfc=c, mec=st.WHITE, mew=0.35, zorder=5)
            axB.text(r["x"] - 0.05, r["value"], f"{r['value']:.2f}", ha="right", va="center", fontsize=st.FS - 0.5,
                     color=st.INK)
    fig.text(4.72 / W, (top_in + 0.11) / H, "b", fontsize=st.FS_TITLE, fontweight="semibold", ha="left", va="bottom")
    fig.text(4.91 / W, (top_in + 0.11) / H, "Share of the page gain recovered", fontsize=st.FS_TITLE,
             fontweight="semibold", ha="left", va="bottom")

    # ---------------- key
    arms_h = [Line2D([], [], color=col[TARGET], lw=1.4, marker="o", ms=4.0, mfc=col[TARGET], mec=st.WHITE, mew=0.6),
              tuple(Line2D([], [], color=col[f], lw=1.4) for f in grafts),
              Line2D([], [], color=REF_COL, lw=0.9)]
    arms_l = [f"{TARGET}: the page from the first step",
              ", ".join(grafts) + ": ref to " + ", ".join(f"{D['g_frac'][f]:g}" for f in grafts)
              + " of the budget, then the page grafted in",
              f"{REF}: no page"]
    marks_h = [Line2D([], [], ls="", marker="o", ms=4.4, mfc=st.INK2, mec=st.WHITE, mew=0.7),
               Line2D([], [], ls="", marker="o", ms=2.5, mfc=st.INK2, mec=st.WHITE, mew=0.35),
               Line2D([], [], ls="", marker="D", ms=5.0, mfc=st.INK2, mec=st.WHITE, mew=0.8),
               Line2D([], [], ls="", marker="o", ms=5.0, mfc=st.WHITE, mec=st.INK2, mew=1.1)]
    marks_l = [f"mean of the {len(seeds)} seeds (a), ratio of the mean differences (b)", "one seed, beside the mean",
               "graft (loss after = before)", "by definition"]
    if rst:
        marks_h.append(Line2D([], [], ls="", marker="s", ms=4.8, mfc=st.WHITE, mec=col["restart"], mew=1.1))
        marks_l.append("restart")
    kw = dict(loc="upper left", frameon=False, fontsize=st.FS - 0.5, handletextpad=0.5, columnspacing=1.15,
              borderaxespad=0.0, borderpad=0.0, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)})
    fig.legend(arms_h, arms_l, bbox_to_anchor=(0.11 / W, 1 - 0.06 / H), ncol=3, handlelength=2.2, **kw)
    fig.legend(marks_h, marks_l, bbox_to_anchor=(0.11 / W, 1 - 0.24 / H), ncol=len(marks_h), handlelength=1.0, **kw)
    st.save(fig, "F12_micrograft")


def report_lines(D, table, paired, recovered):
    """The numbers the caption and FIGURES.md quote."""
    fam, points, grafts = D["fam"], D["points"], D["grafts"]
    last = points[-1]
    r0 = next(iter(fam[REF].values()))
    sh = shares(r0, ["web", "chat"])
    ev = r0["curve"][-1]["eval"]
    # every record and checkpoint was evaluated on the same rows (same tokens per row)
    rows_w = {json.dumps({d: v["w"] for d, v in sorted(c["eval"].items())}) for rs in fam.values() for r in rs.values()
              for c in r["curve"]}
    print(f"F12: mix = " + " + ".join(f"{w:g} x {d}" for d, w in sh.items()) + " (arm.mixture); held-out rows "
          + ", ".join(f"{d} {len(ev[d]['ce'])} rows / {int(sum(ev[d]['w'])):,} tokens" for d in ev)
          + f" ({'the same rows in' if len(rows_w) == 1 else 'DIFFERENT rows across'} every record and checkpoint)"
          + f"; d_model {r0['config']['model']['d_model']}, {r0['tokens_per_step']:,} tokens per step, "
          + f"{r0['steps']:,} steps; parameters ref {r0['n_params']:,}, with the page "
          + f"{next(iter(fam[TARGET].values()))['n_params']:,}")
    for f in [TARGET] + grafts:
        xs = points if f == TARGET else [t for t in points if t > D["g_tok"][f]]
        for t in xs:
            p = paired[(f"{f} - {REF}", t, "mix")]
            print(f"F12: (a) {f} - ref at {t:,}: {p['mean']:+.4f} ± {p['sd']:.4f} (n {p['n']}; per seed "
                  + ", ".join(f"{v:+.4f}" for v in p["per_seed"].values()) + f"; range {min(p['per_seed'].values()):+.4f}"
                  f" to {max(p['per_seed'].values()):+.4f}; relative {100 * p['rel_mean']:+.2f}%)")
    for f in grafts:
        p = paired[(f"{f} - {TARGET}", last, "mix")]
        print(f"F12: {f} - {TARGET} at {last:,}: {p['mean']:+.4f} ± {p['sd']:.4f} (n {p['n']}; per seed "
              + ", ".join(f"{v:+.4f}" for v in p["per_seed"].values()) + ")")
    for f in grafts:
        rc = recovered[(f, "mix")]
        print(f"F12: (b) {f} (graft at {D['g_frac'][f]:g}, step {D['g_step'][f]}, {D['g_tok'][f]:,} tokens): share "
              f"{rc['ratio_of_means']:.3f} (ratio of the mean differences {rc['ref_minus_graft']['mean']:.4f} / "
              f"{rc['ref_minus_target']['mean']:.4f}); per seed "
              + ", ".join(f"{v:.3f}" for v in rc["per_seed"].values()) + f"; mean of the shares {rc['mean']:.3f}")
    for f in [REF, TARGET] + grafts:
        t = table[(f, last, "mix")]
        print(f"F12: loss at {last:,} ({f}): {t['mean']:.4f} ± {t['sd']:.4f} (n {t['n']})")


def main():
    os.makedirs(P.DATA, exist_ok=True)
    D, table, paired, recovered = build()
    report_lines(D, table, paired, recovered)
    draw(D)
    if MISMATCHES:
        print(f"F12: {len(MISMATCHES)} MISMATCH line(s) above")


if __name__ == "__main__":
    main()
