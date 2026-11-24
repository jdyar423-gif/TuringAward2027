"""Build every CSV of paper/data/ from the sources (run with the venv's python -I -B).

    python -I -B make_data.py [--no-extract] [--cutoff paper|newest|STEP] [--blind NAME[,NAME...]]

First it extracts the record files of the newest snapshot of the live run (paths.SNAP_LIVE, with
extract_records.py: train log, run log, manifests, ledgers; no weights) into data/_src/live, and of
the stopped final run into data/_src/final if that is missing; --no-extract keeps the records
extracted last time. A snapshot that is still being copied is refused with exit code 3, and nothing
is written (the records extracted last time stay as they were).

--cutoff fixes the newest record of the run that the CSVs (and so every figure) use. "paper" (the
default) is the newest step of the paper's own live numbers (lead2_work/numbers.json, v9.newest_step),
so that the figures and the text's numbers come from the same records; "newest" keeps every record of
the snapshot; a number keeps the step records up to that step. Records logged after the cut-off
(steps, events, loader counters) and held-out evaluations of later checkpoints are left out. The
cut-off is written to data/cutoff.csv, which the figures print. A cut-off newer than the snapshot
stops the build (exit 4) before anything is written.

--blind adds blind-comparison reports to blind_comparisons.csv beyond the four development sets
(paths.BLIND_DEV): a NAME is the report's file name without ".json" in paths.BLIND_DIR, e.g.
"final-v9-vs-falcon--judge-opus"; a NAME without "--" gets "--judge-opus" appended, so
"final-v9-vs-falcon" works too. A missing report stops the build with a message (exit 2).

Reads the records extracted from the snapshots (extract_records.py; paths in paths.py), the chip
measurement outputs, the TPU ledger, the held-out evaluations, the paper's own analyses
(lead2_work/), the blind-comparison reports, the data-rule scans and the checkpoint manifests.
Writes CSVs into paths.DATA (paper/data/; every series a figure plots, derived series included) and
the extracted records into paths.SRC (data/_src/), and prints the checks it makes; a check that fails
prints a line starting with "MISMATCH" and the build goes on. Nothing else is written.
"""
import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P                                   # noqa: E402
import common as C                                  # noqa: E402
from interventions import INTERVENTIONS, MEASUREMENTS, PHASES  # noqa: E402

MISMATCHES = []


def mismatch(msg):
    MISMATCHES.append(msg)
    print("MISMATCH " + msg)


def utc(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def write_csv(name, rows, cols):
    path = os.path.join(P.DATA, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    print(f"wrote {path} ({len(rows)} rows)")
    return path


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ------------------------------------------------------------------------------------ snapshots
def extract():
    """Records of the newest live snapshot into data/_src/live (replacing the previous extraction only
    when the new one is complete), and of the stopped final run into data/_src/final if missing.
    A snapshot that is still being copied (extract_records.py exits 3) stops the build with exit
    code 3 before anything is written."""
    import shutil
    import extract_records as X

    def one(snap, name):
        new = os.path.join(P.SRC, name + ".new")
        shutil.rmtree(new, ignore_errors=True)
        try:
            meta = X.main(snap, new, quiet=True)
        except SystemExit as e:
            shutil.rmtree(new, ignore_errors=True)
            code = e.code if isinstance(e.code, int) else 1
            sys.stderr.write(f"make_data.py: could not read {snap} (exit {code}); nothing was written. Run again "
                             "later, or with --no-extract to use the records extracted last time.\n")
            sys.exit(code)
        old = os.path.join(P.SRC, name)
        shutil.rmtree(old, ignore_errors=True)
        os.rename(new, old)
        return meta

    os.makedirs(P.SRC, exist_ok=True)
    if not os.path.isdir(P.SRC_FINAL):
        one(P.SNAP_FINAL, "final")
    meta = one(P.SNAP_LIVE, "live")
    print(f"extracted {P.SNAP_LIVE}: snapshot of {utc(meta['created'])} UTC, {meta.get('steps')} steps")


# ------------------------------------------------------------------------------------ cut-off
def cutoff_of(arg, recs):
    """The newest record the CSVs use: dict(kind, step, time, ...). arg "paper": the newest step of the
    paper's live numbers (numbers.json v9.newest_step); "newest": every record; else a step number (the
    newest step record at or before it)."""
    steps = [(cfg, r) for cfg, r in recs if r.get("event") == "step"]
    newest = max(steps, key=lambda cr: cr[1]["time"])
    snap = {}
    ex = os.path.join(P.SRC, "live", "EXTRACTED.json")
    if os.path.exists(ex):
        snap = json.load(open(ex))
    out = dict(snapshot_created_utc=utc(snap["snapshot_created"]) if snap.get("snapshot_created") else "",
               snapshot_steps=snap.get("snapshot_steps", ""), newest_step_in_snapshot=newest[1]["step"],
               newest_utc_in_snapshot=utc(P.T0 + newest[1]["time"]))
    if arg == "newest":
        cfg, r = newest
        out.update(kind="newest", source="--cutoff newest: every record of the snapshot")
    else:
        if arg == "paper":
            nj = json.load(open(P.NUMBERS))["v9.newest_step"]["value"]
            want = int(nj["step"])
            src = f"{P.rel(P.NUMBERS)}, v9.newest_step (the newest step of the paper's live numbers)"
        else:
            want, nj = int(arg), None
            src = f"--cutoff {want}"
        if want > newest[1]["step"]:
            sys.stderr.write(f"make_data.py: the cut-off step {want:,} is newer than the newest step record of the "
                             f"snapshot ({newest[1]['step']:,}); no CSV was written. Wait for a newer snapshot, or "
                             "pass --cutoff newest or a step.\n")
            sys.exit(4)
        cand = [cr for cr in steps if cr[1]["step"] <= want]
        if not cand:
            sys.stderr.write(f"make_data.py: no step record at or before step {want:,}; no CSV was written.\n")
            sys.exit(4)
        cfg, r = max(cand, key=lambda cr: cr[1]["time"])
        if nj is not None:
            same = (r["step"] == want and utc(P.T0 + r["time"])[11:] == nj["utc"] and round(r["p"], 4) == nj["p"]
                    and round(r["loss"], 4) == nj["loss"])
            print(f"  cut-off: step {r['step']:,} at {utc(P.T0 + r['time'])} UTC (p {r['p']:.4f}, loss {r['loss']:.4f}); "
                  f"numbers.json: step {want:,} at {nj['utc']} (p {nj['p']}, loss {nj['loss']}) "
                  f"{'(same record)' if same else ''}")
            if not same:
                mismatch(f"cut-off: the step record of the snapshot differs from numbers.json's v9.newest_step {nj}")
        out.update(kind="paper" if arg == "paper" else "step", source=src)
    out.update(config=cfg, step=r["step"], time_s=round(r["time"], 3), utc=utc(P.T0 + r["time"]), p=round(r["p"], 4),
               loss=round(r["loss"], 4), seq_len=r.get("seq_len"), _time=r["time"])
    out["step_records_after_cutoff"] = sum(1 for _, s in steps if s["time"] > r["time"] + 1e-9)
    return out


def apply_cutoff(recs, cut):
    return [(cfg, r) for cfg, r in recs if r.get("time") is None or r["time"] <= cut["_time"] + 1e-9]


# ------------------------------------------------------------------------------------ train logs
def train_records():
    """(config, record) for every record of the final run's log, then of the live run's log."""
    out = []
    for cfg, root in (("final", P.SRC_FINAL), ("v9", P.SRC_LIVE)):
        path = os.path.join(root, "train", "train_log.jsonl")
        for r in read_jsonl(path):
            out.append((cfg, r))
    return out


def block_fail_draw(rec):
    lb = rec.get("loader_block") or {}
    return lb.get("block"), lb.get("first"), lb.get("last"), lb.get("fail") or {}, lb.get("draws") or {}


BREAKS = ("restore", "setup", "stage", "end")     # events that break the achieved-rate interval


def steps_and_events(recs):
    steps, events = [], []
    prev = None
    nbreak = 0
    for cfg, r in recs:
        ev = r.get("event")
        t = r.get("time")
        if ev == "step":
            m = r.get("micro") or {}
            blk, first, last, fail, draws = block_fail_draw(r)
            row = dict(config=cfg, step=r["step"], tokens=r["step"] * P.TOKENS_PER_STEP,
                       time_s=round(t, 3), utc=utc(P.T0 + t), p=r["p"], lr=r["lr"], loss=r["loss"],
                       loss_ema=r["loss_ema"], seq_len=r["seq_len"], phase=r["phase"],
                       assistant_only=int(bool(r.get("assistant_only"))),
                       chat_frac=r["chat_frac"], library=int(bool(r["library"])),
                       tok_per_s=round(r["tok_per_s"], 1), _rate=r["tok_per_s"], step_s=round(r["step_s"], 5),
                       grad_norm=r["grad_norm"], weight=r["weight"],
                       micro_tokens=m.get("tokens"), tape_gain=m.get("tape_gain"),
                       tape_mass=m.get("tape_mass"), action=r.get("action"),
                       block=blk, block_first=first, block_last=last)
            # achieved rate since the previous step record when no restart lies between them
            if prev is not None and prev["config"] == cfg and prev["_seq"] == nbreak:
                ds = r["step"] - prev["step"]
                dtime = t - prev["time_s"]
                row["interval_steps"] = ds
                row["interval_s"] = round(dtime, 3)
                row["interval_tok_per_s"] = round(ds * P.TOKENS_PER_STEP / dtime, 1) if dtime > 0 else None
            row["_seq"] = nbreak
            steps.append(row)
            prev = row
        else:
            if ev in BREAKS:
                nbreak += 1
            e = dict(config=cfg, event=ev, step=r.get("step"), time_s=round(t, 3) if t is not None else None,
                     utc=utc(P.T0 + t) if t is not None else None, reason=r.get("reason"),
                     path=r.get("path") or r.get("checkpoint"), p=r.get("p"))
            if ev == "restore":
                e["lost_s"] = round(r.get("lost", 0.0), 3)
            if ev == "stage":
                e["phase"] = r.get("phase")
                e["seq_len"] = r.get("seq_len")
                e["assistant_only"] = int(bool(r.get("assistant_only")))
            if ev == "setup":
                comp = r.get("compile") or {}
                e["compile_s"] = round(sum(v.get("seconds", 0) for v in comp.values()), 3)
                led = r.get("ledger") or {}
                e["ledger_train_s"] = led.get("train")
                e["ledger_lost_s"] = led.get("lost")
                e["ledger_setup_s"] = led.get("setup")
                e["train_end_utc"] = utc(r["train_end"]) if r.get("train_end") else None
                e["deadline_utc"] = utc(r["deadline"]) if r.get("deadline") else None
                e["micro_split"] = json.dumps(r.get("micro_split")) if r.get("micro_split") else ""
            events.append(e)
    for s in steps:
        s.pop("_seq", None)
    return steps, events


def intervention_rows(steps, events):
    rows = []
    restores = [e for e in events if e["event"] == "restore"]
    ends = [e for e in events if e["event"] == "end"]
    labelled = {i["step"] for i in INTERVENTIONS} | {m["step"] for m in MEASUREMENTS}
    entries = [dict(i, group="intervention") for i in INTERVENTIONS] + \
              [dict(m, group="measurement", kind="measurement", long=m["short"]) for m in MEASUREMENTS]
    for e in restores:
        if e["step"] not in labelled:
            entries.append(dict(id="U", step=e["step"], kind="unlabelled", short="restart (unlabelled)",
                                long="restore in the train log that matches no curated entry",
                                src="train_log.jsonl", group="unlabelled"))
            labelled.add(e["step"])
    for it in entries:
        s = it["step"]
        r_at = [e for e in restores if e["step"] == s]
        e_at = [e for e in ends if e["step"] == s]
        first_after = next((x for x in steps if x["step"] > s), None)
        row = dict(id=it["id"], group=it["group"], kind=it["kind"], step=s, tokens=s * P.TOKENS_PER_STEP,
                   short=it["short"], long=it["long"], src=it["src"],
                   end_utc=e_at[-1]["utc"] if e_at else "", restore_utc=r_at[-1]["utc"] if r_at else "",
                   event_utc=r_at[-1]["utc"] if r_at else "",
                   restore_attempts=len(r_at),
                   first_step_after=first_after["step"] if first_after else "",
                   first_step_after_utc=first_after["utc"] if first_after else "",
                   config_after=r_at[-1]["config"] if r_at else "")
        if e_at and r_at:
            row["stopped_s"] = round(r_at[-1]["time_s"] - e_at[-1]["time_s"], 1)
        rows.append(row)
    # phase changes of the schedule: the first 'stage' event of each phase (no stop of the run)
    stages = [e for e in events if e["event"] == "stage"]
    for ph in PHASES:
        hit = [e for e in stages if e.get("phase") == ph["phase"]]
        if not hit:
            continue
        e = hit[0]
        first_after = next((x for x in steps if x["config"] == e["config"] and x["step"] > e["step"]), None)
        rows.append(dict(id=ph["id"], group="phase", kind="schedule", step=e["step"],
                         tokens=e["step"] * P.TOKENS_PER_STEP, short=ph["short"], long=ph["long"], src=ph["src"],
                         end_utc="", restore_utc="", event_utc=e["utc"], restore_attempts=0, phase=ph["phase"],
                         p=e["p"], first_step_after=first_after["step"] if first_after else "",
                         first_step_after_utc=first_after["utc"] if first_after else "",
                         config_after=e["config"]))
    rows.sort(key=lambda r: (r["step"], str(r["id"])))
    return rows


# ------------------------------------------------------------------------------------ TPU ledger
def tpu_intervals(now_t):
    rows = []
    notes = {1: "profiling and the glue rewrite; runs discarded; stopped at the user's request",
             2: "run main16 created 17:34:54, trained from 17:46:31; taken back by Colab about 19:29",
             3: "resumed 01:54:14 from step 5,075; graft to v9 at step 29,722; open"}
    with open(P.TPU_INTERVALS) as f:
        for i, line in enumerate(l for l in f if l.strip()):
            a, b, *rest = line.split(maxsplit=2)
            start = float(a)
            end = None if b == "-" else float(b)
            e = end if end is not None else now_t
            rows.append(dict(runtime=i + 1, start_utc=utc(start), end_utc=utc(end) if end else "",
                             open=int(end is None), counted_until_utc=utc(e), seconds=round(e - start),
                             hours=round((e - start) / 3600, 4), ledger_note=(rest[0].strip() if rest else ""),
                             summary=notes.get(i + 1, "")))
    return rows


# ------------------------------------------------------------------------------- chip measurements
def json_line(path, tag):
    with open(path) as f:
        for line in f:
            if line.startswith(tag + " "):
                return json.loads(line[len(tag) + 1:])
    raise ValueError(f"{path}: no {tag} line")


def quantiles(xs):
    xs = sorted(xs)
    n = len(xs)
    q = statistics.quantiles(xs, n=4, method="inclusive") if n >= 2 else [xs[0]] * 3
    return dict(n=n, median_check=round(statistics.median(xs), 4), q1=round(q[0], 4), q3=round(q[2], 4),
                min=round(xs[0], 4), max=round(xs[-1], 4))


def kernel_rows():
    d = json_line(P.CHIP_KERNELS, "KERNELS")
    names = {"": "splash (configured)", "onetile": "onetile", "onetile:save": "onetile:save",
             "onetile:save+relu2": "onetile:save + relu2 (adopted)",
             "onetile:save+relu2+ce": "onetile:save + relu2 + ce", "blocks:save": "splash + save (blocks:save)",
             "+relu2+ce": "splash + relu2 + ce", "+io": "splash + io"}
    rows, steps = [], []
    for r in d["results"]:
        q = quantiles(r["ms_all"])
        rows.append(dict(source="wf/chip_kernels_0745.txt", config="final", kernel=r["kernel"],
                         label=names.get(r["kernel"], r["kernel"]), t=r["t"], rows=r["rows"],
                         tokens_per_piece=r["t"] * r["rows"], ms=round(r["ms"], 4),
                         tok_per_s=round(r["tokens_per_s"], 1), loss=r["loss"],
                         loss_diff=r.get("loss_diff", ""), grad_rel_max=r.get("grad_rel_diff", ""),
                         grad_rel_l2=r.get("grad_rel_l2", ""), speedup=r.get("speedup", 1.0 if r.get("reference") else ""),
                         reference=int(bool(r.get("reference"))), compile_s=r.get("compile_s"), **q))
        for s in r.get("steps", []):
            steps.append(dict(source="wf/chip_kernels_0745.txt", measured_utc="2026-11-24 07:44-07:52",
                              config="final", table_grad="take", kernel=r["kernel"], a=s["a"], rows=s["rows"],
                              t=r["t"], ms=round(s["ms"], 4), tok_per_s=round(s["tokens_per_s"], 1),
                              loss=s["loss"], compile_s=s["compile_s"], timing=P.STEP_TIMING["tpu_kernels"]))
    return rows, steps, d


def v9_rows():
    pieces, steps = [], []
    for key, path in P.V9_MEAS.items():
        mode = key.split(",")[0]
        d = json_line(path, "KERNELS")
        for r in d["results"]:
            q = quantiles(r["ms_all"])
            pieces.append(dict(source=os.path.basename(path), config="v9", table_grad=mode, kernel=r["kernel"],
                               t=r["t"], rows=r["rows"], ms=round(r["ms"], 4),
                               loss_diff=r.get("loss_diff", ""), grad_rel_max=r.get("grad_rel_diff", ""),
                               grad_rel_l2=r.get("grad_rel_l2", ""),
                               tok_per_s=round(r["tokens_per_s"], 1), loss=r["loss"], **q))
            for s in r.get("steps", []):
                steps.append(dict(source=os.path.basename(path),
                                  measured_utc="2026-11-24 10:53-10:56" if "measD" in path else "2026-11-24 09:45-09:58",
                                  config="v9", table_grad=mode, kernel=r["kernel"], a=s["a"], rows=s["rows"],
                                  t=r["t"], ms=round(s["ms"], 4), tok_per_s=round(s["tokens_per_s"], 1),
                                  loss=s["loss"], compile_s=s["compile_s"], timing=P.STEP_TIMING["tpu_kernels"]))
    return pieces, steps


def micro_rows():
    d = json_line(P.MICRO_OUT, "MICRO")
    return [dict(source="micro_out.txt", measured_utc="2026-11-24 03:09-03:15", config="final",
                 table_grad="take", kernel="splash (configured)", a=v["a"], rows=v["rows"], t=v["t"],
                 ms=round(v["ms"], 4), tok_per_s=round(v["tokens_per_s"], 1), loss=v["loss"],
                 timing=P.STEP_TIMING["tpu_micro"])
            for v in d["variants"]]


def flag_rows():
    rows = []
    for path, when in zip(P.FLAGS_OUT, ("2026-11-24 03:48-03:54 (flags_out.txt)", "2026-11-24 03:48-03:54 (flags_out2.txt)")):
        d = json_line(path, "FLAGS")
        for v in d["variants"]:
            for s in v["shapes"]:
                m = re.search(r"limit_kib=(\d+)", v["flags"])
                rows.append(dict(source=os.path.basename(path), variant=v["name"],
                                 vmem_limit_mib=int(m.group(1)) // 1024 if m else "default",
                                 a=s["a"], rows=s["rows"], t=s["t"], ms=round(s["ms"], 4),
                                 tok_per_s=round(s["tokens_per_s"], 1), loss=s["loss"]))
    return rows


def tune_rows():
    """docs/tpu_profile/tpu_tune.v6e.txt: one 32 x 1024 (or 2048) micro-batch value_and_grad, 23 Nov."""
    rows = []
    sect = None
    with open(P.TUNE_OUT) as f:
        for line in f:
            if line.startswith("training, T="):
                sect = int(line.split("=")[1].split(":")[0])
                continue
            m = re.match(r"\s{2}(\S.*?)\s{2,}([\d.]+) ms\s+([\d,]+) tok/s\s+loss ([\d.]+)", line)
            if m and sect:
                rows.append(dict(source="docs/tpu_profile/tpu_tune.v6e.txt", t=sect, variant=m.group(1).strip(),
                                 ms=float(m.group(2)), tok_per_s=int(m.group(3).replace(",", "")),
                                 loss=float(m.group(4))))
            if line.startswith("decoding"):
                sect = None
    return rows


# ------------------------------------------------------------- throughput read from the run logs
RUN_LOGS = [  # (path, configuration, program) - runtime 1 used the code before the glue rewrite until 04:34
    (os.path.join(P.S, "v9.log"), "v9", "before the glue rewrite"),
    (os.path.join(P.S, "final.log"), "final", "before the glue rewrite"),
    (P.FINAL_B_LOG, "final", "after the glue rewrite"),
    (os.path.join(P.SRC_FINAL, "..", "run.log"), "final", "after the glue rewrite"),
]
MONITOR_LOGS = [(os.path.join(P.S, "colab_bes24h_final.log"), "final", "before the glue rewrite")]


def log_points():
    """Benchmark and step-50 lines of the run logs, and the old program's monitor lines (runtime 1)."""
    rows = []
    head = re.compile(r"^===== (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (.*)$")
    bench = re.compile(r"^\[\s*([\d.]+) min\] benchmark: (\S+) T=(\d+): ([\d,]+) tok/s, step ([\d.]+) s")
    step = re.compile(r"^\[\s*([\d.]+) min\] train: step (\d+) p [\d.]+ loss ([\d.]+) T (\d+) tok/s ([\d,]+)")
    for path, cfg, prog in RUN_LOGS:
        t_head = None
        src = P.rel(path)
        with open(path) as f:
            for line in f:
                m = head.match(line)
                if m:
                    t_head = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
                    continue
                if t_head is None:
                    continue
                m = bench.match(line)
                if m:
                    t = t_head + dt.timedelta(minutes=float(m.group(1)))
                    rows.append(dict(source=src, utc=t.strftime("%Y-%m-%d %H:%M:%S"), config=cfg, program=prog,
                                     what="start-up benchmark " + m.group(2), seq_len=int(m.group(3)),
                                     tok_per_s=int(m.group(4).replace(",", "")), step_s=float(m.group(5))))
                    continue
                m = step.match(line)
                if m and int(m.group(2)) == 50:
                    t = t_head + dt.timedelta(minutes=float(m.group(1)))
                    rows.append(dict(source=src, utc=t.strftime("%Y-%m-%d %H:%M:%S"), config=cfg, program=prog,
                                     what="training, step 50", seq_len=int(m.group(4)),
                                     tok_per_s=int(m.group(5).replace(",", ""))))
    mon = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] run RUNNING .*? ([\d.e+]+) tokens, ([\d,]+) steps \| "
                     r"loss ([\d.]+), ([\d,]+) tok/s")
    for path, cfg, prog in MONITOR_LOGS:
        with open(path) as f:
            for line in f:
                m = mon.match(line)
                if m:
                    rows.append(dict(source=path, utc=m.group(1), config=cfg, program=prog,
                                     what=f"training (monitor), step {m.group(3)}", seq_len=1024,
                                     tok_per_s=int(m.group(5).replace(",", ""))))
    rows.sort(key=lambda r: r["utc"])
    return rows


# --------------------------------------------------------------------- live rates by stage (Table 2)
IV_STEP = {str(i["id"]): i["step"] for i in INTERVENTIONS}
# Stages of run main16 in the order they happened: (key, Table 2 label, configuration, from the
# step of intervention lo (exclusive) to that of hi (inclusive; None = open), rows of tokens,
# schedule phase, Table 2's synthetic or benchmark value: (kind, CSV, selector)).
RUN_STAGES = [
    ("rt2_16rows", "16-row micro-batches, runtime 2", "final", None, "1", 1024, None,
     ("start-up benchmark", "throughput_log_points.csv", ("start-up benchmark splash-remat1-ms16-take", 1024, "2026-11-23 17:40"))),
    ("rt3_same", "the same program on runtime 3", "final", "1", "3", 1024, None, None),
    ("pieces", "4 pieces per micro-batch", "final", "3", "5", 1024, None,
     ("synthetic step", "chip_whole_steps.csv", ("micro_out.txt", "take", 64, 4, 1024, "splash (configured)"))),
    ("kernel", "one-tile kernel + ReLU² backward", "final", "5", "7", 1024, None,
     ("synthetic step, with the ce backward on", "chip_whole_steps.csv",
      ("wf/chip_kernels_0745.txt", "take", 64, 4, 1024, "onetile:save+relu2+ce"))),
    ("v9_take", "graft to v9, take table gradient", "v9", "7", "8", 1024, None,
     ("synthetic step", "chip_whole_steps.csv", ("v9_measA.txt", "take", 64, 4, 1024, "onetile:save+relu2"))),
    ("v9_switches", "flat, rows, tokn", "v9", "8", None, 1024, "short",
     ("synthetic step", "chip_whole_steps.csv", ("v9_measD.txt", "take", 64, 4, 1024, "onetile:save+relu2+flat+rows+tokn"))),
    ("v9_2048", "rows of 2,048 tokens", "v9", "8", None, 2048, "long", None),
    ("v9_tail", "chat-heavy tail", "v9", "8", None, None, "tail", None),
]
# Table 2's rows before run main16 (runtime 1): start-up benchmarks and single log lines, no live rate
PRE_STAGES = [
    ("reference", "reference implementation, runtime 1 (32-row micro-batches)", "final",
     ("start-up benchmark splash-remat1-ms32-onehot", 1024, "2026-11-23 03:27"),
     [("training (monitor), step 72", "2026-11-23 04:17"), ("training (monitor), step 301", "2026-11-23 04:28")]),
    ("rewrites", "the rewrites, runtime 1 (32-row micro-batches)", "final",
     ("start-up benchmark splash-remat1-ms32-onehot", 1024, "2026-11-23 04:36"),
     [("training, step 50", "2026-11-23 04:48")]),
]
# the live rates numbers.json gives for the same windows (cross-check of the definition)
NUMBERS_KEY = {"rt2_16rows": "rate.rt2_16rows", "rt3_same": "rate.rt3_same", "pieces": "rate.pieces",
               "kernel": "rate.kernel_to_graft", "v9_take": "rate.v9_take", "v9_switches": "rate.v9_switches_1024",
               "v9_2048": "rate.v9_switches_2048", "final_25300_29700": "rate.final_25300_29700"}


def stage_select(steps, cfg, lo, hi, seq, phase):
    return [s for s in steps if s["config"] == cfg and lo < s["step"] <= hi
            and (seq is None or s["seq_len"] == seq) and (phase is None or s["phase"] == phase)]


def stage_summary(sel):
    rates = [s["_rate"] for s in sel]          # the trainer's logged rate, unrounded
    ach = [s["interval_tok_per_s"] for s in sel if s.get("interval_tok_per_s")]
    return dict(live_median=round(statistics.median(rates)), live_n=len(rates), live_min=round(min(rates)),
                live_max=round(max(rates)), achieved_median=round(statistics.median(ach)) if ach else None,
                achieved_n=len(ach), first_step=sel[0]["step"], last_step=sel[-1]["step"],
                first_utc=sel[0]["utc"], last_utc=sel[-1]["utc"])


def throughput_stages(steps, lp, whole):
    numbers = json.load(open(P.NUMBERS)) if os.path.exists(P.NUMBERS) else {}
    rows = []
    order = 0
    for key, label, cfg, bench, singles in PRE_STAGES:
        what, seq, t = bench
        hit = [r for r in lp if r["what"] == what and r["seq_len"] == seq and r["utc"].startswith(t)]
        assert len(hit) == 1, (key, hit)
        sv = []
        for w, tt in singles:
            sv += [r for r in lp if r["what"] == w and r["utc"].startswith(tt)]
        order += 1
        rows.append(dict(order=order, key=key, part="before run main16", label=label, config=cfg, seq_len=seq,
                         synthetic_tok_per_s=hit[0]["tok_per_s"], synthetic_kind="start-up benchmark",
                         synthetic_source=f"throughput_log_points.csv ({hit[0]['source']}, {hit[0]['utc']})",
                         single_readings="; ".join(f"{r['tok_per_s']} ({r['what']}, {r['utc'][11:16]})" for r in sv),
                         single_readings_values=";".join(str(r["tok_per_s"]) for r in sv)))
    prev = None
    for key, label, cfg, lo_id, hi_id, seq, phase, syn in RUN_STAGES:
        lo = IV_STEP[lo_id] if lo_id else 0
        hi = IV_STEP[hi_id] if hi_id else 10 ** 9
        sel = stage_select(steps, cfg, lo, hi, seq, phase)
        if not sel:
            continue
        order += 1
        row = dict(order=order, key=key, part="run main16 (live)", label=label, config=cfg,
                   from_id=lo_id or "", from_step=lo if lo_id else "",
                   seq_len=seq if seq else sel[0]["seq_len"], phase=phase or sel[0]["phase"],
                   window=f"{lo} < step <= {hi if hi_id else 'end'}" + (f", {seq} tokens" if seq else "")
                   + (f", phase {phase}" if phase else ""), **stage_summary(sel))
        if syn:
            kind, csvname, selector = syn
            if csvname == "throughput_log_points.csv":
                what, sq, t = selector
                hit = [r for r in lp if r["what"] == what and r["seq_len"] == sq and r["utc"].startswith(t)]
                assert len(hit) == 1, (key, hit)
                row.update(synthetic_tok_per_s=hit[0]["tok_per_s"], synthetic_kind=kind,
                           synthetic_source=f"{csvname} ({hit[0]['source']}, {hit[0]['utc']})")
            else:
                src, tg, a, rws, t, kern = selector
                hit = [r for r in whole if r["source"] == src and r["table_grad"] == tg and r["a"] == a
                       and r["rows"] == rws and r["t"] == t and r["kernel"] == kern]
                assert len(hit) == 1, (key, hit)
                row.update(synthetic_tok_per_s=round(hit[0]["tok_per_s"]), synthetic_kind=kind,
                           synthetic_source=f"{csvname} ({src}, {a} x {rws} x {t}, {hit[0]['ms']} ms)")
        if prev is not None:
            row["delta_live"] = row["live_median"] - prev["live_median"]
            row["delta_live_pct"] = round(100 * (row["live_median"] / prev["live_median"] - 1), 4)
        nk = NUMBERS_KEY.get(key)
        if nk and nk in numbers:
            v = numbers[nk]["value"]
            row["numbers_json"] = f"{v['median_tok_per_s']} (n = {v['n']})"
            if v["median_tok_per_s"] != row["live_median"] or v["n"] != row["live_n"]:
                mismatch(f"live rate {key}: {row['live_median']} (n {row['live_n']}) here, numbers.json "
                         f"{v['median_tok_per_s']} (n {v['n']}); the figures and the paper's numbers use different "
                         "records (see --cutoff)")
        rows.append(row)
        prev = row
    # the live window of Table 3 (final after the deflection rule, steps 25,300-29,700)
    sel = stage_select(steps, "final", 25250, 29700, 1024, None)
    if sel:
        row = dict(order="", key="final_25300_29700", part="Table 3 window", label="final, steps 25,300-29,700",
                   config="final", seq_len=1024, phase="short", window="25250 < step <= 29700, 1024 tokens",
                   **stage_summary(sel))
        v = numbers.get("rate.final_25300_29700", {}).get("value")
        if v:
            row["numbers_json"] = f"{v['median_tok_per_s']} (n = {v['n']})"
            if v["median_tok_per_s"] != row["live_median"] or v["n"] != row["live_n"]:
                mismatch(f"live rate final_25300_29700: {row['live_median']} here, numbers.json {v['median_tok_per_s']}")
        rows.append(row)
    return rows


F2_NAMES = {"rt2_16rows": "runtime 2", "rt3_same": "runtime 3, before pieces", "pieces": "4 pieces per micro-batch",
            "kernel": "one-tile kernel", "v9_take": "v9", "v9_switches": "v9 with flat, rows, tokn",
            "v9_2048": "v9, rows of 2,048 tokens", "v9_tail": "v9, chat-heavy tail"}


def f2_segment_rows(steps):
    """Achieved-rate medians per stage under the stretch names make_tables.py reads (tables.md T2)."""
    out = []
    for key, label, cfg, lo_id, hi_id, seq, phase, syn in RUN_STAGES:
        lo = IV_STEP[lo_id] if lo_id else 0
        hi = IV_STEP[hi_id] if hi_id else 10 ** 9
        sel = [s for s in stage_select(steps, cfg, lo, hi, seq, phase) if s.get("interval_tok_per_s")]
        if not sel:
            continue
        vals = [s["interval_tok_per_s"] for s in sel]
        out.append(dict(stretch=F2_NAMES[key], first_step=sel[0]["step"], last_step=sel[-1]["step"], records=len(vals),
                        median_tok_per_s=round(statistics.median(vals), 1), min_tok_per_s=round(min(vals), 1),
                        max_tok_per_s=round(max(vals), 1), from_utc=sel[0]["utc"][:16], to_utc=sel[-1]["utc"][:16]))
    return out


# -------------------------------------------------------------------- F1: centred smoothed loss
def f1_smoothed(steps, events):
    """Per configuration, the step records cut into stretches at every restore and every stage event
    (restarts, the graft, the phase changes of the schedule), each stretch smoothed on its own with
    common.smooth_centred."""
    out = []
    for cfg in ("final", "v9"):
        rows = [s for s in steps if s["config"] == cfg]
        bounds = sorted({e["step"] for e in events if e["config"] == cfg and e["event"] in ("restore", "stage")})
        seg_of = [sum(1 for b in bounds if b < r["step"]) for r in rows]
        for seg in sorted(set(seg_of)):
            sr = [r for r, s in zip(rows, seg_of) if s == seg]
            sm, neff, half, off = C.smooth_centred([r["tokens"] for r in sr], [r["loss"] for r in sr])
            lo = max([b for b in bounds if b < sr[0]["step"]], default=0)
            for r, a, ne, h, o in zip(sr, sm, neff, half, off):
                out.append(dict(config=cfg, stretch=seg, stretch_from_step=lo, step=r["step"], tokens=r["tokens"],
                                seq_len=r["seq_len"], phase=r["phase"], assistant_only=r["assistant_only"],
                                loss_raw=r["loss"], loss_smoothed=round(a, 5), kernel_records_effective=round(ne, 2),
                                half_width_decades=round(h, 5), centre_offset_decades=f"{o:.2e}",
                                drawn=int(ne >= C.NEFF_MIN), loss_ema_trainer=r["loss_ema"],
                                kernel_sigma_decades=C.SIGMA_DEC, kernel_cut_sigmas=C.CUT, neff_min=C.NEFF_MIN))
    return out


# ------------------------------------------------------------------------------ held-out loss
DOMS = ("web", "chat", "proxy")


def rt_row(cfg, c, heldout_meta, source):
    bd = c["by_domain"]
    w = {d: bd[d]["weight"] for d in DOMS}
    wc = (bd["web"]["loss"] * w["web"] + bd["chat"]["loss"] * w["chat"]) / (w["web"] + w["chat"])
    rows_n = {d: (heldout_meta.get(d) or {}).get("rows") for d in DOMS}
    return dict(config=cfg, step=c["step"], checkpoint=c.get("name", ""), evaluated="runtime", variant="on",
                tokens=c["tokens"], train_minutes=round(c["train_seconds"] / 60, 1),   # as Table G1
                web=round(bd["web"]["loss"], 6), chat=round(bd["chat"]["loss"], 6), proxy=round(bd["proxy"]["loss"], 6),
                chat_assistant=round(bd["chat"]["assistant_loss"], 6), web_chat=round(wc, 6), all_rows=round(c["loss"], 6),
                w_web=w["web"], w_chat=w["chat"], w_proxy=w["proxy"], w_all=c.get("weight"),
                rows_web=rows_n["web"], rows_chat=rows_n["chat"], rows_proxy=rows_n["proxy"], source=source)


def heldout_rows(max_step):
    """Held-out loss of the checkpoints on the export check's rows: the runtime's evaluations
    (midrun_eval_1.json, then heldout_v9.jsonl for newer v9 checkpoints) and the local CPU
    evaluations of lead2_work/heldout_analysis.json (with the new paths switched off). Checkpoints
    after the cut-off step (max_step) are left out."""
    rows, seen = [], {}
    meta = {}
    mr = json.load(open(P.MIDRUN_EVAL))
    for run in mr["runs"]:
        cfg = "final" if run["config"].endswith("final.py") else "v9"
        meta = run.get("heldout") or meta
        for c in run["checkpoints"]:
            r = rt_row(cfg, c, run.get("heldout") or {}, "midrun_eval_1.json")
            seen[(cfg, c["step"])] = r
            rows.append(r)
    if os.path.exists(P.HELDOUT_V9):
        for d in read_jsonl(P.HELDOUT_V9):
            for run in (d.get("result") or {}).get("runs", []):
                cfg = "final" if run.get("config", "").endswith("final.py") else "v9"
                for c in run.get("checkpoints", []):
                    if "loss" not in c or "by_domain" not in c:
                        continue
                    r = rt_row(cfg, c, run.get("heldout") or meta, "heldout_v9.jsonl")
                    if (cfg, c["step"]) in seen:
                        o = seen[(cfg, c["step"])]
                        if abs(o["all_rows"] - r["all_rows"]) > 1e-6:
                            mismatch(f"held-out {cfg} {c['step']}: {o['source']} {o['all_rows']} vs heldout_v9.jsonl {r['all_rows']}")
                        continue
                    seen[(cfg, c["step"])] = r
                    rows.append(r)
    for local_file in (P.HELDOUT_LOCAL, getattr(P, "HELDOUT_LOCAL46", None)):
        if not local_file or not os.path.exists(local_file):
            continue
        loc = json.load(open(local_file))
        for name, s in sorted(loc.get("summary", {}).items()):
            cfg = "final" if name.startswith("final") else "v9"
            variant = name.rsplit("_", 1)[1]
            variant = {"on": "on", "pagesoff": "pages off", "pageoff": "token neurons off", "gainoff": "gains off"}.get(variant, variant)
            rt = seen.get((cfg, s["step"]))
            r = dict(config=cfg, step=s["step"], checkpoint="", evaluated="local", variant=variant, tokens=s["tokens"],
                     train_minutes=s["train_minutes"], web=round(s["web"], 6), chat=round(s["chat"], 6),
                     proxy=round(s["proxy"], 6), chat_assistant=round(s["chat_assistant"], 6),
                     web_chat=round(s["web_chat"], 6), all_rows=round(s["all"], 6),
                     w_web=rt["w_web"] if rt else "", w_chat=rt["w_chat"] if rt else "", w_proxy=rt["w_proxy"] if rt else "",
                     rows_web=rt["rows_web"] if rt else "", rows_chat=rt["rows_chat"] if rt else "",
                     rows_proxy=rt["rows_proxy"] if rt else "", source=P.rel(local_file) + " (" + name + ")")
            if variant == "on" and rt and abs(rt["all_rows"] - r["all_rows"]) > 1e-6:
                mismatch(f"held-out {cfg} {s['step']}: local {r['all_rows']} vs runtime {rt['all_rows']}")
            rows.append(r)
    late = sorted({r["step"] for r in rows if r["step"] > max_step})
    if late:
        print(f"  held-out evaluations of checkpoints after the cut-off left out: steps {late}")
    rows = [r for r in rows if r["step"] <= max_step]
    rows.sort(key=lambda r: (r["step"], r["evaluated"] != "runtime", r["variant"]))
    return rows


T975 = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469, 7: 2.3646, 8: 2.3060}  # t(0.975, dof)


def heldout_trend(H):
    """Table 7's line: least squares of final's web + chat held-out loss on training minutes, its
    95% prediction interval (t with n - 2 degrees of freedom), on a grid and at each v9 checkpoint."""
    F = sorted((r for r in H if r["config"] == "final" and r["evaluated"] == "runtime"), key=lambda r: r["step"])
    xs, ys = [r["train_minutes"] for r in F], [r["web_chat"] for r in F]
    n = len(xs)
    xm, ym = sum(xs) / n, sum(ys) / n
    sxx = sum((x - xm) ** 2 for x in xs)
    b = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sxx
    a = ym - b * xm
    dof = n - 2
    s = math.sqrt(sum((y - a - b * x) ** 2 for x, y in zip(xs, ys)) / dof)
    t = T975[dof]

    def pred(x0):
        return a + b * x0, t * s * math.sqrt(1 + 1 / n + (x0 - xm) ** 2 / sxx)

    V = [r for r in H if r["config"] == "v9" and r["variant"] == "on" and r["evaluated"] == "runtime"]
    xmax = max([r["train_minutes"] for r in V] + xs)
    grid = []
    x = xs[0]
    while x <= xmax + 1e-9:
        p, h = pred(x)
        grid.append(dict(train_minutes=round(x, 2), line=round(p, 6), pi95_low=round(p - h, 6), pi95_high=round(p + h, 6),
                         pi95_half=round(h, 6)))
        x += 0.5
    p, h = pred(xmax)
    grid.append(dict(train_minutes=round(xmax, 2), line=round(p, 6), pi95_low=round(p - h, 6), pi95_high=round(p + h, 6),
                     pi95_half=round(h, 6)))
    fit = dict(points=[r["step"] for r in F], intercept=a, slope_per_minute=b, residual_sd=s, n=n, dof=dof, t975=t)
    for r in H:
        if r["config"] == "v9" and r["variant"] == "on":
            p, h = pred(r["train_minutes"])
            r["final_line"] = round(p, 6)
            r["final_line_pi95_half"] = round(h, 6)
            r["gap_to_final_line"] = round(r["web_chat"] - p, 6)
    for g in grid:
        g.update(fit_points=" ".join(map(str, fit["points"])), fit_slope_per_minute=f"{b:.6e}",
                 fit_residual_sd=f"{s:.6e}", fit_dof=dof, fit_t975=t)
    # cross-check against the paper's Table 7 (lead2_work/heldout_analysis.json, line in minutes)
    if os.path.exists(P.HELDOUT_LOCAL):
        ex = json.load(open(P.HELDOUT_LOCAL)).get("extrapolation", {})
        for k, v in ex.items():
            if k.endswith(":last4_minutes"):
                pp, hh = pred(v["x0"])
                ok = abs(pp - v["final_predicted"]) < 1e-4 and abs(hh - v["pi95_halfwidth"]) < 1e-3
                print(f"  Table 7 line at {v['x0']} min: here {pp:.5f} +- {hh:.5f}, heldout_analysis.json "
                      f"{v['final_predicted']:.5f} +- {v['pi95_halfwidth']:.5f} {'(same)' if ok else ''}")
                if not ok:
                    mismatch(f"Table 7 line at {v['x0']} min differs from heldout_analysis.json")
    return grid, fit


# ---------------------------------------------------------------- cost model against the chip
def costmodel():
    """Tables 4 and 5 (lead2_work/predictor_test.json): leave-one-out predictions of the measured
    whole step by each static count, and the predicted against the measured differences."""
    d = json.load(open(P.PREDICTOR))
    rows = d["rows"]
    cols = {"cycles": 1, "bytes": 2, "flop": 3, "reads": 4}
    json_key = {"cycles": "estimated cycles", "bytes": "estimated HBM bytes", "flop": "analytic FLOP per token",
                "reads": "numbers read per token"}
    out = []
    for i, r in enumerate(rows):
        row = dict(config=r[0], est_mcycles_per_step=r[1], hbm_gb_per_step=round(r[2], 3), gflop_per_token=r[3],
                   reads_per_token=r[4], measured_ms=r[5], chip_source=r[6])
        for name, c in cols.items():
            tr = [q for j, q in enumerate(rows) if j != i]
            xs, ys = [q[c] for q in tr], [q[5] for q in tr]
            if max(xs) == min(xs):                     # every other configuration has the same count
                row[f"loo_{name}_ms"] = ""
                row[f"loo_{name}_err"] = ""
                row[f"loo_{name}_note"] = "undefined: the other five have the same value"
            else:
                n = len(xs)
                xm, ym = sum(xs) / n, sum(ys) / n
                b = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum((x - xm) ** 2 for x in xs)
                p = ym + b * (r[c] - xm)
                row[f"loo_{name}_ms"] = round(p, 2)
                row[f"loo_{name}_err"] = round((p - r[5]) / r[5], 4)
            jv = d["loo"][json_key[name]]["loo_rel_err"][i]
            row[f"json_loo_{name}_err"] = "" if jv is None else jv
            mine = row[f"loo_{name}_err"]
            if mine == "" and jv is not None:
                mismatch(f"predictor test, {name}, {r[0]}: predictor_test.json gives a leave-one-out error of {jv}, "
                         "but the fit is undefined (the other five configurations have the same value; the json's "
                         "number comes from a floating-point remainder of about 1e-32 in the sum of squares)")
            elif mine != "" and jv is not None and abs(mine - jv) > 0.0015:
                mismatch(f"predictor test, {name}, {r[0]}: {mine} here, {jv} in predictor_test.json")
        out.append(row)
    summ = {}
    for name in cols:
        e = [abs(r[f"loo_{name}_err"]) for r in out if r[f"loo_{name}_err"] != ""]
        summ[name] = (sum(e) / len(e), max(e), len(e))
        print(f"  leave-one-out {name}: mean {100 * summ[name][0]:.1f}%, largest {100 * summ[name][1]:.1f}% "
              f"over {summ[name][2]} configurations")
    diffs = [dict(change=x["change"], predicted_ms=x["predicted_ms"], measured_ms=x["measured_ms"],
                  ratio=x["ratio"], sign_agrees=int(x["predicted_ms"] * x["measured_ms"] > 0)) for x in d["diffs"]]
    return out, diffs, summ


# --------------------------------------------------------------------------- blind comparisons
OPPONENT = {"falcon": "Falcon-H1-Tiny-90M-Instruct", "mobilellm": "MobileLLM-R1.5-140M"}


def blind_rows(names, graft_step):
    rows = []
    for name in names:
        path = os.path.join(P.BLIND_DIR, name + ".json")
        if not os.path.exists(path):
            sys.exit(f"make_data.py: blind-comparison report {path} not found (names are file names without .json "
                     f"in {P.BLIND_DIR})")
        d = json.load(open(path))
        o = d["overall"]
        pairset, source = name.split("--", 1)
        m = re.search(r"-vs-([a-z0-9]+)$", pairset)
        opp = m.group(1) if m else ""
        sm = re.search(r"step(\d+)", d.get("x", ""))
        step = int(sm.group(1)) if sm else None
        judge = d.get("judge") or {}
        arm = (d.get("y", "").split("/", 1) + [""])[1].split("-")[0]
        rows.append(dict(name=name, study=re.sub(r"-vs-[a-z0-9]+$", "", pairset),
                         split="test" if pairset.startswith("final-") else "dev",
                         pairset=d.get("pairset", pairset), rating_source=source,
                         readings="two readings on exchanged sides" if "orders_agree" in judge else "one reading",
                         x=d.get("x"), y=d.get("y"), our_step=step,
                         our_config="" if step is None else ("final" if step <= graft_step else "v9"),
                         opponent=opp, opponent_label=OPPONENT.get(opp, opp), opponent_arm=arm,
                         n=o["n"], n_pairs=d.get("n_pairs"), n_rated=d.get("n_rated"), wins=o["wins"],
                         losses=o["losses"], ties=o["ties"], both_bad=o["both_bad"],
                         ties_or_both_bad=o["ties"] + o["both_bad"], score=o["score"], score_lo=o["score_ci95"][0],
                         score_hi=o["score_ci95"][1], decisive=o["decisive"], win_rate_decisive=o["win_rate_decisive"],
                         win_rate_decisive_lo=o["win_rate_decisive_ci95"][0],
                         win_rate_decisive_hi=o["win_rate_decisive_ci95"][1], sign_test_p=o["sign_test_p"],
                         orders_agree=judge.get("orders_agree", ""),
                         first_position_pref_mean=judge.get("first_position_pref_mean", ""),
                         source=os.path.relpath(path, P.N)))
        r = rows[-1]
        if r["wins"] + r["losses"] + r["ties_or_both_bad"] != r["n"]:
            mismatch(f"blind {name}: wins + losses + ties + both bad != n")
        if abs((r["wins"] + r["ties_or_both_bad"] / 2) / r["n"] - r["score"]) > 6e-5:
            mismatch(f"blind {name}: score {r['score']} != (wins + ties/2)/n")
    return rows


# ---------------------------------------------------------------------------------- the graft
def graft_facts():
    rows = []
    for path, where in P.GRAFT_CHECK_LOGS:
        shape = ""
        for line in open(path):
            m = re.search(r"check on (\d+) x (\d+) synthetic tokens", line)
            if m:
                shape = f"{m.group(1)} x {m.group(2)}"
            if line.startswith("GRAFT "):
                g = json.loads(line[6:])
                ck = g.get("check") or {}
                rows.append(dict(kind="check", step=g["source_step"], where=where, check_tokens=shape,
                                 source_loss=f"{ck.get('source'):.7f}",
                                 grafted_loss=f"{ck.get('grafted'):.7f}", difference=ck.get("difference"),
                                 config_hash=g["config_hash"], carried_leaves=g["copied_params"],
                                 carried_numbers=g["copied_param_numbers"], new_numbers=g["new_param_numbers"],
                                 source=os.path.relpath(path, P.N) if path.startswith(P.N) else path))
    gg = json.load(open(P.GRAFT_GRAD))
    for b, x in gg["batches"].items():
        s = x["summary"]
        zero = [k for k, v in x["new"].items() if v["exactly_zero"]]
        nonzero = {k: v["max_abs"] for k, v in x["new"].items() if not v["exactly_zero"]}
        rows.append(dict(kind="gradient", step=gg["source_step"], where="development container, CPU float32",
                         source_loss=f"{x['loss_final']:.7f}", grafted_loss=f"{x['loss_v9']:.7f}",
                         difference=x["loss_difference"], rows=x["rows"], batch=b,
                         shared_bit_identical=f"{s['shared_bit_identical']} of {s['shared']}",
                         new_exactly_zero=" ".join(zero),
                         new_nonzero_max_abs=" ".join(f"{k}={v:.2e}" for k, v in nonzero.items()),
                         source=P.rel(P.GRAFT_GRAD)))
    man = json.load(open(P.GRAFT_CKPT))
    g = man["meta"]["guard"]
    rows.append(dict(kind="guard", step=man["step"], where="checkpoint manifest meta.guard", guard_ema=round(g["ema"], 4),
                     guard_count=g["count"], source=P.rel(P.GRAFT_CKPT)))
    live = json.load(open(os.path.join(P.SRC_LIVE, "manifest.json")))
    gr = live.get("graft") or {}
    seed = next((o.split("=", 1)[1] for o in live.get("overrides") or [] if o.startswith("run.seed=")), "")
    rows.append(dict(kind="manifest", step=gr.get("step"), where="live run manifest: 'graft' record; run.seed from "
                     "'overrides'", p=gr.get("p"), tokens=gr.get("tokens"), checkpoint=gr.get("checkpoint"),
                     carried_numbers=gr.get("copied_param_numbers"), new_numbers=gr.get("new_param_numbers"),
                     seed=seed, source="data/_src/live/run/manifest.json"))
    runlog = os.path.join(P.SRC_LIVE, "..", "run.log")
    before_cap, fast, after = None, None, False
    for line in open(runlog):
        if line.startswith("===== "):          # a new start of the run script: its own switches follow
            fast = None if not after else fast
        m = re.match(r"fast: the speed switches of this run are (.*?) \(now on", line)
        if m and not after:
            fast = m.group(1)
        m = re.search(r"at most ([\d.]+) reads of", line)
        if m and not after and before_cap is None:
            before_cap = m.group(1)             # the loader's cap before the graft (max_epochs)
        if "this run continues configs/final.py from step" in line:
            after = True
        m = re.search(r"at most ([\d.]+) reads of [\d.e+]+ tokens over a horizon of ([\d.e+]+) tokens; block 0", line)
        if after and m:
            rows.append(dict(kind="loader", step=gr.get("step"), where="run.log: the cap before the graft (first loader "
                             "line), the first loader block after it, and the speed switches of the start that "
                             "continued the graft", reads_cap=m.group(1), reads_cap_before=before_cap,
                             horizon_tokens=m.group(2), fast_at_graft=fast or "", source="data/_src/live/run.log"))
            break
    return rows, gr


# ----------------------------------------------------------------------------------- data rules
def kinds():
    sys.path.insert(0, P.REPO)
    from bes.data import sources as registry   # read-only import of the project's registry
    return {s.name: s.kind for s in registry.SOURCES}


PATTERNS = ["canned_refusal_now", "cannot_any", "apology_open", "as_ai_open", "i_am_ai", "dont_have",
            "dont_have_access", "need_more", "sure_open"]


def openings_rows(kind_of):
    d = json.load(open(P.OPENINGS))
    rows = []
    for src, r in sorted(d.items()):
        if "error" in r:
            continue
        row = dict(source=src, kind=kind_of.get(src, ""), segments=r["segments"], sampled=r["sampled"],
                   conversations=r["conversations"], turns=r["turns"])
        for p in PATTERNS:
            n = r["patterns"].get(p, 0)
            row[f"{p}_turns"] = n
            row[f"{p}_pct_turns"] = round(100 * n / r["turns"], 4) if r["turns"] else ""
            row[f"{p}_conversations"] = r["conversations_with"].get(p, 0)
        rows.append(row)
    return rows


def deflection_rows(kind_of):
    d = json.load(open(P.DEFLECTION))
    rows = []
    for src, r in sorted(d.items()):
        n = r["conversations"]
        rows.append(dict(source=src, kind=kind_of.get(src, ""), conversations=n, refusal=r["refusal"],
                         deflection=r["deflection"], refusal_pct=round(100 * r["refusal"] / n, 4),
                         deflection_pct=round(100 * r["deflection"] / n, 4),
                         token_share_refusal_pct=round(100 * r["token_share_refusal"], 4),
                         token_share_deflection_pct=round(100 * r["token_share_deflection"], 4)))
    return rows


def loader_drop_rows(recs):
    """Per period and source: reads, refusal and deflection skips counted by the live loader.

    A step record carries the counters of its loader block so far (the block of 64 steps that holds
    the step, up to the last step the loader has prepared). For every (config, block) only the
    record with the largest 'last' is used, so no block is counted twice; blocks are partial (the
    counters stop at the prefetch horizon of the record), which is fine for shares."""
    best = {}
    for cfg, r in recs:
        if r.get("event") != "step":
            continue
        blk, first, last, fail, draws = block_fail_draw(r)
        if blk is None:
            continue
        key = (cfg, blk, first)
        if key not in best or last > best[key][0]:
            best[key] = (last, r["step"], fail, draws)
    periods = [("before rules", 0, 6647), ("refusal rule", 6647, 25280), ("refusal + deflection, final", 25280, 29722),
               ("refusal + deflection, v9", 29722, 10 ** 9)]
    agg = {}
    for (cfg, blk, first), (last, step, fail, draws) in best.items():
        per = next(name for name, lo, hi in periods if lo <= first < hi)
        if per.endswith("v9") != (cfg == "v9"):
            per = "refusal + deflection, v9" if cfg == "v9" else per
        srcs = set(draws) | {k.split("/")[0] for k in fail}
        for s in srcs:
            a = agg.setdefault((per, s), dict(period=per, source=s, blocks=0, draws=0, refusal=0, deflection=0,
                                              heldout=0, too_long=0, other_fail=0))
            a["blocks"] += 1
            a["draws"] += draws.get(s, 0)
            for k, v in fail.items():
                src, why = k.split("/", 1)
                if src != s:
                    continue
                if why in ("refusal", "deflection", "heldout", "too_long"):
                    a[why] += v
                else:
                    a["other_fail"] += v
    rows = []
    for (per, s), a in sorted(agg.items(), key=lambda kv: ([p[0] for p in periods].index(kv[0][0]), kv[0][1])):
        reads = a["draws"] + a["refusal"] + a["deflection"] + a["heldout"] + a["too_long"] + a["other_fail"]
        a["reads"] = reads
        a["refusal_pct_reads"] = round(100 * a["refusal"] / reads, 3) if reads else ""
        a["deflection_pct_reads"] = round(100 * a["deflection"] / reads, 3) if reads else ""
        rows.append(a)
    return rows


# ------------------------------------------------------------------------------------- ledger
def ledger_rows():
    sys.path.insert(0, P.REPO)
    from bes import budget as B
    from bes.roles import role_of

    def manifest(root):
        ck = os.path.join(root, "train", "ckpt")
        newest = sorted(os.listdir(ck))[-1]
        return newest, json.load(open(os.path.join(ck, newest, "manifest.json")))

    out = {}
    for cfg, root in (("final", P.SRC_FINAL), ("v9", P.SRC_LIVE)):
        name, m = manifest(root)
        slots = {}
        for leaf in m["leaves"]:
            if leaf["path"].startswith("state/opt_state/slots/"):
                rest = leaf["path"][len("state/opt_state/slots/"):]
                lp, slot = rest.rsplit("/", 1)
                slots.setdefault(lp.replace("%2F", "/"), []).append(slot)
        for leaf in m["leaves"]:
            p = leaf["path"]
            if not p.startswith("state/params/"):
                continue
            path = p[len("state/params/"):]
            shape = tuple(int(s) for s in leaf["shape"])
            size = math.prod(shape)
            role = role_of(path)
            row = B._ledger_row(path, role.kind)
            if role.kind == "dense":
                step = size
            else:
                reads, _ = B._lookup_rule(path)
                step = min(reads, shape[0]) * (size // shape[0])
            r = out.setdefault(path, dict(leaf=path, shape="x".join(map(str, shape)), numbers=size, row=row,
                                          kind=role.kind, group=role.group, read_per_step=step))
            r[f"in_{cfg}"] = 1
            r[f"ckpt_{cfg}"] = name
            r["optimiser_slots"] = "+".join(sorted(slots.get(path, [])))
    rows = sorted(out.values(), key=lambda r: (B.LEDGER_ORDER.index(r["row"]), r["leaf"]))
    for r in rows:
        r.setdefault("in_final", 0)
        r.setdefault("in_v9", 0)
    return rows


def export_accounting():
    d = json.load(open(os.path.join(P.SRC_FINAL, "final", "export.json")))
    return d["accounting"]


# ------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blind", default="", help="comma-separated extra blind-comparison report names")
    ap.add_argument("--no-extract", action="store_true", help="keep the records extracted last time")
    ap.add_argument("--cutoff", default="paper", help="newest record used: 'paper' (numbers.json's newest step, the "
                                                      "default), 'newest' (all records), or a step number")
    args = ap.parse_args()
    if not (args.cutoff in ("paper", "newest") or args.cutoff.isdigit()):
        sys.stderr.write(f"make_data.py: --cutoff {args.cutoff!r}: give paper, newest or a step number\n")
        sys.exit(2)
    extra = [x.strip() for x in args.blind.split(",") if x.strip()]
    extra = [x if "--" in x else f"{x}--{P.BLIND_DEFAULT_SOURCE}" for x in extra]
    missing = [x for x in P.BLIND_DEV + extra if not os.path.exists(os.path.join(P.BLIND_DIR, x + ".json"))]
    if missing:                                  # checked before anything is written
        sys.stderr.write("make_data.py: blind-comparison report(s) not found in " + P.BLIND_DIR + ": "
                         + ", ".join(m + ".json" for m in missing) + "\n(names are the report file names without "
                         ".json; a name without '--' gets '--" + P.BLIND_DEFAULT_SOURCE + "' appended)\n")
        sys.exit(2)

    if not args.no_extract:
        extract()
    recs_all = train_records()
    cut = cutoff_of(args.cutoff, recs_all)      # exits (4) before anything is written if the snapshot is too old
    recs = apply_cutoff(recs_all, cut)
    os.makedirs(P.DATA, exist_ok=True)
    print(f"cut-off: step {cut['step']:,} at {cut['utc']} UTC ({cut['kind']}; {cut['source']}); "
          f"{cut['step_records_after_cutoff']} newer step records of the snapshot left out")
    write_csv("cutoff.csv", [cut], ["kind", "config", "step", "utc", "time_s", "p", "loss", "seq_len", "source",
                                    "step_records_after_cutoff", "newest_step_in_snapshot", "newest_utc_in_snapshot",
                                    "snapshot_created_utc", "snapshot_steps"])
    steps, events = steps_and_events(recs)
    step_cols = ["config", "step", "tokens", "time_s", "utc", "p", "lr", "loss", "loss_ema", "seq_len", "phase",
                 "assistant_only", "chat_frac", "library", "tok_per_s", "step_s", "interval_steps", "interval_s",
                 "interval_tok_per_s", "grad_norm", "weight", "micro_tokens", "tape_gain", "tape_mass", "action",
                 "block", "block_first", "block_last"]
    write_csv("train_steps.csv", steps, step_cols)
    write_csv("run_events.csv", events, ["config", "event", "step", "time_s", "utc", "reason", "path", "p", "phase",
                                          "seq_len", "assistant_only", "lost_s", "compile_s", "ledger_train_s",
                                          "ledger_lost_s", "ledger_setup_s", "train_end_utc", "deadline_utc",
                                          "micro_split"])
    iv = intervention_rows(steps, events)
    write_csv("interventions.csv", iv, ["id", "group", "kind", "step", "tokens", "short", "end_utc", "restore_utc",
                                        "event_utc", "restore_attempts", "stopped_s", "phase", "p", "first_step_after",
                                        "first_step_after_utc", "config_after", "long", "src"])
    now_t = P.T0 + max(s["time_s"] for s in steps)
    write_csv("tpu_intervals.csv", tpu_intervals(now_t), ["runtime", "start_utc", "end_utc", "open",
                                                           "counted_until_utc", "seconds", "hours", "ledger_note",
                                                           "summary"])
    kr, ks, _ = kernel_rows()
    write_csv("chip_kernels_0745.csv", kr, ["source", "config", "kernel", "label", "t", "rows", "tokens_per_piece",
                                            "ms", "median_check", "q1", "q3", "min", "max", "n", "tok_per_s",
                                            "loss", "loss_diff", "grad_rel_max", "grad_rel_l2", "speedup",
                                            "reference", "compile_s"])
    vp, vs = v9_rows()
    write_csv("chip_v9_pieces.csv", vp, ["source", "config", "table_grad", "kernel", "t", "rows", "ms",
                                         "median_check", "q1", "q3", "min", "max", "n", "tok_per_s", "loss",
                                         "loss_diff", "grad_rel_max", "grad_rel_l2"])
    step_meas = ks + vs + micro_rows()
    write_csv("chip_whole_steps.csv", step_meas, ["source", "measured_utc", "config", "table_grad", "kernel", "a",
                                                  "rows", "t", "ms", "tok_per_s", "loss", "compile_s", "timing"])
    write_csv("chip_vmem_flags.csv", flag_rows(), ["source", "variant", "vmem_limit_mib", "a", "rows", "t", "ms",
                                                   "tok_per_s", "loss"])
    write_csv("chip_tune_2311.csv", tune_rows(), ["source", "t", "variant", "ms", "tok_per_s", "loss"])
    lp = log_points()
    write_csv("throughput_log_points.csv", lp, ["utc", "config", "program", "what", "seq_len",
                                                "tok_per_s", "step_s", "source"])
    print("\nchecks: live rates by stage (Table 2)")
    ts = throughput_stages(steps, lp, step_meas)
    write_csv("throughput_stages.csv", ts, ["order", "key", "part", "label", "config", "from_id", "from_step", "seq_len",
                                            "phase", "window",
                                            "live_median", "live_n", "live_min", "live_max", "achieved_median",
                                            "achieved_n", "first_step", "last_step", "first_utc", "last_utc",
                                            "delta_live", "delta_live_pct", "synthetic_tok_per_s", "synthetic_kind",
                                            "synthetic_source", "single_readings", "single_readings_values",
                                            "numbers_json"])
    gaps = []
    for r in ts:
        if r.get("live_median") and r.get("achieved_median"):
            g = 100 * (r["achieved_median"] / r["live_median"] - 1)
            gaps.append((abs(g), r["key"]))
            print(f"  {r['key']:18s} live {r['live_median']:,} (n {r['live_n']}), achieved {r['achieved_median']:,} "
                  f"(n {r['achieved_n']}): {g:+.3f}%; steps {r['first_step']:,}-{r['last_step']:,}, "
                  f"{r['first_utc']} - {r['last_utc'][11:]}")
    if gaps:
        print(f"  largest gap between the live and the achieved median: {max(gaps)[0]:.3f}% ({max(gaps)[1]})")
    write_csv("f2_segments.csv", f2_segment_rows(steps), ["stretch", "first_step", "last_step", "records",
                                                          "median_tok_per_s", "min_tok_per_s", "max_tok_per_s",
                                                          "from_utc", "to_utc"])
    f1 = f1_smoothed(steps, events)
    write_csv("f1_loss_smoothed.csv", f1, ["config", "stretch", "stretch_from_step", "step", "tokens", "seq_len", "phase",
                                           "assistant_only", "loss_raw", "loss_smoothed", "kernel_records_effective",
                                           "half_width_decades", "centre_offset_decades", "drawn", "loss_ema_trainer",
                                           "kernel_sigma_decades", "kernel_cut_sigmas", "neff_min"])
    print("\nchecks: held-out loss (Table 7, G1)")
    H = heldout_rows(cut["step"])
    grid, fit = heldout_trend(H)
    write_csv("heldout_loss.csv", H, ["config", "step", "checkpoint", "evaluated", "variant", "tokens", "train_minutes",
                                      "web", "chat", "proxy", "chat_assistant", "web_chat", "all_rows", "w_web",
                                      "w_chat", "w_proxy", "w_all", "rows_web", "rows_chat", "rows_proxy",
                                      "final_line", "final_line_pi95_half", "gap_to_final_line", "source"])
    write_csv("heldout_trend.csv", grid, ["train_minutes", "line", "pi95_low", "pi95_high", "pi95_half", "fit_points",
                                          "fit_slope_per_minute", "fit_residual_sd", "fit_dof", "fit_t975"])
    print("\nchecks: cost model against the chip (Tables 4, 5)")
    cm, cd, _ = costmodel()
    write_csv("costmodel_steps.csv", cm, ["config", "measured_ms", "chip_source", "est_mcycles_per_step",
                                          "hbm_gb_per_step", "gflop_per_token", "reads_per_token",
                                          "loo_bytes_ms", "loo_bytes_err", "json_loo_bytes_err",
                                          "loo_cycles_ms", "loo_cycles_err", "json_loo_cycles_err",
                                          "loo_flop_ms", "loo_flop_err", "loo_flop_note", "json_loo_flop_err",
                                          "loo_reads_ms", "loo_reads_err", "loo_reads_note", "json_loo_reads_err"])
    write_csv("costmodel_differences.csv", cd, ["change", "predicted_ms", "measured_ms", "ratio", "sign_agrees"])
    gf, graft = graft_facts()
    write_csv("graft_facts.csv", gf, ["kind", "step", "where", "check_tokens", "source_loss", "grafted_loss",
                                      "difference", "config_hash", "carried_leaves", "carried_numbers", "new_numbers",
                                      "batch", "rows", "shared_bit_identical", "new_exactly_zero",
                                      "new_nonzero_max_abs", "guard_ema", "guard_count", "p", "tokens", "checkpoint",
                                      "seed", "reads_cap", "reads_cap_before", "horizon_tokens", "fast_at_graft",
                                      "source"])
    print("\nchecks: blind comparisons")
    names = P.BLIND_DEV + [x for x in extra if x not in P.BLIND_DEV]
    br = blind_rows(names, int(graft.get("step") or 0))
    write_csv("blind_comparisons.csv", br, ["name", "study", "split", "pairset", "rating_source", "readings", "x", "y",
                                            "our_step", "our_config", "opponent", "opponent_label", "opponent_arm", "n",
                                            "n_pairs", "n_rated", "wins", "ties", "both_bad", "ties_or_both_bad", "losses",
                                            "score", "score_lo", "score_hi", "decisive", "win_rate_decisive",
                                            "win_rate_decisive_lo", "win_rate_decisive_hi", "sign_test_p",
                                            "orders_agree", "first_position_pref_mean", "source"])
    others = sorted(f[:-5] for f in os.listdir(P.BLIND_DIR) if f.startswith("final-") and f.endswith(".json")
                    and f[:-5] not in names)
    if others:
        print("  test-split reports on disk that were NOT added (pass them with --blind): " + ", ".join(others))
    kind_of = kinds()
    op = openings_rows(kind_of)
    cols = ["source", "kind", "segments", "sampled", "conversations", "turns"]
    for p in PATTERNS:
        cols += [f"{p}_turns", f"{p}_pct_turns", f"{p}_conversations"]
    write_csv("openings_scan.csv", op, cols)
    write_csv("deflection_check2.csv", deflection_rows(kind_of),
              ["source", "kind", "conversations", "refusal", "deflection", "refusal_pct", "deflection_pct",
               "token_share_refusal_pct", "token_share_deflection_pct"])
    write_csv("loader_drops_live.csv", loader_drop_rows(recs),
              ["period", "source", "blocks", "reads", "draws", "refusal", "deflection", "heldout", "too_long",
               "other_fail", "refusal_pct_reads", "deflection_pct_reads"])
    led = ledger_rows()
    write_csv("ledger_leaves.csv", led, ["leaf", "shape", "numbers", "row", "kind", "group", "read_per_step",
                                         "in_final", "in_v9", "optimiser_slots", "ckpt_final", "ckpt_v9"])
    acc = export_accounting()
    write_csv("ledger_final_export.csv", acc["rows"], ["row", "total", "step"])

    # ---------------------------------------------------------------------------------- checks
    print("\nchecks")
    fin = [s for s in steps if s["config"] == "final"]
    v9 = [s for s in steps if s["config"] == "v9"]
    print(f"  final step records {len(fin)}: steps {fin[0]['step']}..{fin[-1]['step']}; v9 {len(v9)}: "
          f"{v9[0]['step'] if v9 else '-'}..{v9[-1]['step'] if v9 else '-'} (newest {v9[-1]['utc'] if v9 else '-'} UTC, "
          f"p {v9[-1]['p']:.4f})")
    led_final = json.load(open(os.path.join(P.SRC_FINAL, "train", "ledger.json")))
    assert led_final["tokens"] == led_final["steps"] * P.TOKENS_PER_STEP, "tokens != steps * 262144"
    print(f"  final ledger: {led_final['steps']} steps, {led_final['tokens']:,} tokens "
          f"(= steps x {P.TOKENS_PER_STEP:,}); train {led_final['seconds']['train']:.2f} s, "
          f"lost {led_final['seconds']['lost']:.2f} s")
    fsum = sum(r["numbers"] for r in led if r["in_final"])
    vsum = sum(r["numbers"] for r in led if r["in_v9"])
    shared = sum(r["numbers"] for r in led if r["in_final"] and r["in_v9"])
    new = sum(r["numbers"] for r in led if r["in_v9"] and not r["in_final"])
    vread = sum(r["read_per_step"] for r in led if r["in_v9"])
    fread = sum(r["read_per_step"] for r in led if r["in_final"])
    print(f"  ledger from checkpoints: final {sum(r['in_final'] for r in led)} leaves {fsum:,} (read {fread:,}); "
          f"v9 {sum(r['in_v9'] for r in led)} leaves {vsum:,} (read {vread:,}); shared {shared:,}; new {new:,}")
    assert fsum == 34531720 and vsum == 98036104 and shared == 34531720 and new == 63504384 and vread == 19631240
    exp_rows = {r["row"]: r for r in acc["rows"]}
    assert acc["total"] == 98300111 and acc["active_step"] == 19278959
    assert fsum + exp_rows["library"]["total"] + exp_rows["index"]["total"] == acc["total"]
    print(f"  final export accounting: total {acc['total']:,}, read {acc['active_step']:,} (matches leaves + library + index)")
    for r in iv:
        print(f"  {str(r['id']):>2} step {r['step']:>6}  end {r['end_utc'][11:] or '-':>8}  restore {r['restore_utc'][11:] or '-':>8}"
              f"  event {r['event_utc'][11:] or '-':>8}  attempts {r['restore_attempts']}  stopped {r.get('stopped_s', '-')} s"
              f"  {r['short']}")
    for e in events:
        if e["event"] == "setup":
            print(f"  setup {e['config']:5s} step {e['step']:>6} {e['utc']}  train_end {e['train_end_utc']}  "
                  f"compile {e['compile_s']} s  micro_split {e['micro_split']}")
    print(f"\n{len(MISMATCHES)} mismatch(es)" + ("" if not MISMATCHES else ": see the MISMATCH lines above"))


if __name__ == "__main__":
    main()
