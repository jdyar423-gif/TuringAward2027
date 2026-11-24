"""Build every CSV of paper/data/ from the sources (run with the venv's python -I -B).

    python -I -B make_data.py

Reads the records extracted from the snapshots (extract_records.py; paths in paths.py), the chip
measurement outputs, the TPU ledger, the data-rule scans and the checkpoint manifests. Writes
CSVs into paper/data/ and prints the checks it makes. Nothing outside paper/ is written.
"""
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
from interventions import INTERVENTIONS, MEASUREMENTS  # noqa: E402


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
                       chat_frac=r["chat_frac"], library=int(bool(r["library"])),
                       tok_per_s=round(r["tok_per_s"], 1), step_s=round(r["step_s"], 5),
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
    for it in entries:
        s = it["step"]
        r_at = [e for e in restores if e["step"] == s]
        e_at = [e for e in ends if e["step"] == s]
        first_after = next((x for x in steps if x["step"] > s), None)
        row = dict(id=it["id"], group=it["group"], kind=it["kind"], step=s, tokens=s * P.TOKENS_PER_STEP,
                   short=it["short"], long=it["long"], src=it["src"],
                   end_utc=e_at[-1]["utc"] if e_at else "", restore_utc=r_at[-1]["utc"] if r_at else "",
                   restore_attempts=len(r_at),
                   first_step_after=first_after["step"] if first_after else "",
                   first_step_after_utc=first_after["utc"] if first_after else "",
                   config_after=r_at[-1]["config"] if r_at else "")
        if e_at and r_at:
            row["stopped_s"] = round(r_at[-1]["time_s"] - e_at[-1]["time_s"], 1)
        rows.append(row)
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
                              loss=s["loss"], compile_s=s["compile_s"]))
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
                                  loss=s["loss"], compile_s=s["compile_s"]))
    return pieces, steps


def micro_rows():
    d = json_line(P.MICRO_OUT, "MICRO")
    return [dict(source="micro_out.txt", measured_utc="2026-11-24 03:09-03:15", config="final",
                 table_grad="take", kernel="splash (configured)", a=v["a"], rows=v["rows"], t=v["t"],
                 ms=round(v["ms"], 4), tok_per_s=round(v["tokens_per_s"], 1), loss=v["loss"])
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
                    rows.append(dict(source=os.path.relpath(path, P.N) if path.startswith(P.N) else path,
                                     utc=t.strftime("%Y-%m-%d %H:%M:%S"), config=cfg, program=prog,
                                     what="start-up benchmark " + m.group(2), seq_len=int(m.group(3)),
                                     tok_per_s=int(m.group(4).replace(",", "")), step_s=float(m.group(5))))
                    continue
                m = step.match(line)
                if m and int(m.group(2)) == 50:
                    t = t_head + dt.timedelta(minutes=float(m.group(1)))
                    rows.append(dict(source=path, utc=t.strftime("%Y-%m-%d %H:%M:%S"), config=cfg, program=prog,
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


# ------------------------------------------------------------------------------ held-out loss
def heldout_rows():
    """Held-out loss of the checkpoints (scripts/midrun_eval.py on the runtime's host cores: web 64,
    chat 64, proxy 16 rows; float32, dense attention), from its printed table, plus newer automatic
    evaluations of v9 checkpoints in heldout_v9.jsonl when they hold a checkpoint the log does not."""
    rows = []
    cfg = None
    rx = re.compile(r"^\s+ckpt-\d+-step-(\d+)\s+([\d.e+]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
    with open(P.MIDEVAL_LOG) as f:
        for line in f:
            if line.startswith("/content/runs/main16_final  ("):
                cfg = "final"
            elif line.startswith("/content/runs/main16  ("):
                cfg = "v9"
            m = rx.match(line)
            if m and cfg:
                step = int(m.group(1))
                rows.append(dict(config=cfg, checkpoint=line.split()[0], step=step, tokens=step * P.TOKENS_PER_STEP,
                                 train_minutes=float(m.group(3)), loss=float(m.group(4)), web=float(m.group(5)),
                                 chat=float(m.group(6)), proxy=float(m.group(7)), chat_assistant=float(m.group(8)),
                                 source="mideval_last.log"))
    seen = {(r["config"], r["step"]) for r in rows}
    if os.path.exists(P.HELDOUT_V9):
        for line in open(P.HELDOUT_V9):
            if not line.strip():
                continue
            d = json.loads(line)
            for run in (d.get("result") or {}).get("runs", []):
                if run.get("config") != "configs/v9.py":
                    continue
                for c in run.get("checkpoints", []):
                    step = int(c.get("step", -1))
                    if ("v9", step) in seen or step < 0 or "loss" not in c:
                        continue
                    dom = c.get("by_domain") or {}
                    rows.append(dict(config="v9", checkpoint=c.get("name", ""), step=step, tokens=c.get("tokens"),
                                     train_minutes=round(c.get("train_seconds", 0) / 60, 1) if c.get("train_seconds") else "",
                                     loss=round(c["loss"], 4), web=round((dom.get("web") or {}).get("loss", 0), 4),
                                     chat=round((dom.get("chat") or {}).get("loss", 0), 4),
                                     proxy=round((dom.get("proxy") or {}).get("loss", 0), 4),
                                     chat_assistant=round((dom.get("chat") or {}).get("assistant_loss") or 0, 4),
                                     source="heldout_v9.jsonl"))
                    seen.add(("v9", step))
    rows.sort(key=lambda r: (r["step"]))
    return rows


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
    os.makedirs(P.DATA, exist_ok=True)
    recs = train_records()
    steps, events = steps_and_events(recs)
    step_cols = ["config", "step", "tokens", "time_s", "utc", "p", "lr", "loss", "loss_ema", "seq_len", "phase",
                 "chat_frac", "library", "tok_per_s", "step_s", "interval_steps", "interval_s",
                 "interval_tok_per_s", "grad_norm", "weight", "micro_tokens", "tape_gain", "tape_mass", "action",
                 "block", "block_first", "block_last"]
    write_csv("train_steps.csv", steps, step_cols)
    write_csv("run_events.csv", events, ["config", "event", "step", "time_s", "utc", "reason", "path", "p",
                                          "lost_s", "compile_s", "ledger_train_s", "ledger_lost_s",
                                          "ledger_setup_s", "train_end_utc", "deadline_utc", "micro_split"])
    iv = intervention_rows(steps, events)
    write_csv("interventions.csv", iv, ["id", "group", "kind", "step", "tokens", "short", "end_utc", "restore_utc",
                                        "restore_attempts", "stopped_s", "first_step_after",
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
    write_csv("heldout_loss.csv", heldout_rows(), ["config", "checkpoint", "step", "tokens", "train_minutes", "loss",
                                                   "web", "chat", "proxy", "chat_assistant", "source"])
    step_meas = ks + vs + micro_rows()
    write_csv("chip_whole_steps.csv", step_meas, ["source", "measured_utc", "config", "table_grad", "kernel", "a",
                                                  "rows", "t", "ms", "tok_per_s", "loss", "compile_s"])
    write_csv("chip_vmem_flags.csv", flag_rows(), ["source", "variant", "vmem_limit_mib", "a", "rows", "t", "ms",
                                                   "tok_per_s", "loss"])
    write_csv("chip_tune_2311.csv", tune_rows(), ["source", "t", "variant", "ms", "tok_per_s", "loss"])
    write_csv("throughput_log_points.csv", log_points(), ["utc", "config", "program", "what", "seq_len",
                                                          "tok_per_s", "step_s", "source"])
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
          f"{v9[0]['step'] if v9 else '-'}..{v9[-1]['step'] if v9 else '-'}")
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
              f"  attempts {r['restore_attempts']}  stopped {r.get('stopped_s', '-')} s  {r['short']}")
    for e in events:
        if e["event"] == "setup":
            print(f"  setup {e['config']:5s} step {e['step']:>6} {e['utc']}  train_end {e['train_end_utc']}  "
                  f"compile {e['compile_s']} s  micro_split {e['micro_split']}")


if __name__ == "__main__":
    main()
