"""Write paper/tables.md (T1-T5) from the CSVs of paper/data/ and the curated rows below.

Every curated number carries its source in the row; numbers read from CSVs are not retyped.
"""
import datetime as dt
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import paths as P  # noqa: E402

UTC = dt.timezone.utc


def T(s):
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)


def n(x):
    return f"{int(round(float(x))):,}"


def hms(sec):
    sec = int(round(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h} h {m:02d} min {s:02d} s" if h else f"{m} min {s:02d} s"


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def t1():
    L = C.read("ledger_leaves.csv")
    E = {r["row"]: r for r in C.read("ledger_final_export.csv")}
    names = {"core": "looped core: W_q, W_k, W_v, W_o, W_up, W_down of 2 blocks",
             "head": "output head: P (768 × 512), E_out (8,192 × 512)",
             "small": "norm gains, lanes, loop gains, sink, value gate, trust and candidate projections, critic",
             "emb": "input embedding (8,192 × 768)",
             "ve": "per-loop value embeddings (8,192 × 8 × 128)",
             "gain": "attention-output gains gain/o (8,192 × 2 × 768)",
             "page": "token neurons: page/key, page/val (8,192 × 2 × 8 × 192 each), page/pin, page/pout (2 × 768 × 192 each)",
             "tables": "register (15 fields), candidate-input and trust tables"}
    rows = []
    tot = {"f": [0, 0], "v": [0, 0]}
    for key in ("core", "head", "small", "emb", "ve", "gain", "page", "tables"):
        lf = [r for r in L if r["row"] == key and r["in_final"] == "1"]
        lv = [r for r in L if r["row"] == key and r["in_v9"] == "1"]
        fs, fr = sum(int(r["numbers"]) for r in lf), sum(int(r["read_per_step"]) for r in lf)
        vs, vr = sum(int(r["numbers"]) for r in lv), sum(int(r["read_per_step"]) for r in lv)
        slots = sorted({r["optimiser_slots"] or "none (frozen)" for r in lv})
        tot["f"][0] += fs
        tot["f"][1] += fr
        tot["v"][0] += vs
        tot["v"][1] += vr
        rows.append([key, names[key], len(lf), n(fs) if fs else "–", n(fr) if fs else "–", len(lv), n(vs) if vs else "–",
                     n(vr) if vs else "–", ", ".join(s.replace("m+v", "Adam m, v").replace("mom+norm", "MuonH momentum, norm")
                                                    for s in slots)])
    for key, label in (("library", "written library: passages, question-answer layer, directory, titles, gate"),
                       ("index", "retrieval index (BM25)")):
        r = E[key]
        tot["f"][0] += int(r["total"])
        tot["f"][1] += int(r["step"])
        rows.append([key, label, "–", n(r["total"]), n(r["step"]) + " (worst case, retrieving step)", "–", "–", "–",
                     "not trained"])
    rows.append(["**total**", "", 59, f"**{n(tot['f'][0])}**", f"**{n(tot['f'][1])}**", 64, f"**{n(tot['v'][0])}**",
                 f"**{n(tot['v'][1])}**", ""])
    head = ["row", "what", "final leaves", "final stored", "final read per token", "v9 leaves", "v9 stored",
            "v9 read per token", "optimiser slots"]
    return table(rows, head), tot


def t2():
    seg = {r["stretch"]: r for r in C.read("f2_segments.csv")}
    LP = C.read("throughput_log_points.csv")
    W = C.read("chip_whole_steps.csv")
    K = {(r["kernel"], r["t"]): r for r in C.read("chip_kernels_0745.csv")}
    tune = {(r["t"], r["variant"]): r for r in C.read("chip_tune_2311.csv")}

    def lp(what, t, cfg="final", prog=None):
        hit = [r for r in LP if r["what"] == what and r["utc"].startswith(t) and r["config"] == cfg
               and (prog is None or r["program"] == prog)]
        assert len(hit) == 1, (what, t, hit)
        return int(hit[0]["tok_per_s"])

    def med(name):
        return float(seg[name]["median_tok_per_s"])

    r2, r3a, r3b, r3c, r9 = (med("runtime 2"), med("runtime 3, before pieces"), med("4 pieces per micro-batch"),
                             med("one-tile kernel"), med("v9"))
    r9s = med("v9 with flat, rows, tokn") if "v9 with flat, rows, tokn" in seg else None
    old = lp("start-up benchmark splash-remat1-ms32-onehot", "2026-11-23 03:27")
    new = lp("start-up benchmark splash-remat1-ms32-onehot", "2026-11-23 04:36")
    take = lp("start-up benchmark splash-remat1-ms32-take", "2026-11-23 04:42")
    b16 = lp("start-up benchmark splash-remat1-ms16-take", "2026-11-23 17:40")
    w = {(r["source"], r["a"], r["rows"], r["t"], r["table_grad"]): float(r["ms"]) for r in W}
    m16 = w[("micro_out.txt", "16", "16", "1024", "take")]
    m4 = w[("micro_out.txt", "64", "4", "1024", "take")]
    k_ref, k_new = float(K[("", "1024")]["ms"]), float(K[("onetile:save+relu2", "1024")]["ms"])
    f_step = w[("wf/chip_kernels_0745.txt", "64", "4", "1024", "take")]
    v_step = w[("v9_measA.txt", "64", "4", "1024", "take")]
    v_piece = float([r for r in C.read("chip_v9_pieces.csv") if r["table_grad"] == "take" and r["t"] == "1024"][0]["ms"])
    t_cfg, t_16 = tune[("1024", "configured")], tune[("1024", "micro_seqs=16 (information only)")]
    rows = [
        ["23 Nov 03:27", "1", "final, code before the glue rewrite: start-up benchmark (32-row micro-batches)",
         n(old), "baseline; the plan had assumed 300,000–450,000",
         "final.log; docs/CHANGES.md 19.1"],
        ["23 Nov 04:36", "1", "glue rewrite: rotation as two products, no slices, concatenations or transposes, "
         "bfloat16 feed-forward pre-activations, tuple lanes, tiles 1024/1024/512 with the fused backward kernel; "
         "same values to float rounding", f"{n(new)} (take: {n(take)})",
         "one value-and-gradient of 32 × 1,024: 289.3 → 162.0 ms (1.79×); 62% of a block was glue",
         "docs/tpu_profile/final_b.run.log; docs/TPU_NOTES.md"],
        ["23 Nov 17:40 / 17:48", "2", "micro-batches of 16 rows (optim.micro_seqs=16)",
         f"benchmark {n(b16)}; live {r2:,.0f}",
         f"tune: {t_cfg['ms']} ms per 32 × 1,024 ({n(t_cfg['tok_per_s'])}) against {t_16['ms']} ms per 16 × 1,024 "
         f"({n(t_16['tok_per_s'])}), +8%", "docs/tpu_profile/tpu_tune.v6e.txt; run.log; train_log.jsonl"],
        ["24 Nov 01:55", "3", "same program on the third runtime", f"live {r3a:,.0f}",
         f"{100 * (r3a / r2 - 1):+.1f}% against runtime 2; not investigated", "train_log.jsonl"],
        ["24 Nov 03:31", "3", "each micro-batch cut into 4 pieces on the device (micro_split=4)", f"live {r3b:,.0f}",
         f"synthetic step 16 × 16 × 1,024 {m16:,.1f} ms → 64 × 4 × 1,024 {m4:,.1f} ms "
         f"({100 * (m16 / m4 - 1):+.1f}%); live {100 * (r3b / r3a - 1):+.1f}%", "micro_out.txt; train_log.jsonl"],
        ["24 Nov 07:53", "3", "one-tile attention kernel with saved forward, hand-written ReLU² backward",
         f"live {r3c:,.0f}", f"per piece {k_ref:.3f} → {k_new:.3f} ms (×{k_ref / k_new:.3f}); whole synthetic step "
         f"{f_step:,.1f} ms (with ce on); live {100 * (r3c / r3b - 1):+.1f}%",
         "wf/chip_kernels_0745.txt; train_log.jsonl"],
        ["24 Nov 09:59", "3", "graft to v9 (token pages, output gains; take table gradient)", f"live {r9:,.0f}",
         f"per piece {v_piece:.2f} ms against {k_new:.2f} (×{v_piece / k_new:.2f}); whole step {v_step:,.1f} against "
         f"{f_step:,.1f} ms (×{v_step / f_step:.2f}); live {r9 / r3c:.3f} of final",
         "v9_measA.txt; train_log.jsonl"],
    ]
    if r9s:
        vs = {r["kernel"]: float(r["ms"]) for r in W if r["source"] == "v9_measD.txt"}
        rows.append(["24 Nov 10:57", "3", "v9 speed switches flat, rows, tokn (bit-identical values for flat and rows; "
                     "float32 summation order for tokn)", f"live {r9s:,.0f} ({seg['v9 with flat, rows, tokn']['records']} "
                     "interval readings so far)",
                     f"whole step {v_step:,.1f} → {vs['onetile:save+relu2+flat']:,.1f} (flat) → "
                     f"{vs['onetile:save+relu2+flat+rows']:,.1f} (rows) → {vs['onetile:save+relu2+flat+rows+tokn']:,.1f} ms "
                     f"(tokn); live {100 * (r9s / r9 - 1):+.1f}%", "v9_measD.txt; docs/CHANGES.md 19.7; train_log.jsonl"])
    head = ["when (UTC)", "runtime", "change", "tokens per second", "measurement behind it", "source"]
    return table(rows, head)


def t3():
    K = C.read("chip_kernels_0745.csv")
    by = {(r["kernel"], r["t"]): r for r in K}
    order = [("", "splash kernel (configured)"), ("+io", "splash + io"), ("+relu2+ce", "splash + relu2 + ce"),
             ("blocks:save", "splash + save"), ("onetile", "one-tile"), ("onetile:save", "one-tile + save"),
             ("onetile:save+relu2", "**one-tile + save + relu2 (adopted)**"),
             ("onetile:save+relu2+ce", "one-tile + save + relu2 + ce")]
    rows = []
    for k, lab in order:
        a, b = by[(k, "1024")], by[(k, "2048")]

        def cell(r):
            sp = "ref." if r["reference"] == "1" else f"×{float(r['speedup']):.3f}"
            return f"{float(r['ms']):.3f} [{float(r['min']):.3f}–{float(r['max']):.3f}] {sp}"

        def eq(r):
            if r["reference"] == "1":
                return "–"
            return (f"{float(r['loss_diff']):.1e} / {float(r['grad_rel_max']):.1e} / {float(r['grad_rel_l2']):.1e}"
                    .replace("0.0e+00", "0"))
        rows.append([lab, cell(a), eq(a), cell(b), eq(b)])
    t3a = table(rows, ["configuration", "T = 1,024: ms per piece [min–max of 30] speed-up",
                       "Δloss / grad max rel. / grad L2 rel.", "T = 2,048: ms per piece [min–max] speed-up",
                       "Δloss / grad max rel. / grad L2 rel."])
    W = C.read("chip_whole_steps.csv")
    rows = []
    for r in W:
        rows.append([r["source"], r["measured_utc"], r["config"], r["table_grad"], r["kernel"] or "splash",
                     f"{r['a']} × {r['rows']} × {int(r['t']):,}", f"{float(r['ms']):,.1f}", n(float(r["tok_per_s"]))])
    t3b = table(rows, ["source", "measured (UTC)", "config", "table gradient", "kernel", "pieces × rows × T",
                       "ms per step of 262,144 tokens", "tokens per second (synthetic)"])
    V = C.read("chip_v9_pieces.csv")
    rows = [[r["source"], r["table_grad"], r["kernel"], f"{int(r['rows'])} × {int(r['t']):,}",
             f"{float(r['ms']):.2f} [{float(r['min']):.2f}–{float(r['max']):.2f}]", r["n"]] for r in V]
    t3c = table(rows, ["source", "v9 table gradient", "kernel and switches", "piece",
                       "ms per value-and-gradient [min–max]", "timings"])
    return t3a, t3b, t3c


def t4():
    F = C.read("chip_vmem_flags.csv")
    W = {(r["source"], r["config"], r["table_grad"], r["a"], r["rows"], r["t"]): float(r["ms"])
         for r in C.read("chip_whole_steps.csv")}
    K = {(r["kernel"], r["t"]): float(r["ms"]) for r in C.read("chip_kernels_0745.csv")}
    vm = {(r["variant"], r["a"], r["rows"], r["t"]): float(r["ms"]) for r in F}
    vmem = "; ".join(f"{lim} MiB {vm[(name, '64', '4', '1024')]:,.1f}" for name, lim in
                     (("vmem8", 8), ("vmem16", 16), ("vmem32", 32), ("vmem64", 64), ("vmem96", 96)))
    fk = "wf/chip_kernels_0745.txt"
    rows = [
        ["keep the attention kernel's output through rematerialisation (remat_keep=kernel), splash kernel",
         "161.2 ms against 158.3 ms per 32 × 1,024 micro-batch (−1.8%)", "not adopted",
         "docs/tpu_profile/tpu_tune.v6e.txt; docs/CHANGES.md 19.3"],
        ["one loss chunk per micro-batch (loss_chunk=32768)", "156.4 against 158.3 ms (+1.2%, under the 1.5% bar)",
         "not adopted", "same"],
        ["splash tiles of 2,048 at T = 2,048", "177.5 against 176.1 ms", "not adopted", "same"],
        ["compiler scoped VMEM limit (xla_tpu_scoped_vmem_limit_kib), 64 × 4 × 1,024",
         f"{vmem} ms against {vm[('base', '64', '4', '1024')]:,.1f} ms with the default", "default kept",
         "flags_out.txt, flags_out2.txt; docs/CHANGES.md 19.5"],
        ["hand-written cross-entropy backward (ce) on top of one-tile + save + relu2",
         f"{K[('onetile:save+relu2+ce', '1024')]:.3f} against {K[('onetile:save+relu2', '1024')]:.3f} ms at 1,024 "
         f"(slower); {K[('onetile:save+relu2+ce', '2048')]:.3f} against {K[('onetile:save+relu2', '2048')]:.3f} at "
         "2,048 (faster, within the spread)", "off", fk],
        ["splash accumulators in VMEM (io)", f"×{K[('', '1024')] / K[('+io', '1024')]:.3f} / "
         f"×{K[('', '2048')] / K[('+io', '2048')]:.3f} on the splash kernel; does not apply to the one-tile kernel",
         "superseded", fk],
        ["splash kernel with saved forward (blocks:save)", f"×{K[('', '1024')] / K[('blocks:save', '1024')]:.3f} / "
         f"×{K[('', '2048')] / K[('blocks:save', '2048')]:.3f}", "superseded by one-tile + save", fk],
        ["pieces halved again with the one-tile kernel",
         f"128 × 2 × 1,024 {W[(fk, 'final', 'take', '128', '2', '1024')]:,.1f} against "
         f"{W[(fk, 'final', 'take', '64', '4', '1024')]:,.1f} ms; 128 × 1 × 2,048 "
         f"{W[(fk, 'final', 'take', '128', '1', '2048')]:,.1f} against {W[(fk, 'final', 'take', '64', '2', '2048')]:,.1f} ms",
         "4 pieces kept", fk],
        ["v9: table gradient sorted or onehot instead of take (64 × 4 × 1,024)",
         f"sorted {W[('v9_measB_sorted.txt', 'v9', 'sorted', '64', '4', '1024')]:,.1f}, onehot "
         f"{W[('v9_measC_onehot.txt', 'v9', 'onehot', '64', '4', '1024')]:,.1f} against take "
         f"{W[('v9_measA.txt', 'v9', 'take', '64', '4', '1024')]:,.1f} ms", "take", "v9_measA/B/C"],
        ["v9: pieces of 8 rows (T = 1,024) or 4 rows (T = 2,048)",
         f"{W[('v9_measA.txt', 'v9', 'take', '32', '8', '1024')]:,.1f} against "
         f"{W[('v9_measA.txt', 'v9', 'take', '64', '4', '1024')]:,.1f} ms; "
         f"{W[('v9_measA.txt', 'v9', 'take', '32', '4', '2048')]:,.1f} against "
         f"{W[('v9_measA.txt', 'v9', 'take', '64', '2', '2048')]:,.1f} ms", "4 and 2 rows kept", "v9_measA.txt"],
        ["no rematerialisation (start-up benchmark, runtime 2)", "188,907 against 216,681 tokens per second",
         "not chosen", "run.log (main16, 17:44)"],
        ["code before the rewrite: larger micro-batches", "64 rows 110,855 and 128 rows 104,576 against 32 rows "
         "111,893 tokens per second", "not adopted", "docs/TPU_NOTES.md"],
        ["code before the rewrite: dense instead of flash attention", "303 against 289 ms", "not adopted", "same"],
        ["during the rewrite: one product per head, feed-forward entirely in bfloat16",
         "274.6 ms (listed with the variants that lost); 184.5 against 181.8 ms", "not adopted", "same"],
        ["during the rewrite: 256-wide attention tiles", "kernel 2.2 times slower than with 512", "not adopted",
         "same"],
        ["carry the next loop's input (compile-only estimate, not timed on the chip)", "+4.9% step time (estimate)",
         "dropped", "wf/speed_loops.md"],
    ]
    t4a = table(rows, ["candidate", "measurement", "outcome", "source"])
    rows = [
        ["structural RoPE clocks (clock_planes)", "+0.3% loss when removed (micro ablation), under the pre-registered "
         "1% bar for a new component", "off in configs/final.py", "configs/final.py docstring; docs/E2_ABLATIONS.md"],
        ["confidence channel (use_conf)", "+0.7–0.8% when removed, under the 1% bar", "off", "same"],
        ["v9 lab: query/key pages by token class", "+0.046 nats (worse); value/output pages −0.039 / −0.079",
         "not adopted", "docs/CHANGES.md 17"],
        ["v9: head-wise attention-output gate (after the logit fix)", "−0.02% (neutral) in the micro ablation",
         "off (attn_gate=False)", "docs/CHANGES.md 17; configs/v9.py"],
        ["v9: decoupled decay of the page tables", "−0.62% with decay against −0.61% without (no difference at that "
         "length)", "no decay", "docs/CHANGES.md 17"],
    ]
    t4b = table(rows, ["candidate", "measurement", "outcome", "source"])
    return t4a, t4b


def t5():
    TP = C.read("tpu_intervals.csv")
    IV = C.interventions()
    ev = C.read("run_events.csv")
    S = C.steps()
    rows = []
    used_at_10 = 0
    ten = T("2026-11-24 10:00:00")
    for r in TP:
        s0 = T(r["start_utc"])
        s1 = T(r["end_utc"]) if r["end_utc"] else None
        used_at_10 += ((s1 or ten) - s0).total_seconds() if s0 < ten else 0
    for r in TP:
        s0 = T(r["start_utc"])
        rows.append([r["runtime"], f"{r['start_utc'][5:16]} – {r['end_utc'][5:16] if r['end_utc'] else 'open'}",
                     n(r["seconds"]) + (" (until " + r["counted_until_utc"][11:16] + ")" if r["open"] == "1" else ""),
                     hms(float(r["seconds"])), r["summary"]])
    t5a = table(rows, ["runtime", "counted (UTC, 23–24 Nov)", "TPU seconds", "duration", "use"])
    # gaps
    g = [(T(TP[0]["end_utc"]), T(TP[1]["start_utc"])), (T(TP[1]["end_utc"]), T(TP[2]["start_utc"]))]
    gaps = "; ".join(f"{a:%d %b %H:%M:%S} – {b:%d %b %H:%M:%S}: {hms((b - a).total_seconds())}" for a, b in g)
    # runtime 3 in detail
    rt3 = T(TP[2]["start_utc"])
    first = min(T(e["utc"]) for e in ev if e["event"] == "restore" and e["step"] == "5075" and T(e["utc"]) > rt3)
    last5075 = max(T(e["utc"]) for e in ev if e["event"] == "restore" and e["step"] == "5075")
    rows = [["01:05:55 – " + f"{last5075:%H:%M:%S}", "restore of the step-5,075 snapshot; two data errors of a new "
             f"machine (four restore attempts, the first at {first:%H:%M:%S})", hms((last5075 - rt3).total_seconds())]]
    tot_stop = 0.0
    for r in IV:
        if r["step"] and int(r["step"]) > 5075 and r.get("stopped_s") not in (None, ""):
            tot_stop += float(r["stopped_s"])
            rows.append([f"{r['end_utc'][11:]} – {r['restore_utc'][11:]}", f"{r['id']}: {r['short']} (step {int(r['step']):,})",
                         hms(float(r["stopped_s"]))])
    t5b = table(rows, ["interval (24 Nov, UTC)", "the run was stopped for", "duration"])
    led_f = C.read("run_events.csv")
    budget_end = T(TP[2]["start_utc"]) + dt.timedelta(seconds=86400 - int(TP[0]["seconds"]) - int(TP[1]["seconds"]))
    notes = dict(used_at_10=used_at_10, gaps=gaps, tot_stop=tot_stop, budget_end=budget_end)
    return t5a, t5b, notes


def main():
    t1s, tot = t1()
    t2s = t2()
    t3a, t3b, t3c = t3()
    t4a, t4b = t4()
    t5a, t5b, notes = t5()
    S = C.steps()
    newest = max(s["utc"] for s in S)
    text = f"""# Tables

Generated by `figures/make_tables.py` from `data/*.csv` (built by `figures/make_data.py`) and the
curated rows in that script, each with its source. Re-run `figures/make_all.sh` when the logs grow.
Records of the live run are those up to the cut-off of `data/cutoff.csv` (last step record {newest} UTC). Times are
UTC. "Tokens" are token positions of the training batches: 262,144 per optimiser step (16
micro-batches of 16 × 1,024 or 8 × 2,048), padding included.

## T1. Parameter ledger of final and v9: stored numbers and numbers read per generated token

Standard reading of `docs/ACCOUNTING.md`: every dense leaf counts in full; a lookup table counts the
rows one token reads; the library and its index count the worst-case read of one retrieval, charged
to the step that retrieves. Leaves and shapes from the two checkpoints' manifests
(`train/ckpt/ckpt-000026-step-000029722` of final, `ckpt-000001-step-000030742` of v9), counted with
the row and read rules of `bes/budget.py`; library and index from the final export's accounting
(`final/export.json`, step-5,075 export, real library). Both totals are under the caps of 100,000,000
stored and 20,000,000 read per token.

{t1s}

Notes. (1) final reads {n(tot['f'][1] - 245223)} numbers on a step that does not retrieve; the
{n(tot['f'][1])} include the analytical worst case of one retrieval (245,223), measured retrievals on
500 proxy turns read at most 105,826 (`final/MODEL_CARD.md`). (2) The strict reading (union over the
context window) is not under 20M for either configuration at 2,048 positions: final 20,565,666 on a
held-out web record of 2,048 positions with 883 distinct tokens (`final/MODEL_CARD.md`; 19.62–19.81M
at T = 1,024 by `docs/ACCOUNTING.md`); v9 a median of 23.8M at 1,024 and 26.7M at 2,048 positions,
at most 24.6M and 28.2M, on 40 real documents (`docs/ACCOUNTING.md`, `docs/V9.md` 5). (3) The critic
head (196,864 numbers in `small`) is counted as read although the decoder does not use it.

## T2. Throughput history and its causes

Tokens per second of training. "live": median of the achieved rate between consecutive step records
of the run (262,144 × steps between the records ÷ wall time between them; restarts excluded), so
checkpoint writes are included; start-up benchmarks and synthetic steps are the run's or the scripts'
own timings. The arithmetic ceiling noted in `docs/CHANGES.md` 19.5: about 1.13 GFLOP per token, so
246k tokens/s is about 280 TFLOP/s against a measured matrix-unit peak of 814 TFLOP/s.

{t2s}

## T3. On-chip measurements of the attention kernels and of the whole step

T3a. One value-and-gradient of a piece of 4,096 tokens (4 × 1,024 or 2 × 2,048), configs/final.py,
`table_grad=take`, 24 Nov 07:44–07:52 UTC, one pause of the run (`wf/chip_kernels_0745.txt`): median
of 30 timings with their minimum and maximum, speed-up against the configured splash kernel, and the
equality metrics against it (loss difference; worst relative difference over parameter leaves of the
gradient, maximum and L2). relu2: hand-written ReLU² backward; ce: hand-written cross-entropy
backward; io: splash accumulators in VMEM; save: the forward kernel's output kept through
rematerialisation.

{t3a}

T3b. Whole accumulation steps of 262,144 tokens on a synthetic batch (pieces × rows × T). The final
rows of 07:44 were timed with `ce` on (`onetile:save+relu2+ce`); the run uses `ce` off, whose piece is
0.177 ms faster at 1,024 (T3a). The micro-batch rows of 03:09 use the splash kernel.

{t3b}

T3c. One value-and-gradient of a v9 piece (24 Nov 09:45–09:58 at the switch, `onetile:save+relu2`; and
10:53–10:56 with the v9 switches flat, rows, tokn, `v9_measD.txt`, whose equality metrics against
flat alone are a loss difference of 0 and gradient differences of 9.5e-3 (maximum) and 1.0e-2 (L2)).

{t3c}

## T4. Candidates that did not help, with numbers

T4a. Throughput candidates (measured on the chip unless the row says otherwise).

{t4a}

T4b. Model components that did not clear the bar (CPU micro-ablations and the v9 lab before the run;
not measured in this run).

{t4b}

## T5. TPU time by runtime

The budget is 24 hours of TPU v6e-1 time across runtimes (the user's rule; `docs/CHANGES.md` 19.2:
counted as allocated time, waiting between runtimes not counted). Ledger: `tpu_intervals.txt`.

T5a. Runtimes.

{t5a}

Gaps without a runtime: {notes['gaps']}. Used at 10:00:00 UTC on 24 Nov:
{n(notes['used_at_10'])} s ({notes['used_at_10'] / 3600:.2f} h). The 24 hours end at
{notes['budget_end']:%H:%M:%S} UTC on 24 Nov; the run's deadline is 20:31:49 (15 minutes earlier) and
training ends at 18:33:49 (`run.json`, checkpoint meta). To come [TBD]: training to 18:33:49 (rows of
2,048 tokens from p = 0.67), preference optimisation (60 min), decode selection (35 min), export
(5 min) per the post plan `sample=3 critic=0 prefopt=60 select=35 judge=0 export=5 buffer=15`.

T5b. Runtime 3: TPU held while the run was not training (from the stop record to the restore record
of the train log; the compilation after each restore, 2–51 s, is not included).

{t5b}

Total of the stops after step 5,075: {hms(notes['tot_stop'])}.
"""
    with open(os.path.join(P.PAPER, "tables.md"), "w") as f:
        f.write(text)
    print("wrote", os.path.join(P.PAPER, "tables.md"))


if __name__ == "__main__":
    main()
