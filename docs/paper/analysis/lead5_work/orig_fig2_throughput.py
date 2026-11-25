"""F2: training throughput against wall-clock time (UTC), three runtimes on one time scale.

Data: paper/data/train_steps.csv (achieved rates of run main16 between consecutive step records),
throughput_stages.csv (the live rate of each stretch, Table 2's definition, with n), throughput_log_points.csv
(runtime 1 and start-up benchmarks), interventions.csv, run_events.csv, tpu_intervals.csv, cutoff.csv
(all written by make_data.py). Writes F2_throughput.svg/png.

Grey print: "TPU held, not training" is a flat light grey, "no TPU" a hatch, so the two differ in
pattern and not only by a few units of lightness.
"""
import csv
import datetime as dt
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as st  # noqa: E402
import common as C  # noqa: E402
import paths as P  # noqa: E402

st.setup()
plt = st.plt
import matplotlib.dates as mdates  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator  # noqa: E402

UTC = dt.timezone.utc


def T(s):
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)


def main():
    S = C.steps()
    IV = C.interventions()
    TP = C.read("tpu_intervals.csv")
    LP = C.read("throughput_log_points.csv")
    ev = C.read("run_events.csv")

    t_last = max(T(s["utc"]) for s in S)
    end3 = t_last + dt.timedelta(minutes=10)         # the axis ends at the next half hour after the cut-off
    end3 = end3.replace(minute=0 if end3.minute < 30 else 30, second=0) + dt.timedelta(minutes=30)
    cut = C.cutoff()
    st.matplotlib.rcParams["hatch.linewidth"] = 0.5
    windows = [(T("2026-11-23 02:45:00"), T("2026-11-23 05:15:00")),
               (T("2026-11-23 16:55:00"), T("2026-11-23 19:45:00")),
               (T("2026-11-24 00:50:00"), max(T("2026-11-24 11:00:00"), end3))]
    widths = [(b - a).total_seconds() for a, b in windows]
    fig = plt.figure(figsize=(st.FULL_W, 3.0))
    gs = fig.add_gridspec(1, 3, width_ratios=widths, left=0.075, right=0.985, top=0.775, bottom=0.17, wspace=0.07)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    YMAX = 330

    # runtimes held (an open interval is held to the end of the window)
    held = [(T(r["start_utc"]), T(r["end_utc"]) if r["end_utc"] else windows[2][1]) for r in TP]
    for ax, (a, b) in zip(axes, windows):
        ax.set_xlim(a, b)
        ax.set_ylim(0, YMAX)
        ax.yaxis.set_major_locator(FixedLocator([0, 100, 200, 300]))
        st.grid_y(ax)
        cur = a
        for s0, s1 in held + [(b, b)]:
            if s0 > cur:
                ax.axvspan(cur, min(s0, b), fc=st.WHITE, ec=st.AXIS, hatch="////", lw=0, zorder=0)
            cur = max(cur, s1)
    for ax in axes[1:]:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", left=False, labelleft=False)
    axes[0].set_ylabel("Throughput (10³ tokens/s)")

    # TPU held but not training: each stop record to the next restore record of the train log, the
    # set-up of runtime 2 before its first step, runtime 3 until the step-5,075 restore that worked,
    # and the measurement windows of runtime 1 (docs/CHANGES.md 19.1)
    ends = [T(e["utc"]) for e in ev if e["event"] == "end"]
    rest = [(T(e["utc"]), e) for e in ev if e["event"] == "restore"]
    idle = []
    for t_end in ends:
        t_next = min((t for t, r in rest if t > t_end), default=None)
        if t_next:
            idle.append((t_end, t_next))
    idle.append((held[2][0], max(t for t, r in rest if r["step"] == "5075")))
    idle.append((held[1][0], T("2026-11-23 17:46:31")))
    idle += [(T("2026-11-23 03:34:00"), T("2026-11-23 03:57:00")),
             (T("2026-11-23 04:08:00"), T("2026-11-23 04:17:00"))]
    for t0, t1 in idle:
        for ax in axes:
            ax.axvspan(t0, t1, color=st.SHADE, lw=0, zorder=0.5)

    # run main16: achieved rate between consecutive step records with no restart between them
    segs, cur = [], None
    for s in S:
        if s["interval_tok_per_s"] is None:
            cur = None
            continue
        if cur is None or cur["cfg"] != s["config"]:
            cur = dict(cfg=s["config"], pts=[])
            segs.append(cur)
        cur["pts"].append((T(s["utc"]), s["interval_tok_per_s"] / 1e3))
    for sg in segs:
        col = st.FINAL if sg["cfg"] == "final" else st.V9
        for ax in axes:
            ax.plot([p[0] for p in sg["pts"]], [p[1] for p in sg["pts"]], color=col, lw=1.25, zorder=3)
    # the live rate of each stretch (Table 2: median of the rates the trainer logs with its step records)
    prev = None
    for r in C.read("throughput_stages.csv"):
        if not r["live_median"] or not r["order"]:
            continue
        t0, t1 = T(r["first_utc"]), T(r["last_utc"])
        k = next(i for i, (a, b) in enumerate(windows) if a <= t0 <= b)
        tmid = t0 + (t1 - t0) / 2
        v = int(r["live_median"]) / 1e3
        # below its line when the stretch follows a faster one closely (the label would sit on that line)
        below = prev is not None and v < prev[0] and (t0 - prev[1]).total_seconds() < 3600 and prev[2] == k
        axes[k].text(tmid, v - 9 if below else v + 8, f"{v:.1f}\nn = {int(r['live_n'])}" if below else
                     f"n = {int(r['live_n'])}\n{v:.1f}", ha="center", va="top" if below else "bottom",
                     fontsize=st.FS - 0.5, color=st.INK, zorder=5, linespacing=1.05,
                     bbox=dict(boxstyle="square,pad=0.08", fc=st.WHITE, ec="none"))
        prev = (v, t1, k)

    # runtime 1 and the start-up benchmarks (rows of 1,024 tokens; rematerialised splash plans only)
    for r in LP:
        if r["seq_len"] != "1024":
            continue
        bench = r["what"].startswith("start-up")
        if bench and "splash-remat1" not in r["what"]:
            continue
        t = T(r["utc"])
        k = next(i for i, (a, b) in enumerate(windows) if a <= t <= b)
        col = st.FINAL if r["config"] == "final" else st.V9
        axes[k].plot([t], [int(r["tok_per_s"]) / 1e3], ls="", marker="D" if bench else "o",
                     ms=3.4 if bench else 3.0, mfc=st.WHITE if bench else col, mec=col, mew=0.9, zorder=4)
    a0 = axes[0]
    a0.text(T("2026-11-23 03:02:00"), 128, "before the\nglue rewrite", fontsize=st.FS - 0.5, color=st.INK2,
            ha="left", va="bottom", linespacing=1.1)
    a0.text(T("2026-11-23 04:00:00"), 214, "after", fontsize=st.FS - 0.5, color=st.INK2, ha="center",
            va="bottom")
    a0.text(T("2026-11-23 03:13:27"), 74, "v9", fontsize=st.FS - 0.5, color=st.INK2, ha="center", va="top")

    # restarts (restore times) and phase changes of the schedule (stage events; dashed): tags in lanes
    fig.canvas.draw()
    marks = [(T(r["restore_utc"]), r, "-") for r in IV if r["restore_utc"]]
    marks += [(T(r["event_utc"]), r, (0, (3, 2))) for r in IV if r["group"] == "phase" and r["event_utc"]]
    lane_end = [-1e9, -1e9, -1e9]                 # at most three lanes
    for t, r, ls in sorted(marks, key=lambda m: m[0]):
        k = next(i for i, (a, b) in enumerate(windows) if a <= t <= b)
        ax = axes[k]
        xpix = ax.transData.transform((mdates.date2num(t), 0))[0]
        lab = str(r["id"])
        lane = next((i for i, e in enumerate(lane_end) if xpix - e > 0.19 * fig.dpi), len(lane_end) - 1)
        lane_end[lane] = xpix
        ytag = YMAX + 14 + 22 * lane
        ax.plot([t, t], [0, ytag], color=st.MUTED, lw=0.5, ls=ls, zorder=1, clip_on=False)
        ax.text(t, ytag, lab, ha="center", va="center", fontsize=st.FS - (0.5 if len(lab) > 1 else 0), color=st.INK,
                clip_on=False, zorder=6, bbox=dict(boxstyle="circle,pad=0.15" if len(lab) == 1 else "round,pad=0.15",
                                                   fc=st.WHITE, ec=st.MUTED, lw=0.5))
    # gaps between the runtimes (breaks of the time axis)
    for i in range(2):
        g0, g1 = held[i][1], held[i + 1][0]
        h, m = divmod(int((g1 - g0).total_seconds()) // 60, 60)
        x1, x0 = axes[i].get_position().x1, axes[i + 1].get_position().x0
        yb = axes[i].get_position().y0
        for xx in (x1, x0):
            fig.lines.append(Line2D([xx - 0.004, xx + 0.004], [yb - 0.018, yb + 0.018], transform=fig.transFigure,
                                    color=st.INK2, lw=0.7))
        fig.text((x1 + x0) / 2, axes[i].get_position().y1 - 0.02, f"{h} h {m:02d} min\nwithout\na runtime",
                 ha="center", va="top", fontsize=st.FS - 0.5, color=st.INK2, linespacing=1.1,
                 bbox=dict(boxstyle="square,pad=0.2", fc=st.WHITE, ec="none"))
    a2 = axes[2]
    setups = [e for e in ev if e["event"] == "setup"]
    train_end = max(setups, key=lambda e: float(e["time_s"]))["train_end_utc"]
    deadline = max(setups, key=lambda e: float(e["time_s"]))["deadline_utc"]
    tp = sorted(TP, key=lambda r: r["start_utc"])
    budget_end = T(tp[-1]["start_utc"]) + dt.timedelta(seconds=86400 - sum(int(r["seconds"]) for r in tp[:-1]))
    ended = [e for e in ev if e["event"] == "end" and e["reason"] == "deadline"]
    last = max(S, key=lambda s: s["time_s"])
    if ended:
        note = (f"Training ended at {ended[-1]['utc'][11:]}\n(step {int(ended[-1]['step']):,}). The run's\n"
                f"deadline is {deadline[11:]}; the 24 h\nof TPU time end at {budget_end:%H:%M:%S}.")
    else:
        reserve = round((T(deadline) - T(train_end)).total_seconds() / 60)
        # narrow lines: the note must fit between the restore of step 12,062 (tag 4) and the kernel
        # switch (tag 5), under the 240.1 line, clear of the later labels and tags
        note = (f"Records to step\n{last['step']:,} ({t_last:%H:%M:%S}),\nthe cut-off of the\npaper's numbers.\n"
                f"To come: training\nto {train_end[11:]}, {reserve} min\nof post stages, run\ndeadline "
                f"{deadline[11:]};\nthe 24 h of TPU time\nend at {budget_end:%H:%M:%S}.")
        assert last["step"] == cut["step"], (last["step"], cut["step"])
    a2.text(T("2026-11-24 04:42:00"), 226, note, fontsize=st.FS - 0.5, color=st.INK2, ha="left", va="top",
            linespacing=1.1)
    a2.text(T("2026-11-24 01:07:00"), 12, "restore and\ntwo data errors", fontsize=st.FS - 0.5, color=st.INK2,
            ha="left", va="bottom", linespacing=1.1)

    # time axes: one time scale for the three runtimes; fewer ticks when an hour is narrow
    fig.canvas.draw()
    per_hour = axes[2].get_position().width * fig.get_figwidth() / ((windows[2][1] - windows[2][0]).total_seconds() / 3600)
    roomy = per_hour >= 0.45                       # inches per hour
    step3 = 2 if per_hour >= 0.2 else 4
    for ax, hours in zip(axes, ([3, 4] if roomy else [4], [18, 19] if roomy else [18], range(0, 24, step3))):
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=list(hours), tz=UTC))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=UTC))
    axes[0].set_xlabel("Runtime 1, 23 Nov" if roomy else "Runtime 1", fontsize=st.FS)
    axes[1].set_xlabel("Runtime 2, 23 Nov" if roomy else "Runtime 2", fontsize=st.FS)
    axes[2].set_xlabel("Runtime 3, 24 Nov (UTC)" if roomy else "Runtime 3, 24 Nov (UTC; runtimes 1 and 2: 23 Nov)",
                       fontsize=st.FS)

    handles = [Line2D([], [], color=st.FINAL, lw=1.25), Line2D([], [], color=st.V9, lw=1.25),
               Line2D([], [], ls="", marker="D", ms=3.4, mfc=st.WHITE, mec=st.INK2, mew=0.9),
               Line2D([], [], ls="", marker="o", ms=3.0, mfc=st.INK2, mec=st.INK2),
               Patch(fc=st.SHADE, ec="none"), Patch(fc=st.WHITE, ec=st.AXIS, hatch="////", lw=0)]
    labels = ["final, training", "v9, training", "start-up benchmark", "training, one log line",
              "TPU held, not training", "no TPU"]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.07, 1.0), ncol=6, frameon=False,
               fontsize=st.FS, handlelength=1.4, handletextpad=0.4, columnspacing=1.2)
    st.save(fig, "F2_throughput")
    for r in C.read("throughput_stages.csv"):
        if r["live_median"]:
            print(f"{r['key']:18s} live {r['live_median']} (n {r['live_n']}), achieved {r['achieved_median']} "
                  f"(n {r['achieved_n']}), {r['first_utc']}..{r['last_utc'][11:]}")


if __name__ == "__main__":
    main()
