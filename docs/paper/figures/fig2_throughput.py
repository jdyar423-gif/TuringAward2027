"""F2: training throughput against wall-clock time (UTC), three runtimes on one time scale.

Data: paper/data/train_steps.csv (achieved rates of run main16 between consecutive step records),
throughput_log_points.csv (runtime 1 and start-up benchmarks), interventions.csv, run_events.csv,
tpu_intervals.csv (make_data.py). Writes F2_throughput.* and paper/data/f2_segments.csv.
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
    end3 = t_last + dt.timedelta(minutes=50)
    end3 = end3.replace(minute=0 if end3.minute < 30 else 30, second=0) + dt.timedelta(minutes=30)
    windows = [(T("2026-11-23 02:45:00"), T("2026-11-23 05:15:00")),
               (T("2026-11-23 16:55:00"), T("2026-11-23 19:45:00")),
               (T("2026-11-24 00:50:00"), max(T("2026-11-24 11:00:00"), end3))]
    widths = [(b - a).total_seconds() for a, b in windows]
    fig = plt.figure(figsize=(st.FULL_W, 3.0))
    gs = fig.add_gridspec(1, 3, width_ratios=widths, left=0.075, right=0.985, top=0.80, bottom=0.17, wspace=0.07)
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
                ax.axvspan(cur, min(s0, b), color=st.SHADE2, lw=0, zorder=0)
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
    stretches = [("runtime 2", 0, 5075, 1), ("runtime 3, before pieces", 5075, 9332, 2),
                 ("4 pieces per micro-batch", 9332, 22613, 2), ("one-tile kernel", 22613, 29722, 2),
                 ("v9", 29722, 31912, 2), ("v9 with flat, rows, tokn", 31912, 10 ** 9, 2)]
    seg_rows = []
    for name, lo, hi, k in stretches:
        sel = [s for s in S if lo < s["step"] <= hi and s["interval_tok_per_s"]]
        if not sel:
            continue
        vals = [s["interval_tok_per_s"] for s in sel]
        ts = [T(s["utc"]) for s in sel]
        med = statistics.median(vals)
        seg_rows.append(dict(stretch=name, first_step=sel[0]["step"], last_step=sel[-1]["step"], records=len(vals),
                             median_tok_per_s=round(med, 1), min_tok_per_s=round(min(vals), 1),
                             max_tok_per_s=round(max(vals), 1), from_utc=min(ts).strftime("%Y-%m-%d %H:%M"),
                             to_utc=max(ts).strftime("%Y-%m-%d %H:%M")))
        tmid = min(ts) + (max(ts) - min(ts)) / 2
        axes[k].text(tmid, med / 1e3 + 8, f"{med / 1e3:.1f}", ha="center", va="bottom", fontsize=st.FS,
                     color=st.INK, zorder=5)
    with open(os.path.join(P.DATA, "f2_segments.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(seg_rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(seg_rows)

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

    # interventions and measurement pauses: tags at the restore times, two lanes against overlap
    fig.canvas.draw()
    order = sorted((r for r in IV if r["restore_utc"]), key=lambda r: r["restore_utc"])
    lane_end = [-1e9, -1e9]
    for r in order:
        t = T(r["restore_utc"])
        k = next(i for i, (a, b) in enumerate(windows) if a <= t <= b)
        ax = axes[k]
        xpix = ax.transData.transform((mdates.date2num(t), 0))[0]
        lab = str(r["id"])
        lane = 0 if xpix - lane_end[0] > 0.17 * fig.dpi else 1
        lane_end[lane] = xpix
        ax.plot([t, t], [0, YMAX + (12 if lane == 0 else 38)], color=st.MUTED, lw=0.5, zorder=1, clip_on=False)
        ax.text(t, YMAX + (14 if lane == 0 else 40), lab, ha="center", va="center",
                fontsize=st.FS - (0.5 if len(lab) > 1 else 0), color=st.INK, clip_on=False, zorder=6,
                bbox=dict(boxstyle="circle,pad=0.15" if len(lab) == 1 else "round,pad=0.15", fc=st.WHITE,
                          ec=st.MUTED, lw=0.5))
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
    a2.text(T("2026-11-24 04:45:00"), 205,
            f"Data until the newest\nsnapshot ({t_last:%H:%M}). To come:\ntraining to 18:33:49 (rows of\n"
            "2,048 tokens from p = 0.67),\n118 min of post stages, run\ndeadline 20:31:49; the 24 h of\n"
            "TPU time end at 20:46:50.", fontsize=st.FS - 0.5, color=st.INK2, ha="left", va="top",
            linespacing=1.15)
    a2.text(T("2026-11-24 01:07:00"), 12, "restore and\ntwo data errors", fontsize=st.FS - 0.5, color=st.INK2,
            ha="left", va="bottom", linespacing=1.1)

    # time axes
    for ax, hours in zip(axes, ([3, 4], [18, 19], range(2, 24, 2))):
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=list(hours), tz=UTC))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=UTC))
    axes[0].set_xlabel("Runtime 1, 23 Nov", fontsize=st.FS)
    axes[1].set_xlabel("Runtime 2, 23 Nov", fontsize=st.FS)
    axes[2].set_xlabel("Runtime 3, 24 Nov (UTC)", fontsize=st.FS)

    handles = [Line2D([], [], color=st.FINAL, lw=1.25), Line2D([], [], color=st.V9, lw=1.25),
               Line2D([], [], ls="", marker="D", ms=3.4, mfc=st.WHITE, mec=st.INK2, mew=0.9),
               Line2D([], [], ls="", marker="o", ms=3.0, mfc=st.INK2, mec=st.INK2),
               Patch(fc=st.SHADE, ec="none"), Patch(fc=st.SHADE2, ec="none")]
    labels = ["final, training", "v9, training", "start-up benchmark", "training, one log line",
              "TPU held, not training", "no TPU"]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.07, 1.0), ncol=6, frameon=False,
               fontsize=st.FS, handlelength=1.4, handletextpad=0.4, columnspacing=1.2)
    st.save(fig, "F2_throughput")
    for r in seg_rows:
        print(r)


if __name__ == "__main__":
    main()
