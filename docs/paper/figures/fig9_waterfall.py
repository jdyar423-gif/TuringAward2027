"""F9: throughput waterfall of the paper's Table 2: each change of the run in the order it happened,
the live rate before and after it, and the window each live rate was measured over.

Data: paper/data/throughput_stages.csv (make_data.py): the live rate of each stretch of run main16
(median of the rates the trainer logs with its step records, with n, first and last step and UTC,
and the step of the change that starts it), and Table 2's synthetic or start-up-benchmark value of
the same stage; for the two stages before the run (runtime 1) the start-up benchmarks and single log
lines; interventions.csv (the phase change P1); cutoff.csv (the newest record used). Writes
F9_waterfall.svg/png.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as st  # noqa: E402
import common as C  # noqa: E402

st.setup()
plt = st.plt
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402


def k(v):
    return f"{v / 1e3:.1f}k"


def hm(utc):
    return utc[11:16]


def main():
    R = C.read("throughput_stages.csv")
    pre = [r for r in R if r["part"] == "before run main16"]
    run = [r for r in R if r["part"] == "run main16 (live)"]
    phase = {str(i["phase"]): i for i in C.interventions() if i["group"] == "phase"}
    cut = C.cutoff()

    rows = []                                   # (kind, record) top to bottom
    rows.append(("head", "Before run main16: runtime 1, 23 Nov, 32-row micro-batches (runs discarded; no live rate)"))
    rows += [("pre", r) for r in pre]
    rows.append(("head", "Run main16: live rate = median of the rates the trainer logs every 50th step; "
                         f"records to {cut['label']}"))
    rows += [("run", r) for r in run]
    n = len(rows)
    fig = plt.figure(figsize=(st.FULL_W, 0.29 * n + 0.62))
    ax = fig.add_axes([0.335, 0.62 / (0.29 * n + 0.62), 0.415, 1 - 0.70 / (0.29 * n + 0.62)])
    XMAX = 320.0
    ax.set_xlim(0, XMAX)
    ax.set_ylim(n - 0.5, -0.6)
    ax.xaxis.set_major_locator(plt.FixedLocator([0, 50, 100, 150, 200, 250, 300]))
    st.grid_x(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_xlabel("Training throughput (10³ tokens/s)")
    H = 0.56
    prev_level, prev_y = None, None
    for y, (kind, r) in enumerate(rows):
        if kind == "head":
            ax.text(-0.775, y + 0.12, r, transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=st.FS - 0.5, fontweight="semibold", color=st.INK, clip_on=False)
            prev_level = None
            continue
        col = st.FINAL if r["config"] == "final" else st.V9
        if kind == "pre":
            v = int(r["synthetic_tok_per_s"]) / 1e3
            lo = prev_level if prev_level is not None else 0.0
            ax.add_patch(Rectangle((min(lo, v), y - H / 2), abs(v - lo), H, fc=st.WHITE, ec=col, lw=0.9, zorder=2))
            singles = [int(s) for s in (r["single_readings_values"] or "").split(";") if s]
            for j, s in enumerate(singles):       # several readings: one above the other (they can be ~1k apart)
                dy = 0.0 if len(singles) == 1 else (j - (len(singles) - 1) / 2) * 0.22
                ax.plot([s / 1e3], [y + dy], ls="", marker="o", ms=3.0, mfc=st.INK2, mec=st.WHITE, mew=0.4, zorder=4)
            left2 = "start-up benchmark" + ("; dots: single log lines of training" if r["single_readings_values"] else "")
            right1 = f"benchmark {k(int(r['synthetic_tok_per_s']))}" + (
                f"  ({100 * (v / prev_level - 1):+.1f}%)" if prev_level else "")
            reads = re.findall(r"(\d+) \((training(?: \(monitor\))?), step (\d+), \d\d:\d\d\)", r["single_readings"] or "")
            kind_r = "monitor line" if all("monitor" in w for _, w, _ in reads) else "log line"
            right2 = (f"{kind_r}{'s' if len(reads) > 1 else ''} " + ", ".join(k(int(v)) for v, _, _ in reads)
                      if reads else "")
            label = r["label"].replace(", runtime 1 (32-row micro-batches)", "")
        else:
            v = int(r["live_median"]) / 1e3
            lo = prev_level if prev_level is not None else 0.0
            ax.add_patch(Rectangle((min(lo, v), y - H / 2), abs(v - lo), H, fc=col, ec="none", zorder=2))
            if r["synthetic_tok_per_s"]:
                ax.plot([int(r["synthetic_tok_per_s"]) / 1e3], [y], ls="", marker="D", ms=3.6, mfc=st.WHITE,
                        mec=st.INK2, mew=0.9, zorder=4)
            when = f" (step {int(r['from_step']):,})" if r["from_step"] not in ("", None) else ""
            if r["key"] in ("v9_2048", "v9_tail"):
                ph = phase.get("long" if r["key"] == "v9_2048" else "tail")
                when = f" (p = {float(ph['p']):.2f}, step {int(ph['step']):,})" if ph else ""
            label = r["label"] + when
            left2 = (f"n = {r['live_n']}, steps {int(r['first_step']):,}–{int(r['last_step']):,}, "
                     f"{'23' if r['first_utc'].startswith('2026-11-23') else '24'} Nov {hm(r['first_utc'])}–{hm(r['last_utc'])}")
            pct = (100 * (v / prev_level - 1)) if prev_level is not None else None   # from the medians, rounded once
            right1 = f"{k(int(r['live_median']))} live" + (f"  ({pct:+.1f}%)".replace("-", "−") if pct is not None else "")
            kind_s = r["synthetic_kind"].replace("synthetic step, with the ce backward on", "synthetic, ce on") \
                .replace("synthetic step", "synthetic").replace("start-up benchmark", "benchmark")
            right2 = f"{kind_s} {k(int(r['synthetic_tok_per_s']))}" if r["synthetic_tok_per_s"] else "no synthetic timing"
        if prev_level is not None:              # waterfall connector from the previous level
            ax.plot([prev_level, prev_level], [prev_y + H / 2, y - H / 2], color=st.MUTED, lw=0.6, zorder=1)
        ax.text(-0.02, y - 0.13, label, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=st.FS - 0.5,
                color=st.INK, clip_on=False)
        ax.text(-0.02, y + 0.22, left2, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=st.FS - 0.5, color=st.INK2, clip_on=False)
        ax.text(1.02, y - 0.13, right1, transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=st.FS - 0.5, color=st.INK, clip_on=False)
        ax.text(1.02, y + 0.22, right2, transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=st.FS - 0.5, color=st.INK2, clip_on=False)
        prev_level, prev_y = v, y
    handles = [Patch(fc=st.FINAL, ec="none"), Patch(fc=st.V9, ec="none"), Patch(fc=st.WHITE, ec=st.FINAL, lw=0.9),
               Line2D([], [], ls="", marker="D", ms=3.6, mfc=st.WHITE, mec=st.INK2, mew=0.9),
               Line2D([], [], ls="", marker="o", ms=3.0, mfc=st.INK2, mec=st.INK2)]
    labels = ["final, live", "v9, live", "start-up benchmark", "synthetic or benchmark of the same stage",
              "single log line"]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=st.FS - 0.5,
               handlelength=1.4, handletextpad=0.4, columnspacing=1.1, frameon=False)
    st.save(fig, "F9_waterfall")
    for kind, r in rows:
        if kind == "run":
            print(f"{r['key']:12s} {r['live_median']:>7} n {r['live_n']:>4} {r['delta_live']:>8} {r['delta_live_pct']:>7}%"
                  f"  synthetic {r['synthetic_tok_per_s'] or '-'}  {r['first_utc']}..{r['last_utc'][11:]}")
        elif kind == "pre":
            print(f"{r['key']:12s} benchmark {r['synthetic_tok_per_s']}  single {r['single_readings']}")


if __name__ == "__main__":
    main()
