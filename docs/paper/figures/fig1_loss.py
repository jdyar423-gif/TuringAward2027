"""F1: training loss against training tokens, the dated interventions, the schedule (lr factor, p).

Data: paper/data/train_steps.csv and interventions.csv (make_data.py). Writes F1_loss.svg/png and
paper/data/f1_loss_smoothed.csv (the plotted smoothed series).
"""
import csv
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as st  # noqa: E402
import common as C  # noqa: E402
import paths as P  # noqa: E402

st.setup()
plt = st.plt
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
NEFF_MIN = 4.0          # the smoothed line is drawn where the kernel holds at least this many points


def pow10(x, _pos=None):
    if x <= 0:
        return ""
    e = math.log10(x)
    return f"10{str(int(round(e))).translate(SUP)}" if abs(e - round(e)) < 1e-9 else ""


def main():
    S = C.steps()
    IV = [i for i in C.interventions() if i["group"] == "intervention"]
    seg = {}
    for cfg in ("final", "v9"):
        rows = [s for s in S if s["config"] == cfg]
        xs = [s["tokens"] for s in rows]
        ys = [s["loss"] for s in rows]
        sm, ne = C.smooth_log(xs, ys, with_neff=True) if rows else ([], [])
        seg[cfg] = dict(rows=rows, x=xs, y=ys, sm=sm, neff=ne)

    with open(os.path.join(P.DATA, "f1_loss_smoothed.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["config", "step", "tokens", "loss_raw", "loss_smoothed", "kernel_points_effective",
                    "drawn", "loss_ema_trainer", "kernel_sigma_decades"])
        for cfg in ("final", "v9"):
            d = seg[cfg]
            for r, sm, ne in zip(d["rows"], d["sm"], d["neff"]):
                w.writerow([cfg, r["step"], r["tokens"], r["loss"], round(sm, 5), round(ne, 2),
                            int(ne >= NEFF_MIN or cfg == "v9"), r["loss_ema"], C.SIGMA_DEC])

    fig = plt.figure(figsize=(st.FULL_W, 4.35))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.55, 1.0], width_ratios=[1.0, 1.0],
                          left=0.07, right=0.985, top=0.95, bottom=0.10, hspace=0.16, wspace=0.16)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0], sharex=axA)
    axD = fig.add_subplot(gs[1, 1], sharex=axB)

    raw_kw = dict(s=2.2, linewidths=0, zorder=2)
    colors = {"final": (st.FINAL, st.FINAL_LIGHT), "v9": (st.V9, st.V9_LIGHT)}

    def drawn(d, cfg, lo=0.0):
        """(x, y) of the smoothed line where it is drawn (enough points in the kernel)."""
        pts = [(x, s) for x, s, ne in zip(d["x"], d["sm"], d["neff"])
               if x >= lo and (ne >= NEFF_MIN or cfg == "v9")]
        return [p[0] for p in pts], [p[1] for p in pts]

    # ---------------------------------------------------------------- (a) whole run, log x
    for cfg in ("final", "v9"):
        d = seg[cfg]
        if not d["rows"]:
            continue
        axA.scatter(d["x"], d["y"], color=colors[cfg][1], **raw_kw)
        lx, ly = drawn(d, cfg)
        axA.plot(lx, ly, color=colors[cfg][0], lw=1.25, zorder=3)
    axA.vlines([iv["tokens"] for iv in IV], 2.3, 2.95, color=st.MUTED, lw=0.5, zorder=1)
    xmax = max(seg["v9"]["x"] or seg["final"]["x"])
    axA.set_xscale("log")
    axA.set_xlim(1e7, 1.3e10)
    axA.set_ylim(2.3, 5.0)
    axA.xaxis.set_major_locator(FixedLocator([1e7, 1e8, 1e9, 1e10]))
    axA.xaxis.set_major_formatter(FuncFormatter(pow10))
    axA.xaxis.set_minor_formatter(NullFormatter())
    axA.set_ylabel("Training loss (nats per token)")
    st.grid_y(axA)
    axA.set_title("a   Whole run (logarithmic tokens)")
    xb0, xb1 = 1.0e9, xmax * 1.04
    axA.annotate("", xy=(xb0, 3.06), xytext=(xb1, 3.06),
                 arrowprops=dict(arrowstyle="|-|", color=st.MUTED, lw=0.6, shrinkA=0, shrinkB=0,
                                 mutation_scale=2.5))
    axA.text(math.sqrt(xb0 * xb1), 3.10, "window of b", ha="center", va="bottom", color=st.INK2, fontsize=st.FS)
    # legend (two configurations, raw and smoothed) in the empty lower left
    handles = [Line2D([], [], ls="", marker="o", ms=2.6, mew=0, color=st.FINAL_LIGHT),
               Line2D([], [], color=st.FINAL, lw=1.25),
               Line2D([], [], ls="", marker="o", ms=2.6, mew=0, color=st.V9_LIGHT),
               Line2D([], [], color=st.V9, lw=1.25)]
    labels = ["final, every 50th step", "final, smoothed", "v9, every 50th step", "v9, smoothed"]
    axA.legend(handles, labels, loc="lower left", ncol=1, bbox_to_anchor=(0.0, 0.0), handlelength=1.4,
               handletextpad=0.45, borderaxespad=0.3, labelspacing=0.25, fontsize=st.FS)
    # key of the interventions (upper middle: no data there)
    key = [f"{iv['id']}  {iv['short']}  (step {iv['step']:,})" for iv in IV]
    axA.text(0.43, 0.965, "Interventions (vertical lines; times in F2)", transform=axA.transAxes,
             ha="left", va="top", fontsize=st.FS, color=st.INK, fontweight="semibold")
    axA.text(0.43, 0.905, "\n".join(key), transform=axA.transAxes, ha="left", va="top", fontsize=st.FS,
             color=st.INK, linespacing=1.35)

    # ---------------------------------------------------------------- (b) last part, linear x
    lo = xb0
    ys_win = [y for cfg in ("final", "v9") for x, y in zip(seg[cfg]["x"], seg[cfg]["y"]) if x >= lo]
    ylo = math.floor((min(ys_win) - 0.01) * 20) / 20
    yhi = math.ceil((max(ys_win) + 0.06) * 20) / 20
    for cfg in ("final", "v9"):
        d = seg[cfg]
        pts = [(x, y) for x, y in zip(d["x"], d["y"]) if x >= lo]
        if not pts:
            continue
        axB.scatter([p[0] / 1e9 for p in pts], [p[1] for p in pts], color=colors[cfg][1], **raw_kw)
        lx, ly = drawn(d, cfg, lo)
        axB.plot([x / 1e9 for x in lx], ly, color=colors[cfg][0], lw=1.25, zorder=3)
    for iv in IV:
        x = iv["tokens"] / 1e9
        axB.axvline(x, color=st.MUTED, lw=0.5, zorder=1)
        axB.text(x, yhi - 0.004, str(iv["id"]), ha="center", va="top", fontsize=st.FS, color=st.INK,
                 bbox=dict(boxstyle="circle,pad=0.18", fc=st.WHITE, ec=st.MUTED, lw=0.5))
    axB.set_xlim(lo / 1e9, xb1 / 1e9)
    axB.set_ylim(ylo, yhi)
    axB.set_title("b   From 10⁹ tokens on (linear tokens)")
    st.grid_y(axB)
    fl = seg["final"]
    i5 = min(range(len(fl["x"])), key=lambda j: abs(fl["x"][j] - 5.0e9))
    axB.annotate("final", xy=(fl["x"][i5] / 1e9, fl["sm"][i5]), xytext=(5.45, yhi - 0.13),
                 ha="left", va="bottom", fontsize=st.FS, color=st.INK,
                 arrowprops=dict(arrowstyle="-", color=st.INK2, lw=0.5, shrinkA=1, shrinkB=1))
    vv = seg["v9"]
    if vv["x"]:
        axB.annotate("v9", xy=(vv["x"][-1] / 1e9, vv["sm"][-1]), xytext=(vv["x"][-1] / 1e9 - 0.35, yhi - 0.13),
                     ha="right", va="bottom", fontsize=st.FS, color=st.INK,
                     arrowprops=dict(arrowstyle="-", color=st.INK2, lw=0.5, shrinkA=1, shrinkB=1))

    # ---------------------------------------------------------------- (c), (d) schedule
    allr = seg["final"]["rows"] + seg["v9"]["rows"]
    xs = [r["tokens"] for r in allr]
    for ax, scale in ((axC, 1.0), (axD, 1e9)):
        ax.plot([x / scale for x in xs], [r["lr"] for r in allr], color=st.INK, lw=1.0, zorder=3)
        ax.plot([x / scale for x in xs], [r["p"] for r in allr], color=st.MUTED, lw=1.0, zorder=3)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_locator(FixedLocator([0, 0.5, 1.0]))
        st.grid_y(ax)
    axC.vlines([iv["tokens"] for iv in IV], 0, 1.05, color=st.GRID, lw=0.6, zorder=1)
    for iv in IV:
        axD.axvline(iv["tokens"] / 1e9, color=st.MUTED, lw=0.5, zorder=1)
    axC.set_ylabel("Factor or p")
    axC.set_xlabel("Training tokens (262,144 per optimiser step)")
    axD.set_xlabel("Training tokens (10⁹)")
    axC.text(2.3e7, 0.80, "learning-rate factor", color=st.INK, fontsize=st.FS, ha="left", va="bottom")
    axC.text(1.2e8, 0.04, "progress p", color=st.INK2, fontsize=st.FS, ha="left", va="bottom")
    axD.axhline(0.92, color=st.GRID, lw=0.8, zorder=0)
    axD.text(xb1 / 1e9 - 0.05, 0.905, "p = 0.92: chat-heavy tail (to come)", zorder=4,
             fontsize=st.FS - 0.5, color=st.INK2, ha="right", va="top", bbox=dict(boxstyle="square,pad=0.15", fc=st.WHITE, ec="none"))
    axD.axhline(0.67, color=st.GRID, lw=0.8, zorder=0)
    axD.text(lo / 1e9 + 0.05, 0.632, "p = 0.67: rows of 2,048 tokens (to come)", fontsize=st.FS - 0.5,
             color=st.INK2, ha="left", va="top", zorder=4, bbox=dict(boxstyle="square,pad=0.15", fc=st.WHITE, ec="none"))
    i4 = next(iv for iv in IV if iv["id"] == "4")
    axD.text(i4["tokens"] / 1e9 + 0.07, 0.42, "4: p and the factor jump", fontsize=st.FS - 0.5,
             color=st.INK2, ha="left", va="center", zorder=4, bbox=dict(boxstyle="square,pad=0.15", fc=st.WHITE, ec="none"))
    plt.setp(axA.get_xticklabels(), visible=False)
    plt.setp(axB.get_xticklabels(), visible=False)

    st.save(fig, "F1_loss")

    # ---------------------------------------------------------------- numbers for FIGURES.md
    f, v = seg["final"], seg["v9"]
    first_drawn = next(r["step"] for r, ne in zip(f["rows"], f["neff"]) if ne >= NEFF_MIN)
    print(f"final: steps {f['rows'][0]['step']}..{f['rows'][-1]['step']}, raw loss {f['y'][0]:.4f} at "
          f"{f['x'][0]:.3e} tokens; smoothed line drawn from step {first_drawn}; smoothed at the end {f['sm'][-1]:.4f}")
    for target in (1e8, 3e8, 1e9, 1.33e9, 3e9, 5.9e9, 6.6e9, 7.79e9):
        i = min(range(len(f["x"])), key=lambda j: abs(f["x"][j] - target))
        print(f"  smoothed final near {target:.3g}: step {f['rows'][i]['step']} {f['sm'][i]:.4f} (n_eff {f['neff'][i]:.1f})")
    n = min(21, len(v["y"]))
    if n >= 2:
        a, b = f["y"][-n:], v["y"][:n]
        print(f"last {n} final raw: mean {statistics.mean(a):.4f} sd {statistics.stdev(a):.4f}; first {n} v9 raw: "
              f"mean {statistics.mean(b):.4f} sd {statistics.stdev(b):.4f}; v9 smoothed end {v['sm'][-1]:.4f}")
        print(f"trainer EMA: final end {f['rows'][-1]['loss_ema']:.4f}, v9 end {v['rows'][-1]['loss_ema']:.4f}")
    print(f"y window of b: {ylo}..{yhi}; x window {lo:.2e}..{xb1:.3e}")


if __name__ == "__main__":
    main()
