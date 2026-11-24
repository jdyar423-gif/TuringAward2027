"""F1: training loss against training tokens, every restart and phase change, the schedule.

Data: paper/data/f1_loss_smoothed.csv (raw step losses and the centred smoothed series, written by
make_data.py), train_steps.csv (learning-rate factor, p), interventions.csv and cutoff.csv (the newest
record used). Writes F1_loss.svg/png. If the run has entered the chat-heavy tail (p >= 0.92, loss on assistant tokens
only), those records are drawn in a separate panel with their own scale, since they measure the
loss of other tokens.
"""
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as st  # noqa: E402
import common as C  # noqa: E402

st.setup()
plt = st.plt
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, MaxNLocator  # noqa: E402

SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def pow10(x, _pos=None):
    if x <= 0:
        return ""
    e = math.log10(x)
    return f"10{str(int(round(e))).translate(SUP)}" if abs(e - round(e)) < 1e-9 else ""


def main():
    R = C.read("f1_loss_smoothed.csv")
    for r in R:
        for k in ("step", "tokens", "stretch", "drawn", "assistant_only", "seq_len"):
            r[k] = int(r[k])
        for k in ("loss_raw", "loss_smoothed", "kernel_records_effective"):
            r[k] = float(r[k])
    S = C.steps()
    IV = C.interventions()
    marks = [i for i in IV if i["group"] in ("intervention", "measurement", "unlabelled", "phase")]
    tail = [r for r in R if r["assistant_only"] == 1]
    has_tail = bool(tail)
    main_rows = [r for r in R if r["assistant_only"] == 0]
    xmax = max(r["tokens"] for r in R)
    colors = {"final": (st.FINAL, st.FINAL_LIGHT), "v9": (st.V9, st.V9_LIGHT)}

    fig = plt.figure(figsize=(st.FULL_W, 4.55))
    if has_tail:
        gs = fig.add_gridspec(2, 3, height_ratios=[2.55, 1.0], width_ratios=[1.0, 1.0, 0.34], left=0.07, right=0.985,
                              top=0.955, bottom=0.10, hspace=0.24, wspace=0.17)
    else:
        gs = fig.add_gridspec(2, 2, height_ratios=[2.55, 1.0], width_ratios=[1.0, 1.0], left=0.07, right=0.985,
                              top=0.955, bottom=0.10, hspace=0.24, wspace=0.16)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0], sharex=axA)
    axD = fig.add_subplot(gs[1, 1], sharex=axB)
    raw_kw = dict(s=2.2, linewidths=0, zorder=2)

    def stretches(rows, cfg):
        out = {}
        for r in rows:
            if r["config"] == cfg:
                out.setdefault(r["stretch"], []).append(r)
        return [out[k] for k in sorted(out)]

    def draw_loss(ax, rows, scale, lo=0.0):
        for cfg in ("final", "v9"):
            pts = [r for r in rows if r["config"] == cfg and r["tokens"] >= lo]
            if not pts:
                continue
            ax.scatter([r["tokens"] / scale for r in pts], [r["loss_raw"] for r in pts], color=colors[cfg][1], **raw_kw)
            for sr in stretches(pts, cfg):
                d = [r for r in sr if r["drawn"]]
                if len(d) >= 2:
                    ax.plot([r["tokens"] / scale for r in d], [r["loss_smoothed"] for r in d], color=colors[cfg][0],
                            lw=1.25, zorder=3)

    def vlines(ax, scale, y0, y1, color=st.MUTED, lw=0.5):
        for m in marks:
            ax.plot([m["tokens"] / scale] * 2, [y0, y1], color=color, lw=lw, zorder=1,
                    ls=(0, (3, 2)) if m["group"] == "phase" else "-")

    # ---------------------------------------------------------------- (a) whole run, log x
    draw_loss(axA, main_rows, 1.0)
    ya_lo = math.floor((min(r["loss_raw"] for r in main_rows) - 0.05) * 10) / 10
    ya_hi = 5.0 if max(r["loss_raw"] for r in main_rows) < 4.95 else math.ceil(max(r["loss_raw"] for r in main_rows) * 2) / 2
    vlines(axA, 1.0, ya_lo, ya_lo + 0.25 * (ya_hi - ya_lo))
    axA.set_xscale("log")
    axA.set_xlim(1e7, xmax * 1.35)
    axA.set_ylim(ya_lo, ya_hi)
    axA.xaxis.set_major_locator(FixedLocator([1e7, 1e8, 1e9, 1e10]))
    axA.xaxis.set_major_formatter(FuncFormatter(pow10))
    axA.xaxis.set_minor_formatter(NullFormatter())
    axA.set_ylabel("Training loss (nats per token)")
    st.grid_y(axA)
    axA.set_title("a   Whole run (logarithmic tokens)")
    xb0, xb1 = 1.0e9, xmax * 1.03
    handles = [Line2D([], [], ls="", marker="o", ms=2.6, mew=0, color=st.FINAL_LIGHT),
               Line2D([], [], color=st.FINAL, lw=1.25),
               Line2D([], [], ls="", marker="o", ms=2.6, mew=0, color=st.V9_LIGHT),
               Line2D([], [], color=st.V9, lw=1.25)]
    labels = ["final, every 50th step", "final, centred average", "v9 (from line 7), every 50th step",
              "v9, centred average"]
    axA.legend(handles, labels, loc="lower left", ncol=1, bbox_to_anchor=(0.0, 0.0), handlelength=1.4,
               handletextpad=0.45, borderaxespad=0.3, labelspacing=0.25, fontsize=st.FS)
    # direct label of the v9 cloud (its raw points have the lightness of final's: grey print needs the name)
    v9pts = [r for r in main_rows if r["config"] == "v9"]
    if v9pts:
        axA.text(max(r["tokens"] for r in v9pts) * 1.06, sorted(r["loss_raw"] for r in v9pts)[len(v9pts) // 2], "v9",
                 ha="left", va="center", fontsize=st.FS, color=st.INK)
    # key of the vertical lines (upper middle: no data there)
    key = []
    for m in marks:
        if m["group"] == "intervention":
            key.append(f"{m['id']}  {m['short']}")
    for m in marks:
        if m["group"] == "measurement":
            key.append(f"{m['id']}  measurement pause")
    us = [m for m in marks if m["group"] == "unlabelled"]
    if us:
        key.append("U  other restarts")
    for m in marks:
        if m["group"] == "phase":
            short = m["short"].split(" (")[0].split(", ")[0]
            key.append(f"{m['id']}  {short} (dashed)")
    axA.text(0.43, 0.975, "Vertical lines (Table C2)", transform=axA.transAxes,
             ha="left", va="top", fontsize=st.FS - 0.5, color=st.INK, fontweight="semibold")
    axA.text(0.43, 0.918, "\n".join(key), transform=axA.transAxes, ha="left", va="top", fontsize=st.FS - 0.5,
             color=st.INK, linespacing=1.33)

    # ---------------------------------------------------------------- (b) from 10^9 tokens, linear x
    lo = xb0
    win = [r for r in main_rows if r["tokens"] >= lo]
    yb_lo = math.floor((min(r["loss_raw"] for r in win) - 0.01) * 20) / 20
    yb_top = max(r["loss_raw"] for r in win)
    draw_loss(axB, main_rows, 1e9, lo)
    axB.set_xlim(lo / 1e9, xb1 / 1e9)
    items = [(m["tokens"] / 1e9, str(m["id"]), (0, (3, 2)) if m["group"] == "phase" else "-")
             for m in marks if m["tokens"] >= lo]
    nl = st.tag_lanes(axB, items, y0=0.965, dy=0.075, min_sep_in=0.20, dry=True)
    free = 1.0 - (0.07 + 0.075 * nl)             # share of the axis height below the lowest tag lane
    axB.set_ylim(yb_lo, yb_lo + (yb_top + 0.008 - yb_lo) / free)
    st.tag_lanes(axB, items, y0=0.965, dy=0.075, min_sep_in=0.20, line_to=0.0)
    axB.set_title("b   From 10⁹ tokens on (linear tokens)")
    # direct labels (ink, not series colour) below the clouds, so the configurations read in grey too
    for cfg, x_at in (("final", 0.5 * (lo + 7.0e9)), ("v9", None)):
        pts = [r for r in win if r["config"] == cfg]
        if not pts:
            continue
        if x_at is None:
            x_at = min(r["tokens"] for r in pts) + 0.45 * (max(r["tokens"] for r in pts) - min(r["tokens"] for r in pts))
        near = [r["loss_raw"] for r in pts if abs(r["tokens"] - x_at) < 4e8]
        axB.text(x_at / 1e9, min(near) - 0.012, cfg, ha="center", va="top", fontsize=st.FS, color=st.INK)
    axB.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    st.grid_y(axB)

    # ---------------------------------------------------------------- (e) the tail, own scale
    if has_tail:
        axE = fig.add_subplot(gs[0, 2])
        axF = fig.add_subplot(gs[1, 2], sharex=axE)
        for cfg in ("final", "v9"):
            pts = [r for r in tail if r["config"] == cfg]
            if not pts:
                continue
            axE.scatter([r["tokens"] / 1e9 for r in pts], [r["loss_raw"] for r in pts], color=colors[cfg][1], **raw_kw)
            for sr in stretches(pts, cfg):
                d = [r for r in sr if r["drawn"]]
                if len(d) >= 2:
                    axE.plot([r["tokens"] / 1e9 for r in d], [r["loss_smoothed"] for r in d], color=colors[cfg][0], lw=1.25,
                             zorder=3)
        t0, t1 = min(r["tokens"] for r in tail) / 1e9, max(r["tokens"] for r in tail) / 1e9
        axE.set_xlim(t0 - 0.05 * (t1 - t0 + 0.1), t1 + 0.05 * (t1 - t0 + 0.1))
        ye = [r["loss_raw"] for r in tail]
        axE.set_ylim(math.floor((min(ye) - 0.02) * 20) / 20, math.ceil((max(ye) + 0.04) * 20) / 20)
        axE.set_title("e   p ≥ 0.92", loc="left")
        axE.text(0.04, 0.97, "loss on assistant\ntokens only", transform=axE.transAxes,
                 ha="left", va="top", fontsize=st.FS - 0.5, color=st.INK2, linespacing=1.15)
        axE.xaxis.set_major_locator(MaxNLocator(nbins=3))
        st.grid_y(axE)
        plt.setp(axE.get_xticklabels(), visible=False)
    else:
        axF = None

    # ---------------------------------------------------------------- (c), (d) schedule
    xs = [r["tokens"] for r in S]
    for ax, scale in ((axC, 1.0), (axD, 1e9)) + (((axF, 1e9),) if axF is not None else ()):
        ax.plot([x / scale for x in xs], [r["lr"] for r in S], color=st.INK, lw=1.0, zorder=3)
        ax.plot([x / scale for x in xs], [r["p"] for r in S], color=st.MUTED, lw=1.0, zorder=3)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_locator(FixedLocator([0, 0.5, 1.0]))
        st.grid_y(ax)
    for m in marks:
        axC.plot([m["tokens"]] * 2, [0, 1.05], color=st.GRID, lw=0.6, zorder=1,
                 ls=(0, (3, 2)) if m["group"] == "phase" else "-")
        axD.plot([m["tokens"] / 1e9] * 2, [0, 1.05], color=st.MUTED, lw=0.5, zorder=1,
                 ls=(0, (3, 2)) if m["group"] == "phase" else "-")
    axC.set_ylabel("Factor or p")
    axC.set_title("c   Schedule (logarithmic tokens)")
    axD.set_title("d   Schedule (linear tokens)")
    axC.set_xlabel("Training tokens (262,144 per optimiser step)")
    axD.set_xlabel("Training tokens (10⁹)")
    axC.text(3.3e7, 0.80, "learning-rate factor", color=st.INK, fontsize=st.FS, ha="left", va="bottom")
    axC.text(1.2e8, 0.04, "progress p", color=st.INK2, fontsize=st.FS, ha="left", va="bottom")
    pmax = max(r["p"] for r in S)
    ph = {m["phase"]: m for m in marks if m["group"] == "phase"}
    for pv, key_ in ((0.92, "tail"), (0.67, "long")):
        axD.axhline(pv, color=st.GRID, lw=0.8, zorder=0)
        when = f"step {ph[key_]['step']:,}" if key_ in ph else "to come"
        right = key_ not in ph                 # a phase to come: label at the right end; reached: left end
        # above the line at p = 0.92, below it at 0.67 (above it the factor curve passes at the left end); the
        # white box hides only restart lines, never a curve
        below = key_ == "long"
        axD.text((xb1 / 1e9 - 0.08) if right else (lo / 1e9 + 0.08), pv - 0.03 if below else pv + 0.03,
                 f"p = {pv} ({'P2' if key_ == 'tail' else 'P1'}{'' if key_ in ph else ', to come'})",
                 fontsize=st.FS - 0.5, color=st.INK2, ha="right" if right else "left",
                 va="top" if below else "bottom", zorder=4, bbox=dict(boxstyle="square,pad=0.1", fc=st.WHITE, ec="none"))
    i4 = next((m for m in marks if str(m["id"]) == "4"), None)
    if i4:                                   # between the two curves, where neither passes (no box over data)
        axD.text(i4["tokens"] / 1e9 + 0.15, 0.50, "4: p and the\nfactor jump", fontsize=st.FS - 0.5,
                 color=st.INK2, ha="left", va="center", zorder=4, linespacing=1.1)
    if axF is not None:
        plt.setp(axF.get_yticklabels(), visible=False)
        axF.set_xlabel("10⁹ tokens")
    plt.setp(axA.get_xticklabels(), visible=False)
    plt.setp(axB.get_xticklabels(), visible=False)

    st.save(fig, "F1_loss")

    # ---------------------------------------------------------------- numbers for FIGURES.md
    f = [r for r in R if r["config"] == "final"]
    v = [r for r in R if r["config"] == "v9"]
    first_drawn = next(r["step"] for r in f if r["drawn"])
    print(f"final: steps {f[0]['step']}..{f[-1]['step']}, raw loss {f[0]['loss_raw']:.4f} at step {f[0]['step']}; "
          f"line drawn from step {first_drawn}")
    for target in (3e8, 1e9, 3e9, 5.9e9, 7.6e9):
        cand = [r for r in f if r["drawn"]]
        i = min(cand, key=lambda r: abs(r["tokens"] - target))
        print(f"  centred average near {target:.3g}: step {i['step']} {i['loss_smoothed']:.4f} (n_eff {i['kernel_records_effective']:.1f})")
    lastf = [r for r in f if r["drawn"]][-1]
    firstv = next((r for r in v if r["drawn"]), None)
    print(f"  last drawn final point: step {lastf['step']} {lastf['loss_smoothed']:.4f}; first drawn v9 point: "
          f"{firstv['step'] if firstv else '-'} {firstv['loss_smoothed'] if firstv else float('nan'):.4f}")
    n = min(21, len(v))
    a, b = [r["loss_raw"] for r in f][-n:], [r["loss_raw"] for r in v][:n]
    print(f"last {n} final raw: mean {statistics.mean(a):.4f} sd {statistics.stdev(a):.4f}; first {n} v9 raw: "
          f"mean {statistics.mean(b):.4f} sd {statistics.stdev(b):.4f}")
    for key_ in sorted({(r["config"], r["stretch"]) for r in R}):
        sr = [r for r in R if (r["config"], r["stretch"]) == key_]
        d = [r for r in sr if r["drawn"]]
        print(f"  stretch {key_}: steps {sr[0]['step']}..{sr[-1]['step']} ({len(sr)} records), line drawn "
              f"{d[0]['step'] if d else '-'}..{d[-1]['step'] if d else '-'}; raw mean of the first 5 "
              f"{statistics.mean(r['loss_raw'] for r in sr[:5]):.4f}, last 5 {statistics.mean(r['loss_raw'] for r in sr[-5:]):.4f}")
    full = [r["loss_raw"] - r["loss_smoothed"] for r in R
            if r["drawn"] and abs(float(r["half_width_decades"]) - C.CUT * C.SIGMA_DEC) < 1e-9]
    print(f"scatter of a step's loss around the line where the window is full: sd {statistics.pstdev(full):.4f} "
          f"({len(full)} records)")
    print(f"records: final {len(f)}, v9 {len(v)}; cut-off {C.cutoff()['label']}")
    print(f"p max {pmax:.4f}; tail records {len(tail)}")


if __name__ == "__main__":
    main()
