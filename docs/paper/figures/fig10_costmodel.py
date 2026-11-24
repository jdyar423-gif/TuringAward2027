"""F10: the compiler's cost model against the chip (the paper's Tables 4 and 5).

Data (make_data.py, from lead2_work/predictor_test.json): paper/data/costmodel_steps.csv (six
configurations with a compiled program and a whole-step time on the chip: each static count's
leave-one-out prediction of the measured step) and costmodel_differences.csv (seven changes with a
compile-only prediction and a chip measurement). Writes F10_costmodel.svg/png.

(a) is a dot plot, one row per configuration and one line per count inside it, so that no marker
can hide another: the measured step is a vertical bar, each count's prediction a marker joined to
it by a thin line (its error). (b) puts the predicted against the measured change on log axes.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as st  # noqa: E402
import common as C  # noqa: E402

st.setup()
plt = st.plt
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator  # noqa: E402

SHORT = {"final take 64x4x1024": "final, 4-row pieces", "v9 take 64x4x1024": "v9 as grafted",
         "v9 take 32x8x1024": "v9, 8-row pieces", "v9 sorted 64x4x1024": "v9, sorted",
         "v9 take+flat 64x4x1024": "v9 + flat", "v9 take+flat+rows 64x4x1024": "v9 + flat + rows"}
DIFF_LABEL = {  # change in costmodel_differences.csv -> (short label, label x, label y in ms, ha); the paper's
    # Table 5 names them in full
    "v9 - final (take, 64x4x1024)": ("v9 instead of final", 420, 624, "right"),
    "sorted - take (v9)": ("sorted", 356, 92, "center"),
    "8-row - 4-row pieces (v9, take)": ("8-row (v9)", 66, 37, "left"),
    "8-row - 4-row pieces (final; chip: splash kernels, 03:09)": ("8-row (final)", 18.2, 150, "center"),
    "flat (v9)": ("flat", 24, 35, "right"),
    "rows (v9, on flat)": ("rows", 92, 119, "right"),
    "tokn (v9; predicted on flat, measured on flat+rows)": ("tokn", 8.1, 90, "center"),
}
# count, legend label, offset of its line inside a configuration's row, marker style
SERIES = [
    ("bytes", "estimated HBM bytes (compiler)", -0.24,
     dict(marker="o", ms=4.6, mfc=st.ACCENT, mec=st.ACCENT, mew=0.6)),
    ("cycles", "estimated cycles (compiler)", 0.0,
     dict(marker="s", ms=4.2, mfc=st.WHITE, mec=st.INK, mew=1.0)),
    ("flop", "analytic FLOP or reads per token", 0.24,
     dict(marker="X", ms=4.8, mfc=st.INK2, mec=st.WHITE, mew=0.4)),
]
UP = dict(marker="^", ms=5.2, mfc=st.INK2, mec=st.INK2, mew=0.9)        # (b): the change made the step slower
DOWN = dict(marker="v", ms=5.2, mfc=st.WHITE, mec=st.INK2, mew=1.0)     # (b): ... faster


def main():
    S = C.read("costmodel_steps.csv")
    D = C.read("costmodel_differences.csv")
    fig = plt.figure(figsize=(st.FULL_W, 3.9))
    axA = fig.add_axes([0.175, 0.15, 0.33, 0.63])
    hb = 0.63 * 3.9 / st.FULL_W                     # (b) square: as wide as it is tall
    axB = fig.add_axes([0.972 - hb, 0.15, hb, 0.63])

    # ---------------------------------------------------------------- (a) whole step, leave-one-out
    order = sorted(S, key=lambda r: float(r["measured_ms"]))
    summ = {}
    for count, *_ in SERIES:
        e = [abs(float(r[f"loo_{count}_err"])) for r in S if r[f"loo_{count}_err"] not in ("", None)]
        summ[count] = (100 * sum(e) / len(e), 100 * max(e), len(e))
    for i, r in enumerate(order):
        m = float(r["measured_ms"])
        axA.plot([m, m], [i - 0.40, i + 0.40], color=st.INK, lw=1.4, solid_capstyle="butt", zorder=3)
        for count, label, dy, sty in SERIES:
            v = r[f"loo_{count}_ms"]
            if v in ("", None):                     # FLOP and reads: undefined for final (one final point)
                axA.text(m + 22, i + dy, "FLOP, reads: undefined", ha="left", va="center", fontsize=st.FS - 1.0,
                         color=st.INK2)
                continue
            v = float(v)
            axA.plot([m, v], [i + dy, i + dy], color=st.MUTED, lw=0.7, zorder=2, solid_capstyle="butt")
            axA.plot([v], [i + dy], ls="", zorder=5, **sty)     # above the bar: a marker next to it stays whole
        if i > 0:
            axA.axhline(i - 0.5, color=st.GRID, lw=0.5, zorder=0)
    axA.set_ylim(len(order) - 0.5, -0.5)
    axA.set_yticks(range(len(order)))
    axA.set_yticklabels([f"{SHORT.get(r['config'], r['config'])}\n{float(r['measured_ms']):,.1f} ms" for r in order],
                        fontsize=st.FS - 0.5, linespacing=1.15)
    axA.tick_params(axis="y", length=0)
    axA.spines["left"].set_visible(False)
    axA.set_xlim(450, 1850)
    axA.xaxis.set_major_locator(FixedLocator([600, 800, 1000, 1200, 1400, 1600, 1800]))
    axA.xaxis.set_major_formatter(st.plt.FuncFormatter(st.thousands))
    st.grid_x(axA)
    axA.set_xlabel("Milliseconds per step: measured (bar),\npredicted leave-one-out (markers)", linespacing=1.15)
    axA.set_title("a   Whole step, six configurations (Table 4)", x=-0.53)

    # ---------------------------------------------------------------- (b) differences, magnitudes
    lo_b, hi_b = 5, 1000
    axB.plot([lo_b, hi_b], [lo_b, hi_b], color=st.MUTED, lw=0.8, zorder=1)
    for f in (2, 0.5):
        axB.plot([lo_b, hi_b], [lo_b * f, hi_b * f], color=st.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    for x_at, f, lab in ((5.6, 2, "×2"), (11, 1, "equal"), (22, 0.5, "×½")):
        axB.text(x_at, x_at * f * 1.1, lab, fontsize=st.FS - 0.5, color=st.INK2, ha="left", va="bottom", rotation=45,
                 rotation_mode="anchor", transform_rotates_text=True)
    for r in D:
        p, m = float(r["predicted_ms"]), float(r["measured_ms"])
        axB.plot([abs(m)], [abs(p)], ls="", zorder=3, **(UP if m > 0 else DOWN))
        lab, lx, ly, ha = DIFF_LABEL.get(r["change"], (r["change"], abs(m) * 1.15, abs(p), "left"))
        far = abs(math.log10(ly / abs(p))) > 0.2
        axB.annotate(f"{lab} {float(r['ratio']):.2f}", xy=(abs(m), abs(p)), xytext=(lx, ly), textcoords="data",
                     ha=ha, va="center", fontsize=st.FS - 0.5, color=st.INK, zorder=2.5,   # over the guides only
                     bbox=dict(boxstyle="square,pad=0.08", fc=st.WHITE, ec="none"),
                     arrowprops=dict(arrowstyle="-", color=st.INK2, lw=0.5, shrinkA=1.5, shrinkB=3.5) if far else None)
    n_sign = sum(int(r["sign_agrees"]) for r in D)
    axB.text(0.97, 0.03, f"sign right in {n_sign} of {len(D)}", transform=axB.transAxes, ha="right", va="bottom",
             fontsize=st.FS - 0.5, color=st.INK)
    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.set_xlim(lo_b, hi_b)
    axB.set_ylim(lo_b, hi_b)
    for a in (axB.xaxis, axB.yaxis):
        a.set_major_locator(FixedLocator([10, 30, 100, 300, 1000]))
        a.set_major_formatter(st.plt.FuncFormatter(lambda v, _p: f"{v:,.0f}"))
        a.set_minor_locator(NullLocator())
        a.set_minor_formatter(NullFormatter())
    st.grid_y(axB)
    st.grid_x(axB)
    axB.set_xlabel("Measured change (ms per step, absolute)")
    axB.set_ylabel("Predicted change (ms per step, absolute)")
    axB.set_title("b   Seven changes (Table 5)")

    # keys: one per panel, above it
    hA = [Line2D([], [], color=st.INK, lw=1.4)] + [Line2D([], [], ls="", **sty) for *_, sty in SERIES]
    lA = ["measured on the chip"] + [
        f"{label}: mean error {summ[c][0]:.0f}%, largest {summ[c][1]:.0f}%"
        + ("" if summ[c][2] == len(S) else f" ({summ[c][2]} v9 configurations)") for c, label, _, _ in SERIES]
    fig.legend(hA, lA, loc="upper left", bbox_to_anchor=(0.005, 0.998), ncol=1, fontsize=st.FS - 0.5,
               handlelength=1.3, handletextpad=0.5, labelspacing=0.25, frameon=False)
    hB = [Line2D([], [], ls="", **UP), Line2D([], [], ls="", **DOWN)]
    hB.append(Line2D([], [], ls="", marker=""))
    lB = ["the change made the step slower", "the change made the step faster",
          "number after a label: predicted ÷ measured change"]
    fig.legend(hB, lB, loc="upper left", bbox_to_anchor=(0.60, 0.998), ncol=1, fontsize=st.FS - 0.5,
               handlelength=1.3, handletextpad=0.5, labelspacing=0.25, frameon=False)
    st.save(fig, "F10_costmodel")
    for r in order:
        print(f"{r['config']:30s} measured {r['measured_ms']:>7}  bytes {r['loo_bytes_ms']:>8} ({r['loo_bytes_err']})"
              f"  cycles {r['loo_cycles_ms']:>8} ({r['loo_cycles_err']})  flop {r['loo_flop_ms'] or '-':>8}"
              f" ({r['loo_flop_err'] or '-'})")
    for r in D:
        print(f"  {r['change']:58s} predicted {r['predicted_ms']:>7} measured {r['measured_ms']:>7} ratio {r['ratio']}")
    print("summary:", "; ".join(f"{c}: {summ[c][0]:.1f}% ({summ[c][1]:.1f}%) on {summ[c][2]} of {len(S)}"
                                for c, *_ in SERIES))


if __name__ == "__main__":
    main()
