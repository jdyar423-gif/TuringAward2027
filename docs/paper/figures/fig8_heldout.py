"""F8: held-out loss across the graft on the comparison set of Table 7 (web + chat rows), with final's
line extended past the graft and its 95% prediction interval.

Data (all written by make_data.py): paper/data/heldout_loss.csv (the runtime's evaluations,
midrun_eval_1.json and heldout_v9.jsonl, and the local CPU evaluations of
lead2_work/heldout_analysis.json, pages switched off included) and heldout_trend.csv (the line:
least squares of final's web + chat loss on training minutes through its three checkpoints, and its
95% prediction interval with n - 2 = 1 degree of freedom). Writes F8_heldout.svg/png.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as st  # noqa: E402
import common as C  # noqa: E402

st.setup()
plt = st.plt
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


def main():
    H = C.read("heldout_loss.csv")
    for r in H:
        for k in ("train_minutes", "web_chat"):
            r[k] = float(r[k])
        r["step"] = int(r["step"])
    TR = [{k: float(v) for k, v in r.items() if k in ("train_minutes", "line", "pi95_low", "pi95_high", "pi95_half")}
          for r in C.read("heldout_trend.csv")]
    fit = C.read("heldout_trend.csv")[0]
    F = [r for r in H if r["config"] == "final" and r["evaluated"] == "runtime"]
    V = [r for r in H if r["config"] == "v9" and r["evaluated"] == "runtime" and r["variant"] == "on"]
    L = [r for r in H if r["evaluated"] == "local" and r["variant"] == "on"]
    OFF = [r for r in H if r["evaluated"] == "local" and r["variant"] == "pages off"]
    pts = F + V + OFF
    ylo, yhi = min(r["web_chat"] for r in pts), max(r["web_chat"] for r in pts)
    span = yhi - ylo
    last = max(V, key=lambda r: r["train_minutes"])
    band_last = TR[-1]
    # the axis shows the band at the newest v9 checkpoint unless it is wider than the points' own range
    y0 = max(min(ylo - 0.12 * span, band_last["pi95_low"] - 0.004), ylo - 1.0 * span)
    y1 = min(max(yhi + 0.12 * span, band_last["pi95_high"] + 0.004), yhi + 1.0 * span)

    fig, ax = plt.subplots(figsize=(st.COL_W, 3.15), gridspec_kw=dict(left=0.17, right=0.975, top=0.80, bottom=0.13))
    xs = [t["train_minutes"] for t in TR]
    ax.fill_between(xs, [t["pi95_low"] for t in TR], [t["pi95_high"] for t in TR], color=st.SHADE2, lw=0, zorder=0)
    ax.plot(xs, [t["line"] for t in TR], color=st.MUTED, lw=0.9, zorder=1)
    x_graft = max(r["train_minutes"] for r in F)
    ax.axvline(x_graft, color=st.INK2, lw=0.5, zorder=1)
    ax.text(x_graft + 1.5, y1 - 0.03 * (y1 - y0), "graft", fontsize=st.FS - 0.5, color=st.INK2, va="top", ha="left")
    ax.plot([r["train_minutes"] for r in F], [r["web_chat"] for r in F], ls="-", lw=0.9, color=st.FINAL, marker="o",
            ms=4.2, mec=st.WHITE, mew=0.8, zorder=3)
    ax.plot([r["train_minutes"] for r in V], [r["web_chat"] for r in V], ls="-", lw=0.9, color=st.V9, marker="o",
            ms=4.2, mec=st.WHITE, mew=0.8, zorder=3)
    ax.plot([r["train_minutes"] for r in L], [r["web_chat"] for r in L], ls="", marker="o", ms=7.6, mfc="none",
            mec=st.INK, mew=0.7, zorder=4)
    ax.plot([r["train_minutes"] for r in OFF], [r["web_chat"] for r in OFF], ls="", marker="D", ms=4.2, mfc=st.WHITE,
            mec=st.V9, mew=1.0, zorder=4)
    for r in OFF:
        on = next(v for v in V if v["step"] == r["step"])
        ax.plot([r["train_minutes"]] * 2, [on["web_chat"] + 0.0015, r["web_chat"] - 0.0015], color=st.V9, lw=0.6,
                ls=(0, (2, 1.5)), zorder=2)
    # the gap at the newest v9 checkpoint, rounded: an extrapolation does not deserve four digits
    gap, half = float(last["gap_to_final_line"]), float(last["final_line_pi95_half"])
    # the arrow stands right of the newest checkpoint (and of its dashed pages-off connector), and the
    # x axis leaves room for its two-line label inside the axes
    xg = last["train_minutes"] + 9
    ax.annotate("", xy=(xg, last["web_chat"]), xytext=(xg, float(last["final_line"])),
                arrowprops=dict(arrowstyle="<->", color=st.INK, lw=0.6, shrinkA=0, shrinkB=0, mutation_scale=5))
    ax.text(xg + 2, (last["web_chat"] + float(last["final_line"])) / 2, f"{gap:+.2f}\n± {half:.2f}".replace("-", "−"),
            fontsize=st.FS - 0.5, color=st.INK, va="center", ha="left", linespacing=1.1)
    ax.set_xlim(min(r["train_minutes"] for r in F) - 10, last["train_minutes"] + 62)
    ax.set_ylim(y0, y1)
    st.grid_y(ax)
    ax.set_xlabel("Training time of the run (minutes)")
    ax.set_ylabel("Held-out loss, web + chat (nats per token)")
    handles = [Line2D([], [], color=st.FINAL, marker="o", ms=4.2, mec=st.WHITE, mew=0.8, lw=0.9),
               Line2D([], [], color=st.V9, marker="o", ms=4.2, mec=st.WHITE, mew=0.8, lw=0.9),
               Line2D([], [], ls="", marker="o", ms=7.6, mfc="none", mec=st.INK, mew=0.7),
               Line2D([], [], ls="", marker="D", ms=4.2, mfc=st.WHITE, mec=st.V9, mew=1.0),
               Line2D([], [], color=st.MUTED, lw=0.9), Patch(fc=st.SHADE2, ec="none")]
    labels = ["final (runtime)", "v9 (runtime)", "evaluated again locally", "v9, pages off at evaluation",
              f"final's line ({len(F)} points), extended", "its 95% prediction interval"]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.995), ncol=2, fontsize=st.FS - 0.5,
               handlelength=1.6, handletextpad=0.5, labelspacing=0.35, columnspacing=1.0, frameon=False)
    st.save(fig, "F8_heldout")
    print(f"final's line: slope {float(fit['fit_slope_per_minute']):.3e} per minute, residual sd "
          f"{float(fit['fit_residual_sd']):.2e}, dof {fit['fit_dof']}, t {fit['fit_t975']}")
    for r in V:
        print(f"  v9 {r['step']} at {r['train_minutes']} min: {r['web_chat']:.4f}; line {float(r['final_line']):.4f} "
              f"± {float(r['final_line_pi95_half']):.4f}; gap {float(r['gap_to_final_line']):+.4f}")
    for r in OFF:
        print(f"  pages off {r['step']}: {r['web_chat']:.4f}")
    print(f"  y axis {y0:.4f}..{y1:.4f}; band at {band_last['train_minutes']} min {band_last['pi95_low']:.4f}.."
          f"{band_last['pi95_high']:.4f}")


if __name__ == "__main__":
    main()
