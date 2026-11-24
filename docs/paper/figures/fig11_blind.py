"""F11: the blind pairwise comparisons against the two opponents: wins, ties or "both bad", losses,
and the score (wins + ties/2)/n with its 95% Wilson interval.

Data: paper/data/blind_comparisons.csv (make_data.py, from the bes.eval.proxy reports in
study/proxy/reports/ab/): the four development sets by default; the final test-split sets appear
when make_data.py (or make_all.sh) is given their names with --blind. One row per report, grouped
by opponent, ordered by our checkpoint's step and then by split. Writes F11_blind.svg/png.
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

TIE = st.GRID                 # ties or "both bad": light grey (L* about 89)
LOSS = "#3d3c39"              # opponent preferred: dark grey (L* about 25); wins carry the configuration's colour
OPP_ORDER = ["falcon", "mobilellm"]


def main():
    B = C.read("blind_comparisons.csv")
    for r in B:
        for k in ("n", "wins", "losses", "ties_or_both_bad"):
            r[k] = int(r[k])
        for k in ("score", "score_lo", "score_hi"):
            r[k] = float(r[k])
        r["our_step"] = int(r["our_step"]) if r["our_step"] else None
    rows = []
    for opp in OPP_ORDER + sorted({r["opponent"] for r in B} - set(OPP_ORDER)):
        sel = sorted((r for r in B if r["opponent"] == opp), key=lambda r: (r["split"] != "dev", r["our_step"] or 0))
        if not sel:
            continue
        rows.append(("head", f"against {sel[0]['opponent_label']}"))
        rows += [("row", r) for r in sel]
    n = len(rows)
    height = 0.33 * n + 0.75
    fig = plt.figure(figsize=(st.FULL_W, height))
    bottom, top = 0.42 / height, 1 - 0.33 / height
    axW = fig.add_axes([0.30, bottom, 0.345, top - bottom])
    axS = fig.add_axes([0.685, bottom, 0.165, top - bottom], sharey=axW)
    bar_in = 0.345 * st.FULL_W                 # inches of the 0-100 % axis
    for ax in (axW, axS):
        ax.set_ylim(n - 0.5, -0.6)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.spines["left"].set_visible(False)
    H = 0.58
    for y, (kind, r) in enumerate(rows):
        if kind == "head":
            axW.text(-0.80, y + 0.1, r, transform=axW.get_yaxis_transform(), ha="left", va="center",
                     fontsize=st.FS - 0.5, fontweight="semibold", color=st.INK, clip_on=False)
            continue
        win_col = st.FINAL if r["our_config"] == "final" else (st.V9 if r["our_config"] == "v9" else st.INK2)
        x = 0.0
        segs = ((r["wins"], win_col, st.WHITE, "wins"), (r["ties_or_both_bad"], TIE, st.INK, "ties"),
                (r["losses"], LOSS, st.WHITE, "losses"))
        for cnt, col, txt_col, what in segs:
            w = 100.0 * cnt / r["n"]
            if w > 0:
                axW.barh(y, w, left=x, height=H, color=col, lw=0, zorder=2)
                # a 2-px surface gap between segments
                axW.plot([x + w, x + w], [y - H / 2, y + H / 2], color=st.WHITE, lw=1.2, zorder=3)
                fits = w / 100 * bar_in >= 0.055 * len(str(cnt)) + 0.025
                if fits:
                    axW.text(x + w / 2, y, str(cnt), ha="center", va="center", fontsize=st.FS - 0.5, color=txt_col,
                             zorder=4)
                elif what == "wins":         # too narrow: the count goes just outside the bar's ends
                    axW.text(-0.8, y, str(cnt), ha="right", va="center", fontsize=st.FS - 0.5, color=st.INK, zorder=4)
                elif what == "losses":
                    axW.text(100.8, y, str(cnt), ha="left", va="center", fontsize=st.FS - 0.5, color=st.INK,
                             zorder=4, clip_on=False)
                else:
                    axW.text(x + w / 2, y, str(cnt), ha="center", va="center", fontsize=st.FS - 0.5, color=st.INK,
                             zorder=4)
            x += w
        cfg = r["our_config"] or "ours"
        step = f", step {r['our_step']:,}" if r["our_step"] else ""
        axW.text(-0.05, y - 0.14, f"{cfg}{step}", transform=axW.get_yaxis_transform(), ha="right", va="center",
                 fontsize=st.FS - 0.5, color=st.INK, clip_on=False)
        axW.text(-0.05, y + 0.21, f"{r['split']}, {r['n']} pairs, {r['readings'].replace(' on exchanged sides', '')}",
                 transform=axW.get_yaxis_transform(), ha="right", va="center", fontsize=st.FS - 0.5, color=st.INK2,
                 clip_on=False)
        # score and its 95% Wilson interval
        axS.plot([r["score_lo"], r["score_hi"]], [y, y], color=st.INK, lw=1.0, zorder=3, solid_capstyle="butt")
        for e in (r["score_lo"], r["score_hi"]):
            axS.plot([e, e], [y - 0.13, y + 0.13], color=st.INK, lw=1.0, zorder=3)
        axS.plot([r["score"]], [y], ls="", marker="o", ms=5.0, mfc=win_col, mec=st.WHITE, mew=0.8, zorder=4)
        axS.text(1.03, y, f"{r['score']:.3f} [{r['score_lo']:.3f}, {r['score_hi']:.3f}]", transform=axS.get_yaxis_transform(),
                 ha="left", va="center", fontsize=st.FS - 0.5, color=st.INK, clip_on=False)
    axW.set_xlim(0, 100)
    axW.set_xticks([0, 25, 50, 75, 100])
    st.grid_x(axW)
    axW.set_xlabel("Share of pairs (%); numbers: pairs")
    axS.set_xlim(0, 1)
    axS.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axS.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    axS.axvline(0.5, color=st.MUTED, lw=0.6, zorder=1)
    st.grid_x(axS)
    axS.set_xlabel("Score (wins + ties/2) / n")
    cfgs = [c for c in ("final", "v9") if any(r["our_config"] == c for k, r in rows if k == "row")]
    handles = [Patch(fc=st.FINAL if c == "final" else st.V9, ec="none") for c in cfgs] + \
        [Patch(fc=TIE, ec="none"), Patch(fc=LOSS, ec="none"),
         Line2D([], [], color=st.INK, lw=1.0, marker="o", ms=4.5, mfc=st.INK2, mec=st.WHITE)]
    labels = [f"ours ({c}) preferred" for c in cfgs] + ["tie or both bad", "opponent preferred",
                                                        "score, 95% Wilson interval"]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.012, 1.0), ncol=len(labels), fontsize=st.FS - 0.5,
               handlelength=1.3, handletextpad=0.4, columnspacing=1.0, frameon=False)
    st.save(fig, "F11_blind")
    for k, r in rows:
        if k == "row":
            print(f"{r['name']:42s} {r['our_config']:5s} {r['our_step']} W/T/L {r['wins']}/{r['ties_or_both_bad']}/{r['losses']}"
                  f" score {r['score']:.4f} [{r['score_lo']:.4f}, {r['score_hi']:.4f}] p {r['sign_test_p']}")


if __name__ == "__main__":
    main()
