"""F3: on-chip comparison of the attention-kernel candidates (24 Nov 07:44-07:52 UTC, one pause).

Data: paper/data/chip_kernels_0745.csv (make_data.py from wf/chip_kernels_0745.txt): one
value_and_grad of a piece of 4,096 tokens (4 x 1024 or 2 x 2048), configs/final.py, median of 30
timings, with their minimum and maximum. Writes F3_kernels.svg/png.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as st  # noqa: E402
import common as C  # noqa: E402

st.setup()
plt = st.plt

ORDER = [  # (kernel key, row label) top to bottom
    ("", "splash kernel (configured)"),
    ("+io", "splash + io"),
    ("+relu2+ce", "splash + relu2 + ce"),
    ("blocks:save", "splash + save"),
    ("onetile", "one-tile"),
    ("onetile:save", "one-tile + save"),
    ("onetile:save+relu2", "one-tile + save + relu2"),
    ("onetile:save+relu2+ce", "one-tile + save + relu2 + ce"),
]
ADOPTED = "onetile:save+relu2"


def main():
    rows = C.read("chip_kernels_0745.csv")
    by = {(r["kernel"], int(r["t"])): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(st.COL_W, 2.35), sharey=True,
                             gridspec_kw=dict(left=0.395, right=0.985, top=0.86, bottom=0.17, wspace=0.16))
    n = len(ORDER)
    for ax, t in zip(axes, (1024, 2048)):
        ref = float(by[("", t)]["ms"])
        ax.axvline(ref, color=st.MUTED, lw=0.6, zorder=1)
        for i, (k, lab) in enumerate(ORDER):
            r = by[(k, t)]
            y = n - 1 - i
            ms, lo, hi = float(r["ms"]), float(r["min"]), float(r["max"])
            col = st.FINAL if k == ADOPTED else (st.INK if k == "" else st.MUTED)
            ax.plot([lo, hi], [y, y], color=col, lw=0.9, zorder=2, solid_capstyle="butt")
            ax.plot([ms], [y], ls="", marker="o", ms=4.2, mfc=col, mec=st.WHITE, mew=0.8, zorder=3)
            sp = ref / ms
            txt = "ref." if k == "" else f"×{sp:.2f}"
            ax.text(20.75, y, txt, va="center", ha="left", fontsize=st.FS - 0.5,
                    color=st.INK if k == ADOPTED else st.INK2,
                    fontweight="semibold" if k == ADOPTED else "normal")
        ax.set_xlim(14.0, 22.4)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xticks([14, 16, 18, 20])
        st.grid_x(ax)
        ax.set_title(f"T = {t:,}", fontsize=st.FS_LABEL, fontweight="semibold")
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels([lab for _, lab in reversed(ORDER)])
    for lbl in axes[0].get_yticklabels():
        if lbl.get_text().startswith("one-tile + save + relu2") and not lbl.get_text().endswith("ce"):
            lbl.set_color(st.INK)
            lbl.set_fontweight("semibold")
    axes[0].tick_params(axis="y", length=0)
    axes[1].tick_params(axis="y", length=0)
    fig.supxlabel("Milliseconds per piece of 4,096 tokens (value and gradient)", fontsize=st.FS_LABEL, y=0.02)
    st.save(fig, "F3_kernels")
    for k, lab in ORDER:
        a, b = by[(k, 1024)], by[(k, 2048)]
        print(f"{lab:32s} 1024: {float(a['ms']):7.3f} ms [{a['min']}-{a['max']}] x{float(by[('',1024)]['ms'])/float(a['ms']):.3f}"
              f"  2048: {float(b['ms']):7.3f} [{b['min']}-{b['max']}] x{float(by[('',2048)]['ms'])/float(b['ms']):.3f}"
              f"  dloss {a['loss_diff'] or 0}/{b['loss_diff'] or 0} grad max {a['grad_rel_max']}/{b['grad_rel_max']} l2 {a['grad_rel_l2']}/{b['grad_rel_l2']}")


if __name__ == "__main__":
    main()
