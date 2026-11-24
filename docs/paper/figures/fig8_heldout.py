"""F8 (additional): held-out loss of the checkpoints on both sides of the graft, against training time.

Data: paper/data/heldout_loss.csv (make_data.py from mideval_last.log: scripts/midrun_eval.py on the
runtime's host cores, the export check's rows: web 64, chat 64, proxy 16; float32, dense attention;
newer v9 checkpoints from heldout_v9.jsonl). The grey line is a least-squares line through final's
points, extended past the graft: an extrapolation, not a measurement. Writes F8_heldout.svg/png.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as st  # noqa: E402
import common as C  # noqa: E402

st.setup()
plt = st.plt


def fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def main():
    H = C.read("heldout_loss.csv")
    F = [r for r in H if r["config"] == "final"]
    V = [r for r in H if r["config"] == "v9" and r["train_minutes"]]
    xf, yf = [float(r["train_minutes"]) for r in F], [float(r["loss"]) for r in F]
    xv, yv = [float(r["train_minutes"]) for r in V], [float(r["loss"]) for r in V]
    a, b = fit(xf, yf)
    fig, ax = plt.subplots(figsize=(st.COL_W, 2.25), gridspec_kw=dict(left=0.16, right=0.97, top=0.93, bottom=0.19))
    x_graft = xf[-1]
    ax.axvline(x_graft, color=st.MUTED, lw=0.5, zorder=1)
    ax.text(x_graft + 1.0, max(yf) + 0.0005, "graft (7)", fontsize=st.FS - 0.5, color=st.INK2, va="top")
    xe = [xf[0], max(xv + xf)]
    ax.plot(xe, [a + b * x for x in xe], color=st.MUTED, lw=0.9, zorder=2)
    ax.plot(xf, yf, ls="-", lw=1.0, color=st.FINAL, marker="o", ms=4, mec=st.WHITE, mew=0.8, zorder=3)
    ax.plot(xv, yv, ls="-", lw=1.0, color=st.V9, marker="o", ms=4, mec=st.WHITE, mew=0.8, zorder=3)
    ax.text(xf[0] + 1, yf[0] + 0.0012, "final", fontsize=st.FS, color=st.INK, va="bottom")
    ax.text(xv[-1] - 1, yv[-1] - 0.0016, "v9", fontsize=st.FS, color=st.INK, va="top", ha="right")
    ax.text(xe[1] - 1, a + b * xe[1] + 0.0014, "final's line through its three points,\nextended (extrapolation)",
            fontsize=st.FS - 0.5, color=st.INK2, ha="right", va="bottom", linespacing=1.1)
    gap = (a + b * xv[-1]) - yv[-1]
    ax.annotate("", xy=(xv[-1] + 2.5, yv[-1]), xytext=(xv[-1] + 2.5, a + b * xv[-1]),
                arrowprops=dict(arrowstyle="<->", color=st.INK, lw=0.6, shrinkA=0, shrinkB=0))
    ax.text(xv[-1] + 4, (yv[-1] + a + b * xv[-1]) / 2, f"{gap:.4f}", fontsize=st.FS - 0.5, color=st.INK,
            va="center", ha="left")
    ax.set_xlim(xf[0] - 8, max(xv) + 24)
    st.grid_y(ax)
    ax.set_xlabel("Training time of the run (minutes)")
    ax.set_ylabel("Held-out loss (nats per token)")
    st.save(fig, "F8_heldout")
    print(f"final slope {b:.6f} per minute; extrapolated at {xv[-1]} min: {a + b * xv[-1]:.4f}; v9 {yv[-1]:.4f}; "
          f"gap {gap:.4f}")


if __name__ == "__main__":
    main()
