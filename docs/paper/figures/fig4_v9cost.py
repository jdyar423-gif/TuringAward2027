"""F4: cost of one v9 training step against final's, by table-gradient mode and piece size.

Data: paper/data/chip_whole_steps.csv (make_data.py: wf/chip_kernels_0745.txt for final,
v9_measA/B/C for v9) and train_steps.csv (live step time). One accumulation step = 262,144 tokens,
synthetic batch, kernel onetile:save + relu2 (final's whole step was timed with the ce switch on
as well, see FIGURES.md). Writes F4_v9cost.svg/png.
"""
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as st  # noqa: E402
import common as C  # noqa: E402
import paths as P  # noqa: E402

st.setup()
plt = st.plt


def main():
    W = C.read("chip_whole_steps.csv")
    S = C.steps()

    def ms(cfg, mode, a, rows, t, kernel=None):
        want = kernel or ("onetile:save+relu2+ce" if cfg == "final" else "onetile:save+relu2")
        hit = [r for r in W if r["config"] == cfg and r["table_grad"] == mode and int(r["a"]) == a
               and int(r["rows"]) == rows and int(r["t"]) == t and r["kernel"] == want]
        assert len(hit) == 1, (cfg, mode, a, rows, t, hit)
        return float(hit[0]["ms"])

    groups = [
        (1024, [("final", "take", 64, 4), ("final", "take", 128, 2), ("v9", "take", 64, 4), ("v9", "take", 32, 8),
                ("v9", "onehot", 64, 4), ("v9", "sorted", 64, 4),
                ("v9", "take", 64, 4, "onetile:save+relu2+flat"), ("v9", "take", 64, 4, "onetile:save+relu2+flat+rows"),
                ("v9", "take", 64, 4, "onetile:save+relu2+flat+rows+tokn")]),
        (2048, [("final", "take", 64, 2), ("final", "take", 128, 1), ("v9", "take", 64, 2), ("v9", "take", 32, 4),
                ("v9", "sorted", 64, 2)]),
    ]
    live = {}
    for key, lo, hi in (("final", 22613, 29722), ("v9", 29722, 31912), ("v9 switches", 31912, 10 ** 9)):
        sel = [s for s in S if lo < s["step"] <= hi and s["interval_tok_per_s"]]
        if sel:
            live[key] = statistics.median(P.TOKENS_PER_STEP / s["interval_tok_per_s"] * 1e3 for s in sel)

    fig, axes = plt.subplots(2, 1, figsize=(st.COL_W, 3.55), sharex=True,
                             gridspec_kw=dict(height_ratios=[9, 5], left=0.335, right=0.985, top=0.95,
                                              bottom=0.135, hspace=0.32))
    for ax, (t, rows) in zip(axes, groups):
        n = len(rows)
        base = ms(*(rows[0][:2]), rows[0][2], rows[0][3], t)
        labels = []
        for i, spec in enumerate(rows):
            cfg, mode, a, r = spec[:4]
            kern = spec[4] if len(spec) > 4 else None
            y = n - 1 - i
            v = ms(cfg, mode, a, r, t, kern)
            col = st.FINAL if cfg == "final" else st.V9
            ax.barh(y, v, height=0.56, color=col, lw=0, zorder=2)
            ratio = v / base
            lv = None
            if t == 1024 and a == 64 and mode == "take":
                if cfg == "final":
                    lv = live.get("final")
                elif kern is None:
                    lv = live.get("v9")
                elif kern.endswith("tokn"):
                    lv = live.get("v9 switches")
            txt = f"{v:,.1f}" + ("" if i == 0 else f" (×{ratio:.2f})") + (f"; live {lv:,.0f}" if lv else "")
            ax.text(max(v, lv or 0) + 30, y, txt, va="center", ha="left", fontsize=st.FS - 0.5, color=st.INK)
            if kern:
                labels.append({"onetile:save+relu2+flat": "v9 + flat", "onetile:save+relu2+flat+rows":
                               "v9 + flat + rows", "onetile:save+relu2+flat+rows+tokn":
                               "v9 + flat + rows + tokn"}[kern])
            else:
                labels.append(f"{cfg}, {mode}, {r}-row pieces")
            if lv:
                ax.plot([lv, lv], [y - 0.36, y + 0.36], color=st.INK, lw=1.0, zorder=4, solid_capstyle="butt")
        ax.set_yticks(range(n))
        ax.set_yticklabels(list(reversed(labels)), fontsize=st.FS - 0.5)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_title(f"T = {t:,}", fontsize=st.FS_LABEL, fontweight="semibold")
        st.grid_x(ax)
    axes[0].set_xlim(0, 3050)
    axes[0].set_xticks([0, 500, 1000, 1500, 2000])
    axes[1].set_xlabel("Milliseconds per step of 262,144 tokens")
    st.save(fig, "F4_v9cost")
    print("live median step (ms):", {k: round(v, 1) for k, v in live.items()})
    for t, rows in groups:
        for spec in rows:
            print(f"T={t}", spec, f"{ms(*spec[:4], t, spec[4] if len(spec) > 4 else None):8.1f} ms")


if __name__ == "__main__":
    main()
