"""F4: cost of one v9 training step against final's, by table-gradient mode and piece size.

Data: paper/data/chip_whole_steps.csv (make_data.py: wf/chip_kernels_0745.txt for final,
v9_measA/B/C/D for v9; column "timing": each bar is one timing, the mean of a few whole steps run
back to back, so no spread exists) and throughput_stages.csv (live rates with n: 262,144 / live rate
= the run's own step time; the windows of the paper's Table 3, and the 2,048-token stretch to the
cut-off of cutoff.csv). One accumulation step = 262,144 tokens, synthetic batch, kernel
onetile:save + relu2 (final's whole step was timed with the ce switch on as well, see FIGURES.md).
Writes F4_v9cost.svg/png.
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

    stages = {r["key"]: r for r in C.read("throughput_stages.csv")}
    live, live_n = {}, {}
    for key, stage in (("final", "final_25300_29700"), ("v9", "v9_take"), ("v9 switches", "v9_switches"),
                       ("v9 switches 2048", "v9_2048")):
        if stage in stages and stages[stage]["live_median"]:
            live[key] = P.TOKENS_PER_STEP / int(stages[stage]["live_median"]) * 1e3
            live_n[key] = int(stages[stage]["live_n"])
    groups = [
        (1024, [("final", "take", 64, 4), ("final", "take", 128, 2), ("v9", "take", 64, 4), ("v9", "take", 32, 8),
                ("v9", "onehot", 64, 4), ("v9", "sorted", 64, 4),
                ("v9", "take", 64, 4, "onetile:save+relu2+flat"), ("v9", "take", 64, 4, "onetile:save+relu2+flat+rows"),
                ("v9", "take", 64, 4, "onetile:save+relu2+flat+rows+tokn")]),
        (2048, [("final", "take", 64, 2), ("final", "take", 128, 1), ("v9", "take", 64, 2), ("v9", "take", 32, 4),
                ("v9", "sorted", 64, 2)] + ([("v9", "live only", 64, 2, "live")] if "v9 switches 2048" in live else [])),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(st.COL_W, 4.1), sharex=True,
                             gridspec_kw=dict(height_ratios=[9, 5], left=0.335, right=0.985, top=0.955,
                                              bottom=0.115, hspace=0.25))
    for ax, (t, rows) in zip(axes, groups):
        n = len(rows)
        base = ms(*(rows[0][:2]), rows[0][2], rows[0][3], t)
        labels = []
        for i, spec in enumerate(rows):
            cfg, mode, a, r = spec[:4]
            kern = spec[4] if len(spec) > 4 else None
            y = n - 1 - i
            if kern == "live":                   # no synthetic timing exists: the live tick alone
                lv = live["v9 switches 2048"]
                ax.plot([lv, lv], [y - 0.36, y + 0.36], color=st.INK, lw=1.0, zorder=4, solid_capstyle="butt")
                ax.text(lv + 30, y, f"live {lv:,.0f} (n = {live_n['v9 switches 2048']})",
                        va="center", ha="left", fontsize=st.FS - 0.5, color=st.INK)
                labels.append("v9 + switches, live only")
                continue
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
            nlv = {live.get("final"): live_n.get("final"), live.get("v9"): live_n.get("v9"),
                   live.get("v9 switches"): live_n.get("v9 switches")}.get(lv) if lv else None
            txt = f"{v:,.1f}" + ("" if i == 0 else f" (×{ratio:.2f})") + (f"\nlive {lv:,.0f} (n = {nlv})" if lv else "")
            ax.text(max(v, lv or 0) + 30, y, txt, va="center", ha="left", fontsize=st.FS - 0.5, color=st.INK,
                    linespacing=1.0)
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
    axes[0].set_xlim(0, 2900)
    axes[0].set_xticks([0, 500, 1000, 1500, 2000])
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(st.thousands))
    axes[1].set_xlabel("Milliseconds per step of 262,144 tokens")
    st.save(fig, "F4_v9cost")
    print("live median step (ms):", {k: round(v, 1) for k, v in live.items()})
    for t, rows in groups:
        for spec in rows:
            if len(spec) > 4 and spec[4] == "live":
                continue
            print(f"T={t}", spec, f"{ms(*spec[:4], t, spec[4] if len(spec) > 4 else None):8.1f} ms")


if __name__ == "__main__":
    main()
