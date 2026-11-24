"""F5: how assistant turns open in the tokenised sources, and what the two loader rules remove.

Data: paper/data/openings_scan.csv (openings.json: per source, share of assistant turns whose
opening matches each pattern of the scan script), deflection_check2.csv (deflection_check2.json:
conversations the loader's refusal and deflection functions skip, 30,000 sampled conversations per
chat source), loader_drops_live.csv (what the live loader skipped after the rules went live).
Writes F5_openings.svg/png.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as st  # noqa: E402
import common as C  # noqa: E402

st.setup()
plt = st.plt
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PATTERNS = [  # scan key, column label (the scan's regular expressions: openings_scan.py)
    ("canned_refusal_now", "stock refusal"),
    ("cannot_any", "I can't …"),
    ("apology_open", "Sorry …"),
    ("as_ai_open", "As an AI …"),
    ("i_am_ai", "I'm an AI …"),
    ("dont_have", "I don't have …"),
    ("dont_have_access", "… access, real-time"),
    ("need_more", "Please provide …"),
    ("sure_open", "Sure … (control)"),
]
KIND_LABEL = {"chat": "chat", "rag": "retrieval", "math": "maths", "prefs": "preference\npairs"}


def main():
    op = [r for r in C.read("openings_scan.csv") if int(r["turns"]) > 0]
    order_k = ["chat", "rag", "math", "prefs"]
    op.sort(key=lambda r: (order_k.index(r["kind"]) if r["kind"] in order_k else 9, r["source"]))
    dc = C.read("deflection_check2.csv")
    live = C.read("loader_drops_live.csv")

    fig = plt.figure(figsize=(st.FULL_W, 3.65))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], left=0.16, right=0.985, top=0.80, bottom=0.12,
                          wspace=0.66)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # ------------------------------------------------------------------ (a) heatmap
    bounds = [0, 0.05, 0.5, 1, 2, 5, 10, 20, 50]
    cols = ["#ffffff", st.BLUE_RAMP[0], st.BLUE_RAMP[2], st.BLUE_RAMP[4], st.BLUE_RAMP[6], st.BLUE_RAMP[8],
            st.BLUE_RAMP[10], st.BLUE_RAMP[12]]
    cmap = ListedColormap(cols)
    norm = BoundaryNorm(bounds, cmap.N)
    import numpy as np
    M = np.array([[float(r[f"{k}_pct_turns"]) for k, _ in PATTERNS] for r in op])
    axA.imshow(M, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if v < 0.05:
                txt = "0" if v == 0 else "<0.1"
            elif v < 10:
                txt = f"{v:.1f}"
            else:
                txt = f"{v:.0f}"
            dark = v >= 5
            axA.text(j, i, txt, ha="center", va="center", fontsize=st.FS - 1.0,
                     color=st.WHITE if dark else (st.MUTED if v < 0.05 else st.INK))
    axA.set_xticks(range(len(PATTERNS)))
    axA.set_xticklabels([lab for _, lab in PATTERNS], rotation=38, ha="left", rotation_mode="anchor",
                        fontsize=st.FS - 0.5)
    axA.xaxis.tick_top()
    axA.tick_params(axis="x", length=0, pad=2)
    axA.set_yticks(range(len(op)))
    axA.set_yticklabels([f"{r['source']}" for r in op], fontsize=st.FS - 0.5)
    axA.tick_params(axis="y", length=0)
    for sp in axA.spines.values():
        sp.set_visible(False)
    # kind brackets on the left
    prev, start = None, 0
    for i, r in enumerate(op + [dict(kind=None)]):
        if r["kind"] != prev:
            if prev is not None:
                axA.annotate("", xy=(-0.5, start - 0.45), xytext=(-0.5, i - 0.55), xycoords="data",
                             annotation_clip=False)
                axA.plot([len(PATTERNS) - 0.5 + 0.12] * 2, [start - 0.45, i - 0.55], color=st.INK2, lw=0.7,
                         clip_on=False)
                axA.text(len(PATTERNS) - 0.5 + 0.28, (start + i - 1) / 2, KIND_LABEL.get(prev, prev),
                         ha="left", va="center", fontsize=st.FS - 0.5, color=st.INK2, linespacing=1.05)
            prev, start = r["kind"], i
    for i in range(1, len(op)):
        if op[i]["kind"] != op[i - 1]["kind"]:
            axA.axhline(i - 0.5, color=st.WHITE, lw=2.0)
    fig.text(0.005, 0.975, "a   Share of assistant turns that open with each pattern (%)", fontsize=st.FS_TITLE,
             fontweight="semibold", ha="left", va="top", color=st.INK)
    # colour key
    handles = [Patch(fc=c, ec=st.GRID if c == "#ffffff" else c, lw=0.5) for c in cols]
    labs = ["< 0.05", "0.05–0.5", "0.5–1", "1–2", "2–5", "5–10", "10–20", "≥ 20"]
    axA.legend(handles, labs, loc="upper left", bbox_to_anchor=(-0.2, -0.015), ncol=8, frameon=False,
               fontsize=st.FS - 1.0, handlelength=1.0, handletextpad=0.3, columnspacing=0.7)

    # ------------------------------------------------------------------ (b) what the rules remove
    dc.sort(key=lambda r: float(r["refusal_pct"]) + float(r["deflection_pct"]))
    n = len(dc)
    lv = {}
    for r in live:
        if r["period"].startswith("refusal + deflection"):
            a = lv.setdefault(r["source"], [0, 0, 0])
            a[0] += int(r["reads"])
            a[1] += int(r["refusal"])
            a[2] += int(r["deflection"])
    for i, r in enumerate(dc):
        ref, dfl = float(r["refusal_pct"]), float(r["deflection_pct"])
        axB.barh(i, ref, height=0.6, color=st.MUTED, lw=0, zorder=2)
        axB.barh(i, dfl, left=ref, height=0.6, color=st.ACCENT, lw=0, zorder=2)
        tot = ref + dfl
        axB.text(tot + 0.6, i, f"{tot:.1f}" if tot >= 0.05 else "0", va="center", ha="left",
                 fontsize=st.FS - 0.5, color=st.INK)
        if r["source"] in lv and lv[r["source"]][0] > 0:
            rd, rf, df = lv[r["source"]]
            x = 100 * (rf + df) / rd
            axB.plot([x, x], [i - 0.38, i + 0.38], color=st.INK, lw=1.1, zorder=4, solid_capstyle="butt")
    axB.set_yticks(range(n))
    axB.set_yticklabels([r["source"] for r in dc], fontsize=st.FS - 0.5)
    axB.tick_params(axis="y", length=0)
    axB.set_ylim(-0.6, n - 0.4)
    axB.set_xlim(0, 36)
    st.grid_x(axB)
    axB.set_xlabel("Conversations left out (%)")
    fig.text(0.60, 0.975, "b   Chat conversations the loader leaves out (%)", fontsize=st.FS_TITLE,
             fontweight="semibold", ha="left", va="top", color=st.INK)
    handles = [Patch(fc=st.MUTED, ec="none"), Patch(fc=st.ACCENT, ec="none"),
               Line2D([], [], color=st.INK, lw=1.1)]
    labels = ["stock-refusal rule (from 02:31)", "deflection rule (from 08:36)",
              "skipped by the live loader since 08:36"]
    axB.legend(handles, labels, loc="lower left", bbox_to_anchor=(-0.55, 1.0), ncol=1, frameon=False,
               fontsize=st.FS - 0.5, handlelength=1.3, handletextpad=0.4, labelspacing=0.3)
    st.save(fig, "F5_openings")
    for r in dc:
        rd = lv.get(r["source"])
        live_s = f"live {100 * (rd[1] + rd[2]) / rd[0]:.2f}% of {rd[0]} reads" if rd and rd[0] else "live -"
        print(f"{r['source']:24s} scan refusal {float(r['refusal_pct']):5.2f} + deflection {float(r['deflection_pct']):5.2f}"
              f" = {float(r['refusal_pct']) + float(r['deflection_pct']):5.2f}%  tokens {float(r['token_share_refusal_pct']):.2f}"
              f" + {float(r['token_share_deflection_pct']):.2f}%   {live_s}")


if __name__ == "__main__":
    main()
