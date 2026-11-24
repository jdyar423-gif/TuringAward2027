"""F7: schematic of the model at one token position (configs/final.py and configs/v9.py).

Facts drawn from: docs/SPEC_MODEL_TRAIN.md 4.1-4.3 (forward pass, attention, parameter tree),
docs/V9.md 4-5 (token pages), bes/model/model.py (init_params, block, forward), bes/config.py
(ModelCfg defaults, CAND_SLOTS = 16, REG_FIELDS), and the numbers of paper/data/ledger_leaves.csv and
ledger_final_export.csv (stored / read per token, standard reading of docs/ACCOUNTING.md).
Writes F7_model.svg/png.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as st  # noqa: E402
import common as C  # noqa: E402

st.setup()
plt = st.plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

BLUE_FILL = "#e6f0fd"
ORANGE_FILL = "#fde9e0"
FS = st.FS - 0.5


def main():
    L = C.read("ledger_leaves.csv")
    E = {r["row"]: r for r in C.read("ledger_final_export.csv")}

    def row(name, cfg="v9"):
        rs = [r for r in L if r["row"] == name and r[f"in_{cfg}"] == "1"]
        return sum(int(r["numbers"]) for r in rs), sum(int(r["read_per_step"]) for r in rs)

    core, head, emb, ve, tables, small = (row(k) for k in ("core", "head", "emb", "ve", "tables", "small"))
    gain, page = row("gain"), row("page")
    lib_total = int(E["library"]["total"]) + int(E["index"]["total"])
    lib_read = int(E["library"]["step"]) + int(E["index"]["step"])
    pages_tab = sum(int(r["numbers"]) for r in L if r["leaf"] in ("page/key", "page/val"))
    pages_proj = sum(int(r["numbers"]) for r in L if r["leaf"] in ("page/pin", "page/pout"))
    tot_f = sum(int(r["numbers"]) for r in L if r["in_final"] == "1") + lib_total
    read_f = sum(int(r["read_per_step"]) for r in L if r["in_final"] == "1") + lib_read
    tot_v = sum(int(r["numbers"]) for r in L if r["in_v9"] == "1")
    read_v = sum(int(r["read_per_step"]) for r in L if r["in_v9"] == "1")

    W, H = st.FULL_W, 3.9
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    def box(x, y, w, h, lines, fc=st.WHITE, ec=st.INK2, lw=0.7, bold_first=True, align="left", size=FS,
            pad=0.06, z=2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.05", fc=fc, ec=ec,
                                    lw=lw, zorder=z))
        yy = y + h - pad
        for i, ln in enumerate(lines):
            ax.text(x + (pad if align == "left" else w / 2), yy, ln, ha=align, va="top", fontsize=size,
                    color=st.INK if i == 0 else st.INK2,
                    fontweight="semibold" if (i == 0 and bold_first) else "normal", zorder=z + 1)
            yy -= size / 72 * 1.32

    def arrow(p0, p1, col=st.INK2, lw=0.8, style="-|>", rad=0.0):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=6.5, color=col, lw=lw,
                                     connectionstyle=f"arc3,rad={rad}", zorder=5, shrinkA=0, shrinkB=0))

    ax.text(0.08, H - 0.07, "One token position of the model: stored numbers / numbers read per token (standard reading)",
            ha="left", va="top", fontsize=st.FS_TITLE, fontweight="semibold")

    # ---------------------------------------------------------------- inputs (left column)
    xL, wL = 0.08, 1.80
    box(xL, 3.12, wL, 0.42, ["value embedding Eᵥ[x, r]", f"8 × 128 per token: {ve[0]:,} / {ve[1]:,}"])
    box(xL, 2.62, wL, 0.42, ["token x", f"input embedding E[x]: {emb[0]:,} / {emb[1]:,}"])
    box(xL, 2.04, wL, 0.50, ["registers", "15 fields of the exact layer (role,", "turn, line, sentence, code fence, …)"])
    box(xL, 1.32, wL, 0.64, ["candidate tape, input side", "16 candidate next tokens from exact",
                             "n-gram heads: token row + 6 feature", "rows, projected 64 → 768"])
    ax.text(xL, 1.24, f"register, candidate and trust tables:\n{tables[0]:,} / {tables[1]:,}", ha="left", va="top",
            fontsize=FS, color=st.INK2, linespacing=1.15)
    box(xL, 0.42, wL, 0.42, ["sum of the three input rows, RMS norm", "copied into 2 lanes (hyper-connections)"])
    xb = 0.035
    ax.plot([xb, xb], [2.83, 0.63], color=st.INK2, lw=0.7, zorder=4)
    for yy in (2.83, 2.29, 1.64):
        ax.plot([xb, xL], [yy, yy], color=st.INK2, lw=0.7, zorder=4)
    arrow((xb, 0.63), (xL, 0.63))

    # ---------------------------------------------------------------- looped core (middle)
    xC, wC = 2.12, 2.74
    ax.add_patch(FancyBboxPatch((xC, 0.40), wC, 3.14, boxstyle="round,pad=0.0,rounding_size=0.07", fc="#f7f7f5",
                                ec=st.INK2, lw=0.8, zorder=1))
    ax.text(xC + 0.08, 3.48, "shared looped core: 2 blocks applied 8 times (r = 1 … 8)", ha="left", va="top",
            fontsize=FS, fontweight="semibold")
    ax.text(xC + 0.08, 3.34, f"{core[0]:,} / all (norms, lanes, sink, critic: {small[0]:,} / all)", ha="left",
            va="top", fontsize=FS, color=st.INK2)
    ax.text(xC + 0.08, 3.17, "u = Σₙ αᵣₙ · laneₙ", ha="left", va="top", fontsize=FS, color=st.INK)
    bx, bw = xC + 0.08, wC - 0.36

    def block(y, title):
        box(bx, y, bw, 0.94, [title,
                              "attention: 6 heads × 128 (126 rotary, normalised;",
                              "2 dims of exact key bias); learned sink at BOS",
                              "feed-forward 768 → 3,072 → 768, ReLU²"], fc=st.WHITE)
        ax.add_patch(FancyBboxPatch((bx + 0.05, y + 0.05), bw - 0.10, 0.28,
                                    boxstyle="round,pad=0.0,rounding_size=0.04", fc=ORANGE_FILL, ec=st.V9, lw=0.7,
                                    zorder=3))
        ax.text(bx + 0.09, y + 0.30, "v9: attention output × (1 + gain of x) before Wₒ;",
                ha="left", va="top", fontsize=FS - 0.4, color=st.INK, zorder=4)
        ax.text(bx + 0.09, y + 0.18, "+ 8 private ReLU² neurons of x in 192 dims",
                ha="left", va="top", fontsize=FS - 0.4, color=st.INK, zorder=4)

    block(2.08, "block 1 (+ value embedding of loop r)")
    block(1.04, "block 2")
    arrow((bx + bw / 2, 2.08), (bx + bw / 2, 1.99))
    ax.text(xC + 0.08, 0.94, "lanes ← mixᵣ · lanes + βᵣ · RMSnorm(u) · gᵣ", ha="left", va="top", fontsize=FS,
            color=st.INK)
    ax.text(xC + 0.08, 0.76, "after loop 8: h = RMSnorm(Σ lanes) · g", ha="left", va="top", fontsize=FS,
            color=st.INK2)
    # repeat arrow (loop r -> r + 1) to the right of the blocks
    xa = bx + bw + 0.10
    arrow((xa, 1.10), (xa, 3.02), col=st.INK2, lw=0.7)
    ax.text(xa + 0.06, 2.05, "next loop", rotation=90, ha="left", va="center", fontsize=FS, color=st.INK2)
    # inputs into the core
    arrow((xL + wL + 0.02, 0.63), (xC + 0.06, 0.63))
    arrow((xL + wL + 0.02, 3.25), (bx - 0.01, 2.72), rad=0.15)

    # ---------------------------------------------------------------- head and output law (right)
    xR, wR = 5.00, 1.93
    xg = xC + wC + 0.07
    ax.plot([xC + wC, xg, xg], [0.70, 0.70, 3.20], color=st.INK2, lw=0.8, zorder=5)
    arrow((xg, 3.20), (xR, 3.20))
    box(xR, 2.80, wR, 0.74, ["output head", "z(v) = 15 tanh(h P · Eₒᵤₜ[v] / 15)", "q = softmax(z)",
                             f"P 768 × 512, Eₒᵤₜ 8,192 × 512: {head[0]:,} / all"], pad=0.05)
    arrow((xR + wR / 2, 2.80), (xR + wR / 2, 2.74))
    box(xR, 1.98, wR, 0.76, ["output law with the candidate tape", "p(v) = (q(v) + Σₖ mₖ [cₖ = v]) / (1 + Σₖ mₖ)",
                             "16 candidates cₖ, masses mₖ from trust", "features; exactly q when all mₖ = 0"])
    box(xR, 1.10, wR, 0.80, ["final only: written library", "Simple English Wikipedia passages and",
                             "question-answer pairs behind a gate;", "a retrieved passage enters the context",
                             f"{lib_total:,} / ≤ {lib_read:,} per retrieval"], fc=BLUE_FILL, ec=st.FINAL)
    box(xR, 0.42, wR, 0.60, ["v9 only: token pages (no library)", f"gains {gain[0]:,} / {gain[1]:,}",
                             f"neurons {page[0]:,} / {page[1]:,}"], fc=ORANGE_FILL, ec=st.V9)

    ax.text(0.08, 0.30, f"Totals: final {tot_f:,} stored, {read_f:,} read per token (with the worst-case retrieval);"
            f" v9 {tot_v:,} stored, {read_v:,} read per token.", ha="left", va="top", fontsize=FS, color=st.INK)
    ax.text(0.08, 0.14, "Default decoding (both): 6 particles steered by exact potentials, one chosen by a satisficing "
            "rule; the setting used in the evaluation is chosen after training [TBD].", ha="left", va="top",
            fontsize=FS, color=st.INK2)
    st.save(fig, "F7_model")
    print("core", core, "head", head, "emb", emb, "ve", ve, "tables", tables, "small", small, "gain", gain,
          "page", page, "pages tables", pages_tab, "proj", pages_proj, "lib", lib_total, lib_read)
    print("final", tot_f, read_f, "v9", tot_v, read_v)


if __name__ == "__main__":
    main()
