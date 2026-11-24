"""F6: the graft of run main16 from configs/final.py to configs/v9.py at step 29,722 (schematic,
drawn to scale in stored numbers).

Data: paper/data/ledger_leaves.csv (both checkpoints' manifests), ledger_final_export.csv (library
and index of the final export), run_events.csv, interventions.csv, graft_facts.csv (the graft tool's
checks of Table 6 with their batch shape, the gradient check of Section 6.4, the loss guard, the
loader's read cap before and after the graft, the data seed and the speed switches the run started
with after the graft; all written by make_data.py from the files named in FIGURES.md). No number is
typed here. Writes F6_graft.svg/png.

Colours as in Figure 1 (F7): grey = shared by both configurations, blue = final only (its written
library and index), orange = v9 only (its new leaves).
"""
import json
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as st  # noqa: E402
import common as C  # noqa: E402

st.setup()
plt = st.plt
from matplotlib.patches import Rectangle, FancyArrowPatch  # noqa: E402

ROW_LABEL = {"core": "looped core (2 blocks)", "head": "output head", "emb": "input embedding",
             "ve": "per-loop value embeddings", "tables": "register, candidate, trust tables"}
SHARED = "#c9c8c1"        # grey: shared by both configurations (Figure 1's meaning), L* 80 against blue 50
NEW_LABEL = {"gain/o": ("gain/o", "zeros"),
             "page/key": ("page/key", "normal"),
             "page/val": ("page/val", "zeros"),
             "page/pin+pout": ("page/pin, pout", "uniform")}


def main():
    L = C.read("ledger_leaves.csv")
    E = {r["row"]: r for r in C.read("ledger_final_export.csv")}
    S = C.steps()
    ev = C.read("run_events.csv")
    G = C.read("graft_facts.csv")
    checks = sorted((r for r in G if r["kind"] == "check"), key=lambda r: int(r["step"]))
    grads = [r for r in G if r["kind"] == "gradient"]
    guard = next(r for r in G if r["kind"] == "guard")
    man = next(r for r in G if r["kind"] == "manifest")
    loader = next((r for r in G if r["kind"] == "loader"), None)
    shapes = {r["check_tokens"] for r in checks}
    assert len(shapes) == 1, shapes                  # the three checks ran on the same synthetic batch shape
    shape = shapes.pop().replace(" x ", " × ")
    diffs = {r["difference"] for r in checks}
    assert len(diffs) == 1, diffs                    # one difference for all three checks (0.0)
    diff = diffs.pop()
    iv8 = next(r for r in C.interventions() if str(r["id"]) == "8")
    iv7 = next(r for r in C.interventions() if str(r["id"]) == "7")
    setup9 = [e for e in ev if e["event"] == "setup" and e["config"] == "v9"][0]
    split = json.loads(setup9["micro_split"])            # {"1024": 4, "2048": 4} at the graft
    assert len(set(split.values())) == 1, split
    pieces = next(iter(split.values()))
    shared = {}
    for r in L:
        if r["in_final"] == "1" and r["in_v9"] == "1":
            shared[r["row"]] = shared.get(r["row"], 0) + int(r["numbers"])
    new = {}
    for r in L:
        if r["in_v9"] == "1" and r["in_final"] == "0":
            k = "page/pin+pout" if r["leaf"] in ("page/pin", "page/pout") else r["leaf"]
            new[k] = new.get(k, 0) + int(r["numbers"])
    lib = int(E["library"]["total"]) + int(E["index"]["total"])
    n_shared = sum(shared.values())
    n_new = sum(new.values())
    n_leaves_shared = sum(1 for r in L if r["in_final"] == "1" and r["in_v9"] == "1")
    n_leaves_v9 = sum(1 for r in L if r["in_v9"] == "1")
    new_leaves = [r["leaf"] for r in L if r["in_v9"] == "1" and r["in_final"] == "0"]
    adam = sum(1 for r in L if r["in_final"] == "1" and r["optimiser_slots"] == "m+v")
    muon = sum(1 for r in L if r["in_final"] == "1" and r["optimiser_slots"] == "mom+norm")
    frozen = sum(1 for r in L if r["in_final"] == "1" and not r["optimiser_slots"])

    fig = plt.figure(figsize=(st.FULL_W, 3.55))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 7.0)
    ax.set_ylim(0, 3.55)
    ax.axis("off")
    H = 2.55 / 1.0e8          # inches per stored number (bars drawn to scale)
    y0 = 0.42
    bw = 0.40
    xf, xv = 1.85, 3.05

    def seg(x, y, n, col, hatch=None):
        h = n * H
        ax.add_patch(Rectangle((x, y), bw, h, fc=col, ec=st.WHITE, lw=0.8, hatch=hatch, zorder=2))
        return y + h

    # shared part, same order in both bars
    order = ["core", "head", "emb", "ve", "tables", "small"]
    yf = yv = y0
    marks = {}
    for k in order:
        n = shared[k]
        marks[k] = (yf, yf + n * H)
        yf = seg(xf, yf, n, SHARED)
        yv = seg(xv, yv, n, SHARED)
    y_shared_top = yf
    yl = seg(xf, yf, lib, st.FINAL)
    ynew = {}
    for k in ["gain/o", "page/key", "page/val", "page/pin+pout"]:
        ynew[k] = (yv, yv + new[k] * H)
        yv = seg(xv, yv, new[k], st.V9)
    ax.text(xf + bw / 2, yl + 0.06, "final", ha="center", va="bottom", fontsize=st.FS_TITLE, fontweight="semibold")
    ax.text(xv + bw / 2, yv + 0.06, "v9", ha="center", va="bottom", fontsize=st.FS_TITLE, fontweight="semibold")
    ax.text(xf + bw / 2, y0 - 0.06, f"{n_shared + lib:,}\nstored numbers", ha="center", va="top", fontsize=st.FS - 0.5,
            color=st.INK2, linespacing=1.1)
    ax.text(xv + bw / 2, y0 - 0.06, f"{n_shared + n_new:,}\nstored numbers", ha="center", va="top",
            fontsize=st.FS - 0.5, color=st.INK2, linespacing=1.1)

    # labels of the shared groups (left of the final bar)
    lab_y = {"core": None, "head": None, "emb": None, "ve": None}
    for k in order:
        a, b = marks[k]
        if k in ("tables", "small"):
            continue
        ax.text(xf - 0.08, (a + b) / 2, f"{ROW_LABEL[k]}  {shared[k]:,}", ha="right", va="center",
                fontsize=st.FS - 0.5, color=st.INK)
    a = marks["tables"][0]
    b = marks["small"][1]
    small_lab = textwrap.fill(f"tables {shared['tables']:,}; {C.small_group(L)}: {shared['small']:,}", 33)
    ax.annotate(small_lab,
                xy=(xf - 0.01, (a + b) / 2), xytext=(xf - 0.08, (a + b) / 2 + 0.17), ha="right", va="bottom",
                fontsize=st.FS - 0.5, color=st.INK, linespacing=1.1,
                arrowprops=dict(arrowstyle="-", color=st.INK2, lw=0.5, shrinkA=0, shrinkB=0))
    ax.text(xf - 0.08, y_shared_top + 0.76 * (yl - y_shared_top),     # above the shared-leaves label below it
            f"final only: written library\nand retrieval index {lib:,}\n(not in v9)", ha="right", va="center",
            fontsize=st.FS - 0.5, color=st.INK, linespacing=1.1)
    # the copy: bracket between the bars over the shared part
    for yy in (y0, y_shared_top):
        ax.plot([xf + bw, xv], [yy, yy], color=st.INK2, lw=0.5, zorder=1)
    ax.add_patch(FancyArrowPatch((xf + bw + 0.03, (y0 + y_shared_top) / 2), (xv - 0.03, (y0 + y_shared_top) / 2),
                                 arrowstyle="-|>", mutation_scale=7, color=st.INK, lw=0.8, zorder=3))
    ax.text((xf + bw + xv) / 2, (y0 + y_shared_top) / 2 + 0.05, f"{n_leaves_shared} leaves\ncopied",
            ha="center", va="bottom", fontsize=st.FS - 0.5, color=st.INK, linespacing=1.1)
    ax.text((xf + bw + xv) / 2, (y0 + y_shared_top) / 2 - 0.05, f"{n_shared:,}", ha="center", va="top",
            fontsize=st.FS - 0.5, color=st.INK)
    # labels of the new leaves (right of the v9 bar)
    for k, (a, b) in ynew.items():
        name, init = NEW_LABEL[k]
        yy = (a + b) / 2 if k != "page/pin+pout" else b + 0.05
        ax.text(xv + bw + 0.08, yy, f"{name}  {new[k]:,}  ({init})", ha="left", va="center",
                fontsize=st.FS - 0.5, color=st.INK)
    ax.text(xv + bw + 0.08, (y0 + y_shared_top) / 2,
            f"v9 only: {len(new_leaves)} new leaves, {n_new:,}\nnumbers, at their initialisation\nwith fresh "
            "optimiser slots; the\nvalues and gains are zero, so\nthe function is unchanged", ha="left", va="center",
            fontsize=st.FS - 0.5, color=st.INK2, linespacing=1.12)

    # what else continues, what is new (text column on the right)
    restore = next(e for e in ev if e["event"] == "restore" and e["config"] == "v9")
    stage9 = next(e for e in ev if e["event"] == "stage" and e["config"] == "v9")
    end = [e for e in ev if e["event"] == "end" and e["step"] == str(man["step"])][-1]
    first_v9 = next(s for s in S if s["config"] == "v9")
    last_f = [s for s in S if s["config"] == "final"][-1]
    x0 = 5.10
    n_new_rows = sum(int(g["rows"]) for g in grads)
    blocks = [
        ("Carried with the parameters", [
            f"optimiser slots (Adam m, v: {adam} leaves;",
            f"MuonH momentum, norm: {muon}; frozen: {frozen}),",
            f"step {int(man['step']):,}, update count, time ledger,",
            f"loss guard (average {float(guard['guard_ema']):.3f}), deadline",
            f"{setup9['deadline_utc'][11:]}, training end {setup9['train_end_utc'][11:]},",
            f"{pieces} pieces per micro-batch and the kernel",
            f"switches {loader['fast_at_graft'].replace(' ce=0', '')}",
            f"(the v9 switches came at step {int(iv8['step']):,})",
        ]),
        ("Schedule p = T / (T + R)", [
            f"{float(end['p']):.4f} at the stop ({end['utc'][11:]}), "
            f"{float(stage9['p']):.4f}",
            f"at the restore ({restore['utc'][11:]}): T stood still",
            "while R shrank",
        ]),
        ("New", [
            f"a fresh loader journal (seed {man['seed']}), no",
            f"library rows; told the {int(man['tokens']) / 1e9:.2f} × 10⁹ tokens",
            "before the graft, a source may be read",
            f"{float(loader['reads_cap']):.2f} more times ({loader['reads_cap_before']} × (1 − p), "
            f"p = {float(man['p']):.4f})",
        ]),
        ("Checks (Table 6, Section 6.4)", [
            f"losses on {shape} synthetic tokens",
            f"equal (difference {diff}) at steps",
            ", ".join(f"{int(c['step']):,}" for c in checks[:-1]) + f" and {int(checks[-1]['step']):,} (a re-run",
            "of the same checkpoint, live hash);",
            f"on {n_new_rows} held-out rows the gradients of",
            f"all {grads[0]['shared_bit_identical'].split(' of ')[1]} shared leaves bit-identical",
            "(CPU, float32, dense attention)",
        ]),
    ]
    yy = 3.42
    for title, lines in blocks:
        ax.text(x0, yy, title, ha="left", va="top", fontsize=st.FS - 0.5, fontweight="semibold", color=st.INK)
        yy -= 0.143
        for ln in lines:
            ax.text(x0, yy, ln, ha="left", va="top", fontsize=st.FS - 0.5, color=st.INK2)
            yy -= 0.122
        yy -= 0.06
    ax.text(0.05, 3.48, f"Graft at step {int(man['step']):,} (24 Nov, {iv7['end_utc'][11:]} → "
                        f"{iv7['restore_utc'][11:]} UTC); bars to scale",
            ha="left", va="top", fontsize=st.FS_TITLE, fontweight="semibold", color=st.INK)
    st.save(fig, "F6_graft")
    print("shared", n_shared, n_leaves_shared, "new", n_new, "library", lib, "v9 leaves", n_leaves_v9,
          "adam", adam, "muon", muon, "frozen", frozen)
    print("end", end["utc"], end["p"], "restore", restore["utc"], stage9["p"], "first v9", first_v9["step"],
          first_v9["utc"], first_v9["p"], first_v9["loss"], first_v9["tok_per_s"], "last final", last_f["step"], last_f["p"])


if __name__ == "__main__":
    main()
