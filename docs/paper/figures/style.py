"""Shared look of the paper's figures: one font, one palette, thin marks, recessive axes.

Palette: the validated reference palette of the dataviz method (light surface), slots 1 and 2 for
the two configurations, gray for context. Validated against a white surface with
validate_palette.js (blue #2a78d6 / orange #eb6834: adjacent CVD dE 24.7, normal-vision dE 33.6,
both >= 3:1 contrast). Gray is the de-emphasis ink, not a categorical slot.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P  # noqa: E402

sys.path.append(P.PYLIB)            # private matplotlib (appended: the venv's numpy stays first)
os.environ["MPLCONFIGDIR"] = os.path.join(P.WORK, "mplconfig")   # font cache next to the scripts that run
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
try:
    import matplotlib  # noqa: E402
except ImportError:
    sys.exit(f"style.py: matplotlib not found (the venv has none; the private copy is {P.PYLIB}, or set "
             "PAPER_FIGURES_PYLIB)")

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ink and chrome
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#a8a79f"
SHADE = "#f0efec"
SHADE2 = "#e6e5e0"
WHITE = "#ffffff"

# identity: configurations (same meaning in every figure)
FINAL = "#2a78d6"        # configs/final.py (slot 1, blue)
FINAL_LIGHT = "#9ec5f4"  # blue ramp step 200: raw points of final
V9 = "#eb6834"           # configs/v9.py (slot 2, orange)
V9_LIGHT = "#f6b79c"     # light tint of slot 2 for raw points of v9 (same hue, lighter)
CONTEXT = MUTED          # de-emphasised marks
ACCENT = "#6250d6"       # emphasis colour where no configuration is meant (slot 7, violet)

# sequential ramp (blue, light -> dark) for magnitudes
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

FONT = "Inter"                    # has every glyph used (superscripts, minus, en dash, ≤); outlined in the SVG
FS = 7.0                          # tick labels, annotations (pt)
FS_LABEL = 7.5                    # axis labels
FS_TITLE = 8.0                    # panel titles

COL_W = 3.40                      # one column (in)
DPI = 200                         # PNG resolution
FULL_W = 7.00                     # both columns (in)


def setup():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT, "DejaVu Sans"],
        "font.size": FS,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_TITLE,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 4.0,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.labelsize": FS,
        "ytick.labelsize": FS,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "lines.linewidth": 1.25,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": FS,
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "savefig.dpi": DPI,
        "svg.fonttype": "path",               # glyphs as outlines: the SVG looks the same everywhere
        "svg.hashsalt": "bes-paper",          # stable ids: re-runs give identical SVG for identical data
        "path.simplify": True,
        "axes.formatter.use_mathtext": False,
    })


def grid_y(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def grid_x(ax):
    ax.xaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def save(fig, name):
    """Write FIG/<name>.svg (text outlined) and FIG/<name>.png (DPI dots per inch)."""
    svg = os.path.join(P.FIG_OUT, name + ".svg")
    png = os.path.join(P.FIG_OUT, name + ".png")
    fig.savefig(svg, metadata={"Date": None, "Creator": "paper/figures (matplotlib)"})
    fig.savefig(png, dpi=DPI)
    print("wrote", svg, "and", png)


def thousands(x, _pos=None):
    return f"{x:,.0f}"


def tag_lanes(ax, items, y0=0.975, dy=0.075, min_sep_in=0.19, fontsize=None, line_to=None, linestyle="-",
              dry=False):
    """Labelled tags at data x positions near the top of ax (y in axes coordinates), in as many
    lanes as needed so that two tags in one lane are at least min_sep_in inches apart. items:
    (x, label[, linestyle]). A hairline is drawn from each tag down to line_to (axes y) when given.
    With dry=True nothing is drawn. Returns the number of lanes used (set xlim first)."""
    from matplotlib.transforms import blended_transform_factory
    fig = ax.figure
    fig.canvas.draw()
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    ends = []
    fs = fontsize or FS
    for x, lab, *rest in sorted(items, key=lambda it: it[0]):
        ls = rest[0] if rest else linestyle
        xp = ax.transData.transform((x, 0))[0]
        lane = next((i for i, e in enumerate(ends) if xp - e >= min_sep_in * fig.dpi), len(ends))
        if lane == len(ends):
            ends.append(xp)
        else:
            ends[lane] = xp
        y = y0 - lane * dy
        if dry:
            continue
        if line_to is not None:
            ax.plot([x, x], [line_to, y], transform=tr, color=MUTED, lw=0.5, ls=ls, zorder=1, clip_on=False)
        ax.text(x, y, lab, transform=tr, ha="center", va="center", fontsize=fs - (0.5 if len(lab) > 1 else 0),
                color=INK, zorder=6, clip_on=False,
                bbox=dict(boxstyle="circle,pad=0.16" if len(lab) == 1 else "round,pad=0.16", fc=WHITE, ec=MUTED,
                          lw=0.5))
    return len(ends)
