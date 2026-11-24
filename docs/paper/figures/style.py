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
os.environ["MPLCONFIGDIR"] = os.path.join(P.FIG, "_work", "mplconfig")   # font cache stays under paper/
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import matplotlib  # noqa: E402

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
        "savefig.dpi": 300,
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
    """Write FIG/<name>.svg (text outlined) and FIG/<name>.png (300 dpi)."""
    svg = os.path.join(P.FIG, name + ".svg")
    png = os.path.join(P.FIG, name + ".png")
    fig.savefig(svg, metadata={"Date": None, "Creator": "paper/figures (matplotlib)"})
    fig.savefig(png, dpi=300)
    print("wrote", svg, "and", png)


def thousands(x, _pos=None):
    return f"{x:,.0f}"
