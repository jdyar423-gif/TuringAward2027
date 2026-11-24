"""Readers of the CSVs in paper/data/ and the smoother of F1 (shared by the figure scripts)."""
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P  # noqa: E402


def read(name):
    with open(os.path.join(P.DATA, name)) as f:
        return list(csv.DictReader(f))


def num(x):
    if x in ("", None):
        return None
    try:
        return int(x)
    except ValueError:
        pass
    try:
        return float(x)
    except ValueError:
        return x


def steps():
    rows = read("train_steps.csv")
    out = []
    for r in rows:
        d = {k: num(v) for k, v in r.items()}
        d["config"] = r["config"]
        d["utc"] = r["utc"]
        d["phase"] = r["phase"]
        out.append(d)
    return out


TEXT_COLS = {"utc", "end_utc", "restore_utc", "first_step_after_utc", "short", "long", "src", "kind", "group",
             "config_after", "id"}


def interventions():
    return [{k: (v if k in TEXT_COLS else num(v)) for k, v in r.items()} for r in read("interventions.csv")]


SIGMA_DEC = 0.03     # Gaussian kernel width of the F1 smoother, in decades of tokens


def smooth_log(xs, ys, sigma=SIGMA_DEC, cut=3.0, with_neff=False):
    """Gaussian kernel average of ys over log10(xs) (width sigma decades, truncated at cut*sigma).

    Used per segment (one configuration); points near a segment end average what the segment has
    (one-sided), points with no neighbour within the window keep their own value."""
    lx = [math.log10(x) for x in xs]
    out, neff = [], []
    j0 = 0
    n = len(xs)
    for i in range(n):
        while lx[j0] < lx[i] - cut * sigma:
            j0 += 1
        num_, den, den2 = 0.0, 0.0, 0.0
        j = j0
        while j < n and lx[j] <= lx[i] + cut * sigma:
            w = math.exp(-0.5 * ((lx[j] - lx[i]) / sigma) ** 2)
            num_ += w * ys[j]
            den += w
            den2 += w * w
            j += 1
        out.append(num_ / den)
        neff.append(den * den / den2)          # effective number of points in the kernel
    return (out, neff) if with_neff else out
