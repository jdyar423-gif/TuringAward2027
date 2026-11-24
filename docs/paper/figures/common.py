"""Readers of the CSVs in paper/data/ and the smoother of F1 (shared by make_data.py and the figure
scripts)."""
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


def cutoff():
    """The newest record of the run the CSVs use (data/cutoff.csv, written by make_data.py), with a
    label for the figures: 'step 37,650 (24 Nov, 13:05:57 UTC)'."""
    r = read("cutoff.csv")[0]
    d = {k: num(v) for k, v in r.items()}
    d["utc"] = r["utc"]
    d["label"] = f"step {int(r['step']):,} ({int(r['utc'][8:10])} Nov, {r['utc'][11:]} UTC)"
    return d


def small_group(leaves):
    """The name of the ledger row 'small' (data/ledger_leaves.csv), the same in Figures 1 and 7: its
    leaves are norm gains, attention scalars and the sink, lane weights, the candidate tape's and the
    trust features' 64 x 768 projections and the frozen critic head."""
    n = sum(1 for r in leaves if r["row"] == "small")
    return f"{n} small dense leaves (norms, attention scalars, sink, lanes, tape and trust projections, critic)"


TEXT_COLS = {"utc", "end_utc", "restore_utc", "first_step_after_utc", "short", "long", "src", "kind", "group",
             "config_after", "id", "event_utc", "phase"}


def interventions():
    return [{k: (v if k in TEXT_COLS else num(v)) for k, v in r.items()} for r in read("interventions.csv")]


SIGMA_DEC = 0.03     # Gaussian kernel width of the F1 smoother, in decades of tokens
CUT = 3.0            # the kernel is truncated at CUT * SIGMA_DEC on both sides
NEFF_MIN = 15.0      # the smoothed line is drawn where the kernel averages at least this many records


def smooth_centred(xs, ys, sigma=SIGMA_DEC, cut=CUT):
    """Centred Gaussian kernel average of ys over log10(xs), for the records of ONE stretch.

    At record i the window is [lx_i - h_i, lx_i + h_i] with h_i = min(cut * sigma, lx_i - lx_first,
    lx_last - lx_i): it is symmetric about the record and never reaches past the first or last
    record of the stretch (near a restart it shrinks instead of becoming one-sided). Records are
    evenly spaced in steps, hence denser on the right of a window in log10(tokens); each weight is
    therefore the Gaussian times 1/x (the record's share of log-token length), which puts the
    window's centre of mass on the record itself, so the average does not shift the curve.
    Returns (smoothed values, effective number of records in each window, half-width h_i in
    decades, offset of the window's weighted centre from lx_i in decades)."""
    lx = [math.log10(x) for x in xs]
    n = len(lx)
    out, neff, half, off = [], [], [], []
    for i in range(n):
        h = min(cut * sigma, lx[i] - lx[0], lx[-1] - lx[i])
        num_, den, den2, mom = 0.0, 0.0, 0.0, 0.0
        j = i
        while j >= 0 and lx[j] >= lx[i] - h - 1e-12:
            j -= 1
        j += 1
        while j < n and lx[j] <= lx[i] + h + 1e-12:
            w = math.exp(-0.5 * ((lx[j] - lx[i]) / sigma) ** 2) / xs[j]
            num_ += w * ys[j]
            den += w
            den2 += w * w
            mom += w * (lx[j] - lx[i])
            j += 1
        out.append(num_ / den)
        neff.append(den * den / den2)          # effective number of records in the window
        half.append(h)
        off.append(mom / den)
    return out, neff, half, off
