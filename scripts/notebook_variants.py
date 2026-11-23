"""Notebooks derived from ``notebooks/colab_v6e_24h.ipynb``.

``notebooks/colab_v6e_24h_v9.ipynb`` is the 24-hour notebook with the parameters of
``configs/v9.py`` in cell 1 (configuration, run directory, Drive directory), so that its
"open in Colab" link starts that configuration without an edit. It is written by this script and
never edited by hand: after a change of the 24-hour notebook run

    python scripts/notebook_variants.py

and commit both files (``tests/train/test_export.py`` fails while they differ).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "notebooks" / "colab_v6e_24h.ipynb"
V9 = REPO / "notebooks" / "colab_v6e_24h_v9.ipynb"

#: (text of the 24-hour notebook, text of the v9 notebook): each occurs exactly once in one cell
V9_SUBSTITUTIONS: tuple[tuple[str, str], ...] = (
    ("/notebooks/colab_v6e_24h.ipynb\"", "/notebooks/colab_v6e_24h_v9.ipynb\""),
    ("# BES-100M-A20M: the 24-hour run on one TPU v6e\n",
     "# BES-100M-A20M: the 24-hour run on one TPU v6e, configuration v9\n"),
    ("This notebook runs `scripts/run_24h.py` with `configs/final.py` on a Colab **TPU v6e-1** runtime:\n"
     "downloads, tokenisation, the library, the start-up benchmark and canary, about 19 hours of\n",
     "This notebook runs `scripts/run_24h.py` with `configs/v9.py` (a page of private neurons and\n"
     "output gains per vocabulary id instead of the written library: `docs/V9.md`, `RUNBOOK.md`\n"
     "section 1.4) on a Colab **TPU v6e-1** runtime:\n"
     "downloads, tokenisation, the start-up benchmark and canary, about 19 hours of\n"),
    ("RUN_DIR = \"/content/runs/final\"    # on the local disk",
     "RUN_DIR = \"/content/runs/v9\"       # on the local disk"),
    ("DRIVE_DIR = \"/content/drive/MyDrive/bes_runs/final\"\n",
     "DRIVE_DIR = \"/content/drive/MyDrive/bes_runs/v9\"\n"),
    ("CONFIG = \"configs/final.py\"\n", "CONFIG = \"configs/v9.py\"\n"),
)


def derive(base_text: str, substitutions=V9_SUBSTITUTIONS) -> str:
    """The text of a derived notebook: every substitution applied to the one cell that holds its
    left side (cell sources joined; the file format of the 24-hour notebook is kept)."""
    nb = json.loads(base_text)
    sources = ["".join(c["source"]) for c in nb["cells"]]
    for old, new in substitutions:
        hits = [i for i, s in enumerate(sources) if old in s]
        if len(hits) != 1 or sources[hits[0]].count(old) != 1:
            raise ValueError(f"expected exactly one occurrence of {old!r}, found it in cells {hits}")
        sources[hits[0]] = sources[hits[0]].replace(old, new)
    for c, s in zip(nb["cells"], sources):
        c["source"] = s.splitlines(keepends=True)
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    text = derive(BASE.read_text())
    changed = not V9.exists() or V9.read_text() != text
    if changed:
        V9.write_text(text)
    print(f"{V9.relative_to(REPO)}: {'written' if changed else 'up to date'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
