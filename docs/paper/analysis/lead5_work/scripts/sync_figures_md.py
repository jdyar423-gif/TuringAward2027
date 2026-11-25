"""Fifth pass (copy of lead4_work/scripts/sync_figures_md.py; final captions). Fourth pass (REVIEW_2 items 10 and 11): bring FIGURES.md to the third draft's figure numbers and copy
every caption verbatim from paper.md, so that the two files hold one caption per figure.
For each figure file Fx, the heading "## Figure <old> (`Fx`)..." gets the paper's number, and the
"**Caption** (...)" paragraph (up to the next blank line) is replaced by the paper's caption.
Arguments: paper.md, FIGURES.md (rewritten in place). Prints what it changed.
"""
import re
import sys
from pathlib import Path

paper, figs = Path(sys.argv[1]), Path(sys.argv[2])
P = paper.read_text().split("\n")
# the paper's figure numbers by file, and its captions
num, cap = {}, {}
for i, line in enumerate(P):
    m = re.match(r"!\[Figure ([0-9A-Z]+)\]\(figures/(F\d+)_[a-z0-9]+\.(svg|png)\)", line)
    if m:
        num[m.group(2)] = m.group(1)
for i, line in enumerate(P):
    m = re.match(r"\*\*Figure ([0-9A-Z]+)\.\*\* (.*)", line)
    if m:
        out = [m.group(2)]
        j = i + 1
        while j < len(P) and P[j].strip():
            out.append(P[j])
            j += 1
        cap[m.group(1)] = "\n".join(out)
F = figs.read_text().split("\n")
res, i, changed = [], 0, []
cur = None
while i < len(F):
    line = F[i]
    m = re.match(r"## Figure (.+?) \(`(F\d+)_[a-z0-9]+`\)(.*)", line)
    if m:
        cur = m.group(2)
        new = f"## Figure {num[cur]} (`{line.split('`')[1]}`){m.group(3)}"
        if new != line:
            changed.append(f"heading {cur}: {m.group(1)} -> {num[cur]}")
        res.append(new)
        i += 1
        continue
    if cur and line.startswith("**Caption**"):
        j = i
        while j < len(F) and F[j].strip():
            j += 1
        text = cap[num[cur]]
        res.extend(f"**Caption** (as in `paper.md`, Figure {num[cur]}; copied by `lead5_work/scripts/sync_figures_md.py`). {text}".split("\n"))
        changed.append(f"caption {cur} (Figure {num[cur]}): replaced {j - i} lines")
        i = j
        continue
    res.append(line)
    i += 1
figs.write_text("\n".join(res))
print("\n".join(changed))
print("figures in paper.md:", num)
