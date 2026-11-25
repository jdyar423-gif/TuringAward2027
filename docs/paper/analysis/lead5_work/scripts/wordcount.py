"""Words of the main text (from '## Abstract' to '## References'), two counts:
  - the third draft's convention (RESPONSE_2 'counted without table rules'): whitespace tokens, table rule
    lines left out and the pipes of table rows not counted; this gives 9,570 words, 977 in table rows, for
    the third draft as kept in lead4_work/paper_v3.md (it reported 9,536 and 977);
  - tokens with a letter or digit (dashes, pipes and symbols alone not counted).
Also the words of the first count per section.

    python3 wordcount.py paper.md
"""
import re
import sys

text = open(sys.argv[1]).read()
a = text.index("## Abstract")
b = text.index("## References")
main = text[a:b]
tot = tab = an = an_tab = 0
sec = {}
cur = "Abstract"
for line in main.splitlines():
    s = line.strip()
    if line.startswith("## ") or line.startswith("### "):
        cur = line.lstrip("#").strip()
    if re.match(r"^\|[-| :]+\|$", s):
        continue
    n = len([w for w in line.split() if w != "|"])
    m = sum(1 for w in line.split() if re.search(r"[A-Za-z0-9]", w))
    tot += n
    an += m
    if s.startswith("|"):
        tab += n
        an_tab += m
    sec[cur] = sec.get(cur, 0) + n
print(f"main text {tot} words (third draft's convention), of which {tab} in table rows; "
      f"{an} tokens with a letter or digit, of which {an_tab} in table rows")
for k, v in sec.items():
    print(f"  {v:5d}  {k}")
