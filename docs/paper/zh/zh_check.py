"""Compare a translated part with its English source: headings, table rows and columns, numbers,
citation markers, code spans, [TBD] markers.   python zh_check.py EN.md ZH.md"""
import re
import sys
from collections import Counter

en, zh = (open(p, encoding="utf-8").read() for p in sys.argv[1:3])
bad = 0


def lines(t):
    return t.split("\n")


def heads(t):
    return [re.match(r"#+", l).group(0) for l in lines(t) if l.startswith("#")]


def tables(t):
    out, cur = [], None
    for l in lines(t):
        if l.lstrip().startswith("|"):
            cur = cur or []
            cur.append(l.count("|"))
        elif cur:
            out.append(cur)
            cur = None
    if cur:
        out.append(cur)
    return out


num = re.compile(r"(?<![A-Za-z_])\d[\d,]*(?:\.\d+)?")


def nums(t):
    t = re.sub(r"`[^`\n]*`", " ", t)          # code spans are compared separately
    return Counter(x.rstrip(",") for x in num.findall(t))


def report(name, a, b):
    global bad
    if a != b:
        bad += 1
        if isinstance(a, Counter):
            d1 = a - b
            d2 = b - a
            print(f"  {name}: only in EN {dict(list(d1.items())[:25])}{' ...' if len(d1) > 25 else ''}; only in ZH {dict(list(d2.items())[:25])}{' ...' if len(d2) > 25 else ''}")
        else:
            print(f"  {name}: EN {str(a)[:200]} | ZH {str(b)[:200]}")


report("headings", heads(en), heads(zh))
te, tz = tables(en), tables(zh)
report("number of tables", len(te), len(tz))
if len(te) == len(tz):
    for i, (a, b) in enumerate(zip(te, tz)):
        if a != b:
            bad += 1
            print(f"  table {i + 1}: rows EN {len(a)} ZH {len(b)}; pipes per row differ at rows {[j for j, (x, y) in enumerate(zip(a, b)) if x != y][:8]}")
report("numbers", nums(en), nums(zh))
report("citation markers", Counter(re.findall(r"\[\d+(?:, ?\d+)*\]", en)), Counter(re.findall(r"\[\d+(?:, ?\d+)*\]", zh)))
report("code spans", Counter(re.findall(r"`[^`\n]+`", en)), Counter(re.findall(r"`[^`\n]+`", zh)))
report("TBD markers", len(re.findall(r"\[TBD", en)), len(re.findall(r"\[TBD", zh)))
print(f"{sys.argv[2].split('/')[-1]}: EN {len(lines(en))} lines / ZH {len(lines(zh))} lines; {'OK' if not bad else str(bad) + ' kinds of difference'}")
