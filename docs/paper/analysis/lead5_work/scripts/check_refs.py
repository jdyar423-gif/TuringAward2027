"""Cross-reference check of a paper draft: every table and figure defined is cited outside its own
caption block, every cited table/figure/appendix section exists, and main-text items are cited in the main text."""
import re, sys
t = open(sys.argv[1]).read()
main = t[t.index('## Abstract'):t.index('## References')]
lines = t.splitlines()
defs = {}
for i, l in enumerate(lines):
    m = re.match(r'^\*\*(Table|Figure) ([A-Z]?\d+)[\.\s]', l)
    if m:
        defs[f'{m.group(1)} {m.group(2)}'] = i
cites = {}
for i, l in enumerate(lines):
    for m in re.finditer(r'\b(Tables?|Figures?) ((?:[A-Z]?\d+[a-d]?)(?:(?:, | and | to |–)[A-Z]?\d+[a-d]?)*)', l):
        kind = 'Table' if m.group(1).startswith('Table') else 'Figure'
        for x in re.split(r', | and | to |–', m.group(2)):
            x = re.sub(r'[a-d]$', '', x)
            cites.setdefault(f'{kind} {x}', []).append(i)
# a caption block = from its definition line to the next blank line
def in_own_caption(name, i):
    d = defs[name]
    j = d
    while j < len(lines) and lines[j].strip():
        j += 1
    return d <= i < j
main_start = t[:t.index('## Abstract')].count('\n')
main_end = t[:t.index('## References')].count('\n')
for name, d in sorted(defs.items(), key=lambda kv: kv[1]):
    c = [i for i in cites.get(name, []) if not in_own_caption(name, i)]
    where = 'main' if main_start <= d < main_end else 'appx'
    cm = [i for i in c if main_start <= i < main_end]
    flag = ''
    if not c:
        flag = 'NOT CITED'
    elif where == 'main' and not cm:
        flag = 'not cited in main text'
    print(f'{name:12s} def line {d+1:5d} ({where}) cited {len(c)} times (main {len(cm)}) {flag}')
for name in sorted(set(cites) - set(defs)):
    print('CITED BUT UNDEFINED:', name, [i + 1 for i in cites[name]][:5])
# appendix sections
secs = set(re.findall(r'^\*\*([A-N]\.\d)\b', t, re.M)) | set(re.findall(r'^## Appendix ([A-N])\.', t, re.M))
for m in sorted(set(re.findall(r'Appendix ([A-N](?:\.\d)?)', t))):
    if m not in secs:
        print('APPENDIX REF NOT FOUND:', m)
