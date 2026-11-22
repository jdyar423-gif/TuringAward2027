import json,os,sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'runs'))
def evals(n):
    out={}
    try:
        for ln in open(n+'.log'):
            if ln.startswith('{') and '"mix"' in ln:
                d=json.loads(ln); out[d['tokens']]=(d['mix'],d.get('asst'))
    except FileNotFoundError: pass
    return out
def head(n):
    st=rd=None
    try:
        for ln in open(n+'.log'):
            if 'stored' in ln and 'read/token' in ln:
                parts=ln.replace(',','').split(); st=int(parts[parts.index('stored')+1]); rd=int(parts[parts.index('read/token')+1]); break
    except FileNotFoundError: pass
    return st,rd
base=evals('k_base')
names=sys.argv[1:]
print('%-16s %9s %8s  %s'%('arm','stored','read','mix loss (delta vs base) at 3.3M / 6.6M / 9.8M / 12M tokens'))
for n in names:
    e=evals(n); st,rd=head(n)
    print('%-16s %9s %8s  '%(n,st,rd)+'  '.join('%.4f(%+.3f)'%(e[t][0], e[t][0]-base[t][0]) for t in sorted(e) if t in base))
