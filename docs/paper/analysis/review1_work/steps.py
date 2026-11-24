import json,sys,statistics as st, datetime as dt
F='/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/paper/lead_work/final_snap/run/train/train_log.jsonl'
V='/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/paper/data/_src/live/run/train/train_log.jsonl'
def load(p):
    out=[]
    for line in open(p):
        d=json.loads(line)
        if d.get('event')=='step': out.append(d)
    return out
f=load(F); v=load(V)
print('final steps',len(f), f[0]['step'], f[-1]['step'])
for s in (50,23150,29300,29700,22600,22550,22500):
    for d in f:
        if d['step']==s: print('final',s,'loss',round(d['loss'],4),'tok/s',round(d['tok_per_s']),'p',round(d['p'],5),'lr',round(d['lr'],4),'step_s',round(d['step_s'],4))
print('v9 steps',len(v), v[0]['step'], v[-1]['step'])
for d in v:
    print('v9',d['step'],'loss',round(d['loss'],4),'tok/s',round(d['tok_per_s']),'p',round(d['p'],5),'lr',round(d['lr'],4),'tape_mass',round(d['micro']['tape_mass'],3),'tape_gain',round(d['micro']['tape_gain'],4))
# rates around kernel switch
pre=[d['tok_per_s'] for d in f if 21000<=d['step']<22613]
post=[d['tok_per_s'] for d in f if 22613<d['step']<=25280]
post2=[d['tok_per_s'] for d in f if 25280<d['step']<=29722]
print('pre-kernel median',round(st.median(pre)),'min',round(min(pre)),'max',round(max(pre)), len(pre))
print('post-kernel median',round(st.median(post)),'min',round(min(post)),'max',round(max(post)),len(post))
print('post-deflection median',round(st.median(post2)),'min',round(min(post2)),'max',round(max(post2)),len(post2))
# tape mass medians in p windows
for lo,hi in ((0.40,0.45),(0.45,0.50),(0.40,0.50)):
    tm=[d['micro']['tape_mass'] for d in f if lo<=d['p']<hi]
    tg=[d['micro']['tape_gain'] for d in f if lo<=d['p']<hi]
    print('tape_mass p',lo,hi,'median',round(st.median(tm),3),'mean',round(st.mean(tm),3),'min',round(min(tm),3),'max',round(max(tm),3),'n',len(tm),'| tape_gain median',round(st.median(tg),4))
# losses p in [0.45,0.5)
L=[d['loss'] for d in f if 0.45<=d['p']<0.50]
print('final p .45-.50 n',len(L),'mean',round(st.mean(L),4),'sd',round(st.stdev(L),4),'min',round(min(L),4),'max',round(max(L),4))
