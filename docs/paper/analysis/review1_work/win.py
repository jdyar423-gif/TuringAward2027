import json,statistics as st
F='/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/paper/lead_work/final_snap/run/train/train_log.jsonl'
V='/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/paper/data/_src/live/run/train/train_log.jsonl'
f=[json.loads(l) for l in open(F)]; f=[d for d in f if d.get('event')=='step']
v=[json.loads(l) for l in open(V)]; v=[d for d in v if d.get('event')=='step']
# roles: index? ce_by_role, weight_by_role
for lo,hi in ((0,0.01),(0.05,0.10),(0.15,0.20),(0.25,0.30),(0.35,0.40),(0.45,0.50)):
    W=[d for d in f if lo<=d['p']<hi]
    L=[d['loss'] for d in W]
    ce=[0]*6; w=[0]*6; tg=[]
    for d in W:
        m=d['micro']
        for i in range(6):
            ce[i]+=m['ce_by_role'][i]; w[i]+=m['weight_by_role'][i]
        tg.append(m['tape_gain']*m['weight'])
    tw=sum(d['micro']['weight'] for d in W)
    roles=[round(ce[i]/w[i],3) if w[i] else None for i in range(6)]
    print(lo,hi,'n',len(W),'mean',round(st.mean(L),3),'sd',round(st.stdev(L),3) if len(L)>1 else '-', 'roles',roles,'tape_gain(w)',round(sum(tg)/tw,3))
vm=[d for d in v if d['step']<=31900]; vs=[d for d in v if d['step']>31912]
print('v9 pre-switch rate median',st.median([d['tok_per_s'] for d in vm]),'n',len(vm))
print('v9 post-switch rate median',st.median([d['tok_per_s'] for d in vs]),'n',len(vs))
print('v9 loss mean all',round(st.mean([d['loss'] for d in v]),4),'sd',round(st.stdev([d['loss'] for d in v]),4),'n',len(v))
L21f=[d['loss'] for d in f][-21:]; L21v=[d['loss'] for d in v][:21]
print('last21 final',round(st.mean(L21f),4),round(st.stdev(L21f),4),'first21 v9',round(st.mean(L21v),4),round(st.stdev(L21v),4))
