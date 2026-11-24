import json,statistics as st
F='/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/paper/lead_work/final_snap/run/train/train_log.jsonl'
V='/tmp/claude-0/-home-user-turingaward2027/3615ba93-d69b-5e15-9aac-4ee28e8b4b51/scratchpad/paper/data/_src/live/run/train/train_log.jsonl'
def blocks(p):
    out=[]
    for l in open(p):
        d=json.loads(l)
        if d.get('event')=='step' and 'loader_block' in d:
            lb=d['loader_block']
            if 'subkind_share' in lb: out.append((d['step'],lb['subkind_share']))
    return out
fb=blocks(F); vb=blocks(V)
print('final blocks',len(fb),'keys',sorted(fb[0][1].keys()))
keys=sorted(set(k for _,s in fb for k in s))
for k in keys:
    print(' final',k, round(100*st.mean([s.get(k,0) for _,s in fb]),2))
print('v9 blocks',len(vb))
keys=sorted(set(k for _,s in vb for k in s))
for k in keys:
    vals=[100*s.get(k,0) for _,s in vb]
    print(' v9',k, round(min(vals),2), round(max(vals),2))
# fail counts (refusal/deflection) if present
