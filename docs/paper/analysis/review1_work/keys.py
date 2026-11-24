import json,sys
p=sys.argv[1]
seen={}
for line in open(p):
    d=json.loads(line)
    ev=d.get('event')
    if ev not in seen:
        seen[ev]=d
for ev,d in seen.items():
    dd={k:(v if not isinstance(v,(dict,list)) else type(v).__name__) for k,v in d.items()}
    print(ev, json.dumps(dd)[:1500])
    print()
