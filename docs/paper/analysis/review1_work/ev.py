import json,sys
p=sys.argv[1]
for line in open(p):
    d=json.loads(line)
    if d.get('event') in ('eval','end','restore','setup','stage'):
        if d['event']=='setup':
            d={k:v for k,v in d.items() if k in('event','step','time','train_end','deadline','ledger')}
        print(json.dumps(d)[:900])
