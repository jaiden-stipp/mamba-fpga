import argparse,csv,hashlib,json,os,platform,time
from pathlib import Path
import numpy as np
from benchmark import generate,metrics,REGIMES
from memories import configurations,config_bytes,shared_bytes,predict

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results';RESULTS.mkdir(exist_ok=True)
FAMILIES=['cache','additive','delta','hybrid']

def dump(path,obj):path.write_text(json.dumps(obj,indent=2),encoding='utf-8')

def run(budget):
    configs=configurations(budget)
    rows=[];started=time.time()
    for regime in REGIMES:
        for seed in range(100,103):
            stream=generate(seed,4096,regime);features={}
            for i,cfg in enumerate(configs):
                pred,_=predict(stream,cfg,features)
                rows.append(dict(index=i,regime=regime,seed=seed,**metrics(stream,pred)))
        print('calibration',budget,regime,round(time.time()-started,1),flush=True)
    selected={}
    for family in FAMILIES:
        eligible=[i for i,c in enumerate(configs) if c['family']==family]
        selected[family]=max(eligible,key=lambda i:np.mean([r['all_accuracy'] for r in rows if r['index']==i]))
    # This file is written before ANY held-out stream is evaluated.
    selection=dict(budget=budget,configs=configs,calibration=rows,selected=selected,
                   frozen_selection=True,protocol_sha256=hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest())
    dump(RESULTS/f'selection_{budget}.json',selection)
    test=[]
    for split,seeds,length,stress in [('test',range(200,208),12288,False),('stress',range(300,304),49152,True)]:
        for regime in REGIMES:
            for seed in seeds:
                stream=generate(seed,length,regime,stress);features={}
                # Separate stronger-interface diagnostic. This table is the ONLY
                # method allowed to inspect exact IDs; its allocation is 128B.
                table=np.full(128,16,np.uint8);ip=np.full(length,16,np.uint8)
                for t in range(length):
                    if stream.ops[t]==2:ip[t]=table[stream.entity[t]]
                    else:table[stream.entity[t]]=stream.values[t]
                assert metrics(stream,ip)['all_accuracy']==1.
                if table.nbytes<=budget:test.append(dict(split=split,regime=regime,seed=seed,family='id_table',persistent_bytes=int(table.nbytes),shared_bytes=0,**metrics(stream,ip)))
                for family,i in selected.items():
                    cfg=configs[i];pred,_=predict(stream,cfg,features)
                    test.append(dict(split=split,regime=regime,seed=seed,family=family,config_index=i,
                                     persistent_bytes=config_bytes(cfg),shared_bytes=shared_bytes(cfg),**metrics(stream,pred)))
                if seed==min(seeds):
                    np.savez_compressed(RESULTS/f'trace_{budget}_{split}_{regime}.npz',ops=stream.ops,entity=stream.entity,
                        values=stream.values,truth=stream.truth,previous=stream.previous,edits=stream.edits,age=stream.age,score=stream.score,
                        **{f'prediction_{f}':predict(stream,configs[i],features)[0] for f,i in selected.items()})
            print(split,budget,regime,round(time.time()-started,1),flush=True)
        dump(RESULTS/f'evaluation_{budget}.json',dict(budget=budget,rows=test,elapsed_seconds=time.time()-started,
             python=platform.python_version(),numpy=np.__version__,protocol_sha256=selection['protocol_sha256']))
    print('COMPLETE',budget,round(time.time()-started,1),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--budget',type=int,required=True);a=p.parse_args();run(a.budget)
