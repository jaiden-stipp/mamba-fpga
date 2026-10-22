import argparse,json,hashlib
from pathlib import Path
import numpy as np
from benchmark import generate,metrics,REGIMES
from memories import predict,config_bytes
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'

def calibrate():
    configs=[]
    kinds=[('signature',32,4),('signature',64,8),('int8',64,64),('float32',64,256),('int8_projected',16,16),('int8_projected',32,32)]
    for kind,dim,width in kinds:
        for threshold in list(np.round(np.arange(.30,.951,.05),2))+[.99]:
            configs.append(dict(family='cache',kind=kind,dimension=dim,capacity=(4096-4)//(width+5),threshold=float(threshold)))
    rows=[]
    for regime in REGIMES:
        for seed in range(100,103):
            s=generate(seed,4096,regime);features={}
            for i,c in enumerate(configs):
                p,_=predict(s,c,features);rows.append(dict(index=i,regime=regime,seed=seed,**metrics(s,p)))
        print('refined calibration',regime,flush=True)
    mean=lambda i:np.mean([r['all_accuracy'] for r in rows if r['index']==i])
    selected={'cache_refined':int(max(range(len(configs)),key=mean))}
    for kind,dim,_ in kinds:
        selected[f'cache_{kind}{dim}']=int(max([i for i,c in enumerate(configs) if c['kind']==kind and c['dimension']==dim],key=mean))
    obj=dict(configs=configs,calibration=rows,selected=selected,
             protocol_sha256=hashlib.sha256((ROOT/'CONFIRMATION_PROTOCOL.md').read_bytes()).hexdigest())
    (OUT/'confirmation_selection.json').write_text(json.dumps(obj,indent=2))

def evaluate():
    import torch
    from learned import Delta,predict_neural
    models=[]
    for seed in range(3):
        checkpoint=torch.load(OUT/f'learned_seed{seed}.pt',weights_only=True,map_location='cpu')
        for name,key in [('learned_delta','state'),('untrained_mlp_delta','initial')]:
            m=Delta().cuda();m.load_state_dict(checkpoint[key]);m.eval();models.append((name,seed,m))
    a=json.loads((OUT/'confirmation_selection.json').read_text());b=json.loads((OUT/'selection_4096.json').read_text())
    controls={name:a['configs'][i] for name,i in a['selected'].items()}
    controls.update({name:b['configs'][i] for name,i in b['selected'].items()})
    rows=[]
    for split,seeds,length,stress in [('confirm',range(700,708),12288,False),('confirm_stress',range(800,804),49152,True)]:
        for regime in REGIMES:
            for seed in seeds:
                s=generate(seed,length,regime,stress);features={}
                for name,c in controls.items():
                    p,_=predict(s,c,features)
                    rows.append(dict(split=split,regime=regime,stream_seed=seed,model_seed=-1,family=name,
                                     persistent_bytes=config_bytes(c),**metrics(s,p)))
                for name,ms,m in models:
                    p=predict_neural(s,m)
                    rows.append(dict(split=split,regime=regime,stream_seed=seed,model_seed=ms,family=name,
                                     persistent_bytes=4080,**metrics(s,p)))
            (OUT/'confirmation_evaluation.json').write_text(json.dumps(rows,indent=2))
            print('confirmation',split,regime,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['calibrate','evaluate'],required=True);a=p.parse_args()
    calibrate() if a.stage=='calibrate' else evaluate()
