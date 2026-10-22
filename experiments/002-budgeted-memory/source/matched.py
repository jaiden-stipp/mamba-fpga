"""Post-hoc diagnostic at training-like length/entity count; no tuning."""
import json
from pathlib import Path
import numpy as np
import torch
from benchmark import generate,metrics,REGIMES
from learned import Delta,predict_neural
from memories import predict,config_bytes
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'
models=[]
for seed in range(3):
    a=torch.load(OUT/f'learned_seed{seed}.pt',weights_only=True,map_location='cpu')
    for family,key in [('learned_delta','state'),('untrained_mlp_delta','initial')]:
        m=Delta().cuda();m.load_state_dict(a[key]);models.append((family,seed,m))
sel=json.loads((OUT/'confirmation_selection.json').read_text())
c=sel['configs'][sel['selected']['cache_refined']]
rows=[]
for regime in REGIMES:
    for seed in range(900,916):
        s=generate(seed,256,regime,entities=64);s.score=s.ops==2
        p,_=predict(s,c)
        rows.append(dict(split='matched_diagnostic',regime=regime,stream_seed=seed,model_seed=-1,family='cache_refined',persistent_bytes=config_bytes(c),**metrics(s,p)))
        for family,ms,m in models:
            p=predict_neural(s,m)
            rows.append(dict(split='matched_diagnostic',regime=regime,stream_seed=seed,model_seed=ms,family=family,persistent_bytes=4080,**metrics(s,p)))
    print('matched diagnostic',regime,flush=True)
(OUT/'matched_diagnostic.json').write_text(json.dumps(rows,indent=2))
