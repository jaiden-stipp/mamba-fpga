"""Secondary representation controls, selected only from original calibration.

Added after seeing hybrid stress results. Fresh final streams are therefore the
main interpretation; do not treat this as a preregistered primary analysis.
"""
import json,time
from pathlib import Path
import numpy as np
from benchmark import generate,metrics,REGIMES
from memories import config_bytes,predict
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'

def run():
    a=json.loads((OUT/'selection_4096.json').read_text());chosen={}
    for kind,dim in [('signature',32),('signature',64),('int8',64),('float32',64)]:
        ii=[i for i,c in enumerate(a['configs']) if c['family']=='cache' and c['kind']==kind and c['dimension']==dim]
        i=max(ii,key=lambda i:np.mean([r['all_accuracy'] for r in a['calibration'] if r['index']==i]))
        chosen[f'cache_{kind}{dim}']=a['configs'][i]
    (OUT/'cache_controls_selection.json').write_text(json.dumps(chosen,indent=2))
    rows=[]
    for split,seeds,length,stress in [('final',range(500,508),12288,False),('final_stress',range(600,604),49152,True)]:
        for regime in REGIMES:
            for seed in seeds:
                s=generate(seed,length,regime,stress);features={}
                for name,c in chosen.items():
                    p,_=predict(s,c,features)
                    rows.append(dict(split=split,regime=regime,stream_seed=seed,family=name,persistent_bytes=config_bytes(c),**metrics(s,p)))
            print(split,regime,flush=True)
        (OUT/'cache_controls_evaluation.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':run()
