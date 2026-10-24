"""Check emulator-to-RTL contract, then a 32-token whole-model integration smoke.

Reported language loss is diagnostic; this is not a matched-quality evaluation.
No hardware speed claim can be made from this eager arithmetic emulator.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch_backend import HybridModel,RecurrentBackend
import portable
from regression import dataset,word,SEED

HERE=Path(__file__).resolve().parent

def conformance(traces,evidence):
    results=[]
    for rank in (1,4):
        data,_=dataset(rank,traces);backend=RecurrentBackend()
        expected=[int(x,16) for x in (evidence/f'rank{rank}'/'outputs.hex').read_text().splitlines()]
        state_expected=[int(x,16)&65535 for x in (evidence/f'rank{rank}'/'states.hex').read_text().splitlines()]
        got=[];states=[]
        for t in range(64):
            if t==32:backend=RecurrentBackend()
            tensor=lambda x:torch.from_numpy(np.asarray(x).copy())
            q,k,v=[tensor(data[key][t])[None,None] for key in ('q','k','v')]
            a,b,g=[tensor(data[key][t]).reshape(1,1) for key in ('alpha','beta','gamma')]
            y=backend.step(q,k,v,a,b,g,tensor(data['skip'][t]).reshape(1),tensor(data['increment'][t])[None,None])
            got.extend(word(x) for x in y[0,0].T.numpy().ravel())
            states.extend(backend.cache[1].numpy().view(np.uint16).ravel().tolist())
        assert got==expected,f'Rank {rank}: emulator outputs differ from RTL reference'
        assert states==state_expected,f'Rank {rank}: emulator feedback differs from RTL reference'
        results.append({'rank':rank,'outputs_exact':len(got),'packed_state_values_exact':len(states)})
    return results

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assets',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=HERE/'model_results.json')
    args=ap.parse_args();torch.manual_seed(SEED);portable.ASSETS=args.assets
    records=[];contract=conformance(HERE/'traces',HERE/'evidence')
    print('PASS Torch emulator agrees with all RTL-checked outputs and packed states',flush=True)
    candidates=list(args.assets.rglob('*validation*tokens.npy'));assert len(candidates)==1,candidates
    tokens=np.load(candidates[0])[32768:32801].copy()
    ids=torch.as_tensor(tokens[None],device='cuda:0')
    for name in ('mamba3-siso-187m','mamba3-mimo-187m'):
        model=portable.Model(name,'cuda:0')
        with torch.inference_mode():
            baseline,_,_=model.forward(ids[:,:32])
            decoder=HybridModel(model)
            hidden=torch.stack([decoder.step(ids[:,t]) for t in range(32)],dim=1)
            reference_loss=model.loss(baseline,ids[:,1:33]).mean().item()
            candidate_loss=model.loss(hidden,ids[:,1:33]).mean().item()
            w=model.w('lm_head.weight')
            ref_top=(baseline@w.T).argmax(-1);top=(hidden@w.T).argmax(-1)
        state_bytes=sum(x.cache[1].numel()*2 for x in decoder.backends)
        records.append({'checkpoint':name,'layers':model.nl,'heads_per_layer':model.h,'P':model.p,'N':model.n,'rank':model.rank,
                        'tokens':32,'validation_offset':32768,'reference_perplexity':math.exp(reference_loss),
                        'emulator_perplexity':math.exp(candidate_loss),'relative_perplexity_change':math.expm1(candidate_loss-reference_loss),
                        'argmax_agreement_tokens':int((ref_top==top).sum()),'persistent_main_state_bytes':state_bytes,
                        'hidden_relative_rms':float(torch.linalg.vector_norm(hidden-baseline)/torch.linalg.vector_norm(baseline)),
                        'all_logits_finite':bool(torch.isfinite(hidden@w.T).all())})
        print(json.dumps(records[-1]),flush=True)
        del decoder,model,hidden,baseline,w;torch.cuda.empty_cache()
    result={'status':'PASS','seed':SEED,'backend':'Torch eager arithmetic emulator, not FPGA transport',
            'source_sha256':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                             for name in ('torch_backend.py','evaluate_model.py')},
            'torch_version':torch.__version__,'numpy_version':np.__version__,
            'contract_checks':contract,'models':records,
            'scope':'32-token full-model integration smoke. Dense/normalization/nonlinear/gating/FFN/vocabulary layers execute in Torch. Recurrence arithmetic matches the RTL contract; only sampled tiles were simulated in RTL. No end-to-end hardware speed or matched-quality claim.'}
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')

if __name__=='__main__':main()
