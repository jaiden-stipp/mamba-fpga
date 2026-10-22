import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from portable import Model,ASSETS,ROOT,torch,TAU,validate

RESULTS=ROOT/'results';RESULTS.mkdir(exist_ok=True)

def prepare_data():
    tok=Tokenizer.from_file(str(ASSETS/'tokenizer-llama31/tokenizer.json'))
    for split in ['validation','test']:
        path=ASSETS/f'wikitext2-{split}-tokens.npy'
        if not path.exists():
            rows=pq.read_table(ASSETS/f'wikitext2-{split}.parquet').column('text').to_pylist()
            text='\n\n'.join(rows)
            tokens=np.asarray(tok.encode(text,add_special_tokens=False).ids,dtype=np.int64)
            np.save(path,tokens)
    return tok

def pairs(x,is_mimo):
    a=x.shape[-1]//4
    if is_mimo:return x[...,:a],x[...,x.shape[-1]//2:x.shape[-1]//2+a]
    return x[...,:2*a:2],x[...,1:2*a:2]

@torch.inference_mode()
def calibration(model,token_ids):
    inp=torch.tensor([[128000]+token_ids[:2047].tolist()],device=model.device)
    _,_,traces=model.forward(inp,trace_layers=range(model.nl))
    scores={name:[] for name in ['magnitude','decay_error','phase_relative']}
    stats=[]
    for layer,tr in traces.items():
        inc=tr['increments'].double();err=torch.round(inc/(TAU/256))*(TAU/256)-inc
        cs=inc.cumsum(1);ecs=err.cumsum(1)
        qa,qb=pairs(tr['q_unrot'],model.is_mimo);ka,kb=pairs(tr['k_unrot'],model.is_mimo)
        qnorm=(qa.square()+qb.square()).sum(-2).sqrt()
        knorm=(ka.square()+kb.square()).sum(-2).sqrt()
        vnorm=tr['v'].square().mean((-1,-2)).sqrt()
        mag=(qnorm*knorm).square().mean((0,1))
        alpha=tr['adt'].exp().mean((0,1))
        localerr=(err.square()*(qnorm*knorm).double().square()).mean((0,1))
        decay=localerr/(1-alpha.double().square()).clamp_min(1e-8)[:,None]
        phase=torch.zeros_like(mag,dtype=torch.float64)
        acs=tr['adt'].double().cumsum(1)
        for lag in [1,4,16,64,256]:
            relative=ecs[:,lag:]-ecs[:,:-lag]
            chord=2*torch.sin(relative/2)
            retention=(acs[:,lag:]-acs[:,:-lag]).exp()
            weight=qnorm[:,lag:].double()*knorm[:,:-lag].double()*vnorm[:,:-lag].double()[...,None]*tr['dt'][:,:-lag].double()[...,None]*retention[...,None]
            phase+=(weight*chord).square().mean((0,1))
        for name,val in [('magnitude',mag),('decay_error',decay),('phase_relative',phase)]:scores[name].append(val.float().cpu())
        stats.append({'layer':layer,'mean_alpha':alpha.mean().item(),'mean_increment_abs':inc.abs().mean().item(),'increment_error8_rms':err.square().mean().sqrt().item(),'mean_abs_phase_travel':inc.abs().sum(1).mean().item()})
    masks={}
    for name,vals in scores.items():
        values=torch.stack(vals)
        # Equal per-layer budget: 25% of H*A accumulators at 16 bits, rest 8.
        mask=torch.zeros_like(values,dtype=torch.bool)
        for layer in range(model.nl):
            inds=values[layer].flatten().topk(values[layer].numel()//4).indices
            mask[layer].view(-1)[inds]=True
        masks[name]=mask
    torch.save({'scores':scores,'masks':masks,'stats':stats},RESULTS/f'{model.name}-calibration.pt')
    return {k:v.to(model.device) for k,v in masks.items()},stats

def evaluate(model,tokens,mode,masks=None,starts=None,length=2048):
    if starts is None:starts=[0,4096,8192,12288,16384,20480,24576,28672]
    losses=[];wall=time.time()
    for start in starts:
        sample=tokens[start:start+length]
        assert len(sample)==length
        inp=torch.tensor([[128000]+sample[:-1].tolist()],device=model.device)
        targets=torch.tensor(sample[None],device=model.device)
        y,_,_=model.forward(inp,mode=mode,masks=masks)
        nll=model.loss(y,targets)[0].cpu().numpy()
        losses.append(nll)
    losses=np.stack(losses)
    return losses,{'mode':mode,'starts':starts,'length':length,'tokens':int(losses.size),'nll':float(losses.mean()),'ppl':float(np.exp(losses.mean())),'window_nll':losses.mean(1).tolist(),'seconds':round(time.time()-wall,3)}

def run(name,device,stage='main'):
    prepare_data()
    model=Model(name,device)
    test=np.load(ASSETS/'wikitext2-test-tokens.npy')
    val=np.load(ASSETS/'wikitext2-validation-tokens.npy')
    out_path=RESULTS/f'{name}-{stage}.json'
    rows=json.loads(out_path.read_text()) if out_path.exists() else []
    if stage in ['main','replicate']:
        saved=RESULTS/f'{name}-calibration.pt'
        if saved.exists():masks={k:v.to(device) for k,v in torch.load(saved,weights_only=True)['masks'].items()}
        else:masks,_=calibration(model,val)
        jobs=[(x,None) for x in ['reference','fixed6','fixed8','fixed10','fixed12','fixed16','lookup6','lookup8','lookup10','guard16_8','trig6','trig8','trig10','offset']]
        jobs += [('mixed_'+k,v) for k,v in masks.items()]
        if stage=='replicate':
            jobs=[(x,None) for x in ['reference','fixed8','fixed10','fixed12','guard16_8','guard16_8_trig8']]+[('mixed_'+k,v) for k,v in masks.items()]
        for label,mask in jobs:
            if any(row['label']==label for row in rows):continue
            starts=None if stage=='main' else [32768+4096*i for i in range(32)]
            nll,row=evaluate(model,test,'mixed' if mask is not None else label,masks=mask,starts=starts)
            row['label']=label;row['checkpoint']=name;row['device']=device
            np.save(RESULTS/f'{name}-{stage}-{label}-losses.npy',nll)
            rows.append(row);out_path.write_text(json.dumps(rows,indent=2))
            print(json.dumps(row),flush=True)
    elif stage=='long':
        # Concatenated WikiText test: a length stress test, not coherent-book or
        # retrieval evidence. Only the final 2048 losses in each sequence scored.
        for length in [8192,32768,65536]:
            for mode in ['reference','fixed8','fixed10','fixed12','guard16_8','lookup8','guard16_8_trig8']:
                label=f'{mode}-L{length}'
                if any(row['label']==label for row in rows):continue
                wall=time.time();states=None;nlls=[]
                # Stream 2048-token blocks to keep memory bounded; preserve all
                # layer caches. Last-block predictions include previous context.
                for start in range(0,length,2048):
                    block=test[start:start+2048]
                    prefix=128000 if start==0 else int(test[start-1])
                    inp=torch.tensor([[prefix]+block[:-1].tolist()],device=device)
                    y,states,_=model.forward(inp,mode=mode,states=states)
                    if start==length-2048:
                        nlls=model.loss(y,torch.tensor(block[None],device=device))[0].cpu().numpy()
                row={'checkpoint':name,'label':label,'mode':mode,'length':length,'scored_tokens':len(nlls),'nll':float(nlls.mean()),'ppl':float(np.exp(nlls.mean())),'seconds':round(time.time()-wall,3)}
                np.save(RESULTS/f'{name}-{label}-losses.npy',nlls)
                rows.append(row);out_path.write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--stage',default='main');a=p.parse_args()
    run(a.name,a.device,a.stage)
