"""Train a small existing delta-rule baseline; this is not a new architecture."""
import argparse,json,time,hashlib,copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from benchmark import generate,metrics,REGIMES
from memories import dense,predict,shared_bytes,config_bytes

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False

def batch(rng,B=24,T=256,N=64,regime='noisy'):
    cfg=REGIMES[regime]
    proto=rng.normal(size=(B,N,64)).astype(np.float32)
    proto/=np.linalg.norm(proto,axis=-1,keepdims=True)
    if cfg['cluster']:
        centers=rng.normal(size=(B,16,64)).astype(np.float32);centers/=np.linalg.norm(centers,axis=-1,keepdims=True)
        proto=.90*centers[:,np.arange(N)%16]+.44*proto
        proto/=np.linalg.norm(proto,axis=-1,keepdims=True)
    ids=rng.integers(N,size=(T,B));ops=rng.choice(3,size=(T,B),p=[.6,.1,.3])
    ops[:N]=0;ids[:N]=np.arange(N)[:,None]
    vals=rng.integers(16,size=(T,B));truth=np.zeros((T,B),np.int64)
    state=np.full((B,N),16,np.int64);ix=np.arange(B)
    for t in range(T):
        same=vals[t]==state[ix,ids[t]];vals[t,same]=(vals[t,same]+1)%16
        vals[t,ops[t]==1]=16
        mutate=ops[t]!=2
        state[ix[mutate],ids[t,mutate]]=vals[t,mutate]
        truth[t]=state[ix,ids[t]]
    x=proto[ix[None,:],ids]+cfg['noise']*rng.normal(size=(T,B,64)).astype(np.float32)
    x/=np.linalg.norm(x,axis=-1,keepdims=True)
    # All evaluator-only IDs/prototypes are discarded here.
    return tuple(torch.from_numpy(a).cuda() for a in (x.astype(np.float32),ops,vals,truth))

def codes(values):
    bits=((values[...,None]>>torch.arange(4,device=values.device))&1).float()*2-1
    active=(values!=16).float()
    return torch.cat((bits*active[...,None],active[...,None]),-1)

class Delta(nn.Module):
    def __init__(self,d=204):
        super().__init__();self.d=d
        self.encoder=nn.Sequential(nn.Linear(64,128,bias=False),nn.GELU(),nn.Linear(128,d,bias=False))
        self.write=nn.Parameter(torch.tensor(2.1972246))
    def keys(self,x):
        k=self.encoder(x);k=k-k.mean(-1,keepdim=True)
        return F.normalize(k,dim=-1)
    def forward(self,x,ops,vals,truth):
        k=self.keys(x);targets=codes(vals);S=x.new_zeros(x.shape[1],self.d,5)
        readouts=[]
        for t in range(x.shape[0]):
            read=torch.einsum('bd,bdv->bv',k[t],S)
            readouts.append(read)
            change=(ops[t]!=2).float()[:,None]
            delta=torch.sigmoid(self.write)*change*(targets[t]-read)
            S=S+k[t,:,:,None]*delta[:,None,:]
        reads=torch.stack(readouts)
        mask=ops==2;active=truth!=16
        loss_present=F.binary_cross_entropy_with_logits(4*(reads[...,4][mask]-.5),active[mask].float())
        bitmask=mask&active
        target_bits=((truth[...,None]>>torch.arange(4,device=x.device))&1).float()
        loss_bits=F.binary_cross_entropy_with_logits(4*reads[...,:4][bitmask],target_bits[bitmask])
        return loss_present+loss_bits

@torch.no_grad()
def predict_neural(stream,model):
    # GPU encodes current keys in batches for convenience; the causal memory
    # simulation is then scalar recurrence in Numba. No future key enters state.
    keys=[]
    for start in range(0,len(stream.ops),4096):
        x=torch.from_numpy(stream.keys[start:start+4096]).cuda()
        keys.append(model.keys(x).cpu().numpy()*np.sqrt(model.d))
    k=np.concatenate(keys).astype(np.float32)
    return dense(stream.ops,k,stream.values,1,1.,float(torch.sigmoid(model.write)),.5)

def train(seed):
    torch.manual_seed(seed);np.random.seed(seed)
    rng=np.random.default_rng(10000+seed);model=Delta().cuda()
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    validation=[batch(np.random.default_rng(15000+i),B=12,N=64,regime=r) for i,r in enumerate(REGIMES)]
    history=[];best=float('inf');checkpoint=None;t0=time.time()
    for step in range(1,601):
        regime=list(REGIMES)[int(rng.integers(3))];N=32 if step<201 else 64
        data=batch(rng,N=N,regime=regime)
        opt.zero_grad(set_to_none=True);loss=model(*data);loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step==1 or step%100==0:
            with torch.no_grad():vl=float(torch.stack([model(*v) for v in validation]).mean())
            history.append(dict(step=step,train_loss=float(loss.detach()),validation_loss=vl,seconds=time.time()-t0))
            if vl<best:
                best=vl;checkpoint={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};beststep=step
            print('TRAIN',seed,step,round(float(loss.detach()),4),round(vl,4),round(time.time()-t0,1),flush=True)
    params=sum(p.numel() for p in model.parameters())
    torch.save(dict(state=checkpoint,initial=initial,history=history,seed=seed,best_step=beststep,
        persistent_bytes=4080,shared_parameters=params),OUT/f'learned_seed{seed}.pt')
    (OUT/f'learned_training_{seed}.json').write_text(json.dumps(dict(seed=seed,history=history,best_step=beststep,
        persistent_bytes=4080,shared_parameters=params,shared_parameter_bytes=params*4,
        protocol_sha256=hashlib.sha256((ROOT/'LEARNED_BASELINE_PROTOCOL.md').read_bytes()).hexdigest()),indent=2))
    model.load_state_dict(checkpoint);return model,initial

def evaluate(seed,model,initial):
    raw=Delta().cuda();raw.load_state_dict(initial)
    selection=json.loads((OUT/'selection_4096.json').read_text());rows=[]
    for split,seeds,length,stress in [('final',range(500,508),12288,False),('final_stress',range(600,604),49152,True)]:
        for regime in REGIMES:
            for ss in seeds:
                stream=generate(ss,length,regime,stress)
                for label,m in [('learned_delta',model),('untrained_mlp_delta',raw)]:
                    p=predict_neural(stream,m)
                    rows.append(dict(split=split,regime=regime,stream_seed=ss,model_seed=seed,family=label,persistent_bytes=4080,**metrics(stream,p)))
                if seed==0:
                    features={}
                    for family,i in selection['selected'].items():
                        cfg=selection['configs'][i];p,_=predict(stream,cfg,features)
                        rows.append(dict(split=split,regime=regime,stream_seed=ss,model_seed=-1,family=family,
                                         persistent_bytes=config_bytes(cfg),**metrics(stream,p)))
            print('EVAL',seed,split,regime,flush=True)
        (OUT/f'learned_evaluation_{seed}.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    model,initial=train(a.seed);evaluate(a.seed,model,initial)
