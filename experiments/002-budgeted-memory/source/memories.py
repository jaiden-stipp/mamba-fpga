"""Frozen analytic controls, not trained implementations of named architectures."""
import numpy as np
from numba import njit

@njit(cache=True)
def explicit(ops, keys, values, capacity, threshold, signature=False):
    # Signature keys are physically packed uint8 bytes, not float tensors.
    stored=np.zeros((capacity,keys.shape[1]),dtype=keys.dtype)
    val=np.full(capacity,16,np.uint8)
    stamps=np.zeros(capacity,np.uint32)  # zero means unoccupied
    clock=np.zeros(1,np.uint32)
    pred=np.full(len(ops),16,np.uint8)
    hit=np.zeros(len(ops),np.bool_)
    for t in range(len(ops)):
        clock[0]=t+1
        best=-2.;idx=-1
        for s in range(capacity):
            if stamps[s]==0:continue
            if signature:
                distance=0
                for j in range(keys.shape[1]):
                    bits=int(stored[s,j])^int(keys[t,j])
                    while bits:
                        distance+=bits&1;bits>>=1
                sim=1.-2.*distance/(8*keys.shape[1])
            else:
                dot=0.;norm1=0.;norm2=0.
                for j in range(keys.shape[1]):
                    a=float(stored[s,j]);b=float(keys[t,j])
                    dot+=a*b;norm1+=a*a;norm2+=b*b
                sim=dot/np.sqrt(max(norm1*norm2,1e-20))
            if sim>best:best=sim;idx=s
        found=idx>=0 and best>=threshold
        if ops[t]==2:
            if found:pred[t]=val[idx];hit[t]=True;stamps[idx]=clock[0]
        else:
            if not found:
                idx=int(np.argmin(stamps))
            stored[idx]=keys[t];val[idx]=values[t];stamps[idx]=clock[0]
    return pred,hit

@njit(cache=True)
def dense(ops, keys, values, mode, alpha, beta, presence):
    S=np.zeros((keys.shape[1],5),np.float32)
    pred=np.full(len(ops),16,np.uint8)
    invnorm=1./np.sqrt(keys.shape[1])
    for t in range(len(ops)):
        read=np.zeros(5,np.float32)
        for j in range(keys.shape[1]):
            for c in range(5):read[c]+=keys[t,j]*invnorm*S[j,c]
        if ops[t]==2:
            if read[4]>presence:
                value=0
                for c in range(4):
                    if read[c]>0:value|=1<<c
                pred[t]=value
        else:
            target=np.zeros(5,np.float32)
            if values[t]!=16:
                for c in range(4):target[c]=1. if (values[t]>>c)&1 else -1.
                target[4]=1.
            for j in range(keys.shape[1]):
                for c in range(5):
                    if mode==0:S[j,c]=alpha*S[j,c]+keys[t,j]*invnorm*target[c]
                    else:S[j,c]=alpha*S[j,c]+beta*keys[t,j]*invnorm*(target[c]-alpha*read[c])
    return pred

def transform(keys,kind,dimension):
    # Entire feature map is fixed across requests; no entity-specific codebook.
    if kind in ('signature','dense'):
        projection=np.random.default_rng(743).choice(np.array([-1.,1.],np.float32),(64,dimension))
        signs=keys@projection>=0
        return np.packbits(signs,axis=1) if kind=='signature' else np.where(signs,1,-1).astype(np.int8)
    if kind=='int8':return np.rint(np.clip(keys,-1,1)*127).astype(np.int8)
    if kind=='int8_projected':
        projection=np.random.default_rng(744).normal(size=(64,dimension)).astype(np.float32)
        reduced=keys@projection;reduced/=np.maximum(np.linalg.norm(reduced,axis=1,keepdims=True),1e-12)
        return np.rint(np.clip(reduced,-1,1)*127).astype(np.int8)
    return keys

def config_bytes(cfg):
    if cfg['family'] in ('delta','additive'):return cfg['dimension']*5*4
    if cfg['family']=='hybrid':return config_bytes(cfg['cache'])+config_bytes(cfg['matrix'])
    perkey=(cfg['dimension']+7)//8 if cfg['kind']=='signature' else cfg['dimension']*(1 if cfg['kind'] in ('int8','int8_projected') else 4)
    return 4+cfg['capacity']*(perkey+1+4)

def shared_bytes(cfg):
    if cfg['family']=='hybrid':return shared_bytes(cfg['cache'])+shared_bytes(cfg['matrix'])
    return 64*cfg['dimension']*4 if cfg.get('kind') in ('signature','int8_projected') or cfg['family'] in ('delta','additive') else 0

def predict(stream,cfg,features=None):
    if features is None:features={}
    if cfg['family']=='hybrid':
        cp,ch=predict(stream,cfg['cache'],features)
        dp,_=predict(stream,cfg['matrix'],features)
        return np.where(ch,cp,dp).astype(np.uint8),ch
    kind=cfg.get('kind','dense');dimension=cfg.get('dimension',64)
    key=(kind,dimension)
    if key not in features:features[key]=transform(stream.keys,kind,dimension)
    x=features[key]
    if cfg['family']=='cache':return explicit(stream.ops,x,stream.values,cfg['capacity'],cfg['threshold'],kind=='signature')
    return dense(stream.ops,x,stream.values,int(cfg['family']=='delta'),cfg['alpha'],cfg['beta'],cfg['presence']),None

def configurations(budget):
    out=[]
    for kind,dim,width in [('signature',32,4),('signature',64,8),('int8',64,64),('float32',64,256)]:
        capacity=(budget-4)//(width+5)
        if capacity<1:continue
        for threshold in [.3,.5,.7,.9,.99]:
            out.append(dict(family='cache',kind=kind,dimension=dim,capacity=capacity,threshold=threshold))
    d=budget//20
    for family in ['additive','delta']:
        for alpha in ([.95,.995,1.] if family=='additive' else [.995,1.]):
            for beta in ([1.] if family=='additive' else [.5,1.]):
                for presence in [.25,.5,.75]:
                    out.append(dict(family=family,dimension=d,alpha=alpha,beta=beta,presence=presence))
    # Familiar explicit+compressed control; retain deletions in the event cache.
    for fraction in [.25,.5,.75]:
        cap=int((budget*fraction-4)//69)
        if cap<1:continue
        cb=4+69*cap;d=(budget-cb)//20
        for threshold in [.5,.7,.9]:
            out.append(dict(family='hybrid',cache=dict(family='cache',kind='int8',dimension=64,capacity=cap,threshold=threshold),
                            matrix=dict(family='delta',dimension=d,alpha=1.,beta=1.,presence=.5)))
    assert all(config_bytes(c)<=budget for c in out)
    return out
