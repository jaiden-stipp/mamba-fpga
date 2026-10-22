"""Streaming SET/DELETE/QUERY data. No evaluator state is passed to memories."""
from dataclasses import dataclass
import numpy as np

ABSENT = 16
KEY_DIM = 64

@dataclass
class Stream:
    ops: np.ndarray
    keys: np.ndarray
    values: np.ndarray
    truth: np.ndarray
    previous: np.ndarray
    edits: np.ndarray
    age: np.ndarray
    entity: np.ndarray  # evaluator / separate symbolic track ONLY
    score: np.ndarray

REGIMES = {
    'clean': {'noise': 0., 'cluster': False},
    'noisy': {'noise': .10, 'cluster': False},
    'similar': {'noise': .05, 'cluster': True},
}

def generate(seed, length, regime, stress=False, entities=128):
    cfg=REGIMES[regime]; rng=np.random.default_rng(seed)
    proto=rng.normal(size=(entities,KEY_DIM)).astype(np.float32)
    proto/=np.linalg.norm(proto,axis=1,keepdims=True)
    if cfg['cluster']:
        centers=rng.normal(size=(16,KEY_DIM)).astype(np.float32)
        centers/=np.linalg.norm(centers,axis=1,keepdims=True)
        proto=.90*centers[np.arange(entities)%16]+.44*proto
        proto/=np.linalg.norm(proto,axis=1,keepdims=True)
    ops=np.zeros(length,np.int8); entity=np.zeros(length,np.uint8)
    values=np.full(length,ABSENT,np.uint8); truth=values.copy(); previous=values.copy()
    edits=np.zeros(length,np.int32); age=np.zeros(length,np.int32)
    state=np.full(entities,ABSENT,np.uint8); old=state.copy()
    counts=np.zeros(entities,np.int32); last=np.zeros(entities,np.int32)
    for t in range(length):
        if t<entities: e=t; op=0
        else:
            op=int(rng.choice(3,p=[.60,.10,.30]))
            # Stress shifts the active hot set every 4K events. Queries are a
            # mixture of hot entities and uniform cold entities, not future-aware.
            if stress and rng.random()<(.85 if op!=2 else .60):
                e=int((rng.integers(32)+32*((t//4096)%4))%entities)
            else:e=int(rng.integers(entities))
        if op!=2:
            old[e]=state[e]
            v=ABSENT if op==1 else int(rng.integers(16))
            if op==0 and v==int(state[e]):v=(v+1)%16
            state[e]=v;counts[e]+=1;last[e]=t;values[t]=v
        ops[t]=op;entity[t]=e;truth[t]=state[e];previous[t]=old[e]
        edits[t]=counts[e];age[t]=t-last[e]
    observed=proto[entity]+cfg['noise']*rng.normal(size=(length,KEY_DIM)).astype(np.float32)
    observed/=np.linalg.norm(observed,axis=1,keepdims=True)
    score=(ops==2)&(np.arange(length)>=1024)
    return Stream(ops,observed.astype(np.float32),values,truth,previous,edits,age,entity,score)

def metrics(stream,pred):
    masks={'all':stream.score,'active':stream.score&(stream.truth!=ABSENT),
           'deleted':stream.score&(stream.truth==ABSENT),
           'updated':stream.score&(stream.edits>=2)&(stream.truth!=ABSENT),
           'old_active':stream.score&(stream.truth!=ABSENT)&(stream.age>=512)}
    out={}
    for name,m in masks.items():
        out[name+'_n']=int(m.sum());out[name+'_accuracy']=float(np.mean(pred[m]==stream.truth[m])) if m.any() else None
    stale=stream.score&(stream.edits>=2)&(stream.previous!=ABSENT)&(stream.previous!=stream.truth)
    out['stale_n']=int(stale.sum())
    out['stale_rate']=float(np.mean(pred[stale]==stream.previous[stale])) if stale.any() else None
    out['predicted_absent_rate']=float(np.mean(pred[stream.score]==ABSENT))
    return out
