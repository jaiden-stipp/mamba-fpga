"""Executable whole-model integration boundary for the recurrent RTL tile.

This is a slow eager Torch arithmetic emulator, NOT a hardware driver or a
performance implementation. Dense layers remain in the portable checkpoint
model. Batch/head/P axes vectorize copies of the scalar tile's operation order.
"""
import math
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'source'))
from portable import rms

class RecurrentBackend:
    def __init__(self):self.cache=None

    @torch.inference_mode()
    def step(self,q,k,v,alpha,beta,gamma,skip,increment_words):
        # q/k B,H,R,N; projected v B,H,R,P; per-head alpha/beta/gamma.
        batch,heads,rank,n=q.shape;p=v.shape[-1];angles=increment_words.shape[-1]
        if self.cache is None:
            phase=torch.zeros_like(increment_words,dtype=torch.int64)
            state=torch.zeros((batch,heads,p,n),device=q.device,dtype=torch.float16)
            pk=torch.zeros_like(k);pv=torch.zeros_like(v)
        else:phase,state,pk,pv=self.cache
        phase=(phase+increment_words.to(torch.int64))&65535
        address=torch.round(phase.double()/256).to(torch.int64)&255
        # Generate the exact same FP32 constants as the checked-in RTL ROM.
        table=torch.arange(256,device=q.device,dtype=torch.float64)*math.tau/256
        cs=(torch.round(table.cos()*127).float()/127)[address][:,:,None]
        sn=(torch.round(table.sin()*127).float()/127)[address][:,:,None]
        def rotate(x):
            out=x.clone()
            if rank>1:u=x[...,:angles];w=x[...,n//2:n//2+angles]
            else:u=x[...,:2*angles:2];w=x[...,1:2*angles:2]
            real=u*cs+w*(-sn);imag=u*sn+w*cs
            if rank>1:out[...,:angles]=real;out[...,n//2:n//2+angles]=imag
            else:out[...,:2*angles:2]=real;out[...,1:2*angles:2]=imag
            return out
        q,k=rotate(q),rotate(k)
        current=torch.zeros_like(state,dtype=torch.float32);previous=current.clone()
        for r in range(rank):
            current=current+k[:,:,r,None,:]*v[:,:,r,:,None]
            previous=previous+pk[:,:,r,None,:]*pv[:,:,r,:,None]
        updated=(alpha[:,:,None,None]*state.float()+beta[:,:,None,None]*previous)+gamma[:,:,None,None]*current
        y=torch.zeros_like(v)
        for i in range(n):y=y+updated[:,:,:,i][:,:,None,:]*q[:,:,:,i][:,:,:,None]
        y=y+skip[None,:,None,None]*v
        self.cache=(phase,updated.half(),k,v.clone())
        return y

class HybridModel:
    """Full checkpoint decoder with an explicit replaceable recurrence backend."""
    def __init__(self,model,backend_factory=RecurrentBackend):
        self.m=model;self.backends=[backend_factory() for _ in range(model.nl)]

    @torch.inference_mode()
    def step(self,ids):
        m=self.m;batch=ids.shape[0]
        residual=m.w('backbone.embedding.weight')[ids]
        for layer,backend in enumerate(self.backends):
            pre=f'backbone.layers.{layer}';prefix=pre+'.mixer'
            u=rms(residual,m.w(pre+'.norm.weight'))
            proj=m.linear(u,prefix+'.in_proj')
            z,raw_v,k,q,dd_dt,dd_a,trap,ang=proj.split(
                [m.inner,m.inner,m.n*m.g*m.rank,m.n*m.g*m.rank,m.h,m.h,m.h,m.a],dim=-1)
            dt=F.softplus(dd_dt+m.w(prefix+'.dt_bias'))
            av=-torch.where(dd_a>=0,1+dd_a,1/(1-dd_a.clamp_max(0)))
            alpha=(av.clamp_max(-m.cfg.get('A_floor',1e-4))*dt).exp()
            mix=trap.sigmoid();beta=(1-mix)*dt*alpha;gamma=mix*dt
            q=rms(q.reshape(batch,m.rank,m.g,m.n),m.w(prefix+'.C_norm.weight'))
            k=rms(k.reshape(batch,m.rank,m.g,m.n),m.w(prefix+'.B_norm.weight'))
            q=q.repeat_interleave(m.h//m.g,2).permute(0,2,1,3)+m.w(prefix+'.C_bias')[None]
            k=k.repeat_interleave(m.h//m.g,2).permute(0,2,1,3)+m.w(prefix+'.B_bias')[None]
            v=raw_v.reshape(batch,m.h,m.p)[:,:,None];z=z.reshape(batch,m.h,m.p)[:,:,None]
            if m.is_mimo:
                v=v*m.w(prefix+'.mimo_x')[None];z=z*m.w(prefix+'.mimo_z')[None]
            inc=(torch.tanh(ang.float())[:,None]*dt[...,None]*math.pi).double()
            words=torch.round(inc*65536/math.tau).to(torch.int64)&65535
            y=backend.step(q,k,v,alpha,beta,gamma,m.w(prefix+'.D'),words)
            if m.cfg['is_outproj_norm']:
                y=rms(y,m.w(prefix+'.norm.weight').reshape(m.h,1,m.p))
            y=y*F.silu(z)
            y=(y*m.w(prefix+'.mimo_o')[None]).sum(-2) if m.is_mimo else y.squeeze(-2)
            residual=residual+m.linear(y.reshape(batch,m.inner),prefix+'.out_proj')
            u=rms(residual,m.w(pre+'.norm2.weight'))
            a,gate=m.linear(u,pre+'.mlp.fc1').chunk(2,-1)
            residual=residual+m.linear(a*F.silu(gate),pre+'.mlp.fc2')
        return rms(residual,m.w('backbone.norm_f.weight'))
