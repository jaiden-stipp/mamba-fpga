"""Portable FP32 Mamba-3 research evaluator; no custom CUDA extensions.

Equations and tensor conventions checked against pinned upstream reference tests.
Checkpoint BF16 parameters are converted to FP32; this is not native-kernel parity.
"""
import ast
import json
import math
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

ROOT=Path(__file__).resolve().parent
ASSETS=ROOT/'assets'
TAU=2*math.pi
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.set_float32_matmul_precision('highest')
torch.set_num_threads(8)

def phase_angles(increments, mode='reference', initial=None, mask=None):
    # increments B,L,H,A; accumulator arithmetic is performed in float64 so that
    # fixed-point design error is not confounded with host summation roundoff.
    inc=increments.double()
    if initial is None: initial=torch.zeros_like(inc[:,0])
    if mode.startswith('fixed'):
        bits=int(mode[5:]); step=TAU/(2**bits)
        inc=torch.round(inc/step)*step
    elif mode.startswith('guard'):
        abits=int(mode.split('_')[0][5:]); step=TAU/(2**abits)
        inc=torch.round(inc/step)*step
    elif mode=='mixed':
        assert mask is not None
        lo=torch.round(inc/(TAU/256))*(TAU/256)
        hi=torch.round(inc/(TAU/65536))*(TAU/65536)
        inc=torch.where(mask[None,None],hi,lo)
    phi=inc.cumsum(dim=1)+initial[:,None]
    final=phi[:,-1].remainder(TAU)
    phi=phi.remainder(TAU)
    if mode.startswith('lookup'):
        bits=int(mode[6:]); step=TAU/(2**bits)
        phi=torch.round(phi/step)*step
    elif mode.startswith('guard'):
        bits=int(mode.split('_')[1]); step=TAU/(2**bits)
        phi=torch.round(phi/step)*step
    elif mode=='offset':
        phi=phi+0.731
        final=(final+0.731).remainder(TAU)
    return phi.float(),final

def rotate(x,phi,is_mimo=False,trig_bits=0):
    # x B,L,H,R,N. SISO uses adjacent pairs; MIMO pairs i with i+N/2.
    n=x.shape[-1]; a=phi.shape[-1]
    cs=phi.cos().unsqueeze(-2); sn=phi.sin().unsqueeze(-2)
    if trig_bits:
        qmax=2**(trig_bits-1)-1
        cs=torch.round(cs*qmax)/qmax
        sn=torch.round(sn*qmax)/qmax
    y=x.clone()
    if is_mimo:
        u=x[...,:a];v=x[...,n//2:n//2+a]
        y[...,:a]=u*cs-v*sn
        y[...,n//2:n//2+a]=u*sn+v*cs
    else:
        u=x[...,:2*a:2];v=x[...,1:2*a:2]
        y[...,:2*a:2]=u*cs-v*sn
        y[...,1:2*a:2]=u*sn+v*cs
    return y

def scan(q,k,v,adt,dt,trap,increments,*,is_mimo=False,chunk_size=64,
         mode='reference',states=None,mask=None,return_trace=False):
    """Exact-arithmetic-equivalent chunk evaluation of the three-term update.

    q,k B,L,H,R,N; v B,L,H,R,P; adt,dt,trap B,L,H.
    trap is already sigmoid transformed; no D residual or output gate here.
    State is stored as B,H,N,P. No per-token state quantization in this path.
    """
    b,l,h,r,n=q.shape;p=v.shape[-1]
    assert l%chunk_size==0
    nc=l//chunk_size;t=chunk_size
    phi,angle_end=phase_angles(increments,mode, None if states is None else states[0],mask)
    trig_bits=int(mode.rsplit('trig',1)[1]) if 'trig' in mode else 0
    q=rotate(q,phi,is_mimo,trig_bits);k=rotate(k,phi,is_mimo,trig_bits)
    kp=torch.cat([torch.zeros_like(k[:,:1]) if states is None else states[2][:,None],k[:,:-1]],dim=1)
    vp=torch.cat([torch.zeros_like(v[:,:1]) if states is None else states[3][:,None],v[:,:-1]],dim=1)
    alpha=adt.exp(); beta=(1-trap)*dt*alpha; gamma=trap*dt
    wk=torch.cat([k*gamma[...,None,None],kp*beta[...,None,None]],dim=-2)
    wv=torch.cat([v,vp],dim=-2)
    qc=rearrange(q,'b (c t) h r n -> b h c (t r) n',t=t)
    kc=rearrange(wk,'b (c t) h r n -> b h c (t r) n',t=t)
    vc=rearrange(wv,'b (c t) h r p -> b h c (t r) p',t=t)
    acs=rearrange(adt,'b (c t) h -> b h c t',t=t).cumsum(-1)
    diff=acs[..., :,None]-acs[...,None,:]
    causal=torch.ones(t,t,device=q.device,dtype=torch.bool).tril()
    decay=diff.masked_fill(~causal,-torch.inf).exp().repeat_interleave(r,-2).repeat_interleave(2*r,-1)
    out=((qc@kc.transpose(-1,-2))*decay)@vc
    lastdecay=(acs[...,-1,None]-acs).exp().repeat_interleave(2*r,-1)
    writes=(kc*lastdecay[...,None]).transpose(-1,-2)@vc
    state=torch.zeros((b,h,n,p),device=q.device,dtype=torch.float32) if states is None else states[1].clone()
    incoming=torch.empty_like(writes)
    for c in range(nc):
        incoming[:,:,c]=state
        state=acs[:,:,c,-1,None,None].exp()*state+writes[:,:,c]
    qdecay=acs.exp().repeat_interleave(r,-1)
    out=out+(qc*qdecay[...,None])@incoming
    out=rearrange(out,'b h c (t r) p -> b (c t) h r p',r=r)
    st=(angle_end,state,k[:,-1],v[:,-1])
    trace=None
    if return_trace:
        trace={'q':q,'k':k,'v':v,'adt':adt,'dt':dt,'trap':trap,'increments':increments,'phi':phi,'out':out}
    return out,st,trace

def rms(x,w,eps=1e-5):
    return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)*w

class Model:
    def __init__(self,name,device='cuda:0'):
        self.name=name;self.device=device
        self.config=json.loads((ASSETS/name/'config.json').read_text())
        self.cfg=self.config['ssm_cfg'];self.is_mimo=self.cfg['is_mimo']
        self.rank=self.cfg.get('mimo_rank',1) if self.is_mimo else 1
        self.sd=torch.load(ASSETS/name/'pytorch_model.bin',map_location='cpu',weights_only=True)
        self.sd={k:v.to(device=device,dtype=torch.float32) for k,v in self.sd.items()}
        self.d=self.config['d_model'];self.inner=int(self.d*self.cfg['expand'])
        self.p=self.cfg['headdim'];self.h=self.inner//self.p;self.n=self.cfg['d_state']
        self.a=int(self.n*self.cfg['rope_fraction'])//2
        self.g=self.cfg['ngroups'];self.nl=self.config['n_layer']
        self.used=set()
    def w(self,key):
        self.used.add(key);return self.sd[key]
    def linear(self,x,key):
        out=F.linear(x,self.w(key+'.weight'))
        if key+'.bias' in self.sd:out=out+self.w(key+'.bias')
        return out
    @torch.inference_mode()
    def forward(self,ids,mode='reference',states=None,masks=None,trace_layers=()):
        x=self.w('backbone.embedding.weight')[ids]; residual=x
        all_states=[];traces={}
        for i in range(self.nl):
            pre=f'backbone.layers.{i}';m=pre+'.mixer'
            u=rms(residual,self.w(pre+'.norm.weight'))
            proj=self.linear(u,m+'.in_proj')
            z,v,k,q,dd_dt,dd_a,trap,ang=proj.split([self.inner,self.inner,self.n*self.g*self.rank,self.n*self.g*self.rank,self.h,self.h,self.h,self.a],dim=-1)
            b,l,_=z.shape
            dt=F.softplus(dd_dt+self.w(m+'.dt_bias'))
            av=-torch.where(dd_a>=0,1+dd_a,1/(1-dd_a.clamp_max(0)))
            av=av.clamp_max(-self.cfg.get('A_floor',1e-4));adt=av*dt
            q=rms(q.reshape(b,l,self.rank,self.g,self.n),self.w(m+'.C_norm.weight'))
            k=rms(k.reshape(b,l,self.rank,self.g,self.n),self.w(m+'.B_norm.weight'))
            q=q.repeat_interleave(self.h//self.g,dim=3).permute(0,1,3,2,4)+self.w(m+'.C_bias')[None,None]
            k=k.repeat_interleave(self.h//self.g,dim=3).permute(0,1,3,2,4)+self.w(m+'.B_bias')[None,None]
            vv=v.reshape(b,l,self.h,self.p)
            zz=z.reshape(b,l,self.h,self.p)
            if self.is_mimo:
                vv=vv.unsqueeze(-2)*self.w(m+'.mimo_x')[None,None]
                zz=zz.unsqueeze(-2)*self.w(m+'.mimo_z')[None,None]
            else:
                vv=vv.unsqueeze(-2);zz=zz.unsqueeze(-2)
            increments=torch.tanh(ang.float())[:,:,None,:]*dt[:,:,:,None]*math.pi
            # Use a shared 32-token evaluation chunk for SISO/MIMO; chunk size is
            # independently checked against upstream references and continuation.
            y,st,tr=scan(q,k,vv,adt,dt,trap.sigmoid(),increments,is_mimo=self.is_mimo,
                         chunk_size=32,mode=mode,states=None if states is None else states[i],
                         mask=None if masks is None else masks[i],return_trace=i in trace_layers)
            if tr is not None:
                tr.update({'q_unrot':q,'k_unrot':k,'z':zz,'D':self.w(m+'.D')})
                traces[i]=tr
            all_states.append(st)
            y=y+self.w(m+'.D')[None,None,:,None,None]*vv
            if self.cfg['is_outproj_norm']:
                y=rms(y,self.w(m+'.norm.weight').reshape(self.h,1,self.p))
            y=y*F.silu(zz)
            if self.is_mimo:y=(y*self.w(m+'.mimo_o')[None,None]).sum(-2)
            else:y=y.squeeze(-2)
            residual=residual+self.linear(y.reshape(b,l,self.inner),m+'.out_proj')
            u=rms(residual,self.w(pre+'.norm2.weight'))
            a,g=self.linear(u,pre+'.mlp.fc1').chunk(2,-1)
            residual=residual+self.linear(a*F.silu(g),pre+'.mlp.fc2')
        return rms(residual,self.w('backbone.norm_f.weight')),all_states,traces
    @torch.inference_mode()
    def loss(self,hidden,targets):
        losses=[]
        weight=self.w('lm_head.weight')
        for start in range(0,hidden.shape[1],128):
            logits=F.linear(hidden[:,start:start+128],weight)
            nll=F.cross_entropy(logits.flatten(0,1),targets[:,start:start+128].flatten(),reduction='none')
            losses.append(nll.reshape(targets.shape[0],-1))
        return torch.cat(losses,1)

def load_upstream_refs():
    namespace={'torch':torch,'F':F,'math':math,'Optional':Optional,'Tuple':Tuple,'Tensor':torch.Tensor,'rearrange':rearrange,'repeat':repeat}
    paths=['tests/ops/triton/test_mamba3_siso.py','tests/ops/tilelang/test_mamba3_mimo.py']
    wanted={'_segsum','_pad_zeros','mamba3_siso_step_ref','mamba3_siso_fwd_ref','mamba3_MIMO_step_ref','mamba3_MIMO_chunk_ref'}
    for path in paths:
        tree=ast.parse((ASSETS/'upstream'/path).read_text())
        defs=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name in wanted]
        mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+defs,type_ignores=[])
        exec(compile(ast.fix_missing_locations(mod),str(path),'exec'),namespace)
    return namespace

@torch.inference_mode()
def validate(device='cuda:0'):
    ref=load_upstream_refs();results=[]
    for r in [1,4]:
        torch.manual_seed(810+r)
        b,l,h,n,p,a=1,64,2,16,8,4
        q=torch.randn(b,l,h,r,n,device=device);k=torch.randn_like(q)
        v=torch.randn(b,l,h,r,p,device=device)
        adt=-torch.rand(b,l,h,device=device)*0.15;dt=torch.rand(b,l,h,device=device)*0.1
        trap=torch.randn(b,l,h,device=device);angles=torch.randn(b,l,h,a,device=device)
        inc=angles.tanh()*dt[...,None]*math.pi
        out,st,_=scan(q,k,v,adt,dt,trap.sigmoid(),inc,is_mimo=r>1,chunk_size=16)
        if r==1:
            upstream,ust=ref['mamba3_siso_step_ref'](q.squeeze(-2),k.squeeze(-2),v.squeeze(-2),adt.transpose(1,2),dt.transpose(1,2),trap.transpose(1,2),torch.zeros(h,n,device=device),torch.zeros(h,n,device=device),angles)
            reference=upstream.unsqueeze(-2)
        else:
            # Chunk reference matches shipped MIMO split-half rotation convention.
            da=rearrange(adt,'b (c t) h -> b h c t',t=16).cumsum(-1)
            rev=da[...,-1,None]-da
            reference,_,_=ref['mamba3_MIMO_chunk_ref'](
                q.permute(0,1,3,2,4),k.permute(0,1,3,2,4),v[:,:,:,0],
                torch.zeros(h,r,n,device=device),torch.zeros(h,r,n,device=device),
                torch.ones(h,r,p,device=device),None,None,None,
                inc.double().cumsum(1).float(),rearrange(da,'b h c t -> b h (c t)'),rearrange(rev,'b h c t -> b h (c t)'),
                dt.transpose(1,2),trap.transpose(1,2),None,chunk_size=16,rotary_dim_divisor=4,
                contract_mimo_out=False)
            # MIMO reference uses a common raw V with learned rank up-projection.
            # Recompute our scan using the same rank-expanded V for parity.
            v=v[:,:,:,0:1].expand(-1,-1,-1,r,-1).contiguous()
            out,st,_=scan(q,k,v,adt,dt,trap.sigmoid(),inc,is_mimo=True,chunk_size=16)
            reference=reference.permute(0,1,3,2,4)
        err=(out-reference).abs().max().item()
        rel=((out-reference).square().mean()/reference.square().mean()).sqrt().item()
        assert err<3e-4,(r,err,rel)
        # Compare chunk widths and continuation, all four carried state objects.
        alt,_,_=scan(q,k,v,adt,dt,trap.sigmoid(),inc,is_mimo=r>1,chunk_size=32)
        left,sleft,_=scan(q[:,:32],k[:,:32],v[:,:32],adt[:,:32],dt[:,:32],trap[:,:32].sigmoid(),inc[:,:32],is_mimo=r>1,chunk_size=16)
        right,sright,_=scan(q[:,32:],k[:,32:],v[:,32:],adt[:,32:],dt[:,32:],trap[:,32:].sigmoid(),inc[:,32:],is_mimo=r>1,chunk_size=16,states=sleft)
        spliterr=(torch.cat([left,right],1)-out).abs().max().item()
        chunkerr=(alt-out).abs().max().item()
        offset,_,_=scan(q,k,v,adt,dt,trap.sigmoid(),inc,is_mimo=r>1,chunk_size=16,mode='offset')
        offerr=(offset-out).abs().max().item()
        assert max(spliterr,chunkerr,offerr)<3e-4
        results.append({'rank':r,'upstream_max_abs':err,'upstream_relative_rms':rel,'chunk_width_max_abs':chunkerr,'continuation_max_abs':spliterr,'constant_phase_offset_max_abs':offerr})
    return results

if __name__=='__main__':
    print(json.dumps(validate(),indent=2))
