"""Frozen-activation recurrence tests; these are NOT full-model perplexities."""
import argparse,json,math,time
import numpy as np
from numba import njit,prange,set_num_threads
from portable import Model,ROOT,ASSETS,torch,rotate,phase_angles
from experiment import prepare_data,RESULTS
set_num_threads(8)

@njit(cache=True,parallel=True)
def recurrence(q,k,v,adt,dt,trap,quant_mode,scales):
    # T,H,R,N; V T,H,R,P. Quantize only S, after each token, before readout.
    tmax,hmax,rmax,nmax=q.shape;pmax=v.shape[-1]
    out=np.zeros((tmax,hmax,rmax,pmax),np.float32)
    maxima=np.zeros((hmax,pmax),np.float32)
    clips=np.zeros((hmax,pmax),np.int64)
    for row in prange(hmax*pmax):
        h=row//pmax;p=row%pmax
        st=np.zeros(nmax,np.float32)
        rng=np.uint64(1381+row*101)
        for t in range(tmax):
            aa=np.float32(math.exp(adt[t,h]));bb=np.float32((np.float32(1)-trap[t,h])*dt[t,h]*aa);gg=np.float32(trap[t,h]*dt[t,h])
            maxval=np.float32(0)
            for n in range(nmax):
                now=np.float32(0);prev=np.float32(0)
                for r in range(rmax):
                    now=np.float32(now+np.float32(k[t,h,r,n]*v[t,h,r,p]))
                    if t>0:prev=np.float32(prev+np.float32(k[t-1,h,r,n]*v[t-1,h,r,p]))
                st[n]=np.float32(np.float32(aa*st[n]+bb*prev)+gg*now)
                maxval=max(maxval,abs(st[n]))
            maxima[h,p]=max(maxima[h,p],maxval)
            if quant_mode==1 or quant_mode==2: # FP16 normal/subnormal emulation
                for n in range(nmax):
                    x=st[n];ax=abs(x)
                    if ax>65504:
                        clips[h,p]+=1;st[n]=np.float32(65504 if x>0 else -65504);continue
                    if ax==0:continue
                    step=2.0**max(-24,math.floor(math.log2(ax))-10)
                    u=x/step
                    if quant_mode==1:z=np.rint(u)
                    else:
                        # Per-row reproducible LCG; no shared RNG in parallel loop.
                        rng=rng*np.uint64(6364136223846793005)+np.uint64(1442695040888963407)
                        uniform=float(rng>>np.uint64(11))*(1.0/9007199254740992.0)
                        lo=math.floor(u);z=lo+(1 if uniform<u-lo else 0)
                    st[n]=np.float32(z*step)
            elif quant_mode>=3:
                qmax=127 if quant_mode in (3,4) else 32767
                scale=maxval/qmax if quant_mode in (3,5) else scales[h,p]/qmax
                if scale>0:
                    for n in range(nmax):
                        z=np.rint(st[n]/scale)
                        if z>qmax or z<-qmax:clips[h,p]+=1
                        st[n]=np.float32(min(qmax,max(-qmax,z))*scale)
            for r in range(rmax):
                acc=np.float32(0)
                for n in range(nmax):acc=np.float32(acc+st[n]*q[t,h,r,n])
                out[t,h,r,p]=acc
    return out,maxima,clips

def trace_numpy(tr,model,mode='reference'):
    heads=[0,7,15,23]
    phi,_=phase_angles(tr['increments'],mode)
    q=rotate(tr['q_unrot'],phi,model.is_mimo)[0,:,heads]
    k=rotate(tr['k_unrot'],phi,model.is_mimo)[0,:,heads]
    vals=[q,k,tr['v'][0,:,heads,:,:8],tr['adt'][0,:,heads],tr['dt'][0,:,heads],tr['trap'][0,:,heads]]
    return [np.ascontiguousarray(x.cpu().numpy(),dtype=np.float32) for x in vals]

def run(name,device):
    prepare_data();m=Model(name,device);test=np.load(ASSETS/'wikitext2-test-tokens.npy');val=np.load(ASSETS/'wikitext2-validation-tokens.npy')
    path=RESULTS/f'{name}-state.json';rows=[]
    # Independent frozen trajectories: calibration validation[0:2048], held-out
    # test[196608:204800]. Four heads, all N=128, first eight P channels.
    with torch.inference_mode():
        vinput=torch.tensor([[128000]+val[:2047].tolist()],device=device)
        _,_,cal=m.forward(vinput,trace_layers=[0,5,11])
        tinput=torch.tensor([[128000]+test[196608:196608+8191].tolist()],device=device)
        _,_,traces=m.forward(tinput,trace_layers=[0,5,11])
    for layer in [0,5,11]:
        wall=time.time();cargs=trace_numpy(cal[layer],m);args=trace_numpy(traces[layer],m)
        _,scales,_=recurrence(*cargs,0,np.ones((4,8),np.float32))
        ref,_,_=recurrence(*args,0,scales)
        gpu=traces[layer]['out'][0,:,[0,7,15,23],:,:8].cpu().numpy()
        parity=float(np.sqrt(np.mean((ref-gpu)**2)/np.mean(gpu**2)))
        assert parity<5e-5,parity
        for label,code,phase in [('FP32',0,'reference'),('FP16_RTN',1,'reference'),('FP16_SR',2,'reference'),('INT8_dynamic',3,'reference'),('INT8_static',4,'reference'),('INT16_dynamic',5,'reference'),('INT16_static',6,'reference'),('INT8_dynamic_fixed8',3,'fixed8'),('INT8_dynamic_guard16_8',3,'guard16_8')]:
            aa=args if phase=='reference' else trace_numpy(traces[layer],m,phase)
            pred,_,clips=recurrence(*aa,code,scales)
            delta=pred-ref
            row={'checkpoint':name,'layer':layer,'label':label,'tokens':8192,'heads':[0,7,15,23],'value_channels':list(range(8)),
                 'relative_rms':float(np.sqrt(np.mean(delta.astype(np.float64)**2)/np.mean(ref.astype(np.float64)**2))),
                 'max_abs':float(np.abs(delta).max()),'clips':int(clips.sum()),'reference_parity_relative_rms':parity,'seconds_layer_so_far':round(time.time()-wall,2)}
            rows.append(row);path.write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
    print('COMPLETE',name,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();run(a.name,a.device)
