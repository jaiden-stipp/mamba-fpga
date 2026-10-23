import json,math
import numpy as np
from portable import Model,ROOT,ASSETS,torch
from experiment import prepare_data
from phase_reference import quantized_rotation, pack_vector, pack_golden
HW=ROOT/'hardware'

def hexfile(name,values,width):
    (HW/name).write_text('\n'.join(f'{int(x)&((1<<width)-1):0{(width+3)//4}x}' for x in values)+'\n')

prepare_data();val=np.load(ASSETS/'wikitext2-validation-tokens.npy')
incs=[];xs=[];ys=[]
for name in ['mamba3-siso-187m','mamba3-mimo-187m']:
    m=Model(name,'cuda:0')
    with torch.inference_mode():
        inp=torch.tensor([[128000]+val[:511].tolist()],device='cuda:0')
        _,_,tr=m.forward(inp,trace_layers=[0]);tr=tr[0]
        incs.extend(tr['increments'][0,:,0,0].cpu().numpy().astype(np.float64).tolist())
        kval=tr['k_unrot'][0,:,0,0].cpu().numpy()
        xs.extend(kval[:,0].tolist());ys.extend(kval[:,64 if m.is_mimo else 1].tolist())
    del m,tr;torch.cuda.empty_cache()
inc=np.array(incs);x=np.clip(np.rint(np.array(xs)*1024),-32768,32767).astype(np.int64);y=np.clip(np.rint(np.array(ys)*1024),-32768,32767).astype(np.int64)
# Deliberate signed-data, wrap, and address ties-even boundary cases, then the
# remaining trained-model increments and B-vector coordinates.
corners=[0,128,384,127,129,65535,32768,256,65280,1,65534,65000,50000,30000,10000,40000]
step16=np.rint(inc/(2*np.pi/65536)).astype(np.int64)%65536
step8=np.rint(inc/(2*np.pi/256)).astype(np.int64)%256
step16[:16]=corners;step8[:16]=np.rint(np.array(corners)/256).astype(np.int64)%256
x[:16]=[0,1,-1,32767,-32768,1000,-1000,12345,-12345,32767,-32768,7,-7,512,-512,2048]
y[:16]=x[:16][::-1]
cos=np.rint(np.cos(np.arange(256)*2*np.pi/256)*127).astype(np.int64)
sin=np.rint(np.sin(np.arange(256)*2*np.pi/256)*127).astype(np.int64)
hexfile('cos.hex',cos,8);hexfile('sin.hex',sin,8)
vectors=[];gold=[];p16=0;p8=0;maxrot=0
for i in range(1024):
    guard=quantized_rotation(p16,int(step16[i]),int(x[i]),int(y[i]),16,cos,sin)
    low=quantized_rotation(p8,int(step8[i]),int(x[i]),int(y[i]),8,cos,sin)
    p16,ag,rg,ig=guard;p8,al,rl,il=low
    maxrot=max(maxrot,abs(rg),abs(ig),abs(rl),abs(il))
    vectors.append(pack_vector(int(step16[i]),int(step8[i]),int(x[i]),int(y[i])))
    gold.append(pack_golden(guard,low))
hexfile('vectors.hex',vectors,56);hexfile('golden.hex',gold,116)
metadata={'vectors':1024,'corner_vectors':16,'remaining_vectors':'Validation layer-0 head-0 first rotary pair from official SISO/MIMO checkpoints; x/y Q5.10 inputs.',
          'phase_guard_final':p16,'phase_low_final':p8,'max_product_sum_abs':maxrot,'coefficient_scale':127,'core_scope':'Prequantized phase increments, phase accumulation, lookup and integer rotation; excludes tanh/dt, /127 output scaling, and recurrent SSM.'}
(HW/'vectors_metadata.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata,indent=2))
