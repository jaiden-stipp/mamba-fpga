import json,platform,subprocess,time
import numpy as np
from portable import torch,Model,phase_angles,validate,ASSETS,ROOT
from experiment import RESULTS,prepare_data

@torch.inference_mode()
def run():
    result={'recurrence':validate(),'models':[]}
    prepare_data();ids=np.load(ASSETS/'wikitext2-validation-tokens.npy')
    for name in ['mamba3-siso-187m','mamba3-mimo-187m']:
        model=Model(name,'cuda:0')
        inp=torch.tensor([[128000]+ids[:127].tolist()],device='cuda:0')
        full,_,_=model.forward(inp)
        left,st,_=model.forward(inp[:,:64])
        right,_,_=model.forward(inp[:,64:],states=st)
        delta=torch.cat([left,right],1)-full
        logits_full=full@model.sd['lm_head.weight'].T
        logits_split=torch.cat([left,right],1)@model.sd['lm_head.weight'].T
        logerr=float((logits_full-logits_split).abs().max())
        err=float(delta.abs().max())
        assert err<5e-4 and logerr<2e-3,(name,err,logerr)
        result['models'].append({'checkpoint':name,'hidden_continuation_max_abs':err,'logit_continuation_max_abs':logerr,'all_checkpoint_keys_used':len(set(model.sd)-model.used)==0})
        # lm_head used directly above; count it as used for the audit.
        result['models'][-1]['unused_non_lm_head_keys']=sorted(set(model.sd)-model.used-{'lm_head.weight'})
        result['models'][-1]['all_checkpoint_keys_used']=not result['models'][-1]['unused_non_lm_head_keys']
        del model,full,left,right,st,logits_full,logits_split
        torch.cuda.empty_cache()
    torch.manual_seed(20261019)
    inc=torch.randn(1,65536,2,4,device='cuda:0')*0.03
    for bits in [6,8,10,12,16]:
        phi,_=phase_angles(inc,f'fixed{bits}')
        codes=torch.round(inc.double()/(2*np.pi/2**bits)).to(torch.int64)
        exact=(codes.cumsum(1)%(2**bits)).double()*(2*np.pi/2**bits)
        chord=(phi.double().sin()-exact.sin()).abs().max().item()
        assert chord<1e-5,(bits,chord)
    result['integer_accumulator_equivalence_65536_steps']=True
    result['environment']={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'cuda_runtime':torch.version.cuda,'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], 'tf32':torch.backends.cuda.matmul.allow_tf32}
    (RESULTS/'validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':run()
