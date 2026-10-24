"""Export actual layer-zero/head-zero inputs; no model download or training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'source'))
import portable

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assets',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=HERE/'traces')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    portable.ASSETS=args.assets
    token_path=args.assets/'wikitext2-validation.tokens.npy'
    if not token_path.is_file():
        candidates=list(args.assets.rglob('*validation*tokens.npy'))
        assert len(candidates)==1,candidates
        token_path=candidates[0]
    ids=np.load(token_path)[32768:32800].copy()
    for name in ('mamba3-siso-187m','mamba3-mimo-187m'):
        model=portable.Model(name,'cuda:0')
        with torch.inference_mode():
            _,_,traces=model.forward(torch.as_tensor(ids[None],device='cuda:0'),trace_layers=(0,))
        tr=traces[0]
        cpu=lambda x:x.detach().cpu().numpy()
        dt=tr['dt'][0,:,0];mix=tr['trap'][0,:,0];alpha=tr['adt'][0,:,0].exp()
        output=args.output/(name+'.npz')
        np.savez(output,q=cpu(tr['q_unrot'][0,:,0]),k=cpu(tr['k_unrot'][0,:,0]),
                 v=cpu(tr['v'][0,:,0,:,:8]),alpha=cpu(alpha),
                 beta=cpu((1-mix)*dt*alpha),gamma=cpu(mix*dt),
                 skip=cpu(tr['D'][0]).reshape(1),
                 increment=np.rint(cpu(tr['increments'][0,:,0]).astype(np.float64)*65536/(2*np.pi)).astype(np.int64)&65535,
                 token_ids=ids)
        meta={'checkpoint':name,'layer':0,'head':0,'P_channels':list(range(8)),
              'N':model.n,'rank':model.rank,'tokens':32,'validation_offset':32768,
              'checkpoint_sha256':hashlib.file_digest((args.assets/name/'pytorch_model.bin').open('rb'),'sha256').hexdigest(),
              'tokens_sha256':hashlib.file_digest(token_path.open('rb'),'sha256').hexdigest(),
              'trace_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
              'scope':'Frozen actual model inputs, not full-model quality or acceleration.'}
        output.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf8')
        print(json.dumps(meta),flush=True)
        del model,tr,traces
        torch.cuda.empty_cache()

if __name__=='__main__':main()
