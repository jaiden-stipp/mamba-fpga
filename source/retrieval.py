"""Small controlled one-word recall probe; not a standard retrieval benchmark."""
import argparse,json,time
import numpy as np
from tokenizers import Tokenizer
from portable import Model,ASSETS,ROOT,torch
from experiment import RESULTS,prepare_data

def run(name,device,pilot=False):
    tok=prepare_data();m=Model(name,device)
    filler=np.load(ASSETS/'wikitext2-test-tokens.npy')
    words=['apple','river','music','paper','green','horse','water','stone']
    candidates=[]
    for word in words:
        ids=tok.encode(' '+word,add_special_tokens=False).ids
        assert len(ids)==1,(word,ids)
        candidates.append(ids[0])
    candidate_t=torch.tensor(candidates,device=device)
    query=tok.encode('\nQuestion: What is the secret word?\nAnswer: The secret word is',add_special_tokens=False).ids
    path=RESULTS/f'{name}-retrieval{"-pilot" if pilot else ""}.json'
    rows=json.loads(path.read_text()) if path.exists() else []
    lengths=[512] if pilot else [512,2048,8192]
    modes=['reference'] if pilot else ['reference','fixed8','fixed10','guard16_8','guard16_8_trig8']
    for length in lengths:
      for mode in modes:
        if any(x['length']==length and x['mode']==mode for x in rows):continue
        wall=time.time();cases=[]
        for wi,word in enumerate(words):
          for fraction in ([0.5] if pilot else [0.1,0.5,0.9]):
            needle=tok.encode(f'\nThe secret word is {word}. Remember the secret word.\n',add_special_tokens=False).ids
            fstart=204800+wi*1024
            body=filler[fstart:fstart+length-1-len(query)-len(needle)].tolist()
            at=int(len(body)*fraction)
            ids=[128000]+body[:at]+needle+body[at:]+query
            assert len(ids)==length
            with torch.inference_mode():
              hidden,_,_=m.forward(torch.tensor([ids],device=device),mode=mode)
              logits=hidden[0,-1]@m.sd['lm_head.weight'].T
              eight=logits[candidate_t]
              winner=int(eight.argmax());greedy=int(logits.argmax())
              cases.append({'word':word,'fraction':fraction,'constrained_correct':winner==wi,'greedy_correct':greedy==candidates[wi],
                            'target_prob_eight':float(eight.softmax(0)[wi]),'target_rank':int((logits>logits[candidates[wi]]).sum())+1})
        row={'checkpoint':name,'length':length,'mode':mode,'cases':cases,'n':len(cases),
             'constrained_accuracy':sum(c['constrained_correct'] for c in cases)/len(cases),
             'greedy_accuracy':sum(c['greedy_correct'] for c in cases)/len(cases),
             'mean_target_prob_eight':float(np.mean([c['target_prob_eight'] for c in cases])),
             'seconds':round(time.time()-wall,2)}
        rows.append(row);path.write_text(json.dumps(rows,indent=2))
        print(json.dumps({k:v for k,v in row.items() if k!='cases'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--pilot',action='store_true');a=p.parse_args();run(a.name,a.device,a.pilot)
