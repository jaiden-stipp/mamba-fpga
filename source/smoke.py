import json,time
from portable import Model,ASSETS,torch,validate
from tokenizers import Tokenizer
tok=Tokenizer.from_file(str(ASSETS/'tokenizer-llama31/tokenizer.json'))
text='Mamba is a state space model for sequence modeling. The model maintains a recurrent state and updates it for each new token. '
ids=tok.encode(text*20,add_special_tokens=False).ids
print('vocab',tok.get_vocab_size(),'tokens',ids[:12],flush=True)
for name in ['mamba3-siso-187m','mamba3-mimo-187m']:
 t=time.time();m=Model(name)
 print(name,'keys',len(m.sd),'unique parameters',sum(v.numel() for k,v in m.sd.items() if k!='lm_head.weight'),flush=True)
 x=torch.tensor([[128000]+ids[:255]],device='cuda:0')
 with torch.inference_mode():
  y,states,tr=m.forward(x,trace_layers=[0])
  loss=m.loss(y,torch.tensor([ids[:256]],device='cuda:0'))
  logits=y[:,-1]@m.sd['lm_head.weight'].T
  print('mean nll',loss.mean().item(),'seconds',time.time()-t,'next tokens',[(tok.decode([i]),float(logits[0,i])) for i in logits[0].topk(5).indices],flush=True)
  print('unused',[k for k in m.sd if k not in m.used],flush=True)
 del m,y,states,tr
 torch.cuda.empty_cache()
