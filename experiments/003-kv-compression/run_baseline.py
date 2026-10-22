"""Paired real-model KV compression feasibility experiment; see PROTOCOL.md."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse, gc, hashlib, json, random, re, time, platform
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache, QuantizedCache

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent.parent
ASSETS=Path(os.environ.get('QWEN3_MODEL_DIR', ROOT/'assets'/'Qwen3-4B'))
METHODS=['bf16','hqq4','hqq2']
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False

def cache_for(method, config):
    if method=='bf16': return DynamicCache(config=config)
    return QuantizedCache(backend='hqq',config=config,nbits=int(method[-1]),axis_key=1,axis_value=1,q_group_size=64,residual_length=128)

def storage_stats(cache, detail=False):
    seen=set(); layouts=[]
    def walk(x, name):
        if torch.is_tensor(x):
            s=x.untyped_storage(); ident=(str(x.device),s.data_ptr())
            if s.nbytes() and ident not in seen:
                seen.add(ident); layouts.append({'name':name,'shape':list(x.shape),'dtype':str(x.dtype),'bytes':s.nbytes(),'device':str(x.device)})
        elif isinstance(x,dict):
            for k,v in x.items(): walk(v,f'{name}.{k}')
        elif isinstance(x,(list,tuple)):
            for k,v in enumerate(x): walk(v,f'{name}.{k}')
    for i,layer in enumerate(cache.layers):
        for k,v in vars(layer).items(): walk(v,f'layer{i}.{k}')
    return {'gpu_bytes':sum(x['bytes'] for x in layouts if x['device'].startswith('cuda')),
            'cpu_tensor_bytes':sum(x['bytes'] for x in layouts if x['device']=='cpu'),
            'first_layer_layout':[x for x in layouts if x['name'].startswith('layer0.')] if detail else []}

def make_case(tok,n,pos,seed):
    rng=random.Random(seed)
    target=str(rng.randrange(100000,999999))
    filler=''.join(f'Record {i:05d}: site cedar-{i} has access code {rng.randrange(100000,999999)}.\n' for i in range(n//10))
    marker='DOCUMENT_MARKER'
    user=f'Read the reference records and answer exactly from the document.\n<records>\n{marker}\n</records>\nQuestion: What is the six-digit access code for project ORCHID? Output only the six digits.'
    rendered=tok.apply_chat_template([{'role':'user','content':user}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    pre,post=rendered.split(marker)
    p=tok.encode(pre,add_special_tokens=False); s=tok.encode(post,add_special_tokens=False)
    needle=tok.encode(f'\nThe six-digit access code for project ORCHID is {target}.\n',add_special_tokens=False)
    size=n-len(p)-len(s)-len(needle)
    doc=tok.encode(filler,add_special_tokens=False)[:size]
    assert len(doc)==size
    at=int(len(doc)*pos)
    ids=p+doc[:at]+needle+doc[at:]+s
    assert len(ids)==n and len(s)<64
    return {'id':f'n{n}_p{pos}_s{seed}','n':n,'position':pos,'seed':seed,'target':target,'ids':ids,'question_suffix_tokens':64,'needle_document_token':at}

def sync(): torch.cuda.synchronize()

@torch.inference_mode()
def run(model,tok,case,method):
    gc.collect(); torch.cuda.empty_cache(); sync()
    initial=torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
    ids=torch.tensor([case['ids']],device='cuda')
    # Common BF16 chunked document prefill; compress once at document boundary.
    # This prevents repeated requantization and bounds long-prefill scratch.
    cache=DynamicCache(config=model.config)
    t=time.perf_counter()
    for start in range(0,case['n']-64,2048):
        out=model(input_ids=ids[:,start:min(start+2048,case['n']-64)],past_key_values=cache,use_cache=True,logits_to_keep=1)
        del out
    sync(); document_prefill=time.perf_counter()-t
    t=time.perf_counter()
    if method!='bf16':
        packed=cache_for(method,model.config)
        for i,layer in enumerate(cache.layers):
            packed.update(layer.keys,layer.values,i)
            layer.keys=None; layer.values=None
        cache=packed
    sync(); compression=time.perf_counter()-t
    prefill=document_prefill+compression
    pref_stats=storage_stats(cache)
    prefill_peak=torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t=time.perf_counter()
    out=model(input_ids=ids[:,-64:],past_key_values=cache,use_cache=True,logits_to_keep=1)
    token=out.logits[:,-1].argmax(-1,keepdim=True)
    sync(); question=time.perf_counter()-t
    generated=[int(token.item())]
    del out
    t=time.perf_counter()
    for _ in range(31):
        out=model(input_ids=token,past_key_values=cache,use_cache=True,logits_to_keep=1)
        token=out.logits[:,-1].argmax(-1,keepdim=True)
        generated.append(int(token.item()))
        del out
    sync(); decode=time.perf_counter()-t
    stats=storage_stats(cache,True)
    eos=tok.eos_token_id
    answer_ids=generated[:generated.index(eos)] if eos in generated else generated
    answer=tok.decode(answer_ids,skip_special_tokens=True).strip()
    match=re.search(r'(?<!\d)\d{6}(?!\d)',answer)
    correct=bool(match and match.group()==case['target'])
    row={k:v for k,v in case.items() if k!='ids'}
    row.update(method=method,answer=answer,generated_ids=generated,correct=correct,
      strict_exact=answer==case['target'],prefill_s=prefill,document_prefill_s=document_prefill,compression_s=compression,question_s=question,decode31_s=decode,
      decode_ms_per_token=1000*decode/31,total_s=prefill+question+decode,
      initial_gpu_bytes=initial,prefill_peak_allocated_bytes=prefill_peak,
      decode_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
      final_allocated_bytes=torch.cuda.memory_allocated(),prefill_cache=pref_stats,final_cache=stats,
      cache_logical_values=2*model.config.num_hidden_layers*model.config.num_key_value_heads*model.config.head_dim*(case['n']+31))
    row['effective_bits_per_value']=8*stats['gpu_bytes']/row['cache_logical_values']
    del ids,cache,token
    return row

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--methods',default=','.join(METHODS)); ap.add_argument('--tag',default=''); args=ap.parse_args()
    methods=args.methods.split(',')
    assert all(m in ['bf16','hqq2','hqq4','hqq8'] for m in methods)
    ROOT.mkdir(exist_ok=True)
    print('Loading pinned local Qwen3-4B',flush=True)
    tok=AutoTokenizer.from_pretrained(ASSETS,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(ASSETS,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    env={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(), 'transformers':__import__('transformers').__version__, 'model_manifest':json.loads((ASSETS.parent/'model_manifest.json').read_text()), 'model_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),'argv':__import__('sys').argv}
    (ROOT/(f'environment_{args.tag}.json' if args.tag else 'environment.json')).write_text(json.dumps(env,indent=2))
    for method in methods:
        warm=run(model,tok,make_case(tok,512,.5,999),method)
        print('warmup',method,warm['answer'],flush=True)
    cases=[make_case(tok,n,p,seed) for n in ([4096] if args.smoke else [4096,16384]) for p in ([.5] if args.smoke else [.1,.5,.9]) for seed in ([101] if args.smoke else [101,202])]
    (ROOT/(f'prompts_{args.tag}.json' if args.tag else ('smoke_prompts.json' if args.smoke else 'prompts.json'))).write_text(json.dumps(cases))
    dest=ROOT/(f'raw_{args.tag}.jsonl' if args.tag else ('smoke.jsonl' if args.smoke else 'raw_results.jsonl'))
    with dest.open('w',encoding='utf8') as f:
        for i,case in enumerate(cases):
            shift=i%len(methods)
            order=methods[shift:]+methods[:shift]
            for method in order:
                row=run(model,tok,case,method)
                f.write(json.dumps(row)+'\n'); f.flush()
                print(case['id'],method,'correct',row['correct'],'ms/token',round(row['decode_ms_per_token'],2),'bits',round(row['effective_bits_per_value'],3),flush=True)
    print('FINISHED',str(dest),flush=True)

if __name__=='__main__': main()
