"""Deterministic long-document tasks; no model outputs used to select cases."""
import argparse,json,random,os
from pathlib import Path
from transformers import AutoTokenizer
R=Path(__file__).resolve().parent
B=R.parent.parent

def create(tok,seed,n=16384):
    rng=random.Random(seed)
    names=[f'PROJECT_{chr(65+j)}{chr(65+j)}{chr(65+j)}' for j in range(8)]
    normal={k:str(rng.randrange(100000,999999)) for k in names}
    old={k:str(rng.randrange(100000,999999)) for k in names}
    latest={k:str(rng.randrange(100000,999999)) for k in names}
    constants={k:str(rng.randrange(100000,999999)) for k in names}
    entries=[]
    for j,k in enumerate(names):
        entries.append(((j+.5)/8,f'\nFACT: access code for {k} is {normal[k]}.\n'))
        entries.append(((j+.5)/16,f'\nUPDATE: recovery code for {k} is now {old[k]}.\n'))
        entries.append((.5+(j+.5)/16,f'\nUPDATE: recovery code for {k} is now {latest[k]}.\n'))
        entries.append(((j+.75)/8,f'\n```python\ndef lookup_{k}():\n    return {constants[k]}\n```\n'))
    queries=[]
    for kind in range(4):
        order=names.copy();rng.shuffle(order)
        if kind==0:
            keys=order[:1];q=f'What is the access code for {keys[0]}?';expected=[normal[k] for k in keys]
        elif kind==1:
            keys=order;q='Extract access codes in this order: '+', '.join(keys)+'.';expected=[normal[k] for k in keys]
        elif kind==2:
            keys=order[:1];q=f'What is the latest recovery code for {keys[0]} after applying the updates in document order?';expected=[latest[k] for k in keys]
        else:
            keys=order[:1];q=f'What literal integer does the Python function lookup_{keys[0]} return?';expected=[constants[k] for k in keys]
        queries.append({'kind':['single','multi8','update','code'][kind],'question':q,'expected':expected})
    marker='DOCUMENT_MARKER';frame='QUESTION_MARKER'
    user=f'Document nonce {seed}. Answer only from the document below. FACT access codes, UPDATE recovery codes, and Python function return values are separate fields. Later UPDATE records supersede earlier records for the same project.\n<records>\n{marker}\n</records>\n{frame}\nReturn only a JSON array of decimal strings in requested order, without markdown or explanation.'
    text=tok.apply_chat_template([{'role':'user','content':user}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    pre,after=text.split(marker)
    prefix=tok.encode(pre,add_special_tokens=False)
    suffixes=[tok.encode(after.replace(frame,q['question']),add_special_tokens=False) for q in queries]
    extra=[(pos,tok.encode(s,add_special_tokens=False)) for pos,s in sorted(entries)]
    size=n-len(prefix)-max(map(len,suffixes))-sum(len(x) for _,x in extra)
    filler=''.join(f'Archive row {i:04d}: warehouse juniper-{rng.randrange(1000,9999)} stores batch {rng.randrange(100000,999999)} for routine shipping.\n' for i in range(3000))
    fill=tok.encode(filler,add_special_tokens=False)[:size]
    assert len(fill)==size
    doc=[];last=0
    for pos,ids in extra:
        at=int(size*pos);doc+=fill[last:at]+ids;last=at
    doc+=fill[last:]
    document_ids=prefix+doc
    for q,suffix in zip(queries,suffixes):
        q['ids']=document_ids+suffix;q['input_tokens']=len(q['ids'])
        assert n-128<=q['input_tokens']<=n
    return {'seed':seed,'document_ids':document_ids,'document_tokens':len(document_ids),'queries':queries}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='workload.json');args=ap.parse_args()
    tok=AutoTokenizer.from_pretrained(Path(os.environ.get('QWEN3_MODEL_DIR', B/'experiments/003-kv-compression/assets/Qwen3-4B')),local_files_only=True)
    cases=[create(tok,3001+97*i) for i in range(8)]
    (R/args.out).write_text(json.dumps(cases),encoding='utf8')
    print('Created',len(cases),'documents,',sum(len(x['queries']) for x in cases),'queries, tokens',min(q['input_tokens'] for c in cases for q in c['queries']),max(q['input_tokens'] for c in cases for q in c['queries']))
if __name__=='__main__':main()
