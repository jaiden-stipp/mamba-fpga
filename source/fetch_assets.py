import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
ASSETS.mkdir(exist_ok=True)
REV = 'e9594ce1c732d97440f0332fdc43170a2294dbfa'
MODEL_REVISIONS={'mamba3-siso-187m':'6792c27c00f3bb41506db1066dcd1c51bb0f4b02',
                 'mamba3-mimo-187m':'8fd6e9eb7b795f2e15d7f6353251d0137980c43e'}

def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as f:
        return json.load(f)

def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print('EXISTS', path.name, path.stat().st_size, flush=True)
        return
    temp = path.with_suffix(path.suffix + '.part')
    t = time.time()
    with urllib.request.urlopen(url, timeout=60) as response, temp.open('wb') as out:
        while chunk := response.read(4 * 1024 * 1024):
            out.write(chunk)
    temp.replace(path)
    print('FETCH', path.name, path.stat().st_size, round(time.time()-t, 2), flush=True)

tree = get_json(f'https://api.github.com/repos/state-spaces/mamba/git/trees/{REV}?recursive=1')
paths = [x['path'] for x in tree['tree'] if x['type']=='blob' and
         ('mamba3' in x['path'] or x['path'] in ['LICENSE','mamba_ssm/models/mixer_seq_simple.py','mamba_ssm/modules/block.py','mamba_ssm/modules/mlp.py','mamba_ssm/models/config_mamba.py'])]
(ASSETS/'upstream_tree.json').write_text(json.dumps(paths, indent=2))
for path in paths:
    if path.endswith('.py') or path=='LICENSE':
        fetch(f'https://raw.githubusercontent.com/state-spaces/mamba/{REV}/{path}', ASSETS/'upstream'/path)

manifest = {'source_revision': REV, 'models': {}, 'downloads': []}
for name in ['mamba3-siso-187m','mamba3-mimo-187m']:
    repo = f'state-spaces/{name}'
    meta = {'sha':MODEL_REVISIONS[name]}
    manifest['models'][name] = {'repo':repo, 'revision':meta['sha']}
    for fn in ['README.md','config.json','pytorch_model.bin']:
        fetch(f'https://huggingface.co/{repo}/resolve/{meta["sha"]}/{fn}', ASSETS/name/fn)

tokenizer_repo='NousResearch/Meta-Llama-3.1-8B'
tokenizer_meta={'sha':'1f47e50cdbe801ad8a5174156ec3a0655108fb9f'}
manifest['tokenizer']={'repo':tokenizer_repo,'revision':tokenizer_meta['sha'], 'reason':'Public mirror of the Llama-3.1 tokenizer specified by the official Mamba-3 model card.'}
for fn in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json']:
    fetch(f'https://huggingface.co/{tokenizer_repo}/resolve/{tokenizer_meta["sha"]}/{fn}', ASSETS/'tokenizer-llama31'/fn)
for split in ['test','validation']:
    fetch(f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/{split}-00000-of-00001.parquet', ASSETS/f'wikitext2-{split}.parquet')
for path in ASSETS.rglob('*'):
    if path.is_file() and path.name!='manifest.json':
        h=hashlib.sha256()
        with path.open('rb') as f:
            while block:=f.read(8*1024*1024):h.update(block)
        manifest['downloads'].append({'path':str(path.relative_to(ASSETS)), 'bytes':path.stat().st_size,'sha256':h.hexdigest()})
(ASSETS/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('COMPLETE',flush=True)
