import hashlib,json,zipfile,os
from pathlib import Path
import requests
R=Path(os.environ.get('LLAMA_CPP_DIR', Path(__file__).resolve().parents[2]/'tools/llama.cpp'))
R.mkdir(parents=True,exist_ok=True)
tag='b11751'
s=requests.Session()
j=s.get(f'https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}',timeout=30).json()
ref=s.get(f'https://api.github.com/repos/ggml-org/llama.cpp/git/ref/tags/{tag}',timeout=30).json()
sha=ref['object']['sha']
if ref['object']['type']=='tag': sha=s.get(ref['object']['url'],timeout=30).json()['object']['sha']
receipt={'tag':tag,'commit':sha,'published_at':j['published_at'],'downloads':[]}
def fetch(url,dest,expected=None):
    if not dest.exists():
        print('Downloading',dest.name,flush=True)
        res=s.get(url,stream=True,timeout=60);res.raise_for_status()
        temp=dest.with_suffix(dest.suffix+'.part')
        with temp.open('wb') as f:
            for c in res.iter_content(1024*1024):f.write(c)
        temp.rename(dest)
    h=hashlib.sha256(dest.read_bytes()).hexdigest()
    if expected and expected.startswith('sha256:'): assert h==expected[7:]
    receipt['downloads'].append({'url':url,'file':dest.name,'bytes':dest.stat().st_size,'sha256':h,'release_digest':expected})
def unpack(archive,dest,strip=False):
    dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            parts=Path(item.filename).parts[1:] if strip else Path(item.filename).parts
            if not parts: continue
            target=dest.joinpath(*parts).resolve()
            assert target.is_relative_to(dest.resolve())
            if item.is_dir(): target.mkdir(parents=True,exist_ok=True)
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(item) as src,target.open('wb') as out:
                    import shutil;shutil.copyfileobj(src,out)
for name in [f'llama-{tag}-bin-win-cuda-12.4-x64.zip','cudart-llama-bin-win-cuda-12.4-x64.zip']:
    a=next(a for a in j['assets'] if a['name']==name)
    fetch(a['browser_download_url'],R/name,a.get('digest'))
    unpack(R/name,R/'bin')
source=R/f'llama-{sha}.zip'
fetch(f'https://api.github.com/repos/ggml-org/llama.cpp/zipball/{sha}',source)
unpack(source,R/'source',True)
(R/'engine_manifest.json').write_text(json.dumps(receipt,indent=2))
print('Ready',tag,sha,flush=True)
