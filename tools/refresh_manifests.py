"""Refresh portable per-experiment file-integrity manifests (standard library)."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def include(p):
    if not p.is_file() or '__pycache__' in p.parts or '.cache' in p.parts or p.name=='file_manifest.json':return False
    if p.suffix.lower() in ['.safetensors','.gguf','.pyc','.part','.incomplete']:return False
    if 'assets' in p.parts and (p.suffix.lower() in ['.bin','.parquet','.npy','.pt','.pth'] or p.name=='tokenizer.json'):return False
    return True
def refresh(root):
    entries=[{'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(root.rglob('*')) if include(p)]
    (root/'file_manifest.json').write_text(json.dumps(entries,indent=2),encoding='utf8')
    return len(entries)
if __name__=='__main__':
    registry=json.loads((ROOT/'experiments/registry.json').read_text())
    for exp in registry['experiments']:print(exp['id'],refresh(ROOT/exp['path']),'files')
