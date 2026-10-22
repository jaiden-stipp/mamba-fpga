"""Verify converted model tensors preserve the pinned source values."""
import collections,hashlib,json,sys,os
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
R=Path(__file__).resolve().parent;B=R.parent.parent
sys.path.insert(0,str(B/'tools/llama.cpp/source/gguf-py'))
import gguf
model=Path(os.environ.get('QWEN3_GGUF', B/'assets/models/Qwen3-4B-BF16.gguf'))
reader=gguf.GGUFReader(model)
tensors={t.name:t for t in reader.tensors}
mapping=gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN3,36)
checked=[]
for shard in sorted((Path(os.environ.get('QWEN3_MODEL_DIR', B/'experiments/003-kv-compression/assets/Qwen3-4B'))).glob('model-*.safetensors')):
    with safe_open(shard,framework='pt',device='cpu') as f:
        for name in f.keys():
            mapped=mapping.get_name(name,try_suffixes=('.weight','.bias'))
            assert mapped in tensors,(name,mapped)
            t=tensors[mapped];original=f.get_tensor(name)
            if t.tensor_type==gguf.GGMLQuantizationType.BF16:
                actual=t.data.view(np.uint16).reshape(-1)
                expected=original.view(torch.uint16).numpy().reshape(-1)
            elif t.tensor_type==gguf.GGMLQuantizationType.F32:
                actual=t.data.view(np.float32).reshape(-1)
                expected=original.float().numpy().reshape(-1)
            else:raise AssertionError(('Unexpected quantization',mapped,t.tensor_type))
            assert np.array_equal(actual,expected),mapped
            checked.append({'source':name,'gguf':mapped,'type':t.tensor_type.name,'elements':original.numel()})
assert len(checked)==len(tensors)
h=hashlib.sha256()
with model.open('rb') as f:
    while part:=f.read(8*1024*1024):h.update(part)
receipt={'file':model.name,'bytes':model.stat().st_size,'sha256':h.hexdigest(),'tensors_verified':len(checked),'elements_verified':sum(x['elements'] for x in checked),'types':dict(collections.Counter(x['type'] for x in checked)),'all_tensor_values_preserved':True,'note':'BF16 values compared bitwise; F32-promoted vectors compared exactly. This audits conversion, not engine numerical parity.','tensors':checked}
(R/'model_audit.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps({k:v for k,v in receipt.items() if k!='tensors'},indent=2))
