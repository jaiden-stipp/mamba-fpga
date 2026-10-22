import os, json
from pathlib import Path
os.environ['HF_HUB_DISABLE_XET']='1'
from huggingface_hub import snapshot_download
root=Path(__file__).resolve().parent
repo='Qwen/Qwen3-4B'
revision='1cfa9a7208912126459214e8b04321603b3df60c'
print('Pinned revision:',revision,flush=True)
path=snapshot_download(repo,revision=revision,local_dir=root/'assets'/'Qwen3-4B',allow_patterns=['*.json','*.safetensors','*.txt','LICENSE','README.md'],max_workers=4)
(root/'assets'/'model_manifest.json').write_text(json.dumps({'repo':repo,'revision':revision,'local_path':str(path)},indent=2))
print('Download complete',flush=True)
