"""Package source/recorded evidence; never bundle external model/runtime files."""
import argparse,hashlib,json,zipfile
from pathlib import Path
from refresh_manifests import ROOT,refresh
ap=argparse.ArgumentParser();ap.add_argument('experiment',help='Experiment directory or E001/E002/E003/E004');args=ap.parse_args()
registry=json.loads((ROOT/'experiments/registry.json').read_text())
row=next((x for x in registry['experiments'] if x['id']==args.experiment),None)
folder=(ROOT/row['path']).resolve() if row else Path(args.experiment).resolve()
assert folder.is_relative_to((ROOT/'experiments').resolve()) and (folder/'README.md').is_file()
refresh(folder);manifest=json.loads((folder/'file_manifest.json').read_text())
out=ROOT/'.artifacts';out.mkdir(exist_ok=True);dest=out/(folder.name+'.zip')
with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
    for entry in manifest:
        p=folder/entry['path'];z.write(p,folder.name+'/'+entry['path'])
    z.write(folder/'file_manifest.json',folder.name+'/file_manifest.json')
with zipfile.ZipFile(dest) as z:
    assert z.testzip() is None
    for entry in manifest:assert hashlib.sha256(z.read(folder.name+'/'+entry['path'])).hexdigest()==entry['sha256']
print('Verified',dest,len(manifest),'manifest entries')
