"""Offline repository navigation, provenance and recorded-evidence checks."""
import csv,hashlib,json,re,sys
from pathlib import Path
from urllib.parse import unquote
ROOT=Path(__file__).resolve().parents[1]
errors=[];checks=0
def check(ok,message):
    global checks
    checks+=1
    if not ok:errors.append(message)
registry=json.loads((ROOT/'experiments/registry.json').read_text())
check(len(registry['experiments'])==4,'Expected four documented studies')
check(len({x['id'] for x in registry['experiments']})==4,'Duplicate experiment IDs')
for exp in registry['experiments']:
    folder=ROOT/exp['path']
    for filename in ['README.md',exp['report'],exp['evidence'],'file_manifest.json']:check((folder/filename).is_file(),f"{exp['id']}: missing {filename}")
    manifest=json.loads((folder/'file_manifest.json').read_text())
    for entry in manifest:
        p=(folder/entry['path']).resolve()
        check(p.is_relative_to(folder.resolve()) and p.is_file(),f'Invalid manifest path: {p}')
        if p.is_file():check(p.stat().st_size==entry['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256'],f'Manifest mismatch: {p.relative_to(ROOT)}')
    # Compile project-owned Python without executing GPU/training code or writing pyc.
    for p in folder.rglob('*.py'):
        if 'assets' in p.relative_to(folder).parts:continue
        try:compile(p.read_text(encoding='utf-8-sig'),str(p),'exec');check(True,str(p))
        except (SyntaxError,UnicodeError) as e:check(False,f'{p}: {e}')

# Check curated navigation, while retaining historical generated reports unchanged.
curated=[ROOT/'README.md',ROOT/'ROADMAP.md',ROOT/'PROVENANCE.md',ROOT/'experiments/README.md',ROOT/'experiments/TEMPLATE.md']
for section in ['architecture','ideas','docs']:curated+=list((ROOT/section).rglob('*.md'))
curated+=[ROOT/'literature/README.md']+[ROOT/x['path']/'README.md' for x in registry['experiments']]
for p in curated:
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text(encoding='utf8')):
        target=target.strip('<>').split('#',1)[0]
        if not target or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',target):continue
        local=(p.parent/unquote(target)).resolve()
        check(local.is_relative_to(ROOT) and local.exists(),f'Broken navigation: {p.relative_to(ROOT)} -> {target}')

phase=ROOT/'experiments/001-phase-precision'
hw=json.loads((phase/'hardware_results.json').read_text())
check(len(hw['runs'])==10 and all(r['vectors']==1024 and r['cycles']==1028 and r['guard_errors']==r['low_errors']==0 and r['pass']==1 for r in hw['runs']),'E001 board evidence changed or failed')
low=ROOT/'experiments/003-kv-compression'
main=[json.loads(s) for s in (low/'raw_results.jsonl').read_text().splitlines()]
diag=[json.loads(s) for s in (low/'raw_diagnostic8.jsonl').read_text().splitlines()]
check(len(main)==36 and len(diag)==12,'E003 run counts changed')
check(len({(r['id'],r['method']) for r in main})==36,'Duplicate E003 paired cases')
q4=[r for r in main if r['n']==16384 and r['method']=='hqq4']
check(sum(r['correct'] for r in q4)==4 and sum(r['strict_exact'] for r in q4)==1,'E003 quality summary conflicts with raw evidence')
for row in main+diag:
    n=row['n'];m=row['method']
    expected=row['cache_logical_values']*2 if m=='bf16' else int(2*36*8*128*((n-64)*(int(m[-1])/8+4/64)+95*2))
    check(row['final_cache']['gpu_bytes']==expected,'E003 independent storage formula mismatch')
serv=ROOT/'experiments/004-kv-serving'
valid=json.loads((serv/'validation.json').read_text());audit=json.loads((serv/'model_audit.json').read_text())
check(valid['gpu_serving_runs_completed']==0 and registry['experiments'][3]['status']=='prepared_integration_validated','E004 must remain prepared until GPU measurements exist')
check(audit['tensors_verified']==398 and audit['all_tensor_values_preserved'],'E004 conversion audit missing')
check(json.loads((serv/'client_validation.json').read_text())['first_token_not_buffered'],'SSE timing integration check failed')
cases=json.loads((serv/'workload.json').read_text())
check(len(cases)==8 and sum(len(c['queries']) for c in cases)==32,'E004 workload counts changed')
for case in cases:
    for q in case['queries']:check(q['ids'][:case['document_tokens']]==case['document_ids'] and len(q['ids'])==q['input_tokens'],'E004 document-prefix or input-length mismatch')
with (ROOT/'experiments/002-budgeted-memory/confirmation_results.csv').open(newline='') as f:check(len(list(csv.DictReader(f)))>0,'E002 confirmation table empty')
if errors:
    print('\n'.join(errors));print(f'FAILED: {len(errors)} errors / {checks} checks');sys.exit(1)
print(f'PASS: {checks} checks; four studies; navigation, Python syntax, manifests and recorded evidence consistent; E004 has no GPU serving results.')
