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
check(len(registry['experiments'])==5,'Expected five documented studies')
check(len({x['id'] for x in registry['experiments']})==5,'Duplicate experiment IDs')
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
curated+=[ROOT/'literature/review-2026-10-22.md']+list((ROOT/'experiments/planned').glob('*.md'))
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
check(valid['gpu_serving_runs_completed']==8 and registry['experiments'][3]['status']=='completed_serving_pilot','E004 current status conflicts with GPU measurements')
check(audit['tensors_verified']==398 and audit['all_tensor_values_preserved'],'E004 conversion audit missing')
check(json.loads((serv/'client_validation.json').read_text())['first_token_not_buffered'],'SSE timing integration check failed')
cases=json.loads((serv/'workload.json').read_text())
check(len(cases)==8 and sum(len(c['queries']) for c in cases)==32,'E004 workload counts changed')
for case in cases:
    for q in case['queries']:check(q['ids'][:case['document_tokens']]==case['document_ids'] and len(q['ids'])==q['input_tokens'],'E004 document-prefix or input-length mismatch')
primary=[json.loads(s) for s in (serv/'raw_research_pilot_20261022.jsonl').read_text().splitlines()]
recovery=[json.loads(s) for s in (serv/'raw_research_recovery_20261022.jsonl').read_text().splitlines()]
selected=[r for r in primary if (r['format'],r['slots'])!=('q8_0',8)]+recovery
scored=[r for r in selected if r['phase']!='resident_load']
resident=[r for r in scored if r['phase'].startswith('resident_')]
check(len(primary)==219 and len(recovery)==48 and len(scored)==188,'E004 raw/selected counts changed')
check(len(resident)==124 and all(r['resident_cache_hit'] for r in resident),'E004 resident cache reuse not verified')
check(not any(r.get('error') or r['final'].get('truncated') for r in scored),'E004 recorded HTTP failures or truncation need reporting')
summary=json.loads((serv/'serving_summary.json').read_text())
check(summary['selected_configuration_count']==8 and summary['selected_scored_requests']==len(scored),'E004 summary counts conflict')
for row in summary['rows']:
    rs=[r for r in scored if r['format']==row['format'] and r['slots']==row['slots'] and (r['phase']=='cold')==(row['phase']=='cold')]
    check(row['correct']==sum(r['correct'] for r in rs) and row['qualified']==sum(r['qualified'] for r in rs),'E004 summary differs from raw quality/deadlines')
    tag=row['source_tag'];log=(serv/'logs'/f"{tag}_{row['format']}_c{row['slots']}.log").read_text(encoding='utf8')
    check('offloaded 37/37 layers to GPU' in log and f"K ({row['format']})" in log and f"V ({row['format']})" in log,'E004 placement or KV format not logged')
    check('CUDA0 KV buffer size' in log,'E004 GPU KV allocation absent')
for tag,digest in summary['raw_sha256'].items():check(hashlib.sha256((serv/f'raw_{tag}.jsonl').read_bytes()).hexdigest()==digest,'E004 raw digest mismatch')
guards=json.loads((serv/'configurations_research_pilot_20261022.json').read_text())
check(any(r['format']=='f16' and r['slots']==8 and r['status']=='not_run_capacity_guard' for r in guards),'E004 capacity guard should not be represented as observed OOM')
screen=ROOT/'experiments/005-read-granularity'
with (screen/'results.csv').open(newline='') as f:screen_rows=list(csv.DictReader(f))
check(len(screen_rows)==147,'E005 configuration count changed')
check(all(int(r['modeled_total_fetch_bytes'])>=int(r['logical_payload_read_bytes'])+int(r['metadata_bytes']) for r in screen_rows),'E005 fetch lower bound violated')
ss=json.loads((screen/'summary.json').read_text())
check(ss['evidence_type']=='analytical_address_model_not_hardware_measurement','E005 must not be called measured hardware traffic')
check(ss['source_sha256']==hashlib.sha256((screen/'screen.py').read_bytes()).hexdigest(),'E005 source hash changed since results')
sources=json.loads((ROOT/'literature/sources-2026-10-22.json').read_text())
check(sources['accessible_primary_records']==len(sources['sources'])==38,'Literature source register count changed')
check(len(list((ROOT/'experiments/planned').glob('P*.md')))==7,'Expected six research protocols plus baseline confirmation')
with (ROOT/'experiments/002-budgeted-memory/confirmation_results.csv').open(newline='') as f:check(len(list(csv.DictReader(f)))>0,'E002 confirmation table empty')
if errors:
    print('\n'.join(errors));print(f'FAILED: {len(errors)} errors / {checks} checks');sys.exit(1)
print(f'PASS: {checks} checks; five studies; navigation, syntax, manifests and recorded evidence consistent; E004 has eight GPU configurations and E005 is analytical.')
