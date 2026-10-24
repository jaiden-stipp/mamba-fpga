"""Replay the existing E001 tools with one additional rank-1 state-update sum boundary.

The numerical oracle, primitive test, rank-4 datapath, traces and constraints
are unchanged. Scratch libraries/databases stay under ignored .artifacts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
PRIOR=BASE/'kv_product_stage'
REPO=BASE.parents[2]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')

def prepare():
    inputs=[BASE/'regression.py',BASE/'synthesize.py',*sorted((BASE/'rtl').glob('*')),
            *sorted((HERE/'rtl').glob('*'))]
    identity=hashlib.sha256(b''.join(p.read_bytes() for p in inputs)).hexdigest()[:12]
    stage=REPO/'.artifacts'/f'e001-update-sum-stage-{identity}'/'project/experiments/001-phase-precision/pipeline'
    (stage/'rtl').mkdir(parents=True,exist_ok=True)
    for p in (BASE/'rtl').iterdir():shutil.copy2(p,stage/'rtl'/p.name)
    for p in (HERE/'rtl').iterdir():shutil.copy2(p,stage/'rtl'/p.name)
    for name in ('regression.py','synthesize.py'):shutil.copy2(BASE/name,stage/name)
    # synthesize.py consumes these ROM images; the simulator regenerates its own.
    (stage/'evidence/rank1').mkdir(parents=True,exist_ok=True)
    for name in ('cos_fp32.hex','sin_fp32.hex'):
        shutil.copy2(BASE/'evidence/rank1'/name,stage/'evidence/rank1'/name)
    write_json(HERE/'inputs.json',{
        'baseline_commit':'5ef89a9', 'scratch_id':identity,
        'baseline_files':{str(p.relative_to(BASE)).replace('\\','/'):digest(p)
                          for p in inputs if p.is_relative_to(BASE) and not p.is_relative_to(HERE)},
        'variant_files':{str(p.relative_to(HERE)).replace('\\','/'):digest(p)
                         for p in inputs if p.is_relative_to(HERE)},
        'runner_sha256':digest(Path(__file__)),
        'prior_variant_files':{str(p.relative_to(PRIOR)).replace('\\','/'):digest(p) for p in sorted((PRIOR/'rtl').glob('*'))},
        'traces':{p.name:digest(p) for p in sorted((BASE/'traces').glob('*.npz'))}})
    return stage

def execute(phase):
    stage=prepare()
    output=HERE/('evidence' if phase=='regression' else 'synthesis')
    output.mkdir(exist_ok=True)
    if phase=='regression':
        command=[sys.executable,str(stage/'regression.py'),'--traces',str(BASE/'traces'),
                 '--output',str(output)]
    else:
        command=[sys.executable,str(stage/'synthesize.py'),'--rank','1']
    write_json(output/'execution.json',{'status':'RUNNING','phase':phase,'command':command})
    with (output/'runner.log').open('w',encoding='utf8') as log:
        result=subprocess.run(command,cwd=stage,stdout=log,stderr=subprocess.STDOUT,timeout=3900)
    if phase=='synthesis':
        for p in (stage/'synthesis/ram_banked/rank1').iterdir():
            if p.is_file():shutil.copy2(p,output/p.name)
    else:
        mismatches=[]
        for log in sorted(output.glob('rank*/simulation.log')):
            for line in log.read_text().splitlines():
                if re.match(r'^# MISMATCH',line):
                    mismatches.append({'log':str(log.relative_to(HERE)).replace('\\','/'),'line':line})
                    break
        write_json(output/'first_mismatch.json',{
            'status':'FAIL' if mismatches else ('NONE' if result.returncode==0 else 'RUN_FAILED'),
            'first_by_case':mismatches,'oracle_changed':False,
            'note':'Intentional CORRUPT negative controls are retained separately, not candidate mismatches.'})
    write_json(output/'execution.json',{'status':'PASS' if result.returncode==0 else 'FAIL',
        'phase':phase,'command':command,'exit_code':result.returncode})
    print((output/'runner.log').read_text()[-7000:])
    return result.returncode

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('regression','synthesis'))
    sys.exit(execute(parser.parse_args().phase))
