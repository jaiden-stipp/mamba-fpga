"""Audit the isolated experiment and regenerate its before/after summary."""
import csv
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
PRIOR=BASE/'trig_output_stage'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())

def paths(text):
    found=[]
    for line in text.splitlines():
        cells=[x.strip() for x in line.split(';')][1:-1]
        if len(cells)==8 and re.fullmatch(r'-?\d+\.\d+',cells[0]) and cells[3:5]==['clk','clk']:
            found.append({'slack_ns':float(cells[0]),'from':cells[1],'to':cells[2],
                          'data_delay_ns':float(cells[7])})
    return found

def timing(folder):
    fit=(folder/'pipeline.fit.summary').read_text()
    sta=(folder/'pipeline.sta.rpt').read_text()
    summary=(folder/'pipeline.sta.summary').read_text()
    def number(key):
        return int(re.search(r'^'+re.escape(key)+r'\s*:\s*([\d,]+)',fit,re.M)[1].replace(',',''))
    slacks=[];kind=''
    for line in summary.splitlines():
        if line.startswith('Type'):kind=line.split(':',1)[1]
        if line.startswith('Slack') and re.search(r'\bSetup\b',kind):slacks.append(float(line.split(':',1)[1]))
    corners=[{'fmax_MHz':float(m[1]),'restricted_fmax_MHz':float(m[2])} for m in re.finditer(
        r';\s*(\d+(?:\.\d+)?) MHz\s*;\s*(\d+(?:\.\d+)?) MHz\s*;\s*clk\s*;',sta)]
    assert slacks and corners
    return {'worst_setup_slack_ns':min(slacks),
        'minimum_restricted_fmax_MHz':min(x['restricted_fmax_MHz'] for x in corners),
        'fmax_corners':corners,'fit_logic_elements':number('Total logic elements'),
        'fit_registers':number('Total registers'),
        'fit_multiplier_9bit_elements':number('Embedded Multiplier 9-bit elements'),
        'virtual_pins':number('Total virtual pins'),
        'worst_registered_path':min(paths(sta),key=lambda x:x['slack_ns']),
        'meets_50MHz_setup':min(slacks)>=0}

def main():
    inputs=read(HERE/'inputs.json')
    for name,sha in inputs['baseline_files'].items():assert digest(BASE/name)==sha,name
    for name,sha in inputs['variant_files'].items():assert digest(HERE/name)==sha,name
    for name,sha in inputs['prior_variant_files'].items():assert digest(PRIOR/name)==sha,name
    assert digest(HERE/'run_experiment.py')==inputs['runner_sha256']
    for name,sha in inputs['traces'].items():assert digest(BASE/'traces'/name)==sha,name
    actual=read(HERE/'evidence/summary.json');original=read(PRIOR/'evidence/summary.json')
    assert actual['status']=='PASS' and actual['seed']==original['seed']==0xe001f032
    assert actual['environment']==original['environment']
    for phase,folder in [('regression','evidence'),('synthesis','synthesis')]:
        execution=read(HERE/folder/'execution.json')
        assert execution['phase']==phase and execution['status']=='PASS' and execution['exit_code']==0
    assert actual['arithmetic_vectors']==96112 and actual['arithmetic_result_comparisons']==384448
    assert '# PASS\n' in (HERE/'evidence/arithmetic.log').read_text()
    assert read(HERE/'evidence/first_mismatch.json')['status']=='NONE'
    for name,sha in actual['source_sha256'].items():
        location=HERE/name if (HERE/name).is_file() else BASE/name
        assert digest(location)==sha,name
    cases=[]
    for current,old in zip(actual['cases'],original['cases']):
        rank=current['rank'];folder=HERE/'evidence'/f'rank{rank}';baseline=PRIOR/'evidence'/f'rank{rank}'
        # Identical stimuli and golden files prove the numerical oracle was not loosened.
        for name in ('commands','values','bc','states','outputs','cos_fp32','sin_fp32'):
            assert digest(folder/(name+'.hex'))==digest(baseline/(name+'.hex')),(rank,name)
        expected_states=[int(x,16) for x in (folder/'states.hex').read_text().splitlines()]
        expected_outputs=[int(x,16) for x in (folder/'outputs.hex').read_text().splitlines()]
        state_count=output_count=0
        with (folder/'rtl_results.csv').open() as a,(baseline/'rtl_results.csv').open() as b:
            rows=list(csv.DictReader(a));old_rows=list(csv.DictReader(b))
        assert len(rows)==len(old_rows)==current['state_writes']+current['outputs']
        for row,old_row in zip(rows,old_rows):
            assert all(row[k]==old_row[k] for k in ('kind','token','index','wide','packed'))
            token,index=int(row['token']),int(row['index'])
            if row['kind']=='state':
                assert token==state_count//1024 and index==state_count%1024
                assert (int(row['wide'],16)<<16)|int(row['packed'],16)==expected_states[state_count]
                state_count+=1
            else:
                assert token==output_count//(8*rank) and index==output_count%(8*rank)
                assert int(row['wide'],16)==expected_outputs[output_count]
                output_count+=1
        log=(folder/'simulation.log').read_text();negative=(folder/'negative_control.log').read_text()
        assert '# PASS\n' in log and not re.search(r'^# (FAIL|MISMATCH)',log,re.M)
        assert '# FAIL\n' in negative and 'errors=1 ' in negative and '# PASS' not in negative
        assert state_count==current['state_writes']==65536 and output_count==current['outputs']==512*rank
        assert current['counters']['errors']==current['quantized_reference_max_bit_error']==0
        assert current['counters']['consecutive_write_pairs']==65472
        assert current['counters']['input_bubbles']>0 and current['counters']['output_stalls']>0
        assert current['precision_metrics']==old['precision_metrics']
        assert current['phase_coverage']==old['phase_coverage']
        assert current['token_cycles_without_input_bubbles_or_output_stalls']==[1468 if rank==1 else 1993]
        if rank==1:
            assert '# RESULT READ reset_inflight=checked' in log
            assert '# RESULT READ admits=65536 consumes=65536' in log
            assert '# RESULT ROMOUT admits=2048 consumes=2048' in log
            assert '# RESULT PHASE admits=2048 consumes=2048' in log
            assert '# RESULT SUM admits=65536 consumes=65536' in log
            assert '# RESULT KV reset_inflight=checked' in log
            assert '# RESULT KV admits=65536 consumes=65536 tag_q_checks=65536' in log
            assert '# RESULT ROTATION reset_inflight=checked' in log
            assert '# RESULT ROTATION accepts=2048 adds=2048 tagged_writes=4096 ' in log
        cases.append({'rank':rank,'status':'PASS','state_writes':state_count,'outputs':output_count,
            'max_bit_error':0,'oracle_and_stimuli_byte_identical':True,
            'unstalled_cycles_before':old['token_cycles_without_input_bubbles_or_output_stalls'][0],
            'unstalled_cycles_after':current['token_cycles_without_input_bubbles_or_output_stalls'][0]})
    before_dir=PRIOR/'synthesis';after_dir=HERE/'synthesis'
    for name in ('pipeline.qsf','pipeline.sdc'):assert digest(before_dir/name)==digest(after_dir/name),name
    for folder in (before_dir,after_dir):
        build=read(folder/'build_status.json')
        assert build['status']=='PASS' and not build['board_programmed']
        for name,sha in build['source_sha256'].items():
            overlay=HERE if folder==after_dir else PRIOR
            location=overlay/'rtl'/name if (overlay/'rtl'/name).is_file() else BASE/'rtl'/name
            assert digest(location)==sha,name
    before,after=timing(before_dir),timing(after_dir)
    assert before['worst_setup_slack_ns']==-27.252
    assert before['minimum_restricted_fmax_MHz']==21.16
    prior_comparison=read(PRIOR/'comparison.json')
    assert before==prior_comparison['after']
    retained_reports={name.split('/',1)[1]:sha for name,sha in prior_comparison['artifact_sha256'].items()
                      if name.startswith('synthesis/') and Path(name).suffix in ('.rpt','.summary')}
    for name,sha in retained_reports.items():assert digest(before_dir/name)==sha,name
    provenance=read(HERE/'timing/extraction_provenance.json')
    assert provenance['script_sha256']==digest(HERE/'extract_paths.tcl')
    for label,folder in [('baseline',before_dir),('staged',after_dir)]:
        for name,sha in provenance['projects'][label]['matched_retained_files_sha256'].items():
            if (folder/name).is_file():assert digest(folder/name)==sha,(label,name)
            else:assert read(folder/'build_status.json')['source_sha256'][name]==sha,(label,name)
    for label in ('baseline','staged'):
        log=(HERE/'timing'/(label+'-extraction.log')).read_text()
        assert 'Timing Analyzer was successful. 0 errors, 0 warnings' in log
    isolated={}
    for name in ('baseline-critical','staged-first','staged-second'):
        text=(HERE/'timing'/(name+'.rpt')).read_text()
        assert 'Slow 1200mV 85C Model' in text
        found=paths(text);assert found,name
        isolated[name]=found[0]
        levels=re.search(r'; Number of Logic Levels\s*;\s*;\s*(\d+)',text)
        if levels:isolated[name]['logic_levels']=int(levels[1])
    assert {k:v for k,v in isolated['baseline-critical'].items() if k!='logic_levels'}==before['worst_registered_path']
    evidence=[p for folder in (HERE/'evidence',HERE/'synthesis',HERE/'timing') for p in folder.rglob('*') if p.is_file()]
    result={'status':'PASS','seed':actual['seed'],'baseline_commit':inputs['baseline_commit'],
        'scope':'One additional rank-1 RAM-output boundary before recurrent products. Routed virtual-I/O static timing only; no physical FPGA operation or inference speedup claim.',
        'clock_constraint_MHz':50,'constraints_byte_identical':True,'before':before,'after':after,
        'worst_slack_change_ns':after['worst_setup_slack_ns']-before['worst_setup_slack_ns'],
        'isolated_paths':isolated,'regression':cases,'first_candidate_mismatch':None,
        'artifact_sha256':{str(p.relative_to(HERE)).replace('\\','/'):digest(p) for p in sorted(evidence)},
        'baseline_report_sha256':retained_reports}
    (HERE/'comparison.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('sha256')},indent=2))

if __name__=='__main__':main()
