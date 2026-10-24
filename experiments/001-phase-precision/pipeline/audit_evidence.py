"""Offline audit of retained pipeline evidence; standard library only."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent

def main():
    summary=json.loads((HERE/'evidence/summary.json').read_text())
    assert summary['status']=='PASS' and summary['seed']==0xE001F032
    for name,digest in summary['source_sha256'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
    assert summary['arithmetic_result_comparisons']==4*summary['arithmetic_vectors']
    assert summary['all_half_bit_patterns_tested']==65536
    assert '# PASS\n' in (HERE/'evidence/arithmetic.log').read_text()
    repeated=json.loads((HERE/'repeatability.json').read_text())
    assert repeated['status']=='PASS' and repeated['byte_identical_files']==24
    for name,digest in repeated['sha256'].items():
        assert hashlib.sha256((HERE/'evidence'/name).read_bytes()).hexdigest()==digest,name
    total_writes=total_outputs=0
    for case in summary['cases']:
        rank=case['rank'];folder=HERE/'evidence'/f'rank{rank}'
        expected_states=[int(x,16) for x in (folder/'states.hex').read_text().splitlines()]
        expected_outputs=[int(x,16) for x in (folder/'outputs.hex').read_text().splitlines()]
        state_count=output_count=consecutive=0;previous_cycle=None;previous_token=None
        n,p,tokens=case['N'],case['P'],case['tokens']
        with (folder/'rtl_results.csv').open() as f:
            for row in csv.DictReader(f):
                token,index,cycle=[int(row[k]) for k in ('token','index','cycle')]
                if row['kind']=='state':
                    assert token==state_count//(p*n) and index==state_count%(p*n)
                    assert (int(row['wide'],16)<<16)|int(row['packed'],16)==expected_states[state_count]
                    if previous_token==token:
                        assert cycle==previous_cycle+1
                        consecutive+=1
                    previous_cycle=cycle;previous_token=token;state_count+=1
                else:
                    assert row['kind']=='output'
                    assert token==output_count//(p*rank) and index==output_count%(p*rank)
                    assert int(row['wide'],16)==expected_outputs[output_count]
                    output_count+=1
        assert state_count==len(expected_states)==case['state_writes']==tokens*p*n
        assert output_count==len(expected_outputs)==case['outputs']==tokens*p*rank
        assert consecutive==case['counters']['consecutive_write_pairs']==tokens*(p*n-1)
        assert case['counters']['errors']==case['quantized_reference_max_bit_error']==0
        assert case['counters']['input_bubbles']>0 and case['counters']['output_stalls']>0
        assert case['phase_coverage']['lookup_address_bins']==256
        assert case['token_cycles_without_input_bubbles_or_output_stalls']==[p*n+2*p*rank+3*rank*n//2+n+9]
        log=(folder/'simulation.log').read_text();negative=(folder/'negative_control.log').read_text()
        assert '# PASS\n' in log and not re.search(r'^# (FAIL|MISMATCH)',log,re.M)
        assert '# FAIL\n' in negative and 'errors=1 ' in negative and '# PASS' not in negative
        trace=HERE/'traces'/f'mamba3-{"mimo" if rank==4 else "siso"}-187m.npz'
        assert hashlib.sha256(trace.read_bytes()).hexdigest()==case['trace_sha256']
        meta=json.loads(trace.with_suffix('.json').read_text())
        assert meta['trace_sha256']==case['trace_sha256'] and meta['tokens']==32
        total_writes+=state_count;total_outputs+=output_count
    model=json.loads((HERE/'model_results.json').read_text())
    assert model['status']=='PASS'
    for name,digest in model['source_sha256'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
    for row in model['models']:
        assert row['tokens']==32 and row['layers']==12 and row['heads_per_layer']==24
        assert row['all_logits_finite'] and row['persistent_main_state_bytes']==12*24*64*128*2
        assert math.isclose(row['emulator_perplexity']/row['reference_perplexity']-1,
                            row['relative_perplexity_change'],abs_tol=1e-12)
    for rank in (1,4):
        folder=HERE/'synthesis/ram_banked'/f'rank{rank}'
        build=json.loads((folder/'build_status.json').read_text())
        assert build['status'] in ('PASS','FAIL') and not build['board_programmed']
        for name,digest in build['source_sha256'].items():
            assert hashlib.sha256((HERE/'rtl'/name).read_bytes()).hexdigest()==digest,name
    qor=json.loads((HERE/'synthesis/qor.json').read_text())
    assert len(qor['cases'])==2 and all(x['virtual_pins']>0 for x in qor['cases'])
    for row in qor['cases']:
        folder=HERE/'synthesis/ram_banked'/f"rank{row['rank']}"
        for name,digest in row['report_sha256'].items():
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest,name
        assert row['meets_50MHz_internal_setup']==(row['worst_setup_slack_ns']>=0)
        assert row['allocated_block_memory_bits']==row['M9K_blocks']*9216
    baseline=HERE/'synthesis/register_bank_baseline'
    recorded=json.loads((baseline/'rank1/build_status.json').read_text())
    for name,digest in recorded['source_sha256'].items():
        assert hashlib.sha256((baseline/'source'/name).read_bytes()).hexdigest()==digest,name
    repo=HERE.parents[2]
    for path in HERE.rglob('*.md'):
        for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)',path.read_text(encoding='utf8')):
            target=link.strip('<>').split('#',1)[0]
            if not target or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',target):continue
            resolved=(path.parent/target).resolve()
            assert resolved.is_relative_to(repo) and resolved.exists(),(path,link)
    print(f'PASS recorded evidence: {total_writes} exact state writes, {total_outputs} exact outputs; source/trace hashes, primitive checks, reset/stall logs, negative controls, model integration, synthesis records.')

if __name__=='__main__':main()
