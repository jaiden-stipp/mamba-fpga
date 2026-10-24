"""Seed-locked portable FP32/FP16 RTL regression; NumPy + ModelSim required."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import numpy as np

HERE=Path(__file__).resolve().parent
SEED=0xE001F032
N,P,TOKENS=128,8,64

def write(path,text):
    path.write_text(text,encoding='utf8',newline='\n')

def hexes(path,values,digits):
    write(path,''.join(f'{int(x):0{digits}x}\n' for x in values))

def word(x):
    return int(np.asarray(x,dtype=np.float32).view(np.uint32))

def halfword(x):
    return int(np.asarray(x,dtype=np.float16).view(np.uint16))

def add(a,b):return np.add(a,b,dtype=np.float32)
def mul(a,b):return np.multiply(a,b,dtype=np.float32)

def canon(words):
    result=words.copy()
    result[((words&0x7f800000)==0x7f800000)&((words&0x007fffff)!=0)]=0x7fc00000
    return result

def arithmetic_vectors(stage):
    rng=np.random.default_rng(SEED)
    directed=np.array([0,0x80000000,1,0x80000001,0x007fffff,0x00800000,
        0x00800001,0x3f800000,0xbf800000,0x33800000,0x33000000,0x3f800001,
        0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc00000,0x7fa00001,
        0x387fc000,0x38800000,0x477fe000,0x477ff000,0x3f801000,0x3f803000],dtype=np.uint32)
    aa,bb=np.meshgrid(directed,directed)
    # Every half bit pattern plus 30,000 uniform random FP32 bit pairs and
    # all directed cross-products. Expected arithmetic comes from NumPy.
    hs=np.arange(65536,dtype=np.uint16)
    av=np.concatenate([hs.view(np.float16).astype(np.float32).view(np.uint32),
                       rng.integers(0,2**32,30000,dtype=np.uint32),aa.ravel()])
    bv=np.concatenate([rng.integers(0,2**32,65536,dtype=np.uint32),
                       rng.integers(0,2**32,30000,dtype=np.uint32),bb.ravel()])
    hv=np.arange(len(av),dtype=np.uint32).astype(np.uint16)
    af=av.view(np.float32);bf=bv.view(np.float32)
    with np.errstate(all='ignore'):
        sums=canon(add(af,bf).view(np.uint32));products=canon(mul(af,bf).view(np.uint32))
        halves=af.astype(np.float16).view(np.uint16)
        halves=np.where(np.isnan(af),(av>>16)&0x8000|0x7e00,halves).astype(np.uint16)
        decoded=hv.view(np.float16).astype(np.float32).view(np.uint32)
    # Widening preserves payload/sign, including signaling NaNs; the datapath
    # canonicalizes NaNs in arithmetic and in narrowing, without exception flags.
    is_nan=((hv&0x7c00)==0x7c00)&((hv&1023)!=0)
    decoded[is_nan]=((hv[is_nan].astype(np.uint32)&0x8000)<<16)|0x7f800000|((hv[is_nan].astype(np.uint32)&1023)<<13)
    rows=((int(a)<<160)|(int(b)<<128)|(int(s)<<96)|(int(m)<<64)|(int(h)<<48)|(int(v)<<32)|int(d)
          for a,b,s,m,h,v,d in zip(av,bv,sums,products,halves,hv,decoded))
    hexes(stage/'fp_vectors.hex',rows,48)
    return len(av)

def dataset(rank,traces):
    rng=np.random.default_rng(SEED+rank)
    synthetic={k:rng.normal(0,1,(32,rank,N)).astype(np.float32) for k in ('q','k')}
    synthetic['v']=rng.normal(0,1,(32,rank,P)).astype(np.float32)
    # Directed signed/zero/tiny coordinates alongside normal model-like values.
    for k in ('q','k','v'):
        synthetic[k][0].flat[:8]=[0.,-0.,2**-20,-2**-20,16.,-16.,1.+2**-10,-1.-2**-10]
    synthetic.update(alpha=rng.uniform(.8,.99,32).astype(np.float32),
                     beta=rng.uniform(.001,.03,32).astype(np.float32),
                     gamma=rng.uniform(.001,.03,32).astype(np.float32),
                     skip=rng.normal(0,.1,32).astype(np.float32),
                     increment=rng.integers(0,65536,(32,N//4),dtype=np.int64))
    synthetic['increment'][:4,:4]=[[65535,128,384,0],[1,0,0,0],[65535,65535,1,0],[1,1,65535,0]]
    path=traces/f'mamba3-{"mimo" if rank==4 else "siso"}-187m.npz'
    with np.load(path) as z:
        actual={k:z[k].copy() for k in ('q','k','v','alpha','beta','gamma','skip','increment')}
    actual['skip']=np.repeat(actual['skip'],32)
    return {k:np.concatenate([synthetic[k],actual[k]]) for k in synthetic},path

def build_case(stage,rank,traces):
    data,trace_path=dataset(rank,traces)
    cs=np.rint(np.cos(np.arange(256)*math.tau/256)*127).astype(np.float32)/np.float32(127)
    sn=np.rint(np.sin(np.arange(256)*math.tau/256)*127).astype(np.float32)/np.float32(127)
    hexes(stage/'cos_fp32.hex',[word(x) for x in cs],8)
    hexes(stage/'sin_fp32.hex',[word(x) for x in sn],8)
    commands=[];bc=[];values=[];writes=[];outputs=[];high_outputs=[];early_outputs=[]
    phase=np.zeros(N//4,dtype=np.int64)
    coverage={'positive_wraps':0,'negative_wraps':0,'zero_increments':0,
              'even_halfway_addresses':0,'odd_halfway_addresses':0}
    bins=set()
    state=np.zeros((P,N),dtype=np.float16);pk=np.zeros((rank,N),dtype=np.float32);pv=np.zeros((rank,P),dtype=np.float32)
    high_state=np.zeros((P,N),dtype=np.float64);high_pk=np.zeros((rank,N),dtype=np.float64);high_pv=np.zeros((rank,P),dtype=np.float64)
    for t in range(TOKENS):
        if t==32:
            phase[:]=0;state[:]=0;pk[:]=0;pv[:]=0;high_state[:]=0;high_pk[:]=0;high_pv[:]=0
        a,b,g,d=[np.float32(data[k][t]) for k in ('alpha','beta','gamma','skip')]
        commands.append((word(a)<<96)|(word(b)<<64)|(word(g)<<32)|word(d))
        v=data['v'][t];values.extend(word(x) for x in v.T.ravel())
        q=data['q'][t].copy();k=data['k'][t].copy()
        hq=q.astype(np.float64);hk=k.astype(np.float64)
        inc=data['increment'][t]
        signed_inc=np.where(inc>=32768,inc-65536,inc)
        coverage['positive_wraps']+=int(np.count_nonzero(phase+signed_inc>=65536))
        coverage['negative_wraps']+=int(np.count_nonzero(phase+signed_inc<0))
        coverage['zero_increments']+=int(np.count_nonzero(inc==0))
        phase=(phase+inc)&65535
        addresses=(np.rint(phase/256).astype(np.int64))&255
        bins.update(addresses.tolist())
        coverage['even_halfway_addresses']+=int(np.count_nonzero(((phase&255)==128)&(((phase>>8)&1)==0)))
        coverage['odd_halfway_addresses']+=int(np.count_nonzero(((phase&255)==128)&(((phase>>8)&1)==1)))
        for r in range(rank):
            for pair in range(N//2):
                lo,hi=(pair,pair+N//2) if rank>1 else (2*pair,2*pair+1)
                qx,qy,kx,ky=q[r,lo],q[r,hi],k[r,lo],k[r,hi]
                inc=int(data['increment'][t,pair]) if pair<N//4 else 0
                bc.append((word(qx)<<112)|(word(qy)<<80)|(word(kx)<<48)|(word(ky)<<16)|inc)
                if pair<N//4:
                    c,s=cs[addresses[pair]],sn[addresses[pair]]
                    q[r,lo]=add(mul(qx,c),mul(qy,-s));q[r,hi]=add(mul(qx,s),mul(qy,c))
                    k[r,lo]=add(mul(kx,c),mul(ky,-s));k[r,hi]=add(mul(kx,s),mul(ky,c))
                    c64=math.cos(phase[pair]*math.tau/65536);s64=math.sin(phase[pair]*math.tau/65536)
                    hq[r,lo]=float(qx)*c64-float(qy)*s64;hq[r,hi]=float(qx)*s64+float(qy)*c64
                    hk[r,lo]=float(kx)*c64-float(ky)*s64;hk[r,hi]=float(kx)*s64+float(ky)*c64
        y=np.zeros((rank,P),dtype=np.float32);early=y.copy()
        next_state=np.empty_like(state)
        for p in range(P):
            for n in range(N):
                now=previous=np.float32(0)
                for r in range(rank):
                    now=add(now,mul(k[r,n],v[r,p]));previous=add(previous,mul(pk[r,n],pv[r,p]))
                updated=add(add(mul(a,np.float32(state[p,n])),mul(b,previous)),mul(g,now))
                packed=np.float16(updated);assert np.isfinite(packed)
                next_state[p,n]=packed
                writes.append((word(updated)<<16)|halfword(packed))
                for r in range(rank):
                    y[r,p]=add(y[r,p],mul(updated,q[r,n]))
                    early[r,p]=add(early[r,p],mul(np.float32(packed),q[r,n]))
        y=add(y,mul(d,v));early=add(early,mul(d,v))
        outputs.extend(word(x) for x in y.T.ravel());early_outputs.extend(early.T.ravel())
        high_state=float(a)*high_state+float(b)*(high_pv.T@high_pk)+float(g)*(v.astype(np.float64).T@hk)
        high_y=hq@high_state.T+float(d)*v.astype(np.float64)
        high_outputs.extend(high_y.T.ravel())
        state=next_state;pk=k;pv=v.copy();high_pk=hk;high_pv=v.astype(np.float64)
    hexes(stage/'commands.hex',commands,32);hexes(stage/'values.hex',values,8)
    hexes(stage/'bc.hex',bc,36);hexes(stage/'states.hex',writes,12);hexes(stage/'outputs.hex',outputs,8)
    out=np.asarray(outputs,dtype=np.uint32).view(np.float32).astype(np.float64)
    high=np.asarray(high_outputs);early=np.asarray(early_outputs)
    metrics=[]
    for name,sl in [('synthetic',slice(0,32*P*rank)),('actual_checkpoint_trace',slice(32*P*rank,None))]:
        err=out[sl]-high[sl]
        metrics.append({'dataset':name,'output_max_abs_error_vs_float64':float(np.abs(err).max()),
                        'output_relative_rms_vs_float64':float(np.linalg.norm(err)/np.linalg.norm(high[sl])),
                        'early_narrowing_max_output_change':float(np.abs(out[sl]-early[sl]).max())})
    coverage['lookup_address_bins']=len(bins)
    assert all(coverage[k]>0 for k in coverage),coverage
    return {'rank':rank,'P':P,'N':N,'tokens':TOKENS,'state_writes':len(writes),'outputs':len(outputs),'phase_coverage':coverage,
            'trace_sha256':hashlib.sha256(trace_path.read_bytes()).hexdigest(),'precision_metrics':metrics}

def run(command,stage,timeout=600):
    result=subprocess.run([str(x) for x in command],cwd=stage,capture_output=True,text=True,timeout=timeout)
    return result.returncode,result.stdout+result.stderr

def transcript(text):
    return '\n'.join(x for x in text.splitlines() if re.match(r'^# (?:PASS|FAIL|RESULT |TOKEN |MISMATCH |\*\* (?:Error|Fatal))',x))+'\n'

def require_pass(code,log):
    assert code==0 and re.search(r'^# PASS\s*$',log,re.M) and not re.search(r'^# (?:FAIL|MISMATCH)',log,re.M),log[-7000:]

def execute(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    write(output/'PASS_FAIL.log',f'RUNNING seed=0x{SEED:08x}\n')
    write(output/'summary.json',json.dumps({'status':'RUNNING','seed':SEED})+'\n')
    binaries={k:args.modelsim_bin/(k+'.exe') for k in ('vlib','vlog','vsim')}
    for x in binaries.values():assert x.is_file(),x
    records=[]
    with tempfile.TemporaryDirectory(prefix='e001-pipeline-') as tmp:
        stage=Path(tmp)
        for p in (HERE/'rtl').iterdir():shutil.copy2(p,stage/p.name)
        source_hashes={'rtl/'+name:hashlib.sha256((stage/name).read_bytes()).hexdigest()
                       for name in ('mamba3_recurrent_tile.sv','mamba3_cache_ram.sv','fp32_functions.svh','tb_fp32.sv','tb_recurrent_tile.sv')}
        source_hashes['regression.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        vendor=args.quartus_simlib/'altera_mf.v';assert vendor.is_file(),vendor
        vendor_hash=hashlib.sha256(vendor.read_bytes()).hexdigest()
        code,version=run([binaries['vsim'],'-version'],stage);assert code==0,version
        count=arithmetic_vectors(stage)
        code,log=run([binaries['vlib'],'simlib'],stage);assert code==0,log
        code,log=run([binaries['vlog'],'-sv','-work','simlib','+incdir+.',vendor,
                      'mamba3_cache_ram.sv','mamba3_recurrent_tile.sv','tb_fp32.sv','tb_recurrent_tile.sv'],stage)
        write(output/'compile.log','\n'.join(x for x in log.splitlines() if not x.startswith(('Start time:','End time:','Model Technology')))+'\n')
        assert code==0 and 'Errors: 0' in log,log
        base=[binaries['vsim'],'-c','-onfinish','exit','-lib','simlib']
        ending=['-do','onerror {quit -code 1}; run -all; quit -code 0']
        code,log=run(base+['tb_fp32',f'-gCOUNT={count}']+ending,stage)
        write(output/'arithmetic.log',transcript(log));require_pass(code,log)
        for rank in (1,4):
            info=build_case(stage,rank,args.traces)
            # Retain compact exact stimulus/expected files; scratch simulator
            # databases and the large primitive vector set remain temporary.
            case_out=output/f'rank{rank}';case_out.mkdir(exist_ok=True)
            for name in ('commands','values','bc','states','outputs','cos_fp32','sin_fp32'):
                shutil.copy2(stage/(name+'.hex'),case_out/(name+'.hex'))
            command=base+['tb_recurrent_tile',f'-gRANK={rank}',f'-gTOKENS={TOKENS}']
            code,log=run(command+ending,stage)
            write(case_out/'simulation.log',transcript(log));require_pass(code,log)
            shutil.copy2(stage/'rtl_results.csv',case_out/'rtl_results.csv')
            with (stage/'rtl_results.csv').open() as f:rows=list(csv.DictReader(f))
            assert len(rows)==info['state_writes']+info['outputs']
            # Independent Python re-check of every simulator-exported result.
            expected_states=[int(x,16) for x in (stage/'states.hex').read_text().splitlines()]
            expected_outputs=[int(x,16) for x in (stage/'outputs.hex').read_text().splitlines()]
            for row in rows:
                token,index=int(row['token']),int(row['index'])
                if row['kind']=='state':
                    assert (int(row['wide'],16)<<16)|int(row['packed'],16)==expected_states[token*P*N+index]
                else:assert int(row['wide'],16)==expected_outputs[token*P*rank+index]
            status=re.search(r'RESULT rank=(\d+) tokens=(\d+) writes=(\d+) outputs=(\d+) errors=(\d+) bubbles=(\d+) output_stalls=(\d+) consecutive_write_pairs=(\d+)',log)
            assert status,log[-3000:]
            info['counters']=dict(zip(('rank','tokens','writes','outputs','errors','input_bubbles','output_stalls','consecutive_write_pairs'),map(int,status.groups())))
            cycles=[int(c) for c in re.findall(r'TOKEN token=\d+ cycles=(\d+)',log)]
            assert len(cycles)==TOKENS
            info['token_cycles_without_input_bubbles_or_output_stalls']=sorted(set(cycles[::2]))
            info['token_cycles_with_bubbles_and_stalls']={'min':min(cycles[1::2]),'max':max(cycles[1::2])}
            info['quantized_reference_max_bit_error']=0
            code,badlog=run(command+['+CORRUPT']+ending,stage)
            write(case_out/'negative_control.log',transcript(badlog))
            assert '# FAIL' in badlog and 'errors=1 ' in badlog and '# PASS' not in badlog
            info['negative_control_detected']=True;records.append(info)
    summary={'status':'PASS','seed':SEED,'source_sha256':source_hashes,'arithmetic_vectors':count,
             'environment':{'python':platform.python_version(),'numpy':np.__version__,'generator':'PCG64',
                            'simulator':version.strip(),'altera_mf_sha256':vendor_hash},
             'arithmetic_result_comparisons':4*count,'all_half_bit_patterns_tested':65536,
             'cases':records,'state_storage':'FP16','compute':'FP32 RNE separate multiply/add',
             'precision_reference':'FP64 recurrence and true trigonometry of accumulated 16-bit phase; no FP16 feedback. Same supplied quantized increments, frozen upstream activations.',
             'scope':'Recurrent tile RTL simulation; not full-model or measured FPGA acceleration.'}
    write(output/'summary.json',json.dumps(summary,indent=2)+'\n')
    write(output/'PASS_FAIL.log',f'PASS seed=0x{SEED:08x}\nPrimitive vectors={count}, comparisons={4*count}\n'+
          ''.join(f"PASS rank={x['rank']} tokens={x['tokens']} state_writes={x['state_writes']} outputs={x['outputs']} quantized_reference_max_bit_error=0 negative_control=detected\n" for x in records))
    print(json.dumps(summary,indent=2))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--modelsim-bin',type=Path,default=Path('C:/intelFPGA/18.1/modelsim_ase/win32aloem'))
    ap.add_argument('--quartus-simlib',type=Path,default=Path('C:/intelFPGA_lite/18.1/quartus/eda/sim_lib'))
    ap.add_argument('--traces',type=Path,default=HERE/'traces')
    ap.add_argument('--output',type=Path,default=HERE/'evidence')
    args=ap.parse_args()
    try:execute(args)
    except Exception as error:
        args.output.mkdir(parents=True,exist_ok=True)
        write(args.output/'PASS_FAIL.log',f'FAIL seed=0x{SEED:08x} {type(error).__name__}: {error}\n')
        write(args.output/'summary.json',json.dumps({'status':'FAIL','seed':SEED,'error':str(error)},indent=2)+'\n')
        raise

if __name__=='__main__':main()
