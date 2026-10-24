"""Out-of-context Quartus build, virtual IO except the physical 50MHz clock.

Never programs the board. Build databases stay under ignored .artifacts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rank',type=int,choices=(1,4),default=1)
    ap.add_argument('--quartus-bin',type=Path,default=Path('C:/intelFPGA_lite/18.1/quartus/bin64'))
    args=ap.parse_args()
    source_id=hashlib.sha256(Path(__file__).read_bytes()+b''.join((HERE/'rtl'/name).read_bytes() for name in
                            ('mamba3_recurrent_tile.sv','mamba3_cache_ram.sv','fp32_functions.svh'))).hexdigest()[:12]
    stage=REPO/'.artifacts'/f'e001-recurrent-rank{args.rank}-{source_id}'
    stage.mkdir(parents=True,exist_ok=True)
    for name in ('mamba3_recurrent_tile.sv','mamba3_cache_ram.sv','fp32_functions.svh'):
        shutil.copy2(HERE/'rtl'/name,stage/name)
    for name in ('cos_fp32.hex','sin_fp32.hex'):
        shutil.copy2(HERE/'evidence'/f'rank{args.rank}'/name,stage/name)
    (stage/'pipeline.qpf').write_text('PROJECT_REVISION = "pipeline"\n')
    (stage/'pipeline.sdc').write_text('create_clock -name clk -period 20.000 [get_ports {clk}]\nderive_clock_uncertainty\n')
    (stage/'pipeline.qsf').write_text('''set_global_assignment -name FAMILY "Cyclone IV E"
set_global_assignment -name DEVICE EP4CE115F29C7
set_global_assignment -name TOP_LEVEL_ENTITY mamba3_recurrent_tile
set_global_assignment -name SYSTEMVERILOG_FILE mamba3_recurrent_tile.sv
set_global_assignment -name SYSTEMVERILOG_FILE mamba3_cache_ram.sv
set_global_assignment -name SDC_FILE pipeline.sdc
set_global_assignment -name NUM_PARALLEL_PROCESSORS 4
set_global_assignment -name FITTER_EFFORT "FAST_FIT"
set_global_assignment -name PROJECT_OUTPUT_DIRECTORY output_files
set_location_assignment PIN_Y2 -to clk
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to clk
set_global_assignment -name RESERVE_ALL_UNUSED_PINS "AS INPUT TRI-STATED"
'''+''.join('set_instance_assignment -name VIRTUAL_PIN ON -to '+name+'\n'
           for name in ('reset','start_valid','start_ready','alpha_in[*]','beta_in[*]',
                        'gamma_in[*]','skip_in[*]','v_valid','v_ready','v_in[*]',
                        'bc_valid','bc_ready','qx_in[*]','qy_in[*]','kx_in[*]',
                        'ky_in[*]','increment_in[*]','y_valid','y_ready','y_out[*]',
                        'y_index[*]','done','state_write_valid','state_write_index[*]',
                        'state_write_data[*]','state_write_wide[*]','debug_stage[*]'))+
        f'set_parameter -name RANK {args.rank}\nset_parameter -name P 8\n')
    output=HERE/'synthesis/ram_banked'/f'rank{args.rank}';output.mkdir(parents=True,exist_ok=True)
    # Every output file here is produced by this build script. Avoid retaining
    # stale successful reports when a new build stops earlier in the flow.
    for p in output.iterdir():
        if p.is_file() and p.suffix in ('.rpt','.summary'):p.unlink()
    (output/'build_status.json').write_text(json.dumps({'status':'RUNNING','rank':args.rank,'source_id':source_id})+'\n')
    command=[str(args.quartus_bin/'quartus_sh.exe'),'--flow','compile','pipeline']
    with (stage/'build.log').open('w') as f:
        result=subprocess.run(command,cwd=stage,stdout=f,stderr=subprocess.STDOUT,timeout=3600)
    # Keep vendor reports, not databases or programming images. Build log includes
    # local paths and wall-clock timing and is retained only as diagnostic evidence.
    for p in (stage/'output_files').glob('*'):
        if p.suffix in ('.rpt','.summary'):
            shutil.copy2(p,output/p.name)
    shutil.copy2(stage/'build.log',output/'build.log')
    shutil.copy2(stage/'pipeline.qsf',output/'pipeline.qsf')
    shutil.copy2(stage/'pipeline.sdc',output/'pipeline.sdc')
    record={'rank':args.rank,'P':8,'N':128,'target':'EP4CE115F29C7',
            'clock_constraint_MHz':50,'virtual_io':True,'fitter_effort':'FAST_FIT','board_programmed':False,
            'source_sha256':{name:hashlib.sha256((stage/name).read_bytes()).hexdigest()
                             for name in ('mamba3_recurrent_tile.sv','mamba3_cache_ram.sv','fp32_functions.svh')},
            'quartus_exit_code':result.returncode,'status':'PASS' if result.returncode==0 else 'FAIL'}
    (output/'build_status.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    print((stage/'build.log').read_text(errors='replace')[-5000:])
    sys.exit(0 if result.returncode==0 else 1)

if __name__=='__main__':main()
