"""Extract core resource/timing estimates without equating compile with timing."""
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent

def main():
    cases=[]
    for rank in (1,4):
        folder=HERE/'synthesis/ram_banked'/f'rank{rank}'
        build=json.loads((folder/'build_status.json').read_text())
        assert build['status'] in ('PASS','FAIL'),build
        mapped=(folder/'pipeline.map.summary').read_text()
        fit=(folder/'pipeline.fit.summary').read_text() if (folder/'pipeline.fit.summary').is_file() else ''
        fit_detail=(folder/'pipeline.fit.rpt').read_text(errors='replace') if (folder/'pipeline.fit.rpt').is_file() else ''
        sta=(folder/'pipeline.sta.rpt').read_text(errors='replace') if (folder/'pipeline.sta.rpt').is_file() else ''
        sta_summary=(folder/'pipeline.sta.summary').read_text(errors='replace') if (folder/'pipeline.sta.summary').is_file() else ''
        def number(text,key):
            match=re.search(r'^'+re.escape(key)+r'\s*:\s*([\d,]+)',text,re.M)
            return int(match[1].replace(',','')) if match else None
        virtual=number(fit,'Total virtual pins') or number(mapped,'Total virtual pins')
        assert virtual and virtual>0,'Virtual pin constraints were not applied'
        # Fmax tables list Fmax, restricted Fmax and clock. Keep all corners;
        # do not treat an unconstrained I/O path as a timing-closed interface.
        fmax=[]
        for match in re.finditer(r';\s*(\d+(?:\.\d+)?) MHz\s*;\s*(\d+(?:\.\d+)?) MHz\s*;\s*clk\s*;',sta):
            fmax.append({'fmax_MHz':float(match[1]),'restricted_fmax_MHz':float(match[2])})
        slacks=[];kind=''
        for line in sta_summary.splitlines():
            if re.match(r'^Type\s*:',line):kind=line.split(':',1)[1].strip()
            if re.match(r'^Slack\s*:',line) and re.search(r'\bSetup\b',kind):
                slacks.append(float(line.split(':',1)[1]))
        m9k=re.search(r';\s*M9Ks\s*;\s*([\d,]+)',fit_detail)
        implementation=re.search(r';\s*Total block memory implementation bits\s*;\s*([\d,]+)',fit_detail)
        paths=[]
        for line in sta.splitlines():
            cells=[x.strip() for x in line.split(';')][1:-1]
            if len(cells)==8 and re.fullmatch(r'-?\d+\.\d+',cells[0]) and cells[3:5]==['clk','clk']:
                paths.append({'slack_ns':float(cells[0]),'from':cells[1],'to':cells[2],'data_delay_ns':float(cells[7])})
        cases.append({'rank':rank,'compile_status':build['status'],'virtual_pins':virtual,
            'map_logic_elements':number(mapped,'Total logic elements'),
            'fit_logic_elements':number(fit,'Total logic elements'),
            'fit_registers':number(fit,'Total registers'),
            'fit_memory_bits':number(fit,'Total memory bits'),
            'M9K_blocks':int(m9k[1].replace(',','')) if m9k else None,
            'allocated_block_memory_bits':int(implementation[1].replace(',','')) if implementation else None,
            'fit_multiplier_9bit_elements':number(fit,'Embedded Multiplier 9-bit elements'),
            'fmax_corners':fmax,'minimum_restricted_fmax_MHz':min(x['restricted_fmax_MHz'] for x in fmax) if fmax else None,
            'worst_setup_slack_ns':min(slacks) if slacks else None,
            'worst_reported_registered_path':min(paths,key=lambda x:x['slack_ns']) if paths else None,
            'meets_50MHz_internal_setup':bool(slacks and min(slacks)>=0),
            'report_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.suffix in ('.rpt','.summary')}})
    result={'scope':'Quartus standalone core estimates, virtual non-clock IO, FAST_FIT; no physical-board or complete-model timing measurement.',
            'clock_constraint_MHz':50,'board_programmed':False,'cases':cases}
    (HERE/'synthesis/qor.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
