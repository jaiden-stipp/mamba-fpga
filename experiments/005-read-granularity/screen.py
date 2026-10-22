"""Explicit-address fetch-granularity screen; analytical, not hardware measured."""
import csv, hashlib, json, random
from pathlib import Path

ROOT=Path(__file__).resolve().parent
GROUPS=512
VALUES=64
PLANE_BYTES=VALUES//8
PAYLOAD=GROUPS*VALUES
LAYOUTS={'byte_codes':None,'group_planes':1,'tile4_planes':4,'tile8_planes':8,'tile16_planes':16,'tile64_planes':64,'global_planes':GROUPS}

def schedules():
    mixed=[2]*(GROUPS//2)+[6]*(GROUPS//2)
    shuffled=mixed.copy();random.Random(20261022).shuffle(shuffled)
    return {'uniform2':[2]*GROUPS,'uniform4':[4]*GROUPS,'uniform8':[8]*GROUPS,
            'alternating26':[2,6]*(GROUPS//2),'random26':shuffled,
            'clustered26':mixed,'quarter8_rest2':[8]*(GROUPS//4)+[2]*(3*GROUPS//4)}

def touch(units,start,length,granule):
    units.update(range(start//granule,(start+length-1)//granule+1))

def count(layout,depths,granule):
    units=set();tile=LAYOUTS[layout]
    for group,depth in enumerate(depths):
        if tile is None:
            touch(units,group*VALUES,VALUES,granule)
        else:
            tile_id,offset=divmod(group,tile)
            for plane in range(depth):
                address=tile_id*tile*VALUES+(plane*tile+offset)*PLANE_BYTES
                touch(units,address,PLANE_BYTES,granule)
    payload_units=len(units)
    # All arrays have aligned extents in this fixed sweep.
    touch(units,PAYLOAD,GROUPS*2,granule)
    touch(units,PAYLOAD+GROUPS*2,GROUPS,granule)
    return payload_units,len(units)

def main():
    rows=[];checks=0
    for name,depths in schedules().items():
        logical=sum(depths)*VALUES//8
        for granule in [32,64,128]:
            baseline=count('byte_codes',depths,granule)[1]*granule
            for layout in LAYOUTS:
                payload_units,total_units=count(layout,depths,granule)
                total=total_units*granule
                assert total>=logical+GROUPS*3;checks+=1
                if name=='uniform8':assert total==baseline;checks+=1
                rows.append(dict(schedule=name,layout=layout,granule_bytes=granule,
                    requested_mean_bits=sum(depths)/GROUPS,logical_payload_read_bytes=logical,
                    retained_payload_bytes=PAYLOAD,metadata_bytes=GROUPS*3,
                    modeled_payload_fetch_bytes=payload_units*granule,
                    modeled_total_fetch_bytes=total,modeled_fetch_units=total_units,
                    total_relative_to_byte_codes=total/baseline))
    bykey={(r['schedule'],r['layout'],r['granule_bytes']):r for r in rows}
    assert bykey['uniform4','group_planes',64]['modeled_payload_fetch_bytes']==PAYLOAD;checks+=1
    assert bykey['uniform4','global_planes',64]['modeled_payload_fetch_bytes']==PAYLOAD//2;checks+=1
    with (ROOT/'results.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    selected=[r for r in rows if r['granule_bytes']==64 and r['schedule'] in ['uniform4','random26']]
    summary={'evidence_type':'analytical_address_model_not_hardware_measurement','configurations':len(rows),
             'checks_passed':checks,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'selection':selected,'assumptions':'See PROTOCOL.md. No quality or latency prediction.'}
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    lines=['# E005 analytical results','',
        'These are outputs of an explicit-address model, not measured device traffic, latency or model quality. See [protocol](PROTOCOL.md).','',
        '| Requested schedule | Layout | Logical payload | Modeled total fetch | Relative to byte codes |',
        '|---|---|---:|---:|---:|']
    for r in selected:
        lines.append(f"| {r['schedule']} | {r['layout']} | {r['logical_payload_read_bytes']:,} B | {r['modeled_total_fetch_bytes']:,} B | {r['total_relative_to_byte_codes']:.3f} |")
    lines+=['',f'{len(rows)} configurations; {checks} consistency checks passed. All layouts retain 32,768 payload bytes plus 1,536 metadata bytes.',
        '', 'A four-bit prefix does not imply half the physical bytes: the byte-code layout reads full bytes, and a group-major layout can fetch unused planes in the same unit. Grouping planes across groups helps uniform schedules. Scattered higher-depth groups can touch most units and erase the gain.',
        '', 'Next: replay genuine model-selected depth schedules, measure decoder/scheduler cost and hardware counters, and compare at matched answer quality and total memory. No eight-bit retention or arbitrary prefix is automatically a useful quantizer.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({'configurations':len(rows),'checks_passed':checks,'evidence_type':summary['evidence_type']}))

if __name__=='__main__':main()
