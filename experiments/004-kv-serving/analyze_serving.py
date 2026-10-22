"""Summarize preserved GPU evidence; no timing reconstruction from rounded stdout."""
import argparse,csv,hashlib,json,re,statistics
from pathlib import Path
R=Path(__file__).resolve().parent
PRIMARY='research_pilot_20261022'
RECOVERY='research_recovery_20261022'

def read(name):return json.loads((R/name).read_text(encoding='utf8'))
def records(tag):return [json.loads(s) for s in (R/f'raw_{tag}.jsonl').read_text(encoding='utf8').splitlines()]
def percentile(xs,p):return sorted(xs)[max(0,int(p*(len(xs)-1)))] if xs else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plot',action='store_true');args=ap.parse_args()
    allraw={tag:records(tag) for tag in [PRIMARY,RECOVERY]}
    allwaves={tag:read(f'waves_{tag}.json') for tag in allraw}
    rows=[];checks=0
    for slots in [1,4,8]:
        for fmt in ['f16','q8_0','q4_0']:
            if slots==8 and fmt=='f16':continue
            tag=RECOVERY if (fmt,slots)==('q8_0',8) else PRIMARY
            raw=[r for r in allraw[tag] if r['format']==fmt and r['slots']==slots]
            waves=[w for w in allwaves[tag] if w['format']==fmt and w['slots']==slots]
            assert len(raw)==8+5*slots and len(waves)==6;checks+=2
            log=(R/'logs'/f'{tag}_{fmt}_c{slots}.log').read_text(encoding='utf8')
            assert 'offloaded 37/37 layers to GPU' in log;checks+=1
            assert f'K ({fmt})' in log and f'V ({fmt})' in log;checks+=1
            assert "flash_attn            = enabled" in log;checks+=1
            kv=float(re.search(r'CUDA0 KV buffer size\s*=\s*([0-9.]+) MiB',log).group(1))
            for phase in ['cold','resident']:
                selected=[r for r in raw if r['phase']=='cold'] if phase=='cold' else [r for r in raw if r['phase'].startswith('resident_') and r['phase']!='resident_load']
                active=[w for w in waves if w['phase']=='cold'] if phase=='cold' else [w for w in waves if w['phase'].startswith('resident_') and w['phase']!='resident_load']
                expected_n=8 if phase=='cold' else 4*slots
                assert len(selected)==expected_n;checks+=1
                assert sum(w['requests'] for w in active)==expected_n;checks+=1
                if phase=='resident':assert all(r['resident_cache_hit'] for r in selected);checks+=len(selected)
                assert not any(r.get('error') or r.get('final',{}).get('truncated') for r in selected);checks+=1
                qualified=sum(r['qualified'] for r in selected)
                seconds=sum(w['makespan_s'] for w in active)
                samples=[s['used_MiB'] for w in waves for s in w['gpu_samples'] if 'used_MiB' in s]
                rows.append({'format':fmt,'slots':slots,'phase':phase,'source_tag':tag,'requests':expected_n,
                    'correct':sum(r['correct'] for r in selected),'strict_correct':sum(r['strict_correct'] for r in selected),
                    'qualified':qualified,'deadlines_met':sum(r['deadlines_met'] for r in selected),
                    'resident_hits':sum(bool(r.get('resident_cache_hit')) for r in selected),
                    'active_wave_seconds':seconds,'qualified_per_active_wave_second':qualified/seconds,
                    'median_ttft_s':statistics.median(r['ttft_s'] for r in selected),
                    'p95_ttft_s_nearest_rank':sorted(r['ttft_s'] for r in selected)[min(len(selected)-1,__import__('math').ceil(.95*len(selected))-1)],
                    'median_mean_tpot_s':statistics.median(r['mean_tpot_s'] for r in selected),
                    'output_tokens':sum(r['output_tokens'] for r in selected),'kv_allocation_MiB':kv,
                    'peak_sampled_device_used_MiB':max(samples),
                    'resident_load_s':next(w['makespan_s'] for w in waves if w['phase']=='resident_load')})
    with (R/'serving_summary.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    selected_raw=[r for tag,data in allraw.items() for r in data if ((r['format'],r['slots'])==('q8_0',8))==(tag==RECOVERY)]
    scored=[r for r in selected_raw if r['phase']!='resident_load']
    tasks=[]
    for fmt in ['f16','q8_0','q4_0']:
        for task in sorted({r['task'] for r in scored}):
            rs=[r for r in scored if r['format']==fmt and r['task']==task]
            tasks.append({'format':fmt,'task':task,'requests':len(rs),'correct':sum(r['correct'] for r in rs)})
    summary={'evidence_type':'single_trial_GPU_serving_pilot','date':'2026-10-22',
        'selected_configuration_count':8,'selected_scored_requests':len(scored),'selected_resident_requests':sum(r['phase'].startswith('resident_') for r in scored),
        'all_primary_and_recovery_raw_rows':sum(map(len,allraw.values())),
        'excluded_original_q8_c8_rows':48,'integration_smoke_rows_excluded':3,
        'checks_passed':checks,'rows':rows,'task_breakdown':tasks,
        'notes':['Q8 c8 uses a new complete rerun because the original run lost its final wave/configuration snapshots after a Windows write error. Original raw rows are retained, not silently replaced.',
                 'Resident rate divides qualified responses by the sum of four active wave durations; it excludes initial loading and orchestration gaps. It is not steady-state or identical-arrival serving throughput.',
                 'Populations differ by slot count. Single timing trial and eight independent documents cannot establish matched-quality noninferiority or a general speedup.',
                 'F16 c8 was not executed because of an analytical allocation guard, not an observed OOM.',
                 'Startup logs verify 37/37 reported repeating/output layers offloaded, CUDA0 KV types/allocation and enabled Flash Attention. CUDA_Host buffers also exist. No kernel-dispatch or physical DRAM counter profile was collected.'],
        'raw_sha256':{tag:hashlib.sha256((R/f'raw_{tag}.jsonl').read_bytes()).hexdigest() for tag in allraw}}
    (R/'serving_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    lines=['# E004 GPU serving pilot — 2026-10-22','',
        '**The packed-cache engine loaded eight long contexts with Q8 and Q4, but this pilot does not establish improved quality-matched response throughput.** These are stock-engine baseline measurements, not a new reader or FPGA accelerator.','',
        '## Resident contexts','',
        '| KV | Slots | Correct / requests | Deadline-qualified | Active-wave rate (responses/s) | KV allocated (GiB) | Load time (s) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['phase']=='resident':lines.append(f"| {r['format']} | {r['slots']} | {r['correct']}/{r['requests']} | {r['qualified']} | {r['qualified_per_active_wave_second']:.3f} | {r['kv_allocation_MiB']/1024:.3f} | {r['resident_load_s']:.2f} |")
    lines+=['','Resident rate excludes context loading and inter-wave gaps. Different slot counts use different client populations; compare the same-slot controls before interpreting a capacity curve. All 124 selected resident questions reused the document prefix according to `timings.cache_n`.','',
        '## Identical cold burst','',
        '| KV | Slots | Correct / 8 | Deadline-qualified / 8 | Burst time (s) | Median TTFT (s) |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['phase']=='cold':lines.append(f"| {r['format']} | {r['slots']} | {r['correct']}/8 | {r['qualified']}/8 | {r['active_wave_seconds']:.2f} | {r['median_ttft_s']:.2f} |")
    lines+=['','Every format faces the same eight cold requests; queueing/prefill is included. No correct responses met all three deadlines at four or eight slots. Some early correct responses met the first-token limit but stalled between tokens while other documents were ingested. Scheduling/prefill interference warrants a separate controlled test; this pilot does not isolate its cause.','',
        '## Interpretation and next steps','',
        '- All slots have 17,408 tokens of capacity; inputs are approximately 16K and outputs stop naturally at EOS. Weights, engine, inputs and deadlines are fixed across formats.',
        '- Multi-record extraction is difficult even for F16. Report task-level quality and use held-out RULER/multi-hop/code tasks before making a quality-preservation claim. Compression cannot be blamed for every baseline failure.',
        '- At equal slot count the plain Q4/Q8 paths do not show a reliable useful-rate win over F16 in this one trial. Eight-slot compressed runs show device-resident KV capacity; that alone is not throughput superiority.',
        '- F16 with eight slots needs approximately 26.6 GiB for the conservative weight proxy plus KV, exceeding initial free device memory before scratch. It was guarded out, not run to OOM. Actual Q8/Q4 allocation is verified in startup logs.',
        '- Native logs report CUDA_Host model/output/compute buffers in addition to CUDA0 allocations. We verified reported layer offload and GPU KV, not that every operation or parameter access is GPU-only. No physical-byte/kernel-counter measurement was made.',
        '- Collect repeated trials, a stronger quality set, open-loop arrivals and a tuned serving baseline. Then use P001/P002 to test a specific improvement; do not claim novelty for this comparison.','',
        '## Integrity and recovery','',*summary['notes'], '',
        f"Selected: {len(scored)} scored requests across eight configurations; {summary['all_primary_and_recovery_raw_rows']} raw rows including loads and the retained duplicate Q8 run; {checks} analysis assertions passed.",
        '', 'The failed original CLI smoke and the three older timing-client smoke rows remain archived and are excluded. See [protocol amendment](AMENDMENT-2026-10-22.md), [raw pilot](raw_research_pilot_20261022.jsonl), [recovery raw](raw_research_recovery_20261022.jsonl), [CSV](serving_summary.csv) and [summary JSON](serving_summary.json).']
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,4.3))
        for fmt,color in [('f16','#4263eb'),('q8_0','#0b8f77'),('q4_0','#c57700')]:
            rs=[r for r in rows if r['format']==fmt and r['phase']=='resident']
            axes[0].plot([r['slots'] for r in rs],[100*r['correct']/r['requests'] for r in rs],'o-',label=fmt,color=color)
            axes[1].plot([r['slots'] for r in rs],[r['qualified_per_active_wave_second'] for r in rs],'o-',label=fmt,color=color)
        axes[0].set(ylabel='Correct answers (%)',ylim=(0,100),title='Quality: populations vary with slot count')
        axes[1].set(ylabel='Correct on-time responses / active second',ylim=(0,None),title='Resident active-wave rate; loading excluded')
        for ax in axes:ax.set(xlabel='Allocated long-context slots',xticks=[1,4,8]);ax.grid(alpha=.2);ax.legend()
        fig.suptitle('Qwen3-4B · RTX 3090 · single-trial pilot, not quality-matched superiority',fontsize=11)
        fig.tight_layout();fig.savefig(R/'serving_pilot.png',dpi=170);plt.close(fig)
        lines+=['','![Resident pilot](serving_pilot.png)']
    (R/'serving_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({k:summary[k] for k in ['selected_configuration_count','selected_scored_requests','all_primary_and_recovery_raw_rows','checks_passed']}))

if __name__=='__main__':main()
