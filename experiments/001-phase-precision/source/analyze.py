import csv,json,math,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import ROOT,RESULTS

OUT=ROOT.parent  # Keep generated analysis inside this experiment.
OUT.mkdir(parents=True,exist_ok=True)
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,'savefig.bbox':'tight'})
NAMES=['mamba3-siso-187m','mamba3-mimo-187m'];COLORS=['#1967a3','#c36320']
summary={'models':{},'paired_comparisons':[]}

def bootstrap(delta,seed=20261019):
    delta=np.asarray(delta,dtype=np.float64);rng=np.random.default_rng(seed)
    samples=delta[rng.integers(0,len(delta),size=(20000,len(delta)))].mean(1)
    return np.percentile(samples,[2.5,97.5]).tolist()

def losses(name,stage,label):
    path=RESULTS/f'{name}-{stage}-{label}-losses.npy'
    if not path.exists():path=RESULTS/f'{name}-{label}-losses.npy'
    return np.load(path).astype(np.float64)

csvrows=[]
for name in NAMES:
    summary['models'][name]={}
    for stage in ['main','replicate']:
        rows=json.loads((RESULTS/f'{name}-{stage}.json').read_text())
        ref=losses(name,stage,'reference').mean(1)
        out=[]
        for r in rows:
            w=losses(name,stage,r['label']).mean(1)
            delta=w-ref;ci=bootstrap(delta)
            x={'label':r['label'],'windows':len(w),'tokens':r['tokens'],'nll':float(w.mean()),'ppl':float(np.exp(w.mean())),
               'delta_nll':float(delta.mean()),'relative_ppl_percent':float(100*np.expm1(delta.mean())),
               'delta_nll_95ci':ci,'relative_ppl_percent_95ci':(100*np.expm1(ci)).tolist()}
            out.append(x);csvrows.append({'model':name,'stage':stage,**{k:v for k,v in x.items() if not isinstance(v,list)}})
        summary['models'][name][stage]=out
    for other in ['fixed10','mixed_decay_error','mixed_magnitude']:
        diff=losses(name,'replicate','mixed_phase_relative').mean(1)-losses(name,'replicate',other).mean(1)
        summary['paired_comparisons'].append({'model':name,'contrast':f'phase_relative minus {other}','delta_nll':float(diff.mean()),'delta_nll_95ci':bootstrap(diff)})
    long=json.loads((RESULTS/f'{name}-long.json').read_text())
    for r in long:
        baseline=next(x for x in long if x['length']==r['length'] and x['mode']=='reference')
        r['relative_ppl_percent']=float(100*np.expm1(r['nll']-baseline['nll']))
    summary['models'][name]['long']=long
    summary['models'][name]['state']=json.loads((RESULTS/f'{name}-state.json').read_text())
    summary['models'][name]['retrieval']=json.loads((RESULTS/f'{name}-retrieval.json').read_text())

with (OUT/'language_model_results.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))

labels=['fixed8','fixed10','fixed12','guard16_8_trig8','mixed_magnitude','mixed_decay_error','mixed_phase_relative']
pretty=['8-bit\naccumulator','10-bit\naccumulator','12-bit\naccumulator','16-bit acc.\n8-bit LUT + coeff.','Mixed\nmagnitude','Mixed\ndecay + error','Mixed\nrelative phase']
fig,ax=plt.subplots(figsize=(11,4.4));x=np.arange(len(labels));width=.35
for i,name in enumerate(NAMES):
    rr={r['label']:r for r in summary['models'][name]['replicate']}
    y=np.array([rr[k]['relative_ppl_percent'] for k in labels])
    ci=np.array([rr[k]['relative_ppl_percent_95ci'] for k in labels])
    ax.bar(x+(i-.5)*width,y,width,color=COLORS[i],label=['SISO 187M','MIMO 187M'][i])
    ax.errorbar(x+(i-.5)*width,y,yerr=np.array([y-ci[:,0],ci[:,1]-y]),fmt='none',ecolor='#283342',capsize=3,lw=1)
    for xi,yi,upper in zip(x+(i-.5)*width,y,ci[:,1]):ax.text(xi,max(yi,upper)+.1,f'{yi:.2f}',ha='center',va='bottom',fontsize=8)
ax.set_xticks(x,pretty);ax.set_ylabel('Perplexity increase over matched reference (%)');ax.set_ylim(-.05,6.2)
ax.set_title('Independent text replication: selector does not beat uniform precision',loc='left',weight='bold',pad=15)
ax.legend(frameon=False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.text(.12,-.03,'32 disjoint 2,048-token windows per model. Error bars: paired window-bootstrap 95% intervals.\nMixed policies: 25% of accumulators at 16 bits, 75% at 8 bits (10-bit nominal mean; metadata excluded).',fontsize=8,color='#4e5968')
fig.savefig(FIG/'replication.png');plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
for ax,name in zip(axes,NAMES):
    for mode,color in zip(['fixed8','fixed10','fixed12','guard16_8_trig8'],['#c36320','#ad8540','#1967a3','#148069']):
        rows=[r for r in summary['models'][name]['long'] if r['mode']==mode]
        ax.plot([r['length'] for r in rows],[r['relative_ppl_percent'] for r in rows],marker='o',label=mode.replace('guard16_8_trig8','16 / 8 / 8'))
    ax.set_title(name.replace('mamba3-','').upper());ax.set_xscale('log',base=2);ax.set_xticks([8192,32768,65536],['8K','32K','64K']);ax.set_xlabel('Uninterrupted prefix length (tokens)');ax.grid(alpha=.15)
axes[0].set_ylabel('Perplexity increase over matched reference (%)');axes[1].legend(frameon=False,fontsize=8)
fig.suptitle('Long-prefix stress test: accumulator precision matters',x=.12,ha='left',weight='bold')
fig.text(.12,-.04,'One concatenated WikiText prefix per length; only final 2,048 predictions scored.\nEach checkpoint was trained at 2K context. Different lengths score different text; these are not retrieval results.',fontsize=8,color='#4e5968')
fig.savefig(FIG/'long_prefix.png');plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
for ax,name in zip(axes,NAMES):
    for mode,color in zip(['FP16_RTN','FP16_SR','INT8_dynamic','INT16_dynamic'],['#1967a3','#6e91ad','#c36320','#148069']):
        rr=[r for r in summary['models'][name]['state'] if r['label']==mode]
        ax.plot([r['layer'] for r in rr],[100*r['relative_rms'] for r in rr],marker='o',label=mode)
    ax.set_yscale('log');ax.set_xticks([0,5,11]);ax.set_xlabel('Layer index');ax.set_title(name.replace('mamba3-','').upper());ax.grid(alpha=.15)
axes[0].set_ylabel('SSM output relative RMS error (%)');axes[1].legend(frameon=False,fontsize=8)
fig.suptitle('State quantization remains a separate problem',x=.12,ha='left',weight='bold')
fig.text(.12,-.04,'Frozen real activations over 8,192 tokens; four selected heads and eight value channels.\nOnly state storage is quantized after each token. This is a block diagnostic, not full-model perplexity.',fontsize=8,color='#4e5968')
fig.savefig(FIG/'state_quantization.png');plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
for ax,name in zip(axes,NAMES):
    for mode,color in zip(['reference','fixed8','guard16_8_trig8'],['#283342','#c36320','#148069']):
        rr=[r for r in summary['models'][name]['retrieval'] if r['mode']==mode]
        ax.plot([r['length'] for r in rr],[100*r['constrained_accuracy'] for r in rr],marker='o',label=mode.replace('guard16_8_trig8','16 / 8 / 8'))
    ax.set_xscale('log',base=2);ax.set_xticks([512,2048,8192],['512','2K','8K']);ax.set_ylim(0,105);ax.set_title(name.replace('mamba3-','').upper());ax.set_xlabel('Prompt length (tokens)');ax.axhline(12.5,ls=':',color='#666',label='8-way chance');ax.grid(alpha=.15)
axes[0].set_ylabel('Constrained one-word recall accuracy (%)');axes[1].legend(frameon=False,fontsize=8)
fig.suptitle('Recall probe: low absolute long-context capability limits claims',x=.12,ha='left',weight='bold')
fig.text(.12,-.04,'24 prompts per model/length/mode: eight target words, three needle positions.\nCustom diagnostic; candidate-constrained accuracy differs from unconstrained greedy accuracy.',fontsize=8,color='#4e5968')
fig.savefig(FIG/'recall.png');plt.close(fig)
print(json.dumps({'out':str(OUT),'contrasts':summary['paired_comparisons']},indent=2))
