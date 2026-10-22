import base64, csv, hashlib, json, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import markdown

R=Path(__file__).resolve().parent
rows=[json.loads(s) for s in (R/'raw_results.jsonl').read_text().splitlines()]
assert len(rows)==36 and len({(r['id'],r['method']) for r in rows})==36
diagnostic=[json.loads(s) for s in (R/'raw_diagnostic8.jsonl').read_text().splitlines()] if (R/'raw_diagnostic8.jsonl').exists() else []
assert len(diagnostic) in [0,12]
prompts={p['id']:p for p in json.loads((R/'prompts.json').read_text())}
for r in rows+diagnostic:
    assert len(prompts[r['id']]['ids'])==r['n']
    logical=r['cache_logical_values']
    expected=logical*2 if r['method']=='bf16' else int(2*36*8*128*((r['n']-64)*(int(r['method'][-1])/8+4/64)+95*2))
    assert expected==r['final_cache']['gpu_bytes'],(r['id'],r['method'],expected,r['final_cache']['gpu_bytes'])
    assert r['final_allocated_bytes']>=expected
    assert r['prefill_peak_allocated_bytes']>=r['initial_gpu_bytes']
    if r['method']!='bf16':
        assert any(x['dtype']=='torch.uint8' and '_quantized' in x['name'] for x in r['final_cache']['first_layer_layout'])

groups=[]
for n in [4096,16384]:
    for m in ['bf16','hqq4','hqq2']+(['hqq8'] if diagnostic else []):
        g=[r for r in rows+diagnostic if r['n']==n and r['method']==m]
        groups.append(dict(context=n,method=m,trials=len(g),correct=sum(r['correct'] for r in g),strict_exact=sum(r['strict_exact'] for r in g),cache_GiB=st.median(r['final_cache']['gpu_bytes']/2**30 for r in g),bits=st.median(r['effective_bits_per_value'] for r in g),decode_ms=st.median(r['decode_ms_per_token'] for r in g),total_s=st.median(r['total_s'] for r in g),conversion_s=st.median(r['compression_s'] for r in g),final_allocated_GiB=st.median(r['final_allocated_bytes']/2**30 for r in g),prefill_peak_GiB=st.median(r['prefill_peak_allocated_bytes']/2**30 for r in g),decode_peak_GiB=st.median(r['decode_peak_allocated_bytes']/2**30 for r in g)))
(R/'summary.json').write_text(json.dumps(groups,indent=2))
with (R/'summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=groups[0]);w.writeheader();w.writerows(groups)
fields=['id','n','position','seed','method','target','answer','correct','strict_exact','prefill_s','document_prefill_s','compression_s','question_s','decode31_s','decode_ms_per_token','total_s','effective_bits_per_value','initial_gpu_bytes','prefill_peak_allocated_bytes','decode_peak_allocated_bytes','final_allocated_bytes']
with (R/'results.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows+diagnostic)

colors={'bf16':'#1764a0','hqq4':'#16846e','hqq2':'#c25736','hqq8':'#8661a8'}
fig,axes=plt.subplots(1,3,figsize=(13,3.8))
for m in ['bf16','hqq4','hqq2']:
    g=[x for x in groups if x['method']==m]
    axes[0].plot([x['context'] for x in g],[x['cache_GiB'] for x in g],'o-',label=m,color=colors[m])
    axes[1].plot([x['context'] for x in g],[100*x['correct']/x['trials'] for x in g],'o-',color=colors[m])
    axes[2].plot([x['context'] for x in g],[x['decode_ms'] for x in g],'o-',color=colors[m])
for a in axes:
    a.set_xticks([4096,16384],['4K','16K']); a.set_xlabel('Input tokens'); a.grid(alpha=.2)
axes[0].set_title('Stored GPU KV, metadata included');axes[0].set_ylabel('GiB'); axes[0].legend()
axes[1].set_title('Six paired synthetic queries / length');axes[1].set_ylabel('Retrieval accuracy (%)');axes[1].set_ylim(-5,105)
axes[2].set_title('Complete-model decode, median');axes[2].set_ylabel('ms / token (lower is better)')
fig.suptitle('Qwen3-4B / RTX 3090 — packed storage with BF16 reconstruction',fontsize=12)
fig.tight_layout();fig.savefig(R/'baseline_frontier.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4))
g=[x for x in groups if x['context']==16384 and x['method']!='hqq8']
labels=[x['method'] for x in g]; xs=list(range(len(g))); width=.23
for j,(field,label) in enumerate([('cache_GiB','Persistent KV only'),('final_allocated_GiB','Final PyTorch allocated'),('prefill_peak_GiB','Peak during ingestion/conversion')]):
    ax.bar([x+(j-1)*width for x in xs],[r[field] for r in g],width,label=label)
ax.set_xticks(xs,labels);ax.set_ylabel('GiB');ax.set_title('16K: shrinking KV does not shrink every memory cost');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
fig.tight_layout();fig.savefig(R/'memory_accounting.png',dpi=160);plt.close(fig)

def table(include_diag=False):
    out=['| Context | Cache | Retrieved / 6 | Strict format / 6 | KV GiB | Effective bits | Decode ms/token | Total seconds |','|---|---|---:|---:|---:|---:|---:|---:|']
    for x in groups:
        if (x['method']=='hqq8')!=include_diag: continue
        out.append(f"| {x['context']:,} | {x['method']} | {x['correct']} | {x['strict_exact']} | {x['cache_GiB']:.3f} | {x['bits']:.3f} | {x['decode_ms']:.1f} | {x['total_s']:.2f} |")
    return '\n'.join(out)
at16={x['method']:x for x in groups if x['context']==16384}
b,q=at16['bf16'],at16['hqq4']
ratio=b['cache_GiB']/q['cache_GiB']; slow=q['decode_ms']/b['decode_ms']
memory_table='\n'.join(f"| {x['method']} | {x['final_allocated_GiB']:.2f} | {x['prefill_peak_GiB']:.2f} | {x['decode_peak_GiB']:.2f} | {x['conversion_s']:.2f} |" for x in at16.values() if x['method']!='hqq8')
diagtext=('\n## Exploratory eight-bit diagnostic\n\nAdded after observing failures in the 2/4-bit sweep. It uses the same 12 prompts, no tuned settings, and a separate run. Its sequential timing is not part of the rotated primary comparison. This is a diagnostic of precision sensitivity, not an independent quality confirmation. All 12 eight-bit answers retrieved the correct code and obeyed the strict output format. Timing was unstable: 4K decode ranged from '+f"{min(x['decode_ms_per_token'] for x in diagnostic if x['n']==4096):.1f}"+' to '+f"{max(x['decode_ms_per_token'] for x in diagnostic if x['n']==4096):.1f}"+' ms/token, and 16K from '+f"{min(x['decode_ms_per_token'] for x in diagnostic if x['n']==16384):.1f}"+' to '+f"{max(x['decode_ms_per_token'] for x in diagnostic if x['n']==16384):.1f}"+'. We have not isolated the cause, so the medians below should not be treated as stable performance comparisons. The primary sweep had much less variation, but it too requires controlled independent timing replication.\n\n'+table(True)) if diagnostic else ''
report=f'''# Low-bit long-context memory: first measured baseline

Completed 2026-10-20. **This is an existing-method feasibility experiment, not a novel algorithm or an acceleration breakthrough.**

At 16K tokens, four-bit HQQ reduced stored KV by **{ratio:.2f}×**, but retrieved the correct code in **{q['correct']}/6** questions versus **{b['correct']}/6** for BF16 and took **{slow:.2f}×** as long per decode token. Only **{q['strict_exact']}/6** four-bit answers also obeyed the exact digits-only output format. The full results determine whether this implementation is useful; a smaller tensor alone does not establish that it is.

## What we did

We screened primary literature through October 20, 2026; installed an isolated package overlay without changing the earlier Mamba environment; downloaded and pinned Qwen3-4B; wrote the protocol before collecting the main sweep; ran three cache formats on 12 identical prompts (36 model runs); audited real packed storage and metadata; measured synchronized GPU execution and PyTorch allocation; and retained prompts, answers, raw metrics, source, and plots. A separate 4K smoke run guided a documented ingestion-memory amendment. {('We also ran 12 post-hoc eight-bit diagnostic cases.' if diagnostic else '')}

The model has about four billion parameters. Its weights and activations stay BF16. We change only the historical attention cache. Every prompt contains a randomly generated six-digit target code among distracting records. We vary input length (4K, 16K), target position (10%, 50%, 90%), and two seeds. The same two target numbers repeat across positions/lengths, so these twelve cases are not twelve independent facts. The full model is used; this is not a standalone attention tensor simulation.

All methods ingest the document in BF16 chunks of 2,048 tokens. HQQ compresses history once at the document boundary. The final 64-token question suffix and generated answer then read the compressed history. This avoids the false test where the answer is computed before compression. The protocol amendment and original smoke measurements are retained.

## Results

{table()}

Retrieval requires the first six-digit number in the answer to equal the target; strict format requires the entire stripped answer to equal those six digits. For example, one 16K four-bit answer contained the correct code with extra text and a stray closing thinking tag, while another changed target `709382` to `309382.`. We report both metrics rather than treating correct numbers embedded in format violations as perfect task execution. The retrieval figure uses the preregistered first-number metric.

Times are medians across six cases per context/method, after short warmups. Total is document ingestion + compression + question processing + a fixed 32-token generation workload; 31 further decode passes follow the first token produced by the question. Generation continues after EOS only to hold benchmark work fixed; quality scoring uses text before EOS. Total time is therefore **not** latency to an answer's natural stopping point. One timing observation per prompt, changing thermal state and concurrent computer activity limit precision. No statistically confirmed speedup or broad quality claim follows from this pilot.

![Storage, quality, and decode timing](baseline_frontier.png)

## Where the bytes and time go

HQQ stores packed uint8 tensors containing two- or four-bit codes. Scale and zero-point tensors are BF16. Each 64-value group needs four metadata bytes, which adds 0.5 bits/value to the bulk representation. The 95-token uncompressed tail adds further overhead in this test. Thus the cache is not exactly 2 or 4 effective bits/value. We enumerate all cache tensor backing storages and deduplicate by device and pointer. Audited bytes exactly match an independent closed-form count for every recorded case. CPU/Python object overhead is excluded and separately distinguished from GPU storage.

The HF implementation dequantizes historical keys/values and concatenates BF16 tensors before ordinary SDPA attention. That creates extra reads, writes, launches, and temporary buffers. Source inspection establishes this implementation path; we have not yet measured DRAM traffic with a hardware profiler. It is **not** native INT4/FP4 arithmetic, a fused packed-cache attention kernel, or a faithful KIVI/TurboQuant reproduction. [Versioned HF source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/cache_utils.py).

Whole-model memory accounting at 16K (medians, GiB):

| Cache | Final allocated | Ingestion/conversion peak | Question/decode peak | Conversion seconds |
|---|---:|---:|---:|---:|
{memory_table}

![Memory accounting](memory_accounting.png)

Allocated memory includes model tensors, input tensors, and cache. It excludes CUDA reservation, driver/context overhead, and other processes, so it is not identical to nvidia-smi. Ingestion temporarily keeps the original BF16 cache for every format: the final cache compression ratio cannot be used to claim this pipeline supports proportionately longer prompts. Measured KV savings are also not a measured increase in concurrent requests; we did not run a serving scheduler.

For the 16K four-bit case, final allocated memory falls from {b['final_allocated_GiB']:.2f} to {q['final_allocated_GiB']:.2f} GiB, but the larger of ingestion and decode phase peaks falls only from {max(b['prefill_peak_GiB'],b['decode_peak_GiB']):.2f} to {max(q['prefill_peak_GiB'],q['decode_peak_GiB']):.2f} GiB (approximately {100*(1-max(q['prefill_peak_GiB'],q['decode_peak_GiB'])/max(b['prefill_peak_GiB'],b['decode_peak_GiB'])):.1f}% less). That is why final KV compression and usable peak-memory reduction need separate measurements.
{diagtext}

## What is already published

Low-bit KV storage and asymmetric key/value quantization are established by [KIVI](https://arxiv.org/abs/2402.02750). Joint low-bit execution/cache system design is established by [QServe](https://arxiv.org/abs/2405.04532). Sensitivity-aware layer precision is covered by [KVTuner](https://arxiv.org/abs/2502.04420); query-aware channel precision by [MixKVQ](https://arxiv.org/abs/2512.19206); quantizer-specific bit allocation by [RateQuant](https://arxiv.org/abs/2605.06675). Rotation-based compression is covered by [TurboQuant](https://arxiv.org/abs/2504.19874).

Agent-structure allocation and real serving implementations are covered by [TriAxialKV](https://arxiv.org/abs/2605.17170) and [UltraQuant](https://arxiv.org/abs/2606.20474). Quantization plus self-speculation is covered by [QuantSpec](https://machinelearning.apple.com/research/quantspec). A very recent [ReadKV paper](https://arxiv.org/abs/2610.11245), submitted October 8, separates retained precision from query-dependent fetched precision. Its abstract describes a restricted single-layer latency evaluation; complete-serving benefits cannot be inferred from that number. These sources rule out calling the broad ideas new. The full source register records evidence status; we have not independently reproduced those papers' results.

## Research direction and next decisive result

The promising question is whether **page-aligned precision refinement and reuse can improve complete-model serving at fixed total memory and matched quality**. A possible mechanism would share an immutable progressive cache, retain a small bounded working set of useful refinements, and group page formats to avoid excessive GPU transactions and launch costs. ReadKV, prefix reuse, and tiered compression are close prior work. This is a candidate intersection, **not verified novelty**. The targeted search cannot certify no prior publication exists. Do not start by claiming a new quantizer or a new idea of query-dependent bits.

First reproduce an optimized packed-cache baseline, then measure whether refinement reuse offers headroom beyond ordinary decoded-page LRU. Only develop a new policy if the attainable benefit exceeds its extra memory, selection, and fetch costs. The present HQQ slowdown is a baseline diagnostic, not an explanation of every existing low-bit system's behavior. A faithful stronger baseline may eliminate the proposed advantage, which would be a reason to change direction.

The intended demonstration is two systems answering long-document or code questions under the same GPU budget while a dashboard shows useful requests completed on time, quality, tail latency, and peak memory. Success must be against a strong compressed-cache system, not only BF16. Suggested future targets are ≤1 percentage point absolute loss on each chosen task and ≥20% more deadline-qualified requests/second; these are **unmeasured research goals**. [NEXT_EXPERIMENT.md](NEXT_EXPERIMENT.md) specifies baselines, ablations, resource rationale, and stop conditions.

## How to explain this when presenting

“The model keeps a numerical notebook of the document so it does not have to reread everything from scratch for every word. I first checked whether reducing the notebook's precision preserved exact answers and actually made generation faster. I measured packed bytes, metadata, temporary memory, and complete-model timing. In this baseline, smaller stored history did not translate into faster decoding, and aggressive compression damaged retrieval. The next research step is a reader and memory policy that improves useful serving throughput over strong existing methods.”

Be precise about scope: a transformer cache is context-sized attention memory, not the fixed-size recurrent state from our earlier project. Easy synthetic lookup is a sanity check; it does not establish reasoning, code accuracy, multi-turn robustness, or general long-context capability. We have no trained new method, no optimized fused implementation, and no publishable novelty claim yet. Those are the next scientific and engineering requirements, not details that the presentation can omit.

## Reproduction and evidence

Model: Qwen/Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`; torch 2.8.0+cu128, Transformers 4.57.1, HQQ 0.2.8.post1, one RTX 3090. BF16 weights, SDPA, TF32 disabled, no thinking, deterministic greedy tokens. Model weights are downloaded separately and are excluded from Git. Requirements and model-file SHA256 manifest accompany the package. See [README.md](README.md) for commands.

Raw main evidence: `raw_results.jsonl`, `prompts.json`, `results.csv`, `summary.json`, `environment.json`; separate exploratory evidence: `smoke.jsonl`{(', `raw_diagnostic8.jsonl`, `environment_diagnostic8.json`' if diagnostic else '')}. Every answer, including failures, is retained. `PROTOCOL.md` records the design and amendments. `sources.json` records the literature screen. `validation.json` records integrity checks, not an independent replication.
'''
(R/'report.md').write_text(report,encoding='utf8')
html=markdown.markdown(report,extensions=['tables','fenced_code'])
for name in ['baseline_frontier.png','memory_accounting.png']:
    html=html.replace(f'src="{name}"',f'src="data:image/png;base64,{base64.b64encode((R/name).read_bytes()).decode()}"')
css='body{font:17px/1.65 system-ui;max-width:1060px;margin:50px auto;padding:0 28px;color:#192a38}h1,h2{line-height:1.2}h2{margin-top:2em}table{border-collapse:collapse;font-size:15px}td,th{padding:8px 12px;border-bottom:1px solid #ccd6de;text-align:left}img{width:100%;height:auto}a{color:#175f92}code{font-size:.88em;background:#edf2f5;padding:2px 4px}'
(R/'report.html').write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><title>Low-bit KV research baseline</title><style>{css}</style><body>{html}</body></html>',encoding='utf8')
checks={'main_rows':len(rows),'exploratory_8bit_rows':len(diagnostic),'paired_main_prompts':len(prompts),'cache_byte_formula_matches':len(rows)+len(diagnostic),'packed_uint8_verified':True,'no_duplicate_main_cases':True,'note':'Integrity checks, not independent replication or proof of quality neutrality.'}
(R/'validation.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(groups,indent=2))
