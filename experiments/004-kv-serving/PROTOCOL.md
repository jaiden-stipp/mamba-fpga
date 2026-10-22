# Packed-KV serving experiment — 2026-10-20

Written before collecting serving results. This is a reproduction/feasibility study with an existing optimized engine. It does not introduce a new quantizer or claim novel fused attention.

## Question and primary outcome

Can compressed KV support more simultaneous long-context requests at matched answer quality and response deadlines? Count **correct, deadline-qualified responses per second**, with quality and latency also reported separately. Capacity (whether contexts fit), raw tokens/second, and qualified responses/second are different outcomes.

An affirmative result requires quality to remain within the stated tolerance, and more qualified work over a comparable workload. Fitting eight contexts while four were previously the limit does not itself prove better throughput. A failed/OOM configuration is recorded as unavailable; never silently enable CPU offload or shorten its context.

## Controlled factors

- Qwen3-4B from the exact checkpoint in the preceding baseline. Converted to GGUF BF16; audit conversion metadata and hashes. The converter may retain some small tensors as F32. All comparisons use the same converted file.
- Official llama.cpp build b11751, source commit `84663953535d4323e92b4bc90971aa022c305d06`, CUDA 12.4 Windows binaries. Archive release hashes and binary version. This is a different engine and quantizer from HF HQQ: do not attribute cross-engine differences solely to fusing unpacking.
- KV types: F16 reference, Q8_0, Q4_0, same type for K and V. Model weights fixed; all layers and KV on one RTX 3090. Flash Attention on; no fitting, CPU fallback, speculative decoding, context shifting, or host prompt-cache storage. Requested context/slot count must match engine logs.
- Source inspection locates packed-key dot products and value dequantization inside CUDA Flash Attention implementations. Startup and allocation logs must establish GPU placement and requested format. Without a GPU profiler we do not claim measured DRAM-byte reductions or exact runtime kernel-dispatch verification.

## Workloads and pilot scope

Use fresh deterministic synthetic documents with exact known answers: single-record retrieval, eight-record extraction, latest-update lookup, and constant-return code-symbol lookup. Keep these distinct from the previous two-code needle tests. Target 16,384 prompt tokens, with target records distributed through the document; allow each native context to hold at least 1,024 additional tokens. Disable thinking with the official tokenizer's chat template. Greedy generation stops naturally at EOS, maximum 256 output tokens. Do not force post-EOS text or apply an answer grammar that masks format errors. Record exact input IDs and expected answers.

First pilot: one GPU, slots 1/4/8, three KV formats, eight cold requests arriving together per configuration; then four resident-context rounds per active slot, one per task type. Each document contains all four types of evidence so task distribution remains balanced at every slot setting. Rotate format order by slot setting. Each resident round uses the same document with a new query; require the engine's reported reused-token count (`timings.cache_n`, not final cache occupancy `tokens_cached`) to cover the document, and report cache misses separately. This closed-loop resident experiment measures one request at a time per client with no think time. Its client populations change with slot count and are a capacity/load curve, not identical exogenous arrival traces.

Cold runs disable cache reuse. Resident preloading and any fallback/recomputation are recorded separately; warm-cache latency must not be presented as cold startup latency. Short warmup requests are excluded. Preserve all failure outputs and timing observations. The pilot is not enough to claim ≤1 percentage point noninferiority, workload-general quality, steady-state open-loop throughput, or publication-ready results.

**Provisional application requirements, fixed before timing:** first generated token within 20 seconds; mean post-first-token time per token ≤100 ms; full completion within 45 seconds. These are illustrative interactive-document requirements, not user-specified production SLOs. Exclude load/setup from warm serving latency, but report its cost. Report p95 individual streamed token intervals as a supplementary jitter measure. A response qualifies only if all deadlines and exact semantic answer checks pass. Also report strict output-format accuracy. Prevent prompt truncation; record any request timeout or server failure as a failure, not a dropped row. Client uses a 120-second timeout.

The first pilot establishes feasibility and identifies the strongest compatible cache baseline. Subsequent confirmation should use ≥3 independent timing trials, fresh held-out documents/questions, two model families, 8K–128K contexts supported by the checkpoint, open-loop arrival rates and bursts, and a larger GPU where needed. At most 1 percentage point per-task quality loss and ≥20% more qualified throughput over a strong existing compressed baseline remain **unmeasured future targets**. Approximate noninferiority at this resolution needs many more independent cases and paired confidence bounds.

## GPU access

Preflight found LM Studio and Ollama occupying most GPU VRAM. No other service is terminated or unloaded without user authorization. Prepare and validate the harness while that access is pending. Refuse GPU timing runs if competing allocations leave insufficient memory; do not use WDDM spill/offload as a hidden substitute for GPU capacity. All subprocesses launched by the harness are tracked and cleaned up by their own process handles.

## Research boundary

KIVI, QServe, SAW-INT4, UltraQuant, and ReadKV already cover important combinations of low-bit storage, efficient access, serving layouts, and query-adaptive precision. A successful stock-engine comparison answers feasibility on this machine, not novelty. A potential new refinement-reuse policy should be pursued only after an oracle/trace analysis shows benefits beyond fixed compression and ordinary decoded-page LRU at equal total memory.
