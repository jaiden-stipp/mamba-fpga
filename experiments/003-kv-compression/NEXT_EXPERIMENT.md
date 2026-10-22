# Research direction after the feasibility baseline

## Objective

Serve more simultaneous long-document requests within a fixed GPU memory budget, while preserving useful answers and meeting response deadlines. This is the direct presentation outcome. Cache compression ratio is an intermediate measurement.

The first experiment isolates KV compression. Model weights and activations stay BF16. This keeps us from mistakenly crediting a weight-quantization speedup to a cache method. Joint weight/KV optimization is a later controlled factor, with QServe and QuantSpec as relevant precedents.

## What can still become research

Candidate question, **novelty not established**: can a page-aligned progressive KV reader choose and reuse precision refinements across related requests cheaply enough to improve *complete-model* serving, compared with fixed low-bit and query-adaptive baselines at the same retained bytes and quality?

ReadKV already separates stored bits from read bits and uses the current query to choose precision. Prefix caching, cold-page compression, and load-aware precision also exist. We must not relabel any of those as new. A possible contribution would be a new joint representation/reader/scheduling mechanism that makes refinement reuse economical under actual GPU transactions, shared prefix lifetimes, and changing query topics. This proposal is an intersection to investigate, not a claim that no paper has studied it.

The mechanism to investigate: store one immutable progressive representation of shared history; keep only a bounded working set of decoded or refined pages; choose page refinement using estimated attention error and expected reuse; group compatible page formats for attention so fine-grained decisions do not create excessive launches or divergence. Charge the working set, index, scales, padding, and predictor to the memory budget. Compare against ordinary decoded-page LRU before claiming anything from a learned policy. Refinements recover bits retained in the backing representation; expanding a 2-bit number to BF16 cannot recover information discarded at encoding.

Do not implement a complex controller before measuring whether the target workload has enough cross-query refinement reuse. A trace study should measure reuse distance of useful refinements, topic changes, actual bytes fetched, and attainable oracle benefit. If an oracle cannot beat fixed precision after charging the working set, stop this candidate early. An oracle is an upper-bound diagnostic, not a deployable result. Reusing an approximate prefix must preserve the selected numerical semantics; independently computed full-precision prefixes are not automatically identical to prefixes produced during compressed-history ingestion.

## Decisive experiment

1. Reproduce an optimized packed-cache baseline with a real language model. Verify kernel output against a numerical reference. Inspect whether packed data is decoded inside attention or materialized as a whole BF16 history. Establish BF16 and the engine's best supported quantized-cache baseline. Add faithful rotation/codebook and mixed-precision comparisons where compatible; document implementation differences.
2. Establish quality on held-out long-document QA, exact IDs/numbers, code symbol lookup and executable code tasks, plus multi-turn task switches. Use a validated long-context checkpoint. Keep calibration topics and query families disjoint from test data. Run model replication, not just one model family. The simple needle experiment is an implementation check, not that quality suite.
3. Compare uniform low precision, offline mixed precision, query-adaptive precision, query-adaptive plus ordinary page LRU, and the proposed reuse-aware reader. Hold retained memory and decoded working memory fixed. Include a random allocation control at matched bytes and an oracle for headroom.
4. Sweep context (8K/32K/64K/128K where natively supported), concurrency (1/2/4/8 and saturation), shared-prefix rate, topic shift, and bursty arrivals. Measure qualified requests/second, p50/p95 time to first token, p50/p95 per-output-token latency, task accuracy, allocated/reserved GPU memory, host backing memory, and profiler DRAM traffic. Include all encoding, selection, paging, conversion, and fallback time.
5. Test new representation alone, reader alone, working-set policy alone, and their combination. Repeat independent timing trials and arrival traces; report paired confidence intervals. A better plot at batch one does not establish better serving.

Suggested go/no-go gates, to preregister before that experiment: at most 1 percentage point absolute loss on each selected task relative to BF16, and at least 20% more SLO-qualified requests/second than the strongest compatible existing compressed-cache baseline at the same memory and quality allowance. These are research targets, not observed gains. Choose actual SLOs from the target application's needs before evaluating, not from the fastest result. Report every workload, including regressions.

## Hardware and software

The current machine has two 24GB RTX 3090 GPUs. The first paired baseline uses one GPU. Existing Ubuntu WSL2 sees both GPUs, but no CUDA compiler was found in its shell path. This provides a route for a supported Linux research stack; it is not yet a configured or validated fused-kernel environment.

A defensible resource request is temporary access to an 80GB-class GPU for 8B-scale models, longer contexts, batching, and profiler work, followed by a second architecture for replication. Request Blackwell specifically only if the hypothesis uses native FP4 execution; INT4 packed storage with BF16 arithmetic is a different experiment. AMD CDNA4 would be relevant to reproducing UltraQuant's native path. No hardware purchase is needed to establish the present baseline.

For the current 36-layer/8-KV-head/128-dimensional architecture, BF16 KV is `2 * 36 * 8 * 128 * 2 = 147,456 bytes/token/request` (144 KiB), before allocator overhead. At 64K tokens this is 9 GiB per request, or 36 GiB for four requests, plus approximately 7.49 GiB weights and scratch. This is an architectural extrapolation, **not a claim this checkpoint has validated 64K quality**, nor a measured concurrency result. With four-bit payload and BF16 scale/zero per 64 values the idealized bulk representation is 4.5 bits/value, before full-precision tail, page/index overhead, working sets, and scratch. Larger hardware expands the experiment; it does not make a novelty claim stronger by itself.

The DE2-115 could host a later narrow codec/dataflow experiment if the research mechanism justifies it. It is not the limiting platform or the proposed full-model serving target.

## Presentation explanation

“An LLM keeps an attention notebook for the text it has already read. I am investigating how to make that notebook smaller and cheaper to consult while preserving useful answers. Fewer stored bits do not guarantee faster answers: decoding the notebook can cost more than the reads it saves. My success criterion is more useful requests completed on time on the same GPU, against strong existing compression methods.”

Likely audience questions:

- **Is four-bit inference new?** No. The potential contribution must be a specific new mechanism and measured gain over existing systems.
- **Did you accelerate four-bit arithmetic?** The first HQQ baseline stores packed low-bit KV and reconstructs BF16 for attention. It establishes neither native four-bit arithmetic nor fused low-bit attention performance.
- **Why use an easy retrieval task?** It verifies that compressed history is actually used and detects gross corruption. Broader code/QA tasks and longer multi-turn traces are required for general quality claims.
- **Why can cache shrink much more than total memory?** Weights, temporary activations, attention scratch, and the full-precision tail still occupy GPU memory. The first pipeline also ingests documents in BF16 before compressing them.
- **What connects this to recurrent memory?** Both ask how much past information to retain under a budget, but a transformer KV cache grows with context; it is not a fixed-size recurrent state. This project intentionally studies the more directly deployable transformer inference problem first.
- **What would falsify the idea?** No full-system benefit over a tuned fixed quantizer, no reusable refinements, large task-specific errors, or predictor/working-set overhead erasing saved traffic. A negative baseline is useful for choosing the next experiment, but it is not the intended final research contribution.
