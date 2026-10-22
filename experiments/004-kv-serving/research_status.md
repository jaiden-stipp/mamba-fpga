# Can cheaper compressed-memory reads improve long-context serving?

Status: 2026-10-20. **The serving experiment is prepared and integration-tested. No GPU serving results have been collected in this phase.**

Yes, this can work in appropriate workloads, and the broad result is already published. The open question for our project is whether a specific representation/reader/policy improves **correct responses completed on time** over strong existing compressed-cache systems. The previous HQQ experiment did not show this: its four-bit cache was smaller, but quality and latency deteriorated.

## What the hypothesis actually says

Each decoding step consults attention keys and values from previous tokens. With many long contexts resident at once, reading this history can dominate memory traffic. Compressed cache data can help twice: more histories fit, and each decode may fetch fewer bytes. Both benefits depend on implementation. A reader that reconstructs the entire history in GPU memory adds work and temporary storage. A fused reader can decode small pieces inside attention and avoid that full intermediate history.

This does not guarantee lower latency. Unpacking, scaling, register pressure, query quantization, irregular reads, and additional kernel launches can outweigh the traffic saved. At larger batches the bottleneck can move to arithmetic. On cold requests, document ingestion and queueing can dominate. We therefore distinguish capacity, token throughput, and useful response throughput.

The measured outcome will be:

`qualified response rate = correct responses meeting every response deadline / measured elapsed time`

A configuration must also preserve task quality. A high rate obtained by giving wrong answers is not success. A small pilot can expose large regressions but cannot establish one-percentage-point quality noninferiority. We will need fresh independent cases, confidence bounds, model replication, and stronger workloads for that claim.

## What already exists

- [KIVI](https://arxiv.org/abs/2402.02750) already combines low-bit asymmetric KV quantization with a hardware-conscious implementation and reports real serving throughput gains. Low-bit cache plus higher batch capacity is not a new idea.
- [QServe](https://arxiv.org/abs/2405.04532) already co-designs low-bit weights, activations, KV, and execution. Fixing model weights in our comparison avoids confusing a weight-quantization benefit with a cache benefit.
- [SAW-INT4](https://arxiv.org/abs/2604.19157), submitted April 2026, directly addresses paged layouts, regular access, fused attention, and quality-preserving four-bit KV using lightweight rotation. It is a particularly close baseline for any proposed new serving-compatible codec.
- [UltraQuant](https://arxiv.org/abs/2606.20474) studies context-heavy multi-round agents and hardware-specific four-bit serving paths. Its reported results apply to its tested systems and are not results from our machine.
- [ReadKV](https://arxiv.org/abs/2610.11245), submitted October 8, separates retained bits from query-dependent fetched bits. Query-adaptive refinement itself is already prior work. Its abstract describes restricted single-layer latency evidence; that number is not a complete-model serving result.

A possible new direction is economical reuse of precision refinements across requests with shared histories, accounting for decoded working memory and physical GPU access costs. This remains a **candidate**, not confirmed novelty. Prefix caching and tiered compression are also established, and simply combining named techniques is not sufficient. First show that an oracle can exploit refinement reuse better than fixed quantization and ordinary decoded-page LRU at the same total memory. If that headroom is absent, do not build an elaborate controller.

## The prepared experiment

We pinned official llama.cpp build b11751 (commit `84663953535d4323e92b4bc90971aa022c305d06`) and CUDA 12.4 Windows binaries. Release digests matched the downloaded archives. The engine is an existing optimized baseline; no new kernel is claimed. Its CUDA source has packed-key dot products and value dequantization inside Flash Attention kernels for Q4_0 and Q8_0. Actual GPU placement and format need verification in startup logs, and a profiler is still required to establish runtime kernel dispatch and physical DRAM bytes. [Pinned kernel source](https://github.com/ggml-org/llama.cpp/blob/84663953535d4323e92b4bc90971aa022c305d06/ggml/src/ggml-cuda/fattn-vec.cuh), [server documentation](https://github.com/ggml-org/llama.cpp/blob/84663953535d4323e92b4bc90971aa022c305d06/tools/server/README.md).

The original immutable Qwen3-4B checkpoint was converted to a single GGUF file with BF16 matrices and F32-promoted small vectors. The audit compared all 398 tensors and all 4,022,468,096 parameter values against the source, preserving BF16 bits and F32 values exactly. This verifies conversion, not cross-engine numerical equivalence. The same file is used for F16, Q8_0, and Q4_0 KV comparisons.

Eight fresh synthetic documents provide 32 queries across four task types: single-record lookup, eight-record extraction, latest-updated-fact lookup, and constant-return code-symbol lookup. Each document contains all task types. Inputs are 16,355–16,384 tokens. The official non-thinking chat template supplies identical IDs to every cache format. Natural EOS stopping is used, with a maximum of 256 output tokens. Correctness is scored against known synthetic answers, with strict JSON formatting also reported.

The pilot sweeps 1, 4, and 8 server slots. Cold tests send the same eight requests concurrently and include server queueing. Resident tests load a distinct history into each slot and then ask four successive questions, one from each task type. Resident loading is charged separately and cache misses are recorded. Varying resident-client count is a load curve, not an identical external arrival trace. The protocol fixes provisional deadlines of 20 seconds to first token, 100 ms average post-first-token time per token, and 45 seconds to completion. These are illustrative application requirements; they were not supplied by the user or selected after seeing GPU timing results.

## Why more slots might fit: estimates only

For this architecture, F16 KV payload is `2 × 36 layers × 8 KV heads × 128 dimensions × 2 bytes = 144 KiB/token/request`. GGML Q8_0 uses a two-byte scale plus 32 one-byte values per block; Q4_0 uses a two-byte scale plus 16 packed bytes for 32 values. Bulk effective precision is therefore 8.5 and 4.5 bits/value respectively, before engine allocation overhead. These formats differ from HQQ.

The harness reserves 17,408 tokens per slot, including generation headroom:

| KV format | Estimated KV per slot | Estimated KV for 8 slots |
|---|---:|---:|
| F16 | 2.391 GiB | 19.125 GiB |
| Q8_0 | 1.270 GiB | 10.160 GiB |
| Q4_0 | 0.672 GiB | 5.379 GiB |

Approximately 7.5 GiB of model parameters and additional runtime buffers must also fit. Eight F16 contexts exceed a 24 GiB card before scratch; eight Q8_0 contexts may fit. These are storage calculations, **not measured capacity or throughput results**. More slots could still miss response deadlines, and Q4_0 could fail quality. Actual startup allocation and serving output decide the result.

## Completed checks and current access constraint

The client passed a real local HTTP/SSE test that delivers tokens with known delays. First-token timing was observed before the final response; mean token timing and semantic/format checkers passed. A CPU-only native-engine smoke produced `["123456"]`, matched the HF tokenizer IDs, and exercised prefix reuse. Its timings and simple answer are excluded from GPU performance and long-context quality claims.

Source inspection also exposed an accounting trap: the native `tokens_cached` response field is final cache occupancy, while `timings.cache_n` is the number of prompt tokens actually reused. The harness uses the latter for resident-cache validation.

Preflight found loaded LM Studio and Ollama model processes occupying most GPU memory. The latest GPU-0 check showed about 18,461 MiB allocated and 5,866 MiB free. The serving harness refused to start a GPU timing run. Neither model was unloaded and no existing service was terminated. Temporary unloading needs user authorization because it changes those apps' loaded-model state. No permission to unload them has been received yet.

The remaining immediate step is to free a GPU, run the real 16K smoke, inspect placement/format logs, and execute the pilot. If local resources remain unavailable, the same controlled experiment can be moved to an accessible larger GPU; no remote compute has been rented or assumed. The DE2-115 is not a full-model serving substitute for this experiment.

## What a defensible positive result would look like

First establish a quality-preserving compressed baseline and report how many requests finish correctly on time, along with p50/p95 first-token latency, mean and tail token intervals, completion latency, setup/ingestion cost, allocated memory, and observed failures. Repeat timing trials and then replay identical open-loop arrival traces. Do not claim a serving win from a cache-only byte ratio, one short answer, a CPU fallback, a forced context reduction, or maximum slot count alone.

The presentation demonstration should show two systems answering the same long-document tasks while counters display correct on-time completions and response latency. The concise explanation is: “I am trying to let more people use long documents on the same GPU by making the model's attention memory cheaper to consult. I count useful answers delivered on time, and include the work needed to decode that memory.”

No affirmative result for our system, new precision policy, or publishable novelty claim is established yet. The prepared benchmark makes the next result concrete and reviewable.
