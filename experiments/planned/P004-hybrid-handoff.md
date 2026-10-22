# P004 — Can a GPU–FPGA split repay its transfer cost?

Status: proposed; analytical gate first, new hardware only if the gate passes.

**Question.** Which split, context length and output length make a compressed-memory FPGA engine useful alongside a GPU? [GLITCHES](https://ieee-hpec.org/wp-content/uploads/2024/09/112.pdf) already combines GPU prefill with FPGA decode; [FAST-Prefill](https://arxiv.org/abs/2602.20515) already accelerates dynamic sparse prefill. The split itself is established. A candidate contribution needs a new measured reader, transfer representation or scheduling/dataflow mechanism that beats these appropriate controls.

Compare GPU-only; one-time prefill-to-decode handoff; per-layer attention offload; and an FPGA codec/reader feeding GPU attention. Start with a calibrated lower bound, then replace assumed bandwidth/latency by measurements on available or requested hardware. Never claim that a DE2-115 has PCIe/HBM or can serve a complete 4B model.

For handoff, require `handoff_time + output_tokens * fpga_decode_time < output_tokens * gpu_decode_time`, with setup, conversion and synchronization included. If FPGA decode is slower, no positive output length repays a pure serial handoff; throughput/energy overlap must be tested as a separate system objective. For per-layer offload, sum every query transfer, output transfer and round-trip latency. A fast attention tile can still lose to 36 synchronizations per token.

Use the exact model KV dimensions: `2 * layers * physical_KV_heads * head_dim * context_tokens * bytes_per_scalar`, then add scales, allocation padding and transfer headers. Keep GQA's physical heads separate from its larger query-head count. Include weight residency/duplication, host staging, link topology, pinned-memory setup and GPU-peer DMA support. Measure effective bandwidth for representative sizes and contention; peak link specifications are only upper bounds.

Sweep 8K/32K/128K supported contexts, 32/256/2048 output tokens, batch/session counts and cold/resident scenarios. Produce a break-even surface for transfer bandwidth, FPGA decode latency and output length. Add pessimistic/optimistic bounds so a hardware request names the measured performance needed to change the conclusion.

**Gate.** If an idealized accelerator with zero reader overhead still cannot improve the stated end-to-end latency or energy target by 20%, stop this split. If it can, benchmark the actual proposed engine and the best GPU compressed baseline under identical arrivals and quality. Use hardware budget for HBM capacity/bandwidth, direct transfer support and instrumentation rather than buying a larger board without an experiment it enables.

**Presentation.** A timeline exposes transfer and compute, followed by a plot showing the exact region where the hybrid wins. A well-supported negative break-even result prevents an unproductive architecture; it is not by itself a novel accelerator.
