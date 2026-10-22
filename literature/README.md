# Literature and reading path

New: [2026-10-22 extensive review and experiment recommendations](review-2026-10-22.md), [38-source register](sources-2026-10-22.json), and [ranked experiment protocols](../experiments/planned/README.md). Inspection depth and novelty limits are explicit.

Review focus: memory-aware FPGA acceleration of low-bit inference. Ask how representation, on-chip buffering, and arithmetic dataflow together turn smaller storage into useful full-system gains. Primary links were reviewed on 2026-10-20; this is not an exhaustive novelty certification.

| Order | Primary source | Review focus |
|---|---|---|
| 1 | [FlashAttention](https://arxiv.org/abs/2205.14135) | IO-aware tiling and avoiding intermediates; GPU reasoning to adapt, not an FPGA implementation |
| 2 | [KIVI](https://arxiv.org/abs/2402.02750) | Key/value axes, residual precision, metadata and inference evaluation |
| 3 | [FlightLLM](https://arxiv.org/abs/2401.03868) | FPGA mapping, DSP organization, mixed precision, memory hierarchy and compilation |
| 4 | [SAW-INT4](https://arxiv.org/abs/2604.19157) | 2026 preprint: serving-compatible four-bit layouts, lightweight rotation, fused execution |
| 5 | [ReadKV](https://arxiv.org/abs/2610.11245) | October 2026 preprint: retained vs query-fetched precision; single-layer timing is not complete serving |

For Mamba, continue with [Mamba-2/SSD](https://arxiv.org/abs/2405.21060), [Mamba-3](https://arxiv.org/abs/2603.15569), [FastMamba](https://arxiv.org/abs/2505.18975), and [LowRank-SSM](https://arxiv.org/abs/2608.02954). The latter is a recent FPGA preprint revised October 9; use the revised version for current reported results. These cover recurrence/model structure, quantization/nonlinear approximation, and projection traffic/datapath co-design.

Close prior art also includes [QServe](https://arxiv.org/abs/2405.04532), [KVTuner](https://arxiv.org/abs/2502.04420), [MixKVQ](https://arxiv.org/abs/2512.19206), [UltraQuant](https://arxiv.org/abs/2606.20474), and [QuantSpec](https://machinelearning.apple.com/research/quantspec). Low bits, adaptive precision, query awareness, FPGA mapping, and speculation alone are established ideas.

For each review, record workload/model, precision, bottleneck, dataflow, external/on-chip bytes, logic/DSP/memory cost, achieved clock, quality cost, baseline configuration, full-system result, and limitation. Distinguish measurements, estimates, simulations, and source inspection.

Historical reviews: [Mamba opportunities](research_opportunities.md), [industry landscape](industry-landscape-2026-10-19/research_landscape.md), [landscape sources](industry-landscape-2026-10-19/sources.json), and [KV source register](../experiments/003-kv-compression/sources.json).
