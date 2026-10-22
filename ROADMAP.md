# Research roadmap

Updated 2026-10-22. This is a dependency-based plan, not a schedule or evidence of completion.

| Milestone | State | Exit evidence |
|---|---|---|
| Phase arithmetic and bounded RTL | Completed pilot, E001 | Numerical results, simulation/build evidence, 10 board runs; no full-model speed claim |
| Explicit vs learned updatable memory | Completed synthetic study, E002 | Matched-byte controls and fresh confirmation; no competitive learned mechanism established |
| Real-model KV storage | Completed pilot, E003 | Raw answers, tensor storage, peak allocation and timing; no serving win |
| Optimized packed-cache baseline | Prepared, E004 | Available GPU → 16K smoke → placement/format audit → cold/resident pilot |
| Confirm useful serving improvement | Not started | Repeated trials, held-out tasks, matched quality, identical arrival traces and deadlines |
| Refinement-reuse headroom | Proposed, I001 | Oracle/trace advantage over fixed compression and ordinary LRU at equal total memory |
| Select a useful FPGA target | Proposed, I002 | Whole-model bottleneck, closest prior art and realistic bandwidth/resource model |
| Build a new mechanism | Conditional | Reference validation, ablations, synthesis/board or GPU results and meaningful system benefit |

Next: review FlashAttention and FlightLLM; obtain a free GPU without interrupting another service without authorization; execute E004's frozen pilot; pursue I001/I002 only if measured headroom justifies them.

Hardware requests should specify scientific hypothesis, minimum memory/bandwidth, execution features, software compatibility, experiment count, and the claim current hardware cannot test. No paid compute is rented or assumed.
