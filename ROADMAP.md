# Research roadmap

Updated 2026-10-22. This is a dependency-based plan, not a schedule or evidence of completion.

| Milestone | State | Exit evidence |
|---|---|---|
| Phase arithmetic and bounded RTL | Completed pilot, E001 | Numerical results, simulation/build evidence, 10 board runs; no full-model speed claim |
| Explicit vs learned updatable memory | Completed synthetic study, E002 | Matched-byte controls and fresh confirmation; no competitive learned mechanism established |
| Real-model KV storage | Completed pilot, E003 | Raw answers, tensor storage, peak allocation and timing; no serving win |
| Optimized packed-cache baseline | Completed single-trial E004 pilot | 188 scored requests; 124 resident hits; Q8/Q4 eight-slot allocation; no superiority claim |
| Read-transaction screen | Completed analytical E005 | 147 configurations; real schedules/quality/physical counters remain to be tested |
| Confirm useful serving improvement | Not started | Repeated trials, held-out tasks, matched quality, identical arrival traces and deadlines |
| Refinement-reuse headroom | Proposed, I001 | Oracle/trace advantage over fixed compression and ordinary LRU at equal total memory |
| Select a useful FPGA target | Proposed, I002 | Whole-model bottleneck, closest prior art and realistic bandwidth/resource model |
| Build a new mechanism | Conditional | Reference validation, ablations, synthesis/board or GPU results and meaningful system benefit |

Next: [P000](experiments/planned/P000-serving-confirmation.md) tunes and confirms the serving baseline. Then pursue [P001](experiments/planned/P001-transaction-aware-reader.md) and [P002](experiments/planned/P002-refinement-reuse.md) as one line of work if real-trace/oracle headroom survives. [P003](experiments/planned/P003-recurrent-rounding.md) is the alternative Mamba/RTL direction. See [the literature review](literature/review-2026-10-22.md) and [all protocols](experiments/planned/README.md).

Hardware requests should specify scientific hypothesis, minimum memory/bandwidth, execution features, software compatibility, experiment count, and the claim current hardware cannot test. No paid compute is rented or assumed.
