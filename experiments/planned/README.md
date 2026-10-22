# Next experiments — ranked, 2026-10-22

These are protocols for proposed work, not completed results. [The research review](../../literature/review-2026-10-22.md) explains the literature and novelty limits.

First complete [P000 serving calibration and confirmation](P000-serving-confirmation.md). It addresses the prefill/decode stalls observed in E004 and establishes a fair baseline; it is not a new research contribution.

| Priority | Protocol | Main question | Presentable output | First dependency |
|---|---|---|---|---|
| 1 | [P001 — Transaction-aware reader](P001-transaction-aware-reader.md) | Can useful precision be fetched in efficient memory transactions? | Quality / physical bytes / latency frontier and verified RTL | [E005 address screen](../005-read-granularity/README.md), real Q/K/V traces |
| 2 | [P002 — Refinement reuse](P002-refinement-reuse.md) | Does caching extra precision beat ordinary caching after charging its capacity? | More correct responses per second at fixed memory and deadlines | Strong serving baseline; oracle headroom |
| 3, alternative Mamba thesis | [P003 — Recurrent rounding](P003-recurrent-rounding.md) | Can affordable rounding preserve useful state and avoid longer answers? | Accuracy, answer length, joules/correct answer, RTL cost | Real recurrent feedback; numerical reference |
| Conditional | [P004 — GPU–FPGA split](P004-hybrid-handoff.md) | Can a hardware split repay transfer and synchronization costs? | Measured break-even surface and hardware request | Calibrated bandwidth/latency model |
| Alternative thesis | [P005 — Lookup vs DSP execution](P005-lookup-vs-dsp.md) | When does replacing multiplications with lookup actually help? | Quality-matched throughput/resource frontier | Upstream checkpoint/code audit; calibration budget |
| Lower priority | [P006 — Compression plus speculation](P006-compression-speculation.md) | Does a compressed draft help under long-context memory pressure? | End-to-end accepted-token cost and request goodput | Stable fused cache baseline; exact verifier |

Do P001 and P002 as one linked line of work, or choose P003 if the appointment should remain centered on Mamba and RTL. Do not pursue all six simultaneously. First repeat/extend E004 with a larger independent quality set and identical open-loop arrival traces. Completing a stock-engine benchmark is baseline establishment; the research contribution must be the mechanism that improves over the strongest compatible compressed baseline.

All thresholds in these proposals are **prospective research gates**, not achieved results or guarantees. Register amendments before measurement, preserve failures, and separate exploratory tuning from held-out confirmation.
