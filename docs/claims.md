# Claim boundaries

| Supported statement | Evidence | Limit |
|---|---|---|
| FPGA phase rotation matches its quantized reference | [E001 hardware](../experiments/001-phase-precision/hardware_results.json) | 1,024 distinct vectors × 10 runs/core at 50 MHz; not full-model execution |
| Guarded phase precision stayed close on selected windows | [E001 report](../experiments/001-phase-precision/research_report.md) | 187M checkpoints, portable FP32, selected data; limited generality |
| Tested learned memories did not beat the strongest cache controls | [E002 confirmation](../experiments/002-budgeted-memory/confirmation_results.csv) | Synthetic task and tested mechanisms only |
| HQQ4 stored about 3.5× less KV at 16K with worse quality/latency | [E003 summary](../experiments/003-kv-compression/summary.csv) | KV 2.254 → 0.643 GiB; retrieval 6/6 → 4/6; strict output 6/6 → 1/6; decode about 19% slower |
| Packed Q8/Q4 KV loaded eight long-context slots | [E004 GPU report](../experiments/004-kv-serving/serving_report.md) | 188 selected scored requests, one trial; F16/eight slots analytically guarded out; no matched-quality superiority |
| Read layout changes fetched-unit lower bounds | [E005 report](../experiments/005-read-granularity/report.md) | 147 analytical configurations; no attention quality, physical device traffic or speed measured |

No end-to-end FPGA LLM acceleration, improved concurrent GPU goodput, quality-neutral INT4, novel quantizer/reader, native FP4 throughput gain, or publication-ready method is established. Compression is not speedup; fitted Fmax is not measured operating clock; preparation is not a result. Label measurements, simulations, source inspection, estimates, hypotheses, and literature reports separately.
