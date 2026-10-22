# I001 — Precision-refinement reuse for long-context inference

Status: proposed. Novelty unconfirmed. No reader or scheduling policy has been implemented.

Objective: more correct long-context responses delivered within fixed deadlines on the same GPU budget.

Hypothesis: a bounded working set of useful precision refinements could amortize reconstruction across related queries, while page-aligned grouping avoids expensive fine-grained reads and dispatch. The backing representation must retain the extra information; widening a discarded two-bit value cannot recover it.

Prior work overlaps strongly: ReadKV already separates retained bits from query-dependent reads; MixKVQ already uses query-aware precision; prefix reuse and cold-page compression already exist. See [literature](../literature/README.md). A new contribution would need a specific representation/reader/policy with end-to-end evidence over those methods.

First gate: [E004](../experiments/004-kv-serving/README.md) establishes a quality-preserving optimized serving baseline. Then collect refinement reuse distances and topic changes. Compare an oracle with fixed quantization and ordinary decoded-page LRU at the same total retained + working-set memory. Stop if the oracle has little net benefit.

Only then implement a policy and measure quality, request goodput, tail latency, actual DRAM traffic, and all temporary memory. [Proposed dataflow](../architecture/packed-kv-reader.md).
