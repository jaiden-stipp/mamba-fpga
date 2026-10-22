# P002 — Is extra precision worth keeping between queries?

Status: proposed, high potential but high novelty risk. Linked to I001 and P001.

**Question.** A progressive cache can retain more information than each query reads. Can a small working set of previously useful refinements reduce later read/selection costs enough to offset its capacity, writes and reduced request residency?

**Important distinction.** Retained refinement bits recover information from an existing backing code. Casting a discarded four-bit value to FP16 adds no information. An ordinary decoded-page cache is a required baseline; shared-prefix reuse is also required. This proposal concerns precision working sets for the **same model, token prefix and positions**, not arbitrary KV reuse across unrelated prompts.

**Closest work.** [ReadKV](https://arxiv.org/abs/2610.11245) already separates stored and fetched bits. [KV²](https://arxiv.org/abs/2610.03198) refines token-eviction scores for reusable contexts, a different mechanism. [Budgeted Cache Repair](https://arxiv.org/abs/2610.02233) recomputes selected rows after cross-context reuse. [TensorRT-LLM cold-page compression](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog30_KV_Cache_Compression_in_TensorRT-LLM.md) already compresses offloaded pages. Broad claims about refinement, repair or reuse are therefore unavailable.

**First experiment: oracle headroom.** Collect at least 50 held-out documents with 20 turns each, spanning repeated-topic questions, abrupt topic changes and genuinely unrelated turns. Sweep a precision-working-set budget of 0/5/10/20% of the **same total state allowance**. Include backing codes, metadata, temporary buffers and ordinary prefix state in that allowance. A future-aware offline oracle is an optimistic bound only. Compare no working set, decoded-page LRU, frequency-based admission, randomized matched-budget admission, and a simple causal reuse-distance policy. Charge refinement computation, insertion, lookup, eviction and reloading. Track precision reuse separately from prefix cache hits.

**Gate.** If even the oracle cannot reduce modeled total serving time by 15% while preserving resident capacity and quality, stop. An oracle that assumes free precision decisions or ignores capacity is not useful. If it passes, implement a causal policy and show where it fails under topic changes. Never feed future queries into the evaluated policy.

**Confirmation.** Replay identical external arrival traces with session skew, think time and bursts. Compare at equal total GPU/FPGA memory and identical SLOs. Use at least three timing trials and paired held-out quality evaluation across two model families. Primary result: correct completed responses meeting all deadlines per second. Report each task's quality, resident sessions, tail latency, write/read traffic and rebuild costs alongside it.

**Presentation.** An annotated session timeline shows which precision survives, what a topic switch invalidates, and how much work it saves. A positive result must beat ordinary decoded-page LRU and fixed compressed serving; otherwise the result is a useful negative headroom study, not a new memory system.
