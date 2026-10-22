# P006 — Low-bit drafting under long-context memory pressure

Status: lower-priority extension, not the recommended first contribution.

**Question.** Can a compressed draft reduce total correct-answer latency after paying verification, rejection and memory-residency costs? [QuantSpec](https://machinelearning.apple.com/research/quantspec) already combines hierarchical quantized KV and self-speculation. [SpecKV](https://arxiv.org/abs/2605.02888) already adapts speculation length with compression-aware signals. Adding speculation to low-bit execution is therefore a reproduction or systems study unless a specific new mechanism beats these controls.

Compare ordinary decoding with strong packed KV, an existing self-speculative baseline, fixed draft lengths 1/2/4/8, and a causal controller. Keep the verifier and target output distribution fixed. For a lossless-speculation claim, implement the appropriate acceptance/correction algorithm; matching a few greedy outputs is not a distributional proof. Changing the target to a quantized model changes which distribution is preserved.

Sweep context length, output length, task family and concurrent sessions. Report draft/target cache bytes, duplicate state, rollback buffers, accepted tokens per verification, rejection rate, target invocations, total transfer and all controller overhead. Include a no-draft control with the same total available memory: the draft may displace sessions that otherwise contribute more goodput.

Begin on one GPU. An FPGA draft is conditional on P004's link/synchronization gate; no claim follows from a fast standalone draft. Train/tune controllers on disjoint traces and compare against the best fixed draft length selected on validation data, not merely a deliberately weak default.

**Gate.** Proceed only if it improves complete response latency or correct deadline-qualified throughput by at least 20% over the strongest same-budget baseline, with quality/distribution requirements met. Stop if gains exist only in accepted-tokens-per-step while wall time or request capacity worsens.

**Presentation.** Show the accepted/rejected draft timeline and a full cost waterfall. This direction can produce an understandable demo, but its prior-art overlap and verifier complexity make it less attractive than establishing P001/P002 first.
