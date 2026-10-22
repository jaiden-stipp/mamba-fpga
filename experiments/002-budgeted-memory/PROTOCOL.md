# Updatable recurrent memory under a strict byte budget

Started October 19, 2026. This is the first decision experiment in a research program, not a new architecture claim.

## Decision

Before training another memory model, determine whether repeated updates are actually the bottleneck after comparing against compact explicit storage at equal persistent bytes. Distinguish two interfaces: exact symbolic IDs and noisy perceptual addresses. A method must not receive evaluator-only entity IDs in the latter track.

## Hypotheses, written before running the experiment

1. With exact finite IDs, an ordinary direct-address packed table will make the basic task too easy to support an architectural novelty claim.
2. With noisy vector addresses, storage for addresses, values, and replacement metadata changes the comparison substantially. Compact signatures may be a stronger baseline than storing full floating-point keys.
3. Corrective delta writes should reduce stale-value errors relative to additive writes, but correction alone may not dominate explicit memory at equal bytes.
4. A hybrid that splits the budget between a compressed recurrent state and an explicit recent-event memory may help somewhere, but its advantage must survive a stronger explicit-storage baseline. No novelty is claimed for this familiar hybrid construction.

## Stream

128 entities; 16 independently sampled values plus ABSENT. Initialize every entity once, then SET, DELETE, and QUERY operations. SET always changes the prior value if present. A deletion sets the current binding to ABSENT. Deletion is an operation observed by every method. It is not an external reset or a request to undo arbitrary causal influence.

All methods see the same operations and current observed key. Evaluator-only ground truth holds current and previous bindings, edit counts, and age since last mutation. No labels or future queries enter memory updates. Queries can affect explicit replacement recency. Main scores exclude the initial warmup. Score active and deleted bindings separately and report stale answers and post-update queries so ABSENT predictions cannot inflate the headline unnoticed.

Interfaces: (a) exact uint8 IDs; (b) normalized 64D real vectors with fresh Gaussian observation noise. Perceptual prototypes are newly sampled for every stream and never supplied as a lookup table to methods. Include independent keys and deliberately similar key clusters. This models address uncertainty, not natural language semantics.

Calibration: seeds 100-102, 4,096 events, independent prototypes and values. Freeze configurations before evaluation. Test: seeds 200-207, 12,288 events; stress: seeds 300-303, 49,152 events and changed key/query distributions. Use budgets 256, 1,024, 4,096 bytes per request. Both regimes contain updates, deletion, and retrieval; total stream length is independent of live memory allocation.

## Baselines and scope

- Exact ID table: actual packed uint8 storage (128 bytes; five-bit packing would require only 80 bytes). This receives a stronger interface and is reported separately as a diagnostic.
- Explicit approximate-address caches: 32- and 64-bit sign signatures, int8 64D keys, and FP32 64D keys; uint8 value; uint32 timestamp; four-byte global clock. Nearest-key matching and overwrite thresholds are calibrated. Full arrays including empty capacity count toward the budget. Query misses return ABSENT; this may conflate forgotten and deleted facts and is reflected in active-only metrics.
- Dense additive and delta-rule memories: five FP32 value channels (four signed value bits and a presence channel), random sign-feature keys, no learned parameters. Delta performs a read-correct-write update. An additive model with tuned scalar decay is an algebraic control, not a reproduction of trained Mamba. Dense features and value decoding are shared static functions, not per-request saved history.
- Hybrid: explicit int8-key recent-event cache plus delta matrix, with every allocated byte charged. This is a baseline hypothesis, not an invention.

This experiment does not reproduce the trained Gated DeltaNet, Gated DeltaNet-2, SDM, DART, or HMM models. Their papers constrain novelty and later full-model comparisons. Random feature memories isolate update arithmetic; they are not an architecture leaderboard.

Persistent-byte accounting excludes frozen shared weights/feature transforms, current input and output, and ephemeral computation. Report shared parameters separately. No prior event, prototype table, decoder answer, or uncounted recurrent object is retained. It is a logical array-payload budget, not Python object overhead, total VRAM, peak scratch memory, energy, or a hardware implementation.

## Selection and uncertainty

Select configuration by mean calibration accuracy pooled across the three declared perceptual regimes. Report every tested configuration. No per-test-regime selection. Paired seed-level bootstrap intervals summarize these selected synthetic distributions only. Test seeds are not used to select a method. Stress configurations remain frozen.

Proceed to a learned memory mechanism only if a repeatable gap remains after compact explicit caches and hybrid controls. Potential research target: learn a representation and write rule that separates address uncertainty from interference, improving active-fact retention at a fixed byte budget on new entities and distributions. The first experiment cannot establish that this target is novel or solvable.

## Prior work requiring comparison

- Gated Delta Networks (2024/2025): https://arxiv.org/abs/2412.06464
- Gated DeltaNet-2 (May 2026): https://arxiv.org/abs/2605.22791
- Sparse Delta Memory (July 2026): https://arxiv.org/abs/2607.07386
- Mixture-of-Memories: https://arxiv.org/abs/2502.13685
- DART (August 2026): https://arxiv.org/abs/2608.02032
- Hierarchical Memory Mamba (August 2026): https://arxiv.org/abs/2608.02347
- Recall Scaling Laws (September 2026): https://arxiv.org/abs/2609.07681
- Constant-Memory Recall (September/October 2026): https://arxiv.org/abs/2610.00232

Novelty search is targeted, not exhaustive. Recheck the final mechanism before making any novelty claim.
