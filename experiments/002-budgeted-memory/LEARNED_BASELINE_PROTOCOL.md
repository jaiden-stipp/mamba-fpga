# Learned-address delta baseline: extension fixed before final evaluation

The analytic screen is complete. Its test seeds will not select this learned baseline. This extension is supporting research, not a new update rule: train an MLP address encoder through the existing delta recurrence under a 4,096-byte state cap (204 x 5 FP32 = 4,080 bytes).

Architecture: 64 -> 128 -> 204 bias-free MLP with GELU; center each key across coordinates and L2-normalize. One globally learned write strength beta in (0,1); no hidden memory except the matrix. Value encoding and decoding remain the same four value bits and presence bit as in the analytic screen. Encoder weights are shared across requests, reported separately. Model sees noisy feature vectors, opcodes, and values, never entity IDs or query answers during inference.

Train three independently initialized models (seeds 0, 1, 2), 600 AdamW steps, batch 24, sequence length 256, learning rate 0.001, gradient clipping 1.0. Training streams are new at every batch with 32 or 64 entities and the three previously declared regimes. No test-stream prototype is reused. Query loss supervises active/absent and value bits; every SET and DELETE is applied through the delta rule. Initial writes are part of the input, not a state preload from ground truth.

Evaluate validation every 100 steps on independent fixed batches; keep the lowest validation-loss checkpoint. Evaluate the identically initialized untrained encoder as an ablation. Evaluate compact-cache, additive, delta, and hybrid configurations frozen by the original calibration on exactly the new final streams too.

Final seeds: 500-507, 128 entities, 12,288 events. Length/distribution stress: seeds 600-603, 128 entities, 49,152 events and the previously specified moving hot set. No final test result is used for checkpoint selection or hyperparameters. These distributions are also longer and have more entities than training, so poor results may reflect extrapolation failure; do not confuse them with a proof about learned recurrence in general.

Decision criterion: no architectural advantage claim without beating the strong compact explicit cache on active-fact retention across more than one regime, while counting all persistent bytes. A gain against only the additive control is insufficient. The noisy-key task remains a synthetic identification task and does not demonstrate natural-language memory or a new practical capability.
