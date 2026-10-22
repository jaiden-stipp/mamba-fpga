# P005 — Lookup, low-rank or DSP execution for projection layers

Status: alternative thesis; larger algorithm/calibration commitment than P001.

**Question.** Does a model-quality-matched projection representation use FPGA memory ports and arithmetic more effectively than ordinary low-bit GEMV? This explores the computation itself rather than optimizing a small phase-rotation component.

**Strong prior art.** [LUT-LLM](https://arxiv.org/abs/2511.06174) already trains activation/weight vector-quantized models and implements centroid search plus two-dimensional lookups. [LowRank-SSM v2](https://arxiv.org/abs/2608.02954v2) already jointly selects projection ranks, provides a factorized datapath and optimizes memory delivery. [FlightLLM](https://arxiv.org/abs/2401.03868) supplies a strong arithmetic FPGA reference. Generic lookup multiplication, low rank or mixed precision is not novel.

First verify availability/licenses of author code, training recipe, calibrated weights and hardware scripts. A method requiring retraining cannot be compared as if its model conversion were free. Start with real captured projection inputs and a held-out language-model evaluation; random matrices alone do not establish model usefulness.

Compare dense INT4/INT8 DSP execution, scalar dequantization plus arithmetic, vector-quantized lookup, and rank-reduced projections where appropriate to the model. Match whole-model quality and account for codebooks, index arrays, centroid search, BRAM replication for extra read ports, partial sums, scales, factor intermediates and off-chip transfers. Sweep batch 1/4/16 and projection shapes; the winner for batch-one decode may lose on prefill.

A plausible but unconfirmed mechanism is a single resource-constrained execution fabric that assigns different projection classes to arithmetic or lookup and shares buffers without costly model-wide representation switches. Test a simple offline assignment and its oracle before designing a complex controller. Compare with one uniform representation at equal memory ports, BRAM, DSP and quality; a larger heterogeneous design is not an equal-cost win.

**Gate.** Reproduced layer gains must survive full-model profiling and improve the objective by at least 20% over the best compatible compressed baseline, rather than only over FP16. Report calibration/training compute separately. If on-chip table replication or centroid search dominates, abandon the lookup path or restrict its scope with an honest full-model bound.

**Presentation.** A matrix of projection shape × batch shows the quality-constrained winner, accompanied by synthesis results and a full-model time breakdown. A convincing result explains why memory ports and dataflow change the winner, not just how many multiplications were removed.
