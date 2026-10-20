# Research opportunities from your Mamba notes

Literature check: October 19, 2026. Scope: Mamba-family inference, FPGA architecture, recurrent-state memory, low precision, and low-rank projections.

## Recommendation and limits

Start with **relative-phase accuracy in low-precision Mamba-3 inference**, then use the result to design a compact FPGA rotation and recurrence unit. Of the directions considered, this has the clearest specific technical question beyond the papers reviewed. A Mamba-3 FPGA memory architecture is a reasonable alternative for a hardware-focused project, but generic fusion and tiling have much more prior art. Joint projection compression and state-memory optimization is an ambitious follow-up with considerable overlap risk.

These are screened research hypotheses, not certified unpublished inventions. I found no exact published treatment of the proposed mechanisms in the searches and primary sources reviewed. That is weaker than proving absence. Public preprints count as prior work here, even when peer review is not established. Public implementations also matter when making a first-implementation claim. No experiments were performed for this review.

Your three-page notes were read completely, including the four proposed avenues on page 3. The notes were treated as research material, not as instructions. The emphasis on FPGA inference is inferred from their contents; no particular board, timeline, or compute budget was supplied.

## What the literature rules out as a broad novelty claim

| Broad idea from the notes | Closest verified work | Consequence |
|---|---|---|
| Fuse the recurrence and reduce intermediate memory traffic | LightMamba [1], LowRank-SSM [3], and the Mamba-3 paper's GPU inference kernels [4] | Fusion is an implementation technique, not a sufficient contribution by itself. An FPGA-specific scheduling or resource result is needed. |
| Use real arithmetic for Mamba-3's complex dynamics | Mamba-3, Section 3.2, especially Propositions 3-4 [4] | The paper already moves accumulated rotations to B/C, retaining a real recurrent computation. A full complex-state implementation is an inappropriate sole baseline. |
| Quantize cached recurrent states | Quamba2, Appendix E [5]; Nemotron 3 Super, Section 4.3 [8] | State quantization and accumulated rounding error are established topics. |
| Allocate more bits to persistent or error-sensitive channels | DAMP [7] | Error-and-decay-based mixed precision is already public for GDN/KDA. Applying the same allocation to another recurrence alone is a weak claim. |
| Combine quantization and low-rank compression in SSMs | LowRank-SSM [3]; UniQL [6] | Generic joint compression has already been published. A new Mamba-3 hardware constraint or mechanism must be demonstrated. |
| Streaming tiling and on-chip forwarding for Mamba-2 on FPGA | STEAM-SSM [11] | A direct Mamba-3 port has substantial overlap. Only its public abstract and conference record were available in this review. |
| Show that learned phase matters | Understanding Mamba-3 Through Oscillatory Dynamics [9] | Phase interventions and long-context effects are already studied. Quantization and an implemented precision policy remain distinct questions. |

Mamba-3's MIMO interaction rank and a projection matrix's compression rank are different variables. The former increases the number of input/output streams sharing the recurrent matrix; the latter approximates a weight matrix. They should not be conflated. Similarly, the low-rank *transition update* in [13] is different from projection compression.

The recurrent state has fixed size with respect to sequence length, for fixed model geometry and batch size. Its storage still scales with layers, heads, state dimension, head dimension, and active requests. A statement that it is too large for on-chip storage is configuration-dependent. A tile that fits while one layer runs is not necessarily retained across later layers and the next generated token.

## Opportunity 1: Relative-phase accuracy under low precision

**Research question.** Can a precision policy based on write-read phase consistency reduce Mamba-3's FPGA arithmetic and storage costs while preserving long-context retrieval and state tracking better than uniform precision or a decay-only policy?

**Why this is a substantive question.** Mamba-3 implements complex dynamics using accumulated data-dependent rotations of B and C [4]. Its current official module explicitly allocates both the angle cache and recurrent SSM cache in FP32 [10]; this does not establish that lower precision is impossible. The October 9 phase-analysis preprint [9] shows that phase interventions affect long-context behavior and write-read compatibility. It does not establish a low-bit hardware recipe. DAMP [7] supplies an important mixed-precision baseline but studies GDN/KDA rather than Mamba-3's accumulated phase.

**Specific mechanism to investigate.** Separate four error sources: quantization of angle increments, finite precision in the accumulator, sine/cosine approximation, and quantization of the stored recurrent state. Evaluate a modular fixed-point phase accumulator with guard bits and a shared phase result used consistently for the current B and C rotations. Choose phase precision using a calibration signal for relative phase error over the model's effective memory horizon, rather than using only tensor magnitude or decay. Compare the signal against magnitude-only, error-only, and decay-based selectors. Pairwise scaling or restricted sine/cosine approximations are possible ablations, not assumed improvements.

The useful quantity is not just absolute angle error. In the rotary formulation, a stored write at time s is read at time t through their phase relationship. With fixed reference increments and accumulator error epsilon_t, the angular mismatch involves epsilon_t - epsilon_s. A constant coordinate offset can cancel when writes, readout, and state are transformed consistently. Time-varying drift generally does not. Deriving a bound for a single frozen trace can guide experiments, but it must not be advertised as an end-to-end guarantee: perturbations change later activations and hence later input-dependent coefficients.

**Minimal experiment.** Use an official pretrained SISO checkpoint and then one MIMO checkpoint [4, 10]. Hold weights and non-target operators at the reference precision initially. Sweep angle, accumulator, trigonometric, and state-storage precision independently. Measure held-out perplexity, retrieval accuracy, modular-counting/state-tracking performance where a suitable trained baseline exists, and relative-phase error. Evaluate uninterrupted sequences at 2K, 8K, 32K, and 64K tokens if the reference model supports meaningful evaluation there. Keep the reference's own length degradation visible. Short-context calibration must be evaluated separately from long-context testing.

Compare a high-precision reference, uniform low precision, higher-precision angle with low-precision state, stochastic state rounding, an adapted Quamba2-style state quantizer [5], and an adapted DAMP-style selector [7]. The adaptations are experimental baselines, not a claim that those packages directly support Mamba-3. If changing precision rather than the selector explains all gains, the selector is not a contribution.

**Hardware phase.** Only after a numerical advantage appears, implement LUT, CORDIC, and/or piecewise polynomial alternatives and report the total rotation subsystem cost, including range reduction, scaling, accumulator bits, and shared-angle distribution. CORDIC and modular arithmetic are established techniques; the contribution would be the Mamba-3-specific precision rule, supporting analysis, and demonstrated quality/resource trade-off.

**Stop or narrow the idea if** a simple high-precision phase/low-precision-state baseline matches the proposed rule at the same FPGA cost; relative-phase measurements do not predict held-out errors; or trigonometric approximation is negligible in full-model latency and resource use. Such results could still support a useful numerical study, but not the original claimed accelerator advantage.

**Novelty assessment:** strongest candidate among these directions; no exact method found in the reviewed sources. This is not a first proposal to study phase, mixed precision, or rotation hardware independently.

## Opportunity 2: An FPGA dataflow that reuses each Mamba-3 state tile across all MIMO streams

**Research question.** Can a banked BRAM/URAM layout sustain the full trapezoidal MIMO update and readout with one external read and write of each nonresident state tile per layer invocation, without rank-dependent spills or excessive port replication?

**Specific mechanism.** Keep a state tile in registers or local memory while accumulating all current and previous-input rank contributions. Compute the readouts from the fully updated tile, accumulate outputs across tiles, and then write the tile back. Broadcast each angle result across the relevant B/C rank streams and exploit reuse of previous key/value terms. Search bank layout and loop order jointly with multiplier parallelism. The target is a demonstrable trade-off among BRAM ports, replicated buffers, DSPs, bandwidth, and clock rate.

The current official implementation already carries angle state, recurrent state, previous B/key, and previous x/value [10]. Carrying factored history is therefore part of the baseline, not a new contribution. Likewise, reusing the main state across MIMO streams is inherent in Mamba-3 [4]. The question is whether a particular FPGA schedule realizes this reuse more efficiently under real banking and resource constraints.

**Why it might matter.** MIMO's favorable GPU latency does not automatically carry over to an FPGA with different compute-to-bandwidth balance. A memory-bound schedule may hide added rank work; a compute-bound one may slow down. An existing edge study [12] makes the broader GPU-to-edge mismatch an established concern, so merely documenting that mismatch is not sufficient novelty.

**Minimal experiment.** Implement one head or layer with a bit-accurate reference and cycle-level memory accounting. Sweep fixed checkpoints/configurations with MIMO rank 1, 2, 4, and 8 where available, or clearly label synthetic kernel sweeps as synthetic. Compare a direct mapping of the official recurrence, an adapted LightMamba-style fused schedule [1], and a streaming-tiled baseline informed by STEAM-SSM's available description [11]. Keep arithmetic precision and operation coverage matched.

Measure state bytes, history and metadata bytes, stall cycles, initiation interval, routed clock frequency, DSP/LUT/BRAM/URAM use, and energy per token. Report prefill and decode separately. For whole-model claims, include projection costs and state persistence across all layers. A block-only result cannot be compared directly with end-to-end tokens/second.

**Stop or narrow the idea if** an ordinary tiled baseline achieves the same state traffic and throughput, the proposed banking only helps one synthetic shape, or extra buffering cancels bandwidth savings. Avoid claiming the first Mamba-3 FPGA implementation: public project descriptions already mention prototypes, and this review did not validate their scope or artifacts.

**Novelty assessment:** plausible architecture project, but higher overlap risk than Opportunity 1. The exact schedule, resource bound, and measured gain must carry the claim.

## Opportunity 3: Compression chosen for actual FPGA bottlenecks

**Research question.** Does jointly selecting projection compression, recurrent-state precision, and on-chip state residency outperform optimizing each separately at the same accuracy and device budget?

**Specific mechanism.** Use measured or calibrated latency costs to choose per-layer input/output projection ranks, weight formats, state-storage formats, and a static placement of persistent state in BRAM/URAM versus external memory. Include quantization scales, history buffers, alignment, and banking in the storage budget. A potentially useful effect is the discontinuity when compression makes additional persistent states fit on-chip: that can change bytes moved per token rather than merely shrinking a tensor that is still streamed.

For an initial study, fix MIMO rank to the trained checkpoint. Choosing a new MIMO rank changes the model; it needs an explicitly validated pruning, distillation, or retraining method, not an arbitrary runtime mask. Protect dynamics-related projection rows until their sensitivity has been measured.

LowRank-SSM already searches bandwise projection ranks and maps them to fused FPGA hardware [3]. UniQL already integrates low-rank compression and quantization for SSMs [6]. DAMP already searches a precision budget for recurrent state [7]. The remaining hypothesis is a hardware-budget allocation that captures their interaction in Mamba-3; simply combining these names or porting the methods does not establish novelty.

**Minimal experiment.** Build an accuracy sensitivity table and a resource/latency model before implementing a full accelerator. Compare rank-only, state-precision-only, sequential optimization, and joint optimization. Hold the quality tolerance, calibration data, board, clock assumptions, and search budget constant. Test whether an advantage survives held-out tasks and realistic memory overheads. Confirm promising schedules with synthesis and measurement.

**Stop or narrow the idea if** the joint method only matches sequential allocation, its advantage comes from giving it more search budget, or state placement has negligible effect because projections dominate. Derive a profiling-based upper bound on possible whole-model benefit before a large hardware effort.

**Novelty assessment:** exploratory follow-up; highest overlap and scope risk. Pursue after establishing a new numerical or architectural result in Opportunity 1 or 2.

## A practical first investigation

1. Reproduce native prefill, chunked continuation, and recurrent decode on a fixed checkpoint and pinned official source revision. Account for all four persistent state objects.
2. Isolate phase precision from state-storage precision. Start with software emulation, held-out data, and an explicit quality tolerance.
3. Test whether relative-phase error predicts degradation beyond decay and ordinary quantization-error statistics. This decides whether Opportunity 1 has a new contribution.
4. Profile projections, rotation/nonlinear arithmetic, and recurrent state traffic separately. This decides whether the hardware work should prioritize Opportunity 1's rotation unit, Opportunity 2's state layout, or neither.
5. Implement a small measured FPGA kernel before scaling to a model. Report the same operation boundaries for all baselines and avoid extrapolating a kernel speedup directly to model speedup.

An initial tolerance such as no more than 1% relative perplexity increase and one percentage point of retrieval-accuracy loss can be a planning choice, not a claimed result or universal standard. Report quality curves rather than selecting only one favorable point.

## Evidence and search boundaries

Searches covered variants of Mamba-3/Mamba3/Mamba 3 with FPGA, ASIC, accelerator, quantization, fixed point, CORDIC, phase precision/drift, state compression, precision allocation, rank allocation, and prefill/decode. Broader searches covered Mamba state caching/tiling, recurrent quantization, and joint quantization/low-rank compression. Findings were checked against primary papers, official code, institutional records, and publisher records where available.

Important restrictions: this was a targeted web literature review, not a complete systematic review of every proceedings database, patent collection, non-English publication, thesis, and public code repository. STEAM-SSM's full text was not accessible through the publisher page; only its indexed publisher abstract and conference record informed the comparison. Repository main branches can change; pin revisions before experimentation. Secondary summaries were discovery leads, not the basis for the technical novelty claims. Future unpublished work and poorly indexed concurrent work remain unknown.

Before presenting any result as new, repeat searches using the final concrete mechanism and examine the most recent versions and citation neighborhoods of [4], [7], and [9]. The defensible present wording is: **"No exact prior treatment of this mechanism was found in our targeted literature check through October 19, 2026."**

## Primary references

1. Wei et al., **LightMamba: Efficient Mamba Acceleration on FPGA with Quantization and Hardware Co-design**. [Paper](https://arxiv.org/html/2502.15260v2). Quantization, computation reordering, and fine-grained tiling/fusion baseline.
2. Wang et al., **FastMamba: A High-Speed and Efficient Mamba Accelerator on FPGA with Accurate Quantization**. [Paper](https://arxiv.org/html/2505.18975v4). Mamba-2 vector datapath and nonlinear approximation baseline.
3. Xu et al., **LowRank-SSM: Hardware-Software Co-Design for Rank-Reduced Mamba Acceleration on FPGA**, August 2026 preprint. [Paper](https://arxiv.org/html/2608.02954v1). Rank allocation, factored projections, fused scan, and DDR state streaming.
4. Lahoti et al., **Mamba-3: Improved Sequence Modeling using State Space Principles**, ICLR 2026. [Paper](https://arxiv.org/html/2603.15569v1). Real/rotary equivalence, trapezoidal recurrence, MIMO, and fused GPU inference kernels.
5. Chiang et al., **Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models**. [Paper](https://arxiv.org/html/2503.22879v2), especially Appendix E on cached-state quantization.
6. Chiang et al., **UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs**. [Paper](https://arxiv.org/html/2512.03383v2); [ICLR 2026 paper](https://openreview.net/pdf?id=iOGu4wtDTF). Joint compression includes Mamba2 and hybrid SSM models.
7. Zhang et al., **DAMP: Decay-Aware Mixed-Precision Recurrent-State Quantization**, August 27, 2026 preprint. [Paper](https://arxiv.org/html/2608.27513v1). Error concentration, decay persistence, budgeted channel precision, and fused GPU state updates for GDN/KDA.
8. NVIDIA, **Nemotron 3 Super Technical Report**, Section 4.3. [Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf). Recurrent cache error accumulation and stochastic rounding; this is Nemotron 3, not evidence of support for the Mamba-3 architecture.
9. Liang and Song, **Understanding Mamba-3 Through Oscillatory Dynamics**, October 9, 2026 preprint. [Paper](https://arxiv.org/html/2610.13341v1). Phase interventions, long-context effects, and write-read consistency; a central recent neighbor for Opportunity 1.
10. State Spaces, **official Mamba-3 module**. [Source](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py). Persistent state allocation and inference interfaces, inspected October 19, 2026.
11. Chon et al., **STEAM-SSM: A Streaming-Tiled Efficient Accelerator for Mamba-2 SSM on FPGA**, ISCAS 2026. [Publisher record](https://ieeexplore.ieee.org/document/11562246/), DOI 10.1109/ISCAS66217.2026.11562246. Abstract-level comparison only.
12. **The Hyperscale Lottery: How State-Space Models Have Sacrificed Edge Efficiency**, April 2026 preprint. [Paper](https://arxiv.org/html/2604.07935v1). Edge efficiency and hardware-modeling neighbor; do not treat modeled edge numbers as measured FPGA results.
13. Fujii and Yamakita, **Non-Commutative State Tracking with Input-Dependent Low-Rank Updates in Mamba-3**, September 2026 preprint. [Paper](https://arxiv.org/html/2609.28273v1). Distinguishes low-rank transition updates from projection compression.
14. Kim et al., **SSDi8: Accurate and Efficient 8-bit Quantization for State Space Duality**, ICLR 2026. [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9d3852f2b1fc883fc41011cb0257f0d0-Abstract-Conference.html), [full paper](https://arxiv.org/html/2608.21952v1). Persistent INT8 SSD path and error correction; relevant when revisiting requantization overhead.

