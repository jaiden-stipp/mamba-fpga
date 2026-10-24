# E001: one additional K×V product boundary — 2026-10-24

This isolated candidate builds on `59c59f1`'s rank-1 rotation stage and adds
exactly one stage to the recurrent-state pipeline. It registers current and
previous K×V products before the original add-to-zero operations, with aligned
old-state, Q, address and valid. The previous variant and its evidence remain
unchanged.

Bit-exact regression passes. Routed worst setup slack improves by **7.741 ns**,
but the candidate still fails the 50 MHz setup requirement. No physical FPGA
operation or inference speedup is claimed. See [comparison.json](comparison.json).

## Before and after

Both builds use Quartus Lite 18.1, EP4CE115F29C7, N=128, P=8, rank=1,
FAST_FIT, a 20 ns clock and 447 virtual non-clock pins. QSF and SDC files are
byte-identical. Each design has one routed build; this is not a placement-seed
study. The before column is the previous rotation-stage candidate, not the
original unpipelined implementation.

| Measurement | Rotation-stage baseline | Additional K×V stage |
|---|---:|---:|
| Worst setup slack at 50 MHz | -64.411 ns | -56.670 ns |
| Minimum restricted Fmax across reported corners | 11.85 MHz | 13.04 MHz |
| Fitted logic elements | 28,217 | 27,902 |
| Registers | 2,153 | 2,279 |
| Embedded multiplier 9-bit elements | 105 | 105 |
| Quartus compilation | PASS | PASS |
| Meets 50 MHz setup | FAIL | FAIL |
| Rank-1 unstalled cycles/token | 1,401 | 1,402 |

The multiplier count is the vendor's Cyclone IV nine-bit multiplier-element
count, not a count of modern DSP slices. In simulation the state-compute segment
still writes 1,024 consecutive coordinates, one per cycle, for each token. The
extra stage adds one cycle to pipeline fill/drain latency rather than one cycle
per coordinate.

## Isolated path and change

The [previous TimeQuest report](../rotation_stage/synthesis/pipeline.sta.rpt)
identifies the K-cache RAM read-enable register to `now1[1]`: 84.057 ns data
delay and -64.411 ns setup slack. A full-path extraction from its matching
retained routed database reproduces that exact row and shows 73 logic levels:
[baseline full path](timing/baseline-kram-to-now1.rpt).

```text
before: K/previous-K RAM -> fp_mul(K,V) -> fp_add(+0, product) -> now1/prev1
after:  K/previous-K RAM -> fp_mul(K,V) -> product register -> fp_add(+0, product) -> now1/prev1
```

The [candidate RTL](rtl/mamba3_recurrent_tile.sv) captures `kv_now_product`,
`kv_prev_product`, `kv_old_product` and `kv_q` together. `indices[1]` and
`valid_pipe[1]` identify their shared coordinate. The old-state multiplication
and Q are delayed through the same boundary. Every downstream valid consumer
and address, including internal FP16 RAM writeback and the exposed write port,
is shifted by one cycle. The pipeline drains completely before previous-K copy
and output delivery.

`fp_add(+0, product)` remains explicit: replacing it with a wire can change
signed-zero behavior. Separate multiplication/addition rounding, rank reduction
order, the two state-update additions, current wide-state readout and FP16
writeback are unchanged. The rank-4 pipeline has zero added delay and retains
its original arithmetic and schedule. The earlier phase-rotation boundary is
retained without further changes.

Slow 1200 mV / 85C targeted TimeQuest reports show the routed register split:

| Selected path | Data delay | Setup slack | Logic levels |
|---|---:|---:|---:|
| Baseline K RAM -> `now1[1]` | 84.057 ns | -64.411 ns | 73 |
| Candidate K RAM -> `kv_now_product[10]` | 56.033 ns | -36.484 ns | 38 |
| Candidate `kv_now_product[29]` -> `now1[23]` | 49.833 ns | -29.904 ns | 45 |

See [K RAM to product](timing/staged-kram-to-product.rpt) and
[product to now1](timing/staged-product-to-now1.rpt). Selected-path delays are
not added together or treated as a measured inference rate.

The new global worst path is **`prev2[3]` -> `updated3[1]`**, 76.635 ns data
delay, through the existing state-update additions. No second arithmetic change
was made to pursue that path.

## Verification

Seed `0xe001f032`; ModelSim 10.5b, NumPy 2.2.6, and the same installed
`altera_mf.v` RAM simulation library as the baseline. The wrapper copies and
executes the existing [regression.py](../regression.py), primitive test and FP32
functions without editing them.

| Case | State writes checked | Outputs checked | Maximum bit-reference error | Result |
|---|---:|---:|---:|---|
| Rank 1, 64 tokens | 65,536 | 512 | 0 | PASS |
| Rank 4, 64 tokens | 65,536 | 2,048 | 0 | PASS |

The primitive test passes 96,112 vectors / 384,448 results. The extended
[testbench](rtl/tb_recurrent_tile.sv) checks 65,536 product-stage admissions,
consumptions and independent address/Q alignments. It resets while the new
product stage is valid before state writeback, checks valid is flushed during
clear, and retains the prior rotation reset, in-flight state-update reset,
input bubbles, output stalls, RAM collision checks and bit-exact scoreboards.

Both ranks' seven stimulus/ROM/golden files are byte-identical to the previous
variant. Python independently verifies every exported wide FP32 state, packed
FP16 state and FP32 output, and compares values and indices with baseline CSVs.
Separate FP64 error metrics and phase coverage are unchanged. Deliberately
corrupting one golden state yields exactly one error and FAIL per rank.

No candidate mismatch occurred. The [first-mismatch record](evidence/first_mismatch.json)
is NONE; a failing candidate's first mismatch lines would remain in its
simulation log and be indexed by the wrapper, without changing the oracle.

Read the [PASS/FAIL log](evidence/PASS_FAIL.log), [summary](evidence/summary.json),
[rank-1 simulation](evidence/rank1/simulation.log),
[rank-4 simulation](evidence/rank4/simulation.log),
[build status](synthesis/build_status.json), [fitter summary](synthesis/pipeline.fit.summary),
and [timing summary](synthesis/pipeline.sta.summary).

## Replay and audit

From this directory, using the same installed dependencies:

```powershell
python run_experiment.py regression
python run_experiment.py synthesis
quartus_sta -t extract_paths.tcl <baseline-project-path> <absolute-output-prefix>/baseline baseline
quartus_sta -t extract_paths.tcl <candidate-project-path> <absolute-output-prefix>/staged staged
python analyze.py
```

Project paths identify the `pipeline` project without its extension. The
wrapper stages unchanged shared tools/RTL plus the two candidate overlays in
ignored `.artifacts`. It never programs hardware or overwrites previous
evidence. Databases and generated programming files remain in scratch.
[Input hashes](inputs.json) identify shared, prior and candidate sources and
traces. [Extraction provenance](timing/extraction_provenance.json) binds the
inspected routed databases to retained source/configuration/fitter/timing files;
extraction logs contain actual commands.

`analyze.py` checks source hashes, identical stimuli/oracle, every CSV result,
negative controls, stage checks, exact constraints, prior report hashes and
targeted endpoints before regenerating the comparison. The previous variant
and original implementation retain their independent audits.
