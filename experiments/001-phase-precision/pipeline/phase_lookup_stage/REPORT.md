# E001 timing loop: phase_lookup_stage — 2026-10-24

One additional rank-1 captured-trig/coordinate boundary before rotation multiplication. The parent is `207e8e6` / `update_sum_stage`.
This is an isolated cumulative candidate; earlier RTL and evidence remain unchanged.
No physical FPGA operation or inference speedup is claimed.

## One change

On B/C acceptance, capture all four FP32 coordinates and cosine/sine together with the existing pair/rank/address/valid tags. A new ROTATE_MUL cycle computes the original eight FP32 products, followed by the existing ROTATE_ADD cycle. Keep ready low while the tagged pair is pending. Only the first N/4 rotating pairs use the extra cycle, adding 32 cycles/token. The recurrence pipeline is unchanged. Extend the tag monitor through ROTATE_MUL, check all six captured words, and test resets in both the captured-ROM and existing product stages. Quartus absorbs the trig registers into inferred synchronous-ROM address registers; phase_c is not a routed register name. The initial name-selector failure is retained under timing/initial-selector-*. Actual register selectors bind the phase-to-ROM-address and ROM-address-to-product segments. The registered-ROM inference increases M9Ks from 7 to 9 and logical memory bits from 29,792 to 46,176.

The numerical oracle, separate FP32 multiplication/addition rounding, left-associated
recurrence and FP16 feedback are retained. Rank-4 has no new arithmetic stage.
See [RTL](rtl/mamba3_recurrent_tile.sv), [testbench](rtl/tb_recurrent_tile.sv),
[source hashes](inputs.json) and [comparison](comparison.json).

## Matched routed comparison

Quartus Lite 18.1, EP4CE115F29C7, N=128, P=8, rank=1, FAST_FIT,
20 ns clock and 447 virtual non-clock pins. QSF and SDC are byte-identical.
There is one routed build per design, not a placement-seed study.

| Metric | Parent | Candidate |
|---|---:|---:|
| Worst setup slack at 50 MHz | -36.953 ns | -30.411 ns |
| Worst-corner restricted Fmax estimate | 17.56 MHz | 19.84 MHz |
| Fitted LEs | 28,780 | 28,697 |
| Registers | 2,377 | 2,506 |
| Nine-bit multiplier elements | 105 | 105 |
| Rank-1 unstalled cycles/token | 1403 | 1435 |
| Compile | PASS | PASS |
| Meets 50 MHz setup | FAIL | FAIL |

Worst slack change: **+6.542 ns**.
The multiplier unit is Quartus's Cyclone IV nine-bit element, not a modern DSP slice.
The state-compute segment retains 1,024 consecutive coordinate writes/token in simulation.

## Isolated paths

The baseline full path reproduces the parent's retained worst path. Extra reports
use the slow 1200 mV / 85C model and source/configuration/report-matched databases.

| Selected path | Endpoints | Data ns | Slack ns | Logic levels |
|---|---|---:|---:|---:|
| baseline-critical | `altsyncram:phase_rtl_0|altsyncram_qcc1:auto_generated|ram_block1a0~portb_address_reg0` → `rot_qyc[12]` | 56.475 | -36.953 | 41 |
| staged-first | `phase_rtl_0_bypass[8]` → `altsyncram:cosine_rtl_0|altsyncram_pa91:auto_generated|ram_block1a0~porta_address_reg0` | 15.341 | 4.984 | 9 |
| staged-second | `altsyncram:cosine_rtl_0|altsyncram_pa91:auto_generated|ram_block1a0~porta_address_reg0` → `rot_qyc[15]` | 47.307 | -27.760 | 37 |

These endpoint sets can select different bits; their delays are not added together.
The candidate's global worst path is `altsyncram:cosine_rtl_0|altsyncram_pa91:auto_generated|ram_block1a23~porta_address_reg0` →
`rot_kxs[28]`, 49.931 ns.
See [baseline](timing/baseline-critical.rpt), [first segment](timing/staged-first.rpt),
[second segment](timing/staged-second.rpt), [provenance](timing/extraction_provenance.json),
[fitter](synthesis/pipeline.fit.summary) and [timing](synthesis/pipeline.sta.summary).

## Exact regression

Seed `0xe001f032`, ModelSim 10.5b / NumPy 2.2.6 and the unchanged vendor RAM library.
The existing regression.py, FP32 functions and primitive test are copied without editing.
Both ranks' seven stimulus/ROM/golden files match the parent byte-for-byte.

| Case | State writes | Outputs | Maximum bit-reference error | Result |
|---|---:|---:|---:|---|
| Rank 1, 64 tokens | 65,536 | 512 | 0 | PASS |
| Rank 4, 64 tokens | 65,536 | 2,048 | 0 | PASS |

Primitive checks: 96,112 vectors / 384,448 results. The extra tag/reset monitor
requires `PHASE reset_inflight=checked`, `PHASE admits=2048 consumes=2048`, `SUM admits=65536 consumes=65536`.
Original phase/KV monitors, reset, input pauses, output stalls and collision checks
are retained. Python rechecks every exported FP32/FP16 result and baseline index/value.
FP64-error metrics and phase coverage are unchanged. Each corrupt-golden negative
control fails with exactly one error. There were no candidate mismatches; the
[first-mismatch record](evidence/first_mismatch.json) is NONE and the oracle was not loosened.
Read the [seed-locked log](evidence/PASS_FAIL.log) and [summary](evidence/summary.json).

## Replay

From this directory with the existing local dependencies:

```powershell
python run_experiment.py regression
python run_experiment.py synthesis
quartus_sta -t extract_paths.tcl <project-path> <absolute-report-file> <from-glob> <to-glob>
python analyze.py
```

Use the three selectors in [experiment.json](experiment.json) against the parent
or candidate routed project. Extraction logs retain exact commands; project paths
omit the `.qpf` extension. Databases/programming files remain in ignored `.artifacts`.
The wrapper does not program hardware. The audit verifies source/environment hashes,
unchanged golden files, every CSV result, stage markers, negative controls, exact
constraints, parent reports and extraction provenance before regenerating comparison.json.
