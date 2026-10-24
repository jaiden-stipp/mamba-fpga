# E001 timing loop: trig_output_stage — 2026-10-24

One additional rank-1 registered trig-output/coordinate cycle before rotation multiplication. The parent is `96e4b82` / `phase_lookup_stage`.
This is an isolated cumulative candidate; earlier RTL and evidence remain unchanged.
No physical FPGA operation or inference speedup is claimed.

## One change

Add ROM_OUTPUT between phase capture and ROTATE_MUL. Register all four delayed coordinates and both trig values together; preserve the pair/rank/address/valid lifetime through both wait cycles, multiplication, addition and two writes. Only rotated pairs take the extra cycle (+32 cycles/token). Independently check all six delayed words and reset at the new stage while retaining earlier phase/product/state reset checks. No arithmetic operation or recurrence schedule changes. The initial second path selected trig_c into a sine-product endpoint and had no path; that rejected extraction is retained. The corrected selectors include the captured trig/coordinate inputs of rot_kxs and the ROM-to-capture bank.

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
| Worst setup slack at 50 MHz | -30.411 ns | -27.252 ns |
| Worst-corner restricted Fmax estimate | 19.84 MHz | 21.16 MHz |
| Fitted LEs | 28,697 | 28,687 |
| Registers | 2,506 | 2,699 |
| Nine-bit multiplier elements | 105 | 105 |
| Rank-1 unstalled cycles/token | 1435 | 1467 |
| Compile | PASS | PASS |
| Meets 50 MHz setup | FAIL | FAIL |

Worst slack change: **+3.159 ns**.
The multiplier unit is Quartus's Cyclone IV nine-bit element, not a modern DSP slice.
The state-compute segment retains 1,024 consecutive coordinate writes/token in simulation.

## Isolated paths

The baseline full path reproduces the parent's retained worst path. Extra reports
use the slow 1200 mV / 85C model and source/configuration/report-matched databases.

| Selected path | Endpoints | Data ns | Slack ns | Logic levels |
|---|---|---:|---:|---:|
| baseline-critical | `altsyncram:cosine_rtl_0|altsyncram_pa91:auto_generated|ram_block1a23~porta_address_reg0` → `rot_kxs[28]` | 49.931 | -30.411 | 41 |
| staged-first | `altsyncram:cosine_rtl_0|altsyncram_pa91:auto_generated|ram_block1a0~porta_address_reg0` → `trig_c[16]` | 5.882 | 13.680 | 0 |
| staged-second | `trig_kx[2]` → `rot_kxs[29]` | 45.024 | -25.079 | 35 |

These endpoint sets can select different bits; their delays are not added together.
The candidate's global worst path is `mamba3_cache_ram:cache_bank[0].k_ram|altsyncram:ram|altsyncram_1ed1:auto_generated|ram_block1a0~portb_re_reg` →
`kv_now_product[12]`, 46.871 ns.
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
requires `ROMOUT reset_inflight=checked`, `ROMOUT admits=2048 consumes=2048`, `PHASE admits=2048 consumes=2048`, `SUM admits=65536 consumes=65536`.
Original phase/KV monitors, reset, input pauses, output stalls and collision checks
are retained. Python rechecks every exported FP32/FP16 result and baseline index/value.
FP64-error metrics and phase coverage are unchanged. Each corrupt-golden negative
control fails with exactly one error. There were no candidate mismatches; the
[first-mismatch record](evidence/first_mismatch.json) is NONE and the oracle was not loosened.
Read the [seed-locked log](evidence/PASS_FAIL.log) and [summary](evidence/summary.json).

## Replay

Supplemental [single-operation reports](timing/operator_floor.json) retain the
accumulator-bank add, readout multiplication and final state-add paths. Their
data delays are 44.493, 45.751 and 42.594 ns respectively. Bank selection muxes
and routing are included; the cross-bank accumulator path is a conservative
static path, not a claim of a dynamically observed cross-channel feedback.
These reports motivate inspecting arithmetic-unit boundaries after the memory cut.

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
