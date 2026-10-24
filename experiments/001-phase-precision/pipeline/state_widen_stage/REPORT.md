# E001 timing loop: state_widen_stage — 2026-10-24

Retiming the existing exact FP16-to-FP32 widening operation before the rank-1 read register. The parent is `8fda2a3` / `state_read_stage`.
This is an isolated cumulative candidate; earlier RTL and evidence remain unchanged.
No physical FPGA operation or inference speedup is claimed.

## One change

Capture half_to_float(stored0) into a 32-bit read_state lane buffer instead of capturing the 16-bit code and widening at the multiplier input. Multiply alpha by the already widened register. This moves the existing exact conversion across one existing boundary, adds no stage or token cycles, and changes no numerical operation. Persistent state_mem remains FP16. Declared transient state-buffer width grows by 16 bits; actual fitted registers depend on constant propagation. The read-stage monitor checks the widened word with the same original conversion function (independently tested over all 65,536 half bit patterns by the unchanged primitive regression), while every state/output still uses the unchanged external NumPy golden files. Add explicit 16-bit persistent / 32-bit read-buffer assertions.

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
| Worst setup slack at 50 MHz | -26.971 ns | -25.936 ns |
| Worst-corner restricted Fmax estimate | 21.29 MHz | 21.77 MHz |
| Fitted LEs | 28,713 | 28,651 |
| Registers | 2,876 | 2,879 |
| Nine-bit multiplier elements | 105 | 105 |
| Rank-1 unstalled cycles/token | 1468 | 1468 |
| Compile | PASS | PASS |
| Meets 50 MHz setup | FAIL | FAIL |

Worst slack change: **+1.035 ns**.
The multiplier unit is Quartus's Cyclone IV nine-bit element, not a modern DSP slice.
The state-compute segment retains 1,024 consecutive coordinate writes/token in simulation.

## Isolated paths

The baseline full path reproduces the parent's retained worst path. Extra reports
use the slow 1200 mV / 85C model and source/configuration/report-matched databases.

| Selected path | Endpoints | Data ns | Slack ns | Logic levels |
|---|---|---:|---:|---:|
| baseline-critical | `read_state[6]` → `kv_old_product[30]` | 46.932 | -26.971 | 38 |
| staged-first | `stored0[12]` → `read_state[21]` | 9.987 | 9.943 | 8 |
| staged-second | `read_state[22]` → `kv_old_product[16]` | 42.486 | -22.574 | 34 |

These endpoint sets can select different bits; their delays are not added together.
The candidate's global worst path is `trig_c[1]` →
`rot_kyc[29]`, 45.886 ns.
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
requires `WIDEN conversions=65536 persistent_bits=16 read_bits=32`, `READ admits=65536 consumes=65536`, `ROMOUT admits=2048 consumes=2048`, `PHASE admits=2048 consumes=2048`, `SUM admits=65536 consumes=65536`.
Original phase/KV monitors, reset, input pauses, output stalls and collision checks
are retained. Python rechecks every exported FP32/FP16 result and baseline index/value.
FP64-error metrics and phase coverage are unchanged. Each corrupt-golden negative
control fails with exactly one error. There were no candidate mismatches; the
[first-mismatch record](evidence/first_mismatch.json) is NONE and the oracle was not loosened.
Read the [seed-locked log](evidence/PASS_FAIL.log) and [summary](evidence/summary.json).

## Replay

The selected conversion segment now takes 9.987 ns and the selected old-state
multiply takes 42.486 ns. Persistent main state remains 2 KiB for this tile.
The logical lane buffer grows from 16 to 32 bits; the fitted design adds only
three net registers (2,876 → 2,879), with the multiplier count unchanged.

The final global worst path is one FP32 multiplication, not two serial operators.
Supplemental [single-operation reports](timing/operator_floor.json) reproduce it
and inspect the other arithmetic paths. The external-register loop stops at this
[arithmetic-unit boundary](../ARITHMETIC_LIMIT.md); 50 MHz is still not closed.

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
