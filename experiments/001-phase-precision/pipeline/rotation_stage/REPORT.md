# E001: one rank-1 rotation boundary — 2026-10-24

The isolated candidate registers the eight separately rounded FP32 products of
one rotating B/C pair before their original four FP32 additions. Bit-exact
regression passes. Routed worst setup slack improves by 11.828 ns, but the core
still fails the 50 MHz setup constraint. No physical FPGA operation or inference
speedup is claimed.

The committed `fdbc345` implementation, oracle and reports remain unchanged.
This directory contains the candidate, not a replacement for those historical
results. See [machine-readable comparison](comparison.json).

## Before and after

Both builds use Quartus Lite 18.1, EP4CE115F29C7, N=128, P=8, rank=1,
FAST_FIT, a 20 ns clock, and 447 virtual non-clock pins. QSF and SDC files are
byte-identical. This is one routed build per design, not a placement-seed study.

| Measurement | Retained baseline | One-boundary candidate |
|---|---:|---:|
| Worst setup slack at 50 MHz | -76.239 ns | -64.411 ns |
| Minimum restricted Fmax across reported corners | 10.39 MHz | 11.85 MHz |
| Fitted logic elements | 28,589 | 28,217 |
| Registers | 1,857 | 2,153 |
| Embedded multiplier 9-bit elements | 105 | 105 |
| M9K blocks | 7 | 7 |
| Quartus compilation | PASS | PASS |
| Meets 50 MHz setup | FAIL | FAIL |
| Rank-1 unstalled cycles/token | 1,369 | 1,401 |

The multiplier count is Quartus's Cyclone IV **nine-bit multiplier-element**
count, not a count of modern FPGA DSP slices. The extra stage is used only for
the first N/4 rotating pairs, adding 32 cycles/token. The state-coordinate
compute pipeline still emits 1,024 consecutive writes per token.

## Isolated path and the one change

The [retained TimeQuest report](../synthesis/ram_banked/rank1/pipeline.sta.rpt)
identifies `phase_rtl_0`'s inferred RAM address register to `ky[0]` as the
worst rank-1 path: 95.748 ns data delay, -76.239 ns slack. A fresh full-path
extraction from the matching retained routed database reproduces that exact
row and shows 77 logic levels, phase RAM output and DSP multiplication along
the rotation path: [baseline full path](timing/baseline-phase-to-ky.rpt).

The [candidate RTL](rtl/mamba3_recurrent_tile.sv) changes this sequence:

```text
before: phase/ROM -> fp_mul -> fp_add -> rotated-coordinate register
after:  phase/ROM -> fp_mul -> product register -> fp_add -> rotated-coordinate register
```

All eight products and the pair/rank/rounded-phase-address/valid tags are
captured on the accepted B/C handshake. `ROTATE_ADD` consumes the registered
products one cycle later. Tagged pair/rank select the two subsequent RAM writes;
valid clears after the high-coordinate write and on reset. Input ready stays
low while the pair is pending. Unrotated pairs bypass the new stage, preserving
their raw FP32 bits; rank-4 keeps the original rotation path.

The current-state pipeline, state address/valid shift registers, recurrence
ordering, output accumulation and FP16 writeback are unchanged. No FMA,
reassociation, rounding change or relaxed comparison is introduced.

Targeted slow-85C TimeQuest extractions confirm the actual register boundary:

| Path set | Data delay | Setup slack | Logic levels |
|---|---:|---:|---:|
| Baseline phase RAM -> `ky[0]` | 95.748 ns | -76.239 ns | 77 |
| Candidate phase RAM -> product bank, worst endpoint `rot_qyc[15]` | 65.682 ns | -46.220 ns | 46 |
| Candidate K product bank -> `ky`, worst `rot_kxs[1]` -> `ky[1]` | 57.379 ns | -37.507 ns | 47 |

See [phase-to-products](timing/staged-phase-to-products.rpt) and
[products-to-ky](timing/staged-products-to-ky.rpt). These are worst paths within
different selected endpoint sets; their delays are not additive latency or an
inference-rate measurement.

The candidate's global worst path is now K-cache RAM read-enable register ->
`now1[1]`, with 84.057 ns data delay. Its -64.411 ns slack limits the core before
either selected rotation segment. No second arithmetic change was made.

## Exact verification

Seed `0xe001f032`, ModelSim 10.5b, NumPy 2.2.6. The runner copies and executes
the existing [regression.py](../regression.py), primitive test and arithmetic
functions without editing them. The candidate testbench adds tag/address/valid
checks and aborts a token while the product stage is valid; it retains the
original in-flight state-update reset, input bubbles, output stalls, state and
output scoreboards. Every clear cycle also checks that rotation valid is zero.

| Case | State writes checked | Outputs checked | Maximum bit-reference error | Result |
|---|---:|---:|---:|---|
| Rank 1, 64 tokens | 65,536 | 512 | 0 | PASS |
| Rank 4, 64 tokens | 65,536 | 2,048 | 0 | PASS |

The primitive run checks 96,112 vectors / 384,448 arithmetic results. Rank 1
checks 2,048 stage acceptances, 2,048 add cycles and 4,096 tagged B/C writes.
Both ranks' seven stimulus/ROM/golden files are byte-identical to the retained
baseline. Python independently checks every exported FP32 state update, packed
FP16 state and FP32 output, and compares values/indices against baseline CSVs.
The separate FP64-error metrics are unchanged.

Each deliberately corrupted-golden run fails with exactly one error. There
were no candidate mismatches; [first-mismatch record](evidence/first_mismatch.json)
is explicitly NONE. On a failing candidate run, the existing simulator's
first mismatch lines are retained in its simulation log and indexed by the
wrapper; the oracle is never changed automatically.

Read the [seed-locked PASS/FAIL log](evidence/PASS_FAIL.log),
[regression summary](evidence/summary.json), [rank-1 log](evidence/rank1/simulation.log),
[rank-4 log](evidence/rank4/simulation.log), [build status](synthesis/build_status.json),
[fitter summary](synthesis/pipeline.fit.summary), and
[timing summary](synthesis/pipeline.sta.summary).

## Replay and audit

From this directory, with the same installed NumPy/ModelSim/Quartus dependencies:

```powershell
python run_experiment.py regression
python run_experiment.py synthesis
```

The wrapper stages unmodified baseline tools/shared RTL plus the two explicit
candidate RTL/testbench overlays under ignored `.artifacts`. It never programs
the board or overwrites baseline evidence. [Input hashes](inputs.json) identify
the actual sources and traces used. Databases and programming files remain in
scratch; only reports/configuration/logs are retained here.

For a completed routed project, extract the selected paths with:

```powershell
quartus_sta -t extract_paths.tcl <baseline-project-path> <absolute-output-prefix>/baseline baseline
quartus_sta -t extract_paths.tcl <candidate-project-path> <absolute-output-prefix>/staged staged
python analyze.py
```

Project paths identify the `pipeline` project without its extension. Extraction
uses the slow 1200 mV / 85C model. The extraction logs preserve the actual
commands. The baseline database matched the retained core/source, QSF, SDC,
fitter and timing report hashes; its full path reproduced the retained slack.
`analyze.py` verifies source hashes, unchanged oracle/stimuli, every CSV result,
negative controls, exact constraints, report provenance and isolated endpoints
before regenerating `comparison.json`. Baseline evidence has its independent
unchanged audit in `../audit_evidence.py`.
