# E001 deterministic phase-corner regression — 2026-10-23

**PASS.** Seed `0xe0014096` generates 4,096 distinct new vectors with no overlap
with the original 1,024 packed input vectors. Each vector is checked on both
the 16-bit guarded accumulator and the 8-bit accumulator against its own
bit-accurate quantized reference. Both cores have **zero mismatches and maximum
integer output error 0**. LUT addresses, output counts, final phases, and harness
cycle counts are checked as well.

This result is RTL simulation using ModelSim Altera Starter Edition 10.5b.
It is not an additional FPGA board run, synthesis result, or full-model evaluation.

## Coverage and execution

| New vector category | Count | Directed coverage |
|---|---:|---|
| Phase wraparound | 1,024 | 512 positive and 512 negative crossings of the 16-bit modular boundary |
| Signed extrema | 1,024 | Both axes at -32,768 / +32,767, all four extreme/extreme pairs, and directed signed increment extrema |
| Zero increments | 1,024 | Both phase accumulators hold state, including zero and extreme coordinates |
| Rounding boundaries | 1,024 | Every LUT bin at fractional phase 127, 128, 129, and 255; even/odd halfway ties and address 255-to-0 carry |

The original workload passes separately: 1,024 vectors / 1,028 cycles. The new
workload passes from reset: 4,096 vectors / 4,100 cycles. Total positive-test
coverage is **5,120 vectors / 10,240 core results**. Counts do not include the
intentional negative-control rerun.

The PRNG is explicitly specified LCG32 arithmetic, rather than an implicit
library RNG. Generation is repeated and compared before simulation. The runner
recomputes every original golden word with the shared integer reference, checks
the sine/cosine ROMs, and rejects duplicate new inputs or overlap with legacy data.
An intentional corruption flips one expected real-output bit for each core at
new vector 0; both RTL comparators detect exactly one error and report FAIL.
The regression treats that observed failure as a passing negative control.

## Error against the higher-precision rotation model

Both RTL cores and this reference begin the new sequence at phase zero.
The reference accumulates the underlying signed increments exactly on a
2^24-ticks-per-turn integer grid, then uses float64 sine/cosine without LUT address
rounding or coefficient quantization. Each core receives independently rounded
16-bit or 8-bit increments derived from those same higher-precision increments.

Inputs are the same exact signed Q5.10 coordinates in both models. Normalize RTL
product sums by `127 * 1024` before comparing, so the following errors are in
**input coordinate units**, not raw product integers or model loss/perplexity.

| Core | Maximum absolute component error | Maximum 2-D L2 error | Component RMSE | Worst component-error vector (zero-based) |
|---|---:|---:|---:|---:|
| Guarded accumulator, 16 bits | 0.627718459294 | 0.645217383348 | 0.146337934317 | 1,339 |
| Low-precision accumulator, 8 bits | 79.7722015277 | 87.8470320492 | 14.2782079942 | 3,686 |

| Category | Guarded maximum component error | 8-bit maximum component error |
|---|---:|---:|
| Phase wraparound | 0.189032173038 | 0.189032173038 |
| Signed extrema | 0.627718459294 | 4.86608610344 |
| Zero increments | 0.0536940421806 | 1.03920918705 |
| Rounding boundaries | 0.619129308002 | 79.7722015277 |

The large 8-bit error is compatible with exact RTL/reference agreement: the RTL
implements the intended low-precision arithmetic, while repeated increment
rounding loses phase information. The new stream intentionally stresses these
boundaries; it is not a representative language-model activation distribution.
Categories share accumulated state within the new stream. Zero increments hold
the existing phase and its existing numerical error; they do not reset it.

These numerical errors include increment quantization, accumulation differences,
phase-to-address rounding, and trig coefficient quantization. They exclude
additional input quantization, tanh/dt preprocessing, the SSM recurrence, and
full-model effects. Legacy raw floating increments are unavailable, so this
higher-precision comparison applies only to the separately reset new sequence.
No acceptable-quality bound is asserted for these synthetic errors; PASS refers
to RTL correctness and the declared coverage/control checks.

## Evidence and reproduction

- [PASS/FAIL log](evidence/phase_corner_regression/PASS_FAIL.log): seed, counts, maximum errors, and data hashes.
- [Summary](evidence/phase_corner_regression/summary.json): category coverage, numerical metrics, tool version, source and artifact SHA-256 values.
- [New per-vector comparisons](evidence/phase_corner_regression/per_vector_comparison.csv): every RTL result, expected integer result/address, PASS/FAIL, and signed error against float64.
- [Legacy per-vector comparisons](evidence/phase_corner_regression/legacy_per_vector_comparison.csv): the retained 1,024-vector regression.
- [New vector definitions](evidence/phase_corner_regression/new_vectors.jsonl): input coordinates, packed inputs, higher-precision increments/outputs, category, and phase states.
- [Input ROM](evidence/phase_corner_regression/vectors.hex) and [golden ROM](evidence/phase_corner_regression/golden.hex): 4,096 new cases.
- [Compilation](evidence/phase_corner_regression/compile.log), [legacy simulation](evidence/phase_corner_regression/legacy_transcript.log), [new simulation](evidence/phase_corner_regression/extended_transcript.log), and [expected negative-control failure](evidence/phase_corner_regression/negative_control_transcript.log).
- [Runner](source/phase_corner_regression.py), [shared reference](source/phase_reference.py), and [reproduction command](REPRODUCING.md).

Transcripts retain result/error lines and omit wall-clock banners and temporary
paths. Per-vector integer results and ROM files are byte-reproducible; float64
metrics may differ by tiny libm rounding amounts across operating systems.
Original board ROMs, bitstream, and historical resource/timing measurements are
retained. The core arithmetic RTL is unchanged; the controller/testbench are
extended and have not been resynthesized or revalidated on board.
