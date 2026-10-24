# E001 rank-1 timing loop

Each successor targets the parent's measured global worst registered path, keeps
the bit-exact arithmetic contract, runs both ranks through the existing ModelSim
regression, and uses the same Quartus Lite 18.1 / EP4CE115F29C7 / FAST_FIT /
20 ns / virtual-I/O constraints. Each retained iteration has a separate commit.
These are routed estimates, not physical FPGA operation or inference speedups.

| Cumulative variant | Worst setup ns | Fmax MHz | LEs | Registers | 9-bit multipliers | Unstalled cycles/token |
|---|---:|---:|---:|---:|---:|---:|
| [Original](REPORT.md) | -76.239 | 10.39 | 28,589 | 1,857 | 105 | 1,369 |
| [Rotation products](rotation_stage/REPORT.md) | -64.411 | 11.85 | 28,217 | 2,153 | 105 | 1,401 |
| [K×V products](kv_product_stage/REPORT.md) | -56.670 | 13.04 | 27,902 | 2,279 | 105 | 1,402 |
| [State-update sum](update_sum_stage/REPORT.md) | -36.953 | 17.56 | 28,780 | 2,377 | 105 | 1,403 |
| [Registered trig lookup](phase_lookup_stage/REPORT.md) | -30.411 | 19.84 | 28,697 | 2,506 | 105 | 1,435 |
| [Trig output register](trig_output_stage/REPORT.md) | -27.252 | 21.16 | 28,687 | 2,699 | 105 | 1,467 |
| [RAM output register](state_read_stage/REPORT.md) | -26.971 | 21.29 | 28,713 | 2,876 | 105 | 1,468 |
| [Retimed state widening](state_widen_stage/REPORT.md) | -25.936 | 21.77 | 28,651 | 2,879 | 105 | 1,468 |

All retained variants pass 131,072 state writes, 2,560 outputs and 384,448
primitive arithmetic checks with zero bit-reference error; the seven stimulus/
ROM/golden files per rank remain identical. Reset and tag monitors grow with the
added stages. Each corrupt-golden control fails as intended. None meets 50 MHz.

The registered trig lookup infers synchronous ROM address registers and raises
M9Ks from 7 to 9 (logical memory 29,792 → 46,176 bits). Its initial path extraction
used an RTL register name that synthesis absorbed into the ROM; the audit rejected
that extraction, and its diagnostics remain retained beside the corrected reports.

The widening retime keeps persistent state FP16 and adds no cycles. Its declared
lane buffer is 16 bits wider, but synthesis adds only three net fitted registers.
The conversion path takes 9.987 ns and the selected old-state multiply 42.486 ns.

The latest measured worst path is `trig_c[1]` → `rot_kyc[29]`, **one FP32
multiplication**, 45.886 ns / 37 logic levels. This is the [stopping point for
external register cuts](ARITHMETIC_LIMIT.md): the next relevant implementation
change is an internally pipelined arithmetic unit, with latency-aware scheduling
for accumulator feedback. Single-operation paths and their actual source/report
hashes are retained. This is not a claim that 50 MHz was achieved or cannot be
achieved by a different implementation. No numerical equality or timing constraint
was relaxed. The latest verified candidate is [state_widen_stage](state_widen_stage/REPORT.md).
