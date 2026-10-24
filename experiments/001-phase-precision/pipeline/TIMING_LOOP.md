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

All retained variants pass 131,072 state writes, 2,560 outputs and 384,448
primitive arithmetic checks with zero bit-reference error; the seven stimulus/
ROM/golden files per rank remain identical. Reset and tag monitors grow with the
added stages. Each corrupt-golden control fails as intended. None meets 50 MHz.

The registered trig lookup infers synchronous ROM address registers and raises
M9Ks from 7 to 9 (logical memory 29,792 → 46,176 bits). Its initial path extraction
used an RTL register name that synthesis absorbed into the ROM; the audit rejected
that extraction, and its diagnostics remain retained beside the corrected reports.

The latest measured worst path is `read_state[6]` → `kv_old_product[30]`,
46.932 ns data delay through FP16 widening and FP32 multiplication. The next
iteration moves widening before the read-stage register, keeping the persistent
state FP16 while adding only 16 transient register bits and no token cycles.
Supplemental single-operation paths already take roughly 42–45 ns in this build;
ordinary inter-operator boundaries are approaching their useful limit. No change
to numerical association, oracle tolerances or constraints is permitted as a
shortcut to a better result.
