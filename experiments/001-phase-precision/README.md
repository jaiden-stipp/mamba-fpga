# E001 — Mamba-3 phase precision and DE2-115

Status: completed numerical pilot and board validation. No new phase algorithm or full-model FPGA acceleration is established.

Read the [report](research_report.md), [HTML report](research_report.html), [reproduction instructions](REPRODUCING.md), [model table](language_model_results.csv), and [hardware summary](hardware_results.json). The original [delivery ZIP](downloads/mamba_phase_precision_reproducibility.zip) is retained. See the [RTL architecture](../../architecture/phase-rotator.md) and [original opportunity audit](../../literature/research_opportunities.md).

The 16/8/8 phase configuration produced +0.0093% / +0.0130% relative perplexity for SISO/MIMO on 32 selected disjoint 2K windows per model. The FPGA tested 1,024 distinct vectors ten times at 50 MHz with zero arithmetic mismatches against each core's own quantized reference. Mixed phase heuristics did not beat uniform 10-bit accumulated phase.

The FPGA covers phase accumulation, LUT addressing, sine/cosine lookup, and signed rotation. It excludes the SSM recurrence and full model. Model results use portable FP32 reference equations, not native optimized-kernel parity. Selected-window evaluation and JTAG timing have explicit limitations.

## Reproduce

`source/` holds scripts and pinned upstream code; `source/results/` holds analysis inputs; `source/hardware/` holds RTL, Quartus project, ROMs, testbench, and original board/build files. `evidence/` and `figures/` hold report artifacts.

From this directory with the documented dependencies:

```sh
python source/analyze.py
python source/make_report.py
```

Recorded-input analysis needs no model download. Analysis now writes into this experiment directory. Inference needs separately downloaded checkpoints; follow REPRODUCING.md and verify hashes. Quartus/ModelSim operations require vendor tools and target-device checks.
