# Mamba-3 phase precision and DE2-115 experiments

An initial investigation of phase precision in the official 187M-parameter Mamba-3 SISO and MIMO models, with a tested phase-rotation design for the Terasic DE2-115 FPGA board.

This repository contains the completed numerical study, recorded results, source code, and FPGA evidence. It does not establish a new algorithm or guaranteed unpublished research direction. Separating phase-accumulator precision from lookup precision is an established technique; the report discusses related work and the limits of the experiment.

## Main findings

- A 16-bit phase accumulator with an 8-bit lookup address and 8-bit sine/cosine coefficients stayed close to the reference: relative perplexity increased by **0.0093% (SISO)** and **0.0130% (MIMO)** across 32 selected, disjoint 2K-token WikiText windows per model.
- The investigated mixed 8/16-bit selection policies did not beat uniform 10-bit accumulated phase on mean perplexity. These results do not support a positive adaptive-precision algorithm claim.
- A DE2-115 phase-rotation test passed **1,024 distinct vectors**, repeated ten times, with zero arithmetic mismatches in both cores. Each core was compared against its own quantized arithmetic reference.
- In the shared fitted test harness, the 16-bit accumulator core used 14 more logic elements and 19 more registers than the 8-bit core, with the same reported M9K and DSP counts. These are hierarchy measurements within one build, not separate accelerator implementations.

The FPGA implements phase accumulation, rounded lookup, and signed rotation. It does **not** run the SSM recurrence or a complete language model. The language-model measurements use a portable FP32 implementation checked against official reference equations, rather than the native optimized kernels. Selected-window perplexity is not a canonical whole-test-set benchmark. The report includes these and other limitations.

## Read the results

- [Full research report](research_report.md): methods, tables, figures, related work, limitations, and next experiments.
- [Standalone HTML report](research_report.html): download and open locally for a self-contained version with embedded figures.
- [Research-opportunity literature audit](research_opportunities.md).
- [Reproduction instructions](REPRODUCING.md).
- [Complete original reproducibility ZIP](downloads/mamba_phase_precision_reproducibility.zip).

| Path | Contents |
| --- | --- |
| `source/` | Experiment, validation, analysis, and report-generation scripts |
| `source/hardware/` | Verilog, Quartus project, ROM vectors, testbench, and board-test Tcl |
| `source/results/` | Recorded numerical inputs for rebuilding analyses |
| `evidence/` | Raw results, asset hashes, build reports, board runs, and tested `phase_lab.sof` |
| `figures/` | Scientific figures used in the report |
| `language_model_results.csv` | Tabular language-model results |
| `summary.json`, `hardware_results.json` | Machine-readable summaries |

## Reproduce

See [REPRODUCING.md](REPRODUCING.md) for the full workflow. Run the Python scripts from `source/`; the dependency lock is `source/requirements-lock.txt`. The existing scripts generate reports under `../outputs/phase_precision_results` relative to the repository root. The recorded inputs are included, so rebuilding the analysis does not require downloading model weights. Re-running inference does require the separately downloaded assets and an appropriate Python/PyTorch environment.

The tested hardware project is `source/hardware/phase_lab.qpf`, targeting `EP4CE115F29C7` with Quartus Lite 18.1 and a 50 MHz board clock. The programming image is `evidence/phase_lab.sof`. Verify the device and pin map before using it on another board. The test used volatile configuration; board flash was not modified.

## Provenance and redistribution

Model, source, and tokenizer revisions are recorded in [evidence/asset_manifest.json](evidence/asset_manifest.json). The dataset download URLs resolve a moving branch; compare the recorded hashes before claiming reproduction. Model weights, datasets, Python environments, and vendor tool installations are not redistributed here. Small recorded experimental tensors and losses are included.

Upstream source retains its [Apache-2.0 license](source/assets/upstream/LICENSE). Vendor tools and IP remain subject to their applicable licenses. The downloadable ZIP preserves the original delivered package; this repository additionally provides this overview, reproduction notes, and the research-opportunity audit.
