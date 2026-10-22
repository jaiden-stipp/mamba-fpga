# FPGA and ML inference research

A research workspace for FPGA/RTL acceleration, numerical precision, and memory-efficient inference. The repository began with Mamba-3 phase arithmetic and now includes bounded-memory experiments and transformer KV-cache studies.

**Start here:** [experiment index](experiments/README.md), [architecture](architecture/README.md), [research ideas](ideas/README.md), [reading path](literature/README.md), and [roadmap](ROADMAP.md).

## Research status

| ID | Investigation | Status | What the evidence supports |
|---|---|---|---|
| [E001](experiments/001-phase-precision/README.md) | Mamba-3 phase precision and DE2-115 RTL | Completed pilot + hardware validation | Numerical tradeoffs and a verified phase-rotation component; no full-model FPGA acceleration claim |
| [E002](experiments/002-budgeted-memory/README.md) | Useful, updatable memory under a byte budget | Completed synthetic study | Learned-memory controls and ordinary caches compared on fresh confirmation streams; no competitive new memory method established |
| [E003](experiments/003-kv-compression/README.md) | Low-bit KV storage in Qwen3-4B | Completed feasibility pilot | Smaller stored KV can coexist with worse accuracy and slower decoding |
| [E004](experiments/004-kv-serving/README.md) | Packed-KV serving at fixed deadlines | Prepared; integration checks passed | Model conversion, HTTP/SSE timing, and workload preparation; **zero GPU serving runs** recorded |

The current question is whether compressed memory can support more useful, simultaneous long-context requests at matched quality and response deadlines. This broad direction already has substantial prior work. Candidate contributions and novelty checks belong in [ideas/](ideas/README.md); measured claims belong in [docs/claims.md](docs/claims.md).

## Repository map

```text
experiments/    Per-study protocols, source, recorded results, evidence, and reports
ideas/          Hypotheses, closest prior work, decision gates, and idea template
architecture/   Implemented RTL scope, proposed dataflows, and design decisions
literature/     Reading path, source registers, and historical landscape review
docs/           Research method, reproducibility, provenance, and claim boundaries
tools/          Lightweight repository validation and experiment packaging
.github/        Validation workflow and structured research issue templates
ROADMAP.md      Milestones tied to evidence and dependencies
```

## Quick checks

From the repository root, with Python 3.12 or later:

```sh
python tools/validate_repository.py
```

This checks navigation, study manifests, recorded row counts, and status consistency without GPU access or downloaded models. Experiment-specific commands and dependencies are in each experiment's README. Reports and raw results are included; large external models, datasets, CUDA binaries, environments, and generated vendor databases stay outside Git.

## Selected measured results

- **E001:** a 16-bit phase accumulator / 8-bit lookup / 8-bit coefficients produced +0.0093% and +0.0130% relative perplexity on selected SISO/MIMO evaluation windows. The DE2-115 datapaths matched their own quantized references on 1,024 distinct vectors repeated over 10 runs at 50 MHz.
- **E003:** at 16K context, HQQ4 stored 0.643 GiB of KV versus 2.254 GiB for BF16, while retrieving 4/6 codes versus 6/6 and taking about 19% longer per decode token. Only 1/6 HQQ4 responses followed the strict output format. Cache-only compression is not whole-model memory reduction or inference acceleration.

See [claims and limitations](docs/claims.md) before reusing numbers in a presentation or resume. The recorded studies are small and have explicit scope limits; prepared experiments are not results.

## Working conventions

Give experiments and ideas stable IDs, freeze an evaluation protocol before measuring, keep exploratory changes visible, and preserve failed outputs. A new experiment starts from [the experiment template](experiments/TEMPLATE.md); a new hypothesis starts from [the idea template](ideas/TEMPLATE.md). Link related literature and architecture decisions. Archive conclusions even when an idea fails.

## Provenance and licensing

Original evidence is retained with documented repository-portability edits in [PROVENANCE.md](PROVENANCE.md). Upstream files keep their existing licenses, including the [Mamba source Apache-2.0 license](experiments/001-phase-precision/source/assets/upstream/LICENSE). Model and vendor-tool licenses apply separately. No blanket license is asserted over third-party material; see [licensing notes](docs/licensing.md).
