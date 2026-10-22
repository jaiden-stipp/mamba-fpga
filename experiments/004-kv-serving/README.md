# E004 — Packed-KV serving with quality and response deadlines

Status: completed single-trial GPU serving pilot on 2026-10-22. **188 selected scored requests across eight configurations.** Q8/Q4 loaded eight long-context slots; matched-quality throughput superiority is not established. Original preparation checks and all failed/recovery evidence are retained.

Start with [GPU results](serving_report.md), [CSV](serving_summary.csv), [summary JSON](serving_summary.json), and the [protocol amendment](AMENDMENT-2026-10-22.md). Run `python analyze_serving.py --plot` to regenerate the report/figure from raw evidence (Matplotlib is optional for plotting).

Historical preparation: [research status](research_status.md), [HTML status](research_status.html), [protocol](PROTOCOL.md), [validation](validation.json), [model audit](model_audit.json), [client test](client_validation.json), and [CPU smoke](cpu_smoke.json). The historical preparation files establish integration only; the new GPU report contains the measurements and their limitations. [Proposed architecture](../../architecture/packed-kv-reader.md).

The [new streaming validation](client_validation_20261022.json) includes a 50 KB terminal event. [Preparation validation](preparation_validation.json) preserves the earlier zero-GPU-run snapshot; [current validation](validation.json) records the completed pilot.

## Pilot comparison

Official llama.cpp b11751, commit `84663953535d4323e92b4bc90971aa022c305d06`, CUDA 12.4 Windows binaries; same BF16-weight Qwen3-4B GGUF with F16/Q8_0/Q4_0 KV. Slots 1/4/8, about 16K input, natural EOS, cold and resident histories. Provisional deadlines: 20 s first token, 100 ms mean post-first-token time/token, 45 s completion. They are illustrative requirements fixed before timing, not measured results or user-specified production SLOs.

No new attention kernel or refinement policy exists. Source inspection is not runtime dispatch or DRAM-traffic profiling.

## Assets and setup

Install [requirements.txt](requirements.txt). Defaults from repository root:

| Asset | Location | Override |
|---|---|---|
| HF model | `experiments/003-kv-compression/assets/Qwen3-4B/` | `QWEN3_MODEL_DIR` |
| Official engine/source | `tools/llama.cpp/` | `LLAMA_CPP_DIR` during setup |
| Serving binary | `tools/llama.cpp/bin/llama-server.exe` | `LLAMA_SERVER` |
| BF16 GGUF | `assets/models/Qwen3-4B-BF16.gguf` | `QWEN3_GGUF` |

Run `python setup_engine.py` here to download official binaries and matching source with release-digest checks. Download the HF model using E003, create `assets/models/` at repository root, then convert from that root:

```sh
python tools/llama.cpp/source/convert_hf_to_gguf.py experiments/003-kv-compression/assets/Qwen3-4B --outfile assets/models/Qwen3-4B-BF16.gguf --outtype bf16
```

External weights, binaries, and downloaded source are ignored. Rerun the audit for a new local conversion; the recorded audit describes the original external GGUF.

## Run from this directory

```sh
python test_client.py
python audit_model.py
python cpu_smoke.py
python build_workload.py
python serve_benchmark.py --smoke --slots 1 --formats f16 --tag gpu_smoke
python serve_benchmark.py --tag pilot
```

The harness binds to 127.0.0.1, starts hidden own-server processes, and cleans only their handles. It refuses competing GPU allocations by default, disables auto fitting/context reduction, and never unloads another service. An analytical capacity guard prevents impossible allocations from being mistaken for GPU-resident results; it is not a measured OOM. Use a fresh `--tag` for every run; existing raw tags are refused. JSON snapshots use replacement/retry for transient Windows locks. Use `--gpu 1` only when available. Inspect placement, actual formats, and context-size logs before accepting GPU timing.

`raw_<tag>.jsonl` preserves responses and timing; `waves_<tag>.json` records goodput, cache hits, makespan and GPU samples; `configurations_<tag>.json` records commands/setup failures. Resident hits use `timings.cache_n`, not final occupancy. Report cold and resident timing separately and retain loading costs. The small synthetic pilot cannot prove 1-point quality noninferiority or publishable novelty.
