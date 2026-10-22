# E003 — Low-bit KV storage feasibility baseline

Status: completed pilot: 36 paired main runs plus 12 post-hoc eight-bit diagnostics. Existing HF HQQ methods are used; no new codec or fused kernel is claimed.

Read the [report](report.md), [HTML report](report.html), [protocol](PROTOCOL.md), [results](results.csv), and [summary](summary.csv). [NEXT_EXPERIMENT.md](NEXT_EXPERIMENT.md) preserves the original proposal; the prepared implementation is now [E004](../004-kv-serving/README.md).

At 16K, HQQ4 reduced stored KV from 2.254 to 0.643 GiB, retrieved 4/6 codes versus 6/6 for BF16, and took about 19% longer per decode token. Only 1/6 outputs obeyed the strict format. Overall peak allocated GPU memory fell about 5%. Eight-bit diagnostics restored all 12 answers but had unstable timing. No serving improvement or quality noninferiority was established.

## Source and reproduction

`run_baseline.py` runs actual Qwen3-4B inference; `analyze.py` audits bytes and generates reports. `raw_results.jsonl`, `raw_diagnostic8.jsonl`, `prompts.json`, `environment*.json`, and `model_files_sha256.json` retain original evidence. `sources.json` records the dated literature screen; smoke data remains separate.

Install [requirements.txt](requirements.txt) with the CUDA PyTorch index. From this directory:

```sh
python analyze.py
# Separately, for inference (requires downloads and GPU):
python download.py
python run_baseline.py
python run_baseline.py --methods hqq8 --tag diagnostic8
```

The downloader pins Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c` under `assets/Qwen3-4B/`. Set `QWEN3_MODEL_DIR` to reuse other local assets and keep `model_manifest.json` alongside the model folder. Models and datasets stay out of Git. Inference overwrites chosen result files, so preserve independent reruns.

The implementation reconstructs BF16 history before ordinary SDPA. Timing uses 32 generated tokens, continuing after EOS to hold work fixed; quality stops at EOS. It does not measure natural-stop response latency, native INT4 arithmetic, faithful KIVI, or production serving.
