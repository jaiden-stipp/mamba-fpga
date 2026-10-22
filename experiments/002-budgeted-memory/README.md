# E002 — Useful updatable memory under a storage budget

Status: completed synthetic study with calibration, learned controls, and fresh confirmation. The tested learned memories did not establish a competitive advantage over the strongest ordinary caches.

Read the [report](research_report.md), [HTML report](research_report.html), [confirmation results](confirmation_results.csv), and [summary](summary.json). This is synthetic memory research, not a language-model or FPGA speed benchmark. [Related hypothesis](../../ideas/I003-updatable-recurrent-memory.md).

Protocols: [initial](PROTOCOL.md), [learned baselines](LEARNED_BASELINE_PROTOCOL.md), and [confirmation](CONFIRMATION_PROTOCOL.md). Source: [benchmark](source/benchmark.py), [methods](source/memories.py), [learned controls](source/learned.py), and [confirmation driver](source/confirm.py).

## Reproduce

Install [requirements.txt](requirements.txt). From this directory:

```sh
python source/test_memories.py
python source/test_learned.py
python source/report.py
```

The report uses recorded `results/`. Training/calibration are separate expensive commands: inspect CLI help and frozen protocols before running. Preserve selected configurations and held-out splits. Budgets count actual per-request array payload; shared parameters and scratch have separate accounting. Synthetic results do not establish general language-model memory capability.
