# Experiment index

Each directory is a bounded investigation, not a claim that the broad research idea is new. Status reflects recorded evidence as of 2026-10-22.

| ID | Directory | Status | Main evidence |
|---|---|---|---|
| E001 | [Phase precision](001-phase-precision/README.md) | Completed pilot / board validated | Selected-window model measurements, RTL, Quartus reports, and 10 board runs |
| E002 | [Budgeted memory](002-budgeted-memory/README.md) | Completed synthetic study | Protocols, trained controls, confirmation streams, and byte accounting |
| E003 | [KV compression](003-kv-compression/README.md) | Completed feasibility pilot | 36 paired runs + 12 exploratory eight-bit runs on Qwen3-4B |
| E004 | [Packed-KV serving](004-kv-serving/README.md) | Completed single-trial pilot | 188 selected scored requests across eight configurations; raw outputs, logs and recovery retained |
| E005 | [Read granularity](005-read-granularity/README.md) | Completed analytical screen | 147 address-model configurations; 170 consistency checks; not hardware measured |

Review [the six ranked research protocols and baseline-confirmation plan](planned/README.md) before selecting the next study.

Use [TEMPLATE.md](TEMPLATE.md) for the next experiment. Preserve stable IDs even if a study is retired. Protocol changes after a pilot must be explicitly labeled. Machine-readable status and evidence paths are maintained in [registry.json](registry.json).
