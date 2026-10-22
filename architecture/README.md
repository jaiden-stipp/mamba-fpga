# Architecture

Architecture documents distinguish implemented components from proposed systems.

| Document | Status |
|---|---|
| [System scope](system-scope.md) | Boundaries between software experiments, FPGA block validation, and serving |
| [Phase rotator](phase-rotator.md) | Implemented and validated on DE2-115 |
| [Packed-KV reader](packed-kv-reader.md) | Proposed system; stock-engine baseline prepared |
| [ADR 0001](decisions/0001-evidence-and-scope.md) | Evidence/status conventions and no full-model claim from block results |

RTL lives with its owning experiment, currently [E001 hardware](../experiments/001-phase-precision/source/hardware/phase_rotator.v). This avoids duplicating an implementation between architecture notes and experimental evidence.
