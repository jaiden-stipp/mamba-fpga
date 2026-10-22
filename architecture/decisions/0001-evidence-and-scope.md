# ADR 0001 — Separate proposals, numerical evidence, hardware evidence, and serving results

Date: 2026-10-22. Status: accepted as repository organization.

## Context

The work spans model math, low-bit storage, RTL, FPGA builds, and serving preparation. A block test, cache compression ratio, or source-level kernel inspection supports a narrower claim than end-to-end model acceleration.

## Decision

- Keep each study's protocols, source, results, and evidence under one stable experiment ID.
- Put untested mechanisms in ideas/ and proposed architecture documents.
- Record implementation status and evidence class explicitly.
- Preserve negative results and exploratory/confirmation distinctions.
- Keep large external assets and environments out of version control.
- Keep the current stock-engine serving study marked prepared until real GPU results exist.

## Consequences

Navigation and claim review become easier, while original recorded reports may retain historical paths. Portability changes are documented in PROVENANCE.md. Stronger claims require new experiments, not a README status edit.
