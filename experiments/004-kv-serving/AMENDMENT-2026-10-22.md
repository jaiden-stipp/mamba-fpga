# Serving protocol amendment — 2026-10-22

Recorded before the full GPU pilot. Preserve the original protocol and all smoke outputs.

- The installed b11751 engine accepts `--load-mode none` in place of the older `--no-mmap` option. The original smoke tag `research_smoke_20261022` failed before inference; these are startup failures, not model failures.
- Tag `research_smoke_v2_20261022` exercised one cold case per format successfully. Its byte-at-a-time SSE client added approximately 0.6 seconds parsing the final event, which included a redundant 47 KB prompt. These three runs are integration evidence only, excluded from performance comparisons.
- The corrected client reads available chunks through urllib3 `read1`, preserves immediate token delivery, and requests only timing/answer/status fields in the final event. A local HTTP test with a 50 KB terminal event verifies that first-token delivery is not buffered and mean token time follows the server's deliberate delays. Deadlines and answer scoring remain unchanged.
- Enable engine verbosity 4 to preserve actual placement, context, KV type and allocation logs.
- Before launching, apply an analytical device-capacity guard: GGUF file size plus exact nominal KV allocation for this checkpoint must not exceed the initial free VRAM. GGUF size is a conservative weight proxy, not measured resident allocation; scratch memory is additional. F16 with eight 17,408-token slots exceeds this bound by several GiB. Record `not_run_capacity_guard`, not observed OOM, and do not permit host spill as a capacity result.
- Collect tag `research_pilot_20261022` using the original slots 1/4/8 and formats F16/Q8_0/Q4_0. It is a single timing trial on eight synthetic documents, not a quality noninferiority test or publication-ready throughput claim.

Further changes, if needed, must receive a new tag and an explicit explanation. The earlier HQQ study remains a separate engine, codec and protocol.

## Recovery after the pilot

All eight runnable configurations completed their requests. A transient Windows `OSError: [Errno 22] Invalid argument` prevented the last Q8/eight-slot configuration and its wave summaries from being saved. Its 48 raw rows were already flushed and remain in the original pilot file. No GPU timing is reconstructed from rounded terminal output.

Before repeating that configuration, the harness was changed to publish JSON snapshots through a temporary file plus replacement, retry transient locks, save after each wave, and refuse an existing raw tag. The new tag `research_recovery_20261022` repeats only Q8/eight slots. Analysis uses the new run for that configuration regardless of its outcome and retains the original run as excluded duplicate evidence. This is operational recovery, not an independent confirmation trial of the complete study. No deadlines, inputs, scoring or inference settings changed.
