# From recurrent tile to complete accelerated decode

The target remains: serve more concurrent long-context requests at matched
answer quality and response deadlines by making recurrent memory cheaper to
read. This plan concerns research implementation, not a study schedule.

## Current boundary

The RTL owns phase, previous inputs, FP16 main state, recurrence, current-output
reduction, and D skip for one resident head/P slice. A replaceable software
backend invokes the same arithmetic inside both complete 187M checkpoint
decoders. The software backend is an emulator; no FPGA transport is implemented.

Dense input/output projections, normalization, nonlinear coefficient generation,
MIMO projections/gating/output normalization, FFN, embedding and vocabulary
projection remain in Torch. The original E001 board result still applies only
to the older stand-alone phase rotator.

## Gate 1: close the larger arithmetic pipeline

The completed first builds fit but miss 50 MHz: estimated worst-corner Fmax is
10.39 MHz for SISO and 4.87 MHz for rank-4 MIMO. SISO is limited by phase lookup
through rotation; MIMO by the rank reduction into `now1`. These are measured
static-timing findings, not predictions of complete-model speed.

Subsequent isolated [rank-1 timing iterations](TIMING_LOOP.md) reach 21.77 MHz
estimated Fmax while preserving the tested bit contract. The latest limiting
path is now one FP32 multiplication; the [arithmetic-unit boundary report](ARITHMETIC_LIMIT.md)
records why internal unit pipelines and feedback scheduling are the next work.
The first-build findings above remain historical; 50 MHz is still not closed.

Use the retained Quartus reports to identify the actual critical path, logic,
DSP and RAM use. Replace long combinational floating-point functions with
appropriately staged implementations while retaining this regression as the
contract. Phase rotation currently combines multiplication and addition in a
load cycle; rank reduction and output accumulation also need careful staging.
Changing the reduction tree or using FMA requires new expected arithmetic and
a separate numerical comparison, not simply loosening bit-equality checks.

The concrete next design is pipelined FP32 multiply/add units, registered
coefficient reads, and registered rank reductions with aligned state/address/
valid tags. Preserve the left-to-right rank sum if retaining the current bit
contract. A balanced tree would change association and must be a distinct
numerical variant.

A multi-cycle output adder introduces a feedback hazard: consecutive N values
cannot update the same accumulator every clock before its previous result
returns. Interleave independent P channels, issuing N-major with P varying
inside each N, so the same channel recurs only every P cycles. If P is at least
the adder latency, this can retain one state coordinate per cycle while
preserving each output's original N summation order. P=64 also amortizes the
shared B/C and phase caches. This scheduler is proposed, not implemented in the
current P-major prototype. Its acceptance test must check tagged addresses
rather than assume monotonically increasing write addresses.

Acceptance: a routed design with a verified operating clock and a physical
self-test reproducing the RTL-checked trace. Report cold reset, input loading,
compute, and output drain separately. A virtual-I/O timing build is preliminary.

## Gate 2: scale state ownership and add transport

Add request/layer/head IDs, an explicit context layout and DMA with backpressure
and fault handling. Keep compressed state in its compressed format across the
transport. Measure actual transferred bytes, burst efficiency, and conversion
cost. Preserve previous-input and phase caches per request.

Add prefill-state import/export as well: a GPU-prefilled prompt must transfer
the complete initial state, phase and previous-input caches into the correct
request context. The current tile starts from zero and consumes consecutive
tokens; it cannot yet accept a prefilled context. Count that transfer in TTFT.

Share B/C, phase and previous K across P slices of the same head. Blindly
replicating P=8 instances duplicates these caches and the rotation work. A
P=64 head engine or shared broadcast buffers can avoid that duplication.
Similarly, cache raw previous V plus projection weights when that reduces
total cost; the present rank-projected previous-V cache is a prototype choice.

Acceptance: host-to-FPGA replay of consecutive tokens and interleaved request
contexts, with exact quantized checks, transfer-byte counters, reset/recovery
checks, and board timestamps. JTAG self-test demonstrates correctness; it is
not the proposed high-throughput serving transport.

## Gate 3: integrate and profile complete decode

Implement the physical backend behind `RecurrentBackend.step`. Initially retain
dense work on the GPU, and count every synchronization and transfer. Test that
hybrid boundary before investing in a complete FPGA dense engine: a faster
recurrence can still make the overall decoder slower if transfer dominates.

If dense work dominates, implement a reusable matrix engine with tiled external
weights, double buffering, normalization/nonlinear approximations, residual and
gating paths, and vocabulary projection. Reuse that engine for the model's
input/output projections and FFN. Its formats need their own checkpoint-level
quality validation. State compression does not eliminate the weight traffic.

Acceptance: identical tokenizer/checkpoint/task inputs; complete prompt prefill
and autoregressive generation; full time-to-first-token and inter-token timing,
not only recurrent kernel timing. Verify the actual optimized baseline's
dispatch: some existing Mamba-3 kernels already fuse rotation/recurrence/readout.
Fusion alone is not a novelty claim.

## Gate 4: the acceleration experiment

Before running, freeze workload, request lengths, concurrency sweep, admission
policy, answer metric, quality tolerance, deadlines and repetition count. Use
fresh evaluation tasks beyond the diagnostic corpus windows already inspected.

Compare a competent optimized uncompressed/fused baseline, FP16 memory, and any
quality-qualified lower-bit candidate. Record TTFT, p50/p95/p99 inter-token
latency, completed requests meeting the deadline, throughput, total memory,
actual bandwidth, and energy if measurable. Count weights, metadata, all
request caches, staging, queues, wide tile buffers, transfer buffers and any
error-feedback sidecars. Use warmups, repeated runs and uncertainty intervals.

Acceptance: more completed concurrent requests at the same quality/deadline
contract, or a clearly bounded negative result explaining the bottleneck.

## Hardware proposal trigger

Request newer hardware when the measured design shows a specific limit: lack
of sufficient RAM for the chosen residency policy, too few arithmetic lanes,
external bandwidth, or transport latency. Specify required capacity/bandwidth
from the workload ledger and measured utilization. PCIe plus sufficient onboard
DRAM/HBM and suitable floating-point DSP resources are capabilities to evaluate;
no particular board is presumed necessary before those measurements.

For perspective, both 187M configurations have 12*24*64*128 = 2,359,296 main
state coordinates per request. One lane processing one coordinate per cycle
would need at least 47.19 ms at a hypothetical 50 MHz for that arithmetic alone.
This excludes loading, dense layers, transfers and scheduling. Parallelism and
shared buffers are therefore architectural requirements; a single correct tile
does not imply an impressive whole-model token rate.
