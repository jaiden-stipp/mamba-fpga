# E001 recurrent pipeline implementation — 2026-10-23

The phase experiment now has a functioning recurrent-tile RTL implementation and an executable whole-model arithmetic integration boundary. This is a substantially larger hardware block than the original phase rotator. Complete-model FPGA inference and an inference acceleration claim remain unestablished.

## What was implemented

- Guarded phase accumulation and lookup, with FP32 B/C coordinates and the existing 16/8/8 phase/trigonometric policy.
- Resident FP16 main-state RAM, widening on read, the original current-plus-previous-input recurrence, FP32 readout before storage rounding, FP16 writeback, and the D skip term.
- Persistent phase and previous K/V caches, token sequencing, paused input streams, stable output under backpressure, and reset/clear behavior.
- Portable finite-width FP32 multiplication/addition and FP16 conversion, with explicit separate-operation rounding. Arithmetic is implemented in synthesizable integer logic, not simulation-only real variables.
- A replaceable recurrence backend inside complete SISO/MIMO checkpoint decoders. The supplied backend is a Torch arithmetic emulator, not FPGA transport.

See the [interface and block diagram](README.md), [RTL](rtl/mamba3_recurrent_tile.sv), and [full-decoder gates](FULL_DECODER_PLAN.md).

## Verification

Seed: `0xe001f032`. ModelSim primitive regression: **96,112 input pairs / 384,448 checked results**, including all 65,536 FP16 bit patterns. The four checks per pair are FP32 add, FP32 multiply, FP32-to-FP16, and FP16-to-FP32. NaN policies are explicit in the source; this is not an exhaustive certification of all FP32 inputs.

| RTL case | Tokens | Exact state writes | Exact outputs | Max bit-reference error | Unstalled cycles/token |
|---|---:|---:|---:|---:|---:|
| Rank 1, P=8, N=128 | 64 | 65,536 | 512 | 0 | 1,369 |
| Rank 4, P=8, N=128 | 64 | 65,536 | 2,048 | 0 | 1,993 |

Each case contains 32 deterministic synthetic tokens followed by a reset and 32 actual checkpoint-input tokens. Actual traces select layer 0, head 0, P channels 0–7, full N=128, from WikiText-2 validation offset 32,768. The traces are frozen upstream inputs; sampled RTL replay alone is not a full-model quality evaluation. Exact stimulus, reference, simulator output and checkpoint hashes are retained.

The testbench also aborts an in-flight update and resets the core, inserts input pauses, and stalls output consumption. Every token has 1,024 consecutive state writes: one coordinate per cycle in the compute segment. Deliberately corrupting one expected state code yields exactly one reported error per rank, demonstrating that the scoreboard rejects a bad result. Python independently rechecks every exported result; the runner requires PASS markers and correct counts rather than trusting simulator exit status alone.

A second clean simulator build/run reproduced all 24 retained evidence files byte-for-byte; see [repeatability hashes](repeatability.json). Read the [PASS/FAIL log](evidence/PASS_FAIL.log), [summary](evidence/summary.json), [SISO simulation](evidence/rank1/simulation.log) and [MIMO simulation](evidence/rank4/simulation.log).

## Error against higher precision

Bit equality above is against the quantized, explicitly ordered reference. The separate higher-precision comparison uses FP64 state feedback, FP64 recurrence/readout and true trigonometry of the accumulated 16-bit phase. It retains the same supplied quantized increments and frozen upstream activations. It therefore measures storage/arithmetic/lookup approximation, but does not include error from quantizing original continuous phase increments or changed upstream activations.

| Dataset | Rank | Maximum absolute output error | Relative output RMS error |
|---|---:|---:|---:|
| synthetic | 1 | 0.331288778 | 0.305911% |
| actual checkpoint trace | 1 | 0.0344715546 | 0.256661% |
| synthetic | 4 | 0.23610393 | 0.283865% |
| actual checkpoint trace | 4 | 0.0848789701 | 0.664103% |

These are block-output errors, not answer-accuracy percentages. Narrowing before the current readout changes the actual-trace outputs by a maximum of 0.00236988068 / 0.00207400322 (SISO/MIMO), supporting the decision to read out the wider update.

## Whole-model integration smoke

The CPU arithmetic emulator exactly reproduces every RTL-checked output and packed state. The full decoder then executes all 12 layers, 24 heads/layer, P=64 and N=128 on the GPU with the same arithmetic contract. Dense and nonlinear operations execute in Torch; no FPGA offload occurs.

| Checkpoint | Relative perplexity change | Baseline argmax agreement | Main-state bytes |
|---|---:|---:|---:|
| mamba3-siso-187m | +0.117228% | 31/32 | 4,718,592 |
| mamba3-mimo-187m | +0.070883% | 31/32 | 4,718,592 |

This is a 32-token teacher-forced integration check on a previously inspected corpus. It confirms functioning feedback and interfaces, not matched quality, long-context capability, or generated-answer correctness. The candidate combines guarded phase, FP16 feedback and explicit sequential operation ordering; the baseline is the portable chunked FP32 evaluator. Main-state storage is 4.5 MiB versus 9 MiB in FP32. Previous K/V, phase, weights, transient B/C and wide pipeline storage are additional; the prototype caches FP32 projected previous V rather than the native raw-V representation.

See [model results](model_results.json) and [integration source](torch_backend.py).

## Synthesis and timing

| Core | Fitted LEs | Registers | 9-bit multipliers | M9Ks | Worst-corner Fmax estimate | Meets 50 MHz setup? |
|---|---:|---:|---:|---:|---:|---|
| Rank 1 | 28,589 | 1,857 | 105 | 7 | 10.39 MHz | **NO** |
| Rank 4 | 51,940 | 4,641 | 168 | 16 | 4.87 MHz | **NO** |

**Both cores fit, and both fail the 50 MHz setup requirement.** Compilation PASS is not timing closure. These estimates do not show an operating FPGA or complete-model acceleration. M9Ks above count allocated physical blocks; logical memory bits in the summaries below are not the whole physical allocation.

Rank 1 worst reported registered path: `altsyncram:phase_rtl_0|altsyncram_qcc1:auto_generated|ram_block1a0~portb_address_reg0` to `ky[0]`, 95.748 ns data delay and -76.239 ns setup slack.
Rank 4 worst reported registered path: `mamba3_cache_ram:cache_bank[0].k_ram|altsyncram:ram|altsyncram_1ed1:auto_generated|ram_block1a0~portb_re_reg` to `now1[12]`, 205.096 ns data delay and -185.431 ns setup slack.

For SISO the limiting path crosses phase lookup and FP32 rotation. For MIMO it crosses the rank reduction into `now1`. Deeper arithmetic pipelines and an appropriate accumulation schedule are required; changing state placement alone did not remove those paths.

The matched SISO [register-bank control](synthesis/register_bank_baseline/README.md) used 43,253 LEs and 14,272 registers. Explicit RAM banking reduced fitted logic to 28,589 LEs (**33.9% lower**) and registers to 1,857 (**87.0% lower**), using 7 M9Ks instead of 4. Both retained FP16 main state and the same mathematical operations. The control and candidate both have a 10.39 MHz worst-corner Fmax estimate, so this is a measured resource improvement, not a measured clock or inference-speed improvement. Extra room for parallel engines is a future opportunity, not a demonstrated serving gain.

Rank 1: Quartus build status **PASS**. See [build record](synthesis/ram_banked/rank1/build_status.json) and [vendor log](synthesis/ram_banked/rank1/build.log).

```text
Analysis & Synthesis Status : Successful - Fri Oct 23 19:43:35 2026
Quartus Prime Version : 18.1.0 Build 625 09/12/2018 SJ Lite Edition
Revision Name : pipeline
Top-level Entity Name : mamba3_recurrent_tile
Family : Cyclone IV E
Total logic elements : 29,092
    Total combinational functions : 28,249
    Dedicated logic registers : 1,889
Total registers : 1889
Total pins : 1
Total virtual pins : 447
Total memory bits : 29,792
Embedded Multiplier 9-bit elements : 105
Total PLLs : 0
```


```text
Fitter Status : Successful - Fri Oct 23 19:45:26 2026
Quartus Prime Version : 18.1.0 Build 625 09/12/2018 SJ Lite Edition
Revision Name : pipeline
Top-level Entity Name : mamba3_recurrent_tile
Family : Cyclone IV E
Device : EP4CE115F29C7
Timing Models : Final
Total logic elements : 28,589 / 114,480 ( 25 % )
    Total combinational functions : 28,249 / 114,480 ( 25 % )
    Dedicated logic registers : 1,857 / 114,480 ( 2 % )
Total registers : 1857
Total pins : 1 / 529 ( < 1 % )
Total virtual pins : 447
Total memory bits : 29,792 / 3,981,312 ( < 1 % )
Embedded Multiplier 9-bit elements : 105 / 532 ( 20 % )
Total PLLs : 0 / 4 ( 0 % )
```


```text
------------------------------------------------------------
Timing Analyzer Summary
------------------------------------------------------------

Type  : Slow 1200mV 85C Model Setup 'clk'
Slack : -76.239
TNS   : -31398.404

Type  : Slow 1200mV 85C Model Hold 'clk'
Slack : 0.342
TNS   : 0.000

Type  : Slow 1200mV 85C Model Minimum Pulse Width 'clk'
Slack : 9.550
TNS   : 0.000

Type  : Slow 1200mV 0C Model Setup 'clk'
Slack : -67.793
TNS   : -27609.999

Type  : Slow 1200mV 0C Model Hold 'clk'
Slack : 0.344
TNS   : 0.000

Type  : Slow 1200mV 0C Model Minimum Pulse Width 'clk'
Slack : 9.557
TNS   : 0.000

Type  : Fast 1200mV 0C Model Setup 'clk'
Slack : -28.077
TNS   : -9539.228

Type  : Fast 1200mV 0C Model Hold 'clk'
Slack : 0.138
TNS   : 0.000

Type  : Fast 1200mV 0C Model Minimum Pulse Width 'clk'
Slack : 9.199
TNS   : 0.000

------------------------------------------------------------
```

Rank 4: Quartus build status **PASS**. See [build record](synthesis/ram_banked/rank4/build_status.json) and [vendor log](synthesis/ram_banked/rank4/build.log).

```text
Analysis & Synthesis Status : Successful - Fri Oct 23 19:47:32 2026
Quartus Prime Version : 18.1.0 Build 625 09/12/2018 SJ Lite Edition
Revision Name : pipeline
Top-level Entity Name : mamba3_recurrent_tile
Family : Cyclone IV E
Total logic elements : 53,660
    Total combinational functions : 51,089
    Dedicated logic registers : 4,769
Total registers : 4769
Total pins : 1
Total virtual pins : 447
Total memory bits : 66,656
Embedded Multiplier 9-bit elements : 168
Total PLLs : 0
```


```text
Fitter Status : Successful - Fri Oct 23 19:50:39 2026
Quartus Prime Version : 18.1.0 Build 625 09/12/2018 SJ Lite Edition
Revision Name : pipeline
Top-level Entity Name : mamba3_recurrent_tile
Family : Cyclone IV E
Device : EP4CE115F29C7
Timing Models : Final
Total logic elements : 51,940 / 114,480 ( 45 % )
    Total combinational functions : 51,089 / 114,480 ( 45 % )
    Dedicated logic registers : 4,641 / 114,480 ( 4 % )
Total registers : 4641
Total pins : 1 / 529 ( < 1 % )
Total virtual pins : 447
Total memory bits : 66,656 / 3,981,312 ( 2 % )
Embedded Multiplier 9-bit elements : 168 / 532 ( 32 % )
Total PLLs : 0 / 4 ( 0 % )
```


```text
------------------------------------------------------------
Timing Analyzer Summary
------------------------------------------------------------

Type  : Slow 1200mV 85C Model Setup 'clk'
Slack : -185.431
TNS   : -80356.467

Type  : Slow 1200mV 85C Model Hold 'clk'
Slack : 0.282
TNS   : 0.000

Type  : Slow 1200mV 85C Model Minimum Pulse Width 'clk'
Slack : 9.549
TNS   : 0.000

Type  : Slow 1200mV 0C Model Setup 'clk'
Slack : -167.153
TNS   : -70907.974

Type  : Slow 1200mV 0C Model Hold 'clk'
Slack : 0.286
TNS   : 0.000

Type  : Slow 1200mV 0C Model Minimum Pulse Width 'clk'
Slack : 9.555
TNS   : 0.000

Type  : Fast 1200mV 0C Model Setup 'clk'
Slack : -82.203
TNS   : -25870.714

Type  : Fast 1200mV 0C Model Hold 'clk'
Slack : 0.101
TNS   : 0.000

Type  : Fast 1200mV 0C Model Minimum Pulse Width 'clk'
Slack : 9.199
TNS   : 0.000

------------------------------------------------------------
```


Builds target EP4CE115F29C7 using Quartus Lite 18.1 and FAST_FIT effort, with a 50 MHz constraint and virtual I/O except the physical clock. No board was programmed. A requested 50 MHz constraint is not an achieved operating frequency. Post-fit estimates with virtual I/O are preliminary and do not demonstrate physical transfer or serving performance. See the [parsed resource/timing record](synthesis/qor.json).

For the tested P=8 geometry, unstalled simulated token cost is `P*N + 2*P*rank + 3*rank*N/2 + N + 9` cycles. It includes command, V/BC loading, rotation, state work, previous-cache copy and output drain; it excludes the initial P*N-cycle clear, physical DMA, other heads/layers and every dense/nonlinear layer. One-element-per-cycle state processing must not be presented as one model token per cycle.

## Research interpretation

The useful result is that the larger recurrence is now executable, bit-checked RTL with actual checkpoint trace coverage. FP16 state is decoded into local FP32 computation; no full-state FP32 shadow or materialized global dequantization pass is stored in the RTL. This provides a concrete base for testing whether memory savings survive arithmetic, transfer and scheduling costs.

Fusion is already present in existing Mamba-3 implementations; see the prior [dispatch/literature investigation](../../006-mamba3-architecture/fused_decode/REPORT.md). The open contribution is a measured storage/quality/complete-decode tradeoff on the proposed hardware. Current evidence does not establish novelty, faster complete decode, matched answer quality, improved concurrent serving, or reduced energy.

The next implementation gates are timing closure, real context/transport handling, shared B/C and previous-K buffers across P slices, complete decode profiling, and a frozen quality/deadline experiment. Those gates and hardware-proposal triggers are specified in [FULL_DECODER_PLAN.md](FULL_DECODER_PLAN.md).
