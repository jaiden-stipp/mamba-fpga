"""Render the implementation report from recorded machine-readable evidence."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent

def main():
    r=json.loads((HERE/'evidence/summary.json').read_text())
    m=json.loads((HERE/'model_results.json').read_text())
    assert r['status']==m['status']=='PASS'
    lines=['# E001 recurrent pipeline implementation — 2026-10-23','',
           'The phase experiment now has a functioning recurrent-tile RTL implementation and an executable whole-model arithmetic integration boundary. This is a substantially larger hardware block than the original phase rotator. Complete-model FPGA inference and an inference acceleration claim remain unestablished.','',
           '## What was implemented','',
           '- Guarded phase accumulation and lookup, with FP32 B/C coordinates and the existing 16/8/8 phase/trigonometric policy.',
           '- Resident FP16 main-state RAM, widening on read, the original current-plus-previous-input recurrence, FP32 readout before storage rounding, FP16 writeback, and the D skip term.',
           '- Persistent phase and previous K/V caches, token sequencing, paused input streams, stable output under backpressure, and reset/clear behavior.',
           '- Portable finite-width FP32 multiplication/addition and FP16 conversion, with explicit separate-operation rounding. Arithmetic is implemented in synthesizable integer logic, not simulation-only real variables.',
           '- A replaceable recurrence backend inside complete SISO/MIMO checkpoint decoders. The supplied backend is a Torch arithmetic emulator, not FPGA transport.','',
           'See the [interface and block diagram](README.md), [RTL](rtl/mamba3_recurrent_tile.sv), and [full-decoder gates](FULL_DECODER_PLAN.md).','',
           '## Verification','',
           f"Seed: `0x{r['seed']:08x}`. ModelSim primitive regression: **{r['arithmetic_vectors']:,} input pairs / {r['arithmetic_result_comparisons']:,} checked results**, including all 65,536 FP16 bit patterns. The four checks per pair are FP32 add, FP32 multiply, FP32-to-FP16, and FP16-to-FP32. NaN policies are explicit in the source; this is not an exhaustive certification of all FP32 inputs.",'',
           '| RTL case | Tokens | Exact state writes | Exact outputs | Max bit-reference error | Unstalled cycles/token |',
           '|---|---:|---:|---:|---:|---:|']
    for c in r['cases']:
        lines.append(f"| Rank {c['rank']}, P={c['P']}, N={c['N']} | {c['tokens']} | {c['state_writes']:,} | {c['outputs']:,} | 0 | {c['token_cycles_without_input_bubbles_or_output_stalls'][0]:,} |")
    lines+=['','Each case contains 32 deterministic synthetic tokens followed by a reset and 32 actual checkpoint-input tokens. Actual traces select layer 0, head 0, P channels 0–7, full N=128, from WikiText-2 validation offset 32,768. The traces are frozen upstream inputs; sampled RTL replay alone is not a full-model quality evaluation. Exact stimulus, reference, simulator output and checkpoint hashes are retained.','',
            'The testbench also aborts an in-flight update and resets the core, inserts input pauses, and stalls output consumption. Every token has 1,024 consecutive state writes: one coordinate per cycle in the compute segment. Deliberately corrupting one expected state code yields exactly one reported error per rank, demonstrating that the scoreboard rejects a bad result. Python independently rechecks every exported result; the runner requires PASS markers and correct counts rather than trusting simulator exit status alone.','',
            'A second clean simulator build/run reproduced all 24 retained evidence files byte-for-byte; see [repeatability hashes](repeatability.json). Read the [PASS/FAIL log](evidence/PASS_FAIL.log), [summary](evidence/summary.json), [SISO simulation](evidence/rank1/simulation.log) and [MIMO simulation](evidence/rank4/simulation.log).','',
            '## Error against higher precision','',
            'Bit equality above is against the quantized, explicitly ordered reference. The separate higher-precision comparison uses FP64 state feedback, FP64 recurrence/readout and true trigonometry of the accumulated 16-bit phase. It retains the same supplied quantized increments and frozen upstream activations. It therefore measures storage/arithmetic/lookup approximation, but does not include error from quantizing original continuous phase increments or changed upstream activations.','',
            '| Dataset | Rank | Maximum absolute output error | Relative output RMS error |',
            '|---|---:|---:|---:|']
    for c in r['cases']:
        for p in c['precision_metrics']:
            lines.append(f"| {p['dataset'].replace('_',' ')} | {c['rank']} | {p['output_max_abs_error_vs_float64']:.9g} | {100*p['output_relative_rms_vs_float64']:.6f}% |")
    lines+=['','These are block-output errors, not answer-accuracy percentages. Narrowing before the current readout changes the actual-trace outputs by a maximum of '+
            ' / '.join(f"{c['precision_metrics'][1]['early_narrowing_max_output_change']:.9g}" for c in r['cases'])+
            ' (SISO/MIMO), supporting the decision to read out the wider update.','',
            '## Whole-model integration smoke','',
            'The CPU arithmetic emulator exactly reproduces every RTL-checked output and packed state. The full decoder then executes all 12 layers, 24 heads/layer, P=64 and N=128 on the GPU with the same arithmetic contract. Dense and nonlinear operations execute in Torch; no FPGA offload occurs.','',
            '| Checkpoint | Relative perplexity change | Baseline argmax agreement | Main-state bytes |',
            '|---|---:|---:|---:|']
    for model in m['models']:
        lines.append(f"| {model['checkpoint']} | {100*model['relative_perplexity_change']:+.6f}% | {model['argmax_agreement_tokens']}/32 | {model['persistent_main_state_bytes']:,} |")
    lines+=['','This is a 32-token teacher-forced integration check on a previously inspected corpus. It confirms functioning feedback and interfaces, not matched quality, long-context capability, or generated-answer correctness. The candidate combines guarded phase, FP16 feedback and explicit sequential operation ordering; the baseline is the portable chunked FP32 evaluator. Main-state storage is 4.5 MiB versus 9 MiB in FP32. Previous K/V, phase, weights, transient B/C and wide pipeline storage are additional; the prototype caches FP32 projected previous V rather than the native raw-V representation.','',
            'See [model results](model_results.json) and [integration source](torch_backend.py).','',
            '## Synthesis and timing','']
    qor=json.loads((HERE/'synthesis/qor.json').read_text())
    lines+=['| Core | Fitted LEs | Registers | 9-bit multipliers | M9Ks | Worst-corner Fmax estimate | Meets 50 MHz setup? |',
            '|---|---:|---:|---:|---:|---:|---|']
    for c in qor['cases']:
        lines.append(f"| Rank {c['rank']} | {c['fit_logic_elements']:,} | {c['fit_registers']:,} | {c['fit_multiplier_9bit_elements']} | {c['M9K_blocks']} | {c['minimum_restricted_fmax_MHz']:.2f} MHz | {'YES' if c['meets_50MHz_internal_setup'] else '**NO**'} |")
    lines+=['','**Both cores fit, and both fail the 50 MHz setup requirement.** Compilation PASS is not timing closure. These estimates do not show an operating FPGA or complete-model acceleration. M9Ks above count allocated physical blocks; logical memory bits in the summaries below are not the whole physical allocation.','']
    for c in qor['cases']:
        path=c['worst_reported_registered_path']
        lines.append(f"Rank {c['rank']} worst reported registered path: `{path['from']}` to `{path['to']}`, {path['data_delay_ns']:.3f} ns data delay and {path['slack_ns']:.3f} ns setup slack.")
    lines+=['','For SISO the limiting path crosses phase lookup and FP32 rotation. For MIMO it crosses the rank reduction into `now1`. Deeper arithmetic pipelines and an appropriate accumulation schedule are required; changing state placement alone did not remove those paths.','',
            'The matched SISO [register-bank control](synthesis/register_bank_baseline/README.md) used 43,253 LEs and 14,272 registers. Explicit RAM banking reduced fitted logic to 28,589 LEs (**33.9% lower**) and registers to 1,857 (**87.0% lower**), using 7 M9Ks instead of 4. Both retained FP16 main state and the same mathematical operations. The control and candidate both have a 10.39 MHz worst-corner Fmax estimate, so this is a measured resource improvement, not a measured clock or inference-speed improvement. Extra room for parallel engines is a future opportunity, not a demonstrated serving gain.','']
    for rank in (1,4):
        folder=HERE/'synthesis/ram_banked'/f'rank{rank}'
        status=json.loads((folder/'build_status.json').read_text()) if (folder/'build_status.json').is_file() else {'status':'NOT_COMPLETE'}
        lines.append(f"Rank {rank}: Quartus build status **{status['status']}**. See [build record](synthesis/ram_banked/rank{rank}/build_status.json) and [vendor log](synthesis/ram_banked/rank{rank}/build.log).")
        for filename in ('pipeline.map.summary','pipeline.fit.summary','pipeline.sta.summary'):
            p=folder/filename
            if p.is_file():lines+=['','```text',p.read_text(errors='replace').strip(),'```','']
    lines+=['','Builds target EP4CE115F29C7 using Quartus Lite 18.1 and FAST_FIT effort, with a 50 MHz constraint and virtual I/O except the physical clock. No board was programmed. A requested 50 MHz constraint is not an achieved operating frequency. Post-fit estimates with virtual I/O are preliminary and do not demonstrate physical transfer or serving performance. See the [parsed resource/timing record](synthesis/qor.json).','',
            'For the tested P=8 geometry, unstalled simulated token cost is `P*N + 2*P*rank + 3*rank*N/2 + N + 9` cycles. It includes command, V/BC loading, rotation, state work, previous-cache copy and output drain; it excludes the initial P*N-cycle clear, physical DMA, other heads/layers and every dense/nonlinear layer. One-element-per-cycle state processing must not be presented as one model token per cycle.','',
            '## Research interpretation','',
            'The useful result is that the larger recurrence is now executable, bit-checked RTL with actual checkpoint trace coverage. FP16 state is decoded into local FP32 computation; no full-state FP32 shadow or materialized global dequantization pass is stored in the RTL. This provides a concrete base for testing whether memory savings survive arithmetic, transfer and scheduling costs.','',
            'Fusion is already present in existing Mamba-3 implementations; see the prior [dispatch/literature investigation](../../006-mamba3-architecture/fused_decode/REPORT.md). The open contribution is a measured storage/quality/complete-decode tradeoff on the proposed hardware. Current evidence does not establish novelty, faster complete decode, matched answer quality, improved concurrent serving, or reduced energy.','',
            'The next implementation gates are timing closure, real context/transport handling, shared B/C and previous-K buffers across P slices, complete decode profiling, and a frozen quality/deadline experiment. Those gates and hardware-proposal triggers are specified in [FULL_DECODER_PLAN.md](FULL_DECODER_PLAN.md).','']
    (HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf8',newline='\n')

if __name__=='__main__':main()
