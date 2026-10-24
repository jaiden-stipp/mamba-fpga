# Timing loop stopping point: arithmetic-unit pipelines

Five additional matched rank-1 iterations progressed from 13.04 to 21.77 MHz
worst-corner Fmax estimate, with worst setup slack improving from -56.670 to
-25.936 ns at the unchanged 50 MHz target. This is a result for the current
routed virtual-I/O implementation, not a universal frequency limit or an FPGA
operation/inference-speed measurement. See the [complete table](TIMING_LOOP.md).

## Why another external register is insufficient here

The final global worst path is **`trig_c[1]` → `rot_kyc[29]`**, 45.886 ns,
37 logic levels. Both ends are already registers; between them is one call to
the portable FP32 multiplication function. The matched [full path](state_widen_stage/timing/operator-rotation-multiply.rpt)
reproduces the global slack, endpoints and delay. Registering its input/output
again does not divide the multiplier's internal normalization/product/rounding
logic. The next relevant cut must be inside the arithmetic unit.

Other selected paths in this same routed candidate also exceed the 20 ns target:

| Datapath | Data delay | Setup slack |
|---|---:|---:|
| One rotation multiply | 45.886 ns | -25.936 ns |
| One readout multiply | 45.131 ns | -25.165 ns |
| One final state addition | 42.185 ns | -22.271 ns |
| Accumulator-bank addition | 42.702 ns | -22.781 ns |

The [operator ledger](state_widen_stage/timing/operator_floor.json) binds these
reports to the compiled source. The accumulator's worst selected path crosses
banks and includes mux/routing; it is a conservative static path rather than
proof that those two banks interact dynamically. The preceding read-stage build
also retained a same-bank accumulator path, `accum[0][0][28]` →
`accum[0][0][24]`, 43.434 ns: [report](state_read_stage/timing/operator-accumulator-feedback.rpt).
These observations motivate a different implementation, not relaxing constraints.

## What the compressed-state iteration established

The fourth iteration exposed FP16 widening plus multiplication as the worst
path. The fifth retimed that exact widening operation:

```text
before: FP16 RAM -> FP16 read register -> widen -> FP32 multiply
after:  FP16 RAM -> widen -> FP32 read register -> FP32 multiply
```

Persistent main state remains 16-bit: 2 KiB for this P=8/N=128 tile. Other
caches and pipeline buffers are additional. The declared transient state buffer
grows by 16 bits; synthesis adds only three net fitted registers in this build
(2,876 → 2,879), with 105 nine-bit multiplier elements unchanged. This is not a
claim that all implementations will have the same physical register delta.
The new conversion segment takes 9.987 ns, and the selected old-state multiply
segment takes 42.486 ns. No extra token cycle is added. All tested state/output
bits, FP64-error metrics and persistent-state format remain unchanged.

## Next implementation contract

The external-register loop stops here because progressing toward 50 MHz now
requires internally pipelined FP32 multiply/add units and their latency-aware
scheduling, as anticipated in [the full-decoder plan](FULL_DECODER_PLAN.md).
This work is not implemented or claimed by the retained timing-loop variants.

1. Split multiplication into registered internal stages, carrying sign,
   exponent, special-value classification, mantissa product, rounding data,
   coordinate/pair IDs and valid. Preserve the original RNE/subnormal/NaN/
   signed-zero behavior and separate multiplication/addition contract.
2. Validate each new unit against the independent primitive golden data before
   integrating it into rotation, state products and readout. Keep every latency
   change explicit in reset/drain/backpressure and address checks.
3. Pipeline addition without changing summation association. The accumulator is
   a feedback dependency: a multi-cycle adder cannot consume the same channel's
   next N coordinate before its previous result returns.
4. Interleave independent P channels with N outer/P inner, provided P is at
   least the adder latency, or explicitly stall issue. Interleaving can retain
   each channel's original increasing-N sum order. A changed issue order needs
   exact golden checks by coordinate address, complete unique-write coverage,
   and per-channel ordering assertions; it must not loosen numerical equality.
5. Repeat the same regression and routed timing loop at the arithmetic-unit
   level. Count the changed buffer resources and cycle latency; do not infer
   whole-model acceleration from frequency alone.

All five completed iterations retain their sources, raw reports, seed-locked
PASS/FAIL evidence and separate commits. No candidate numerical mismatch occurred.
Two rejected timing-selector attempts are retained in their respective iteration
directories; corrected extractions use actual routed names/connected endpoint
sets. They were audit failures in path selection, not numerical failures.
