# Register-bank control implementation

This is the earlier SISO control build, under the same 50 MHz constraint,
FAST_FIT effort and virtual non-clock I/O policy as the RAM-banked implementation.
It uses inferred Verilog arrays for current B/C and previous K; Quartus mapped
those arrays into registers. The current implementation instead instantiates
the vendor synchronous RAM primitive for those banks. Both use FP16 main state
and the same FP32 arithmetic, rotation policy and recurrence equations.

The retained source snapshot matches the hashes in
[the build record](rank1/build_status.json). Its measured fitted use is
43,253 logic elements, 14,272 registers and 17,504 logical RAM bits, with
105 nine-bit multiplier elements. The constrained 50 MHz setup requirement is
not met. This control is an implementation comparison, not a novelty claim or
a complete-inference benchmark.

To reproduce, copy `source/` into a fresh directory under the repository's
ignored `.artifacts/`, copy `rank1/pipeline.qsf` and `rank1/pipeline.sdc` there,
and run Quartus Lite 18.1 `quartus_sh --flow compile pipeline` in that copy.
The source snapshot includes the exact FP32 trigonometric ROM files and QPF.
Do not run builds directly in this evidence folder.

Incomplete rank-4 control builds were stopped after the explicit RAM design
superseded them. No completed rank-4 control fit or timing claim is made.
