# E005 — Does a low-bit read actually fetch fewer bytes?

Status: executable analytical screening model. It does **not** measure DRAM, execute attention, establish answer quality or implement an FPGA accelerator.

Read [the protocol](PROTOCOL.md), run `python experiments/005-read-granularity/screen.py`, then inspect [the report](report.md), [CSV](results.csv) and [machine-readable summary](summary.json). No external dependencies or model downloads are required. Future model-trace and RTL stages are in [P001](../planned/P001-transaction-aware-reader.md).

The model counts the minimum distinct aligned fetch units touched by explicit byte addresses for the same requested prefix schedule under different eight-bit progressive layouts. It exposes transaction waste that an average-bit count hides. ReadKV already distinguishes logical and physical traffic; this is a design-screening tool, not a claim to have discovered that distinction.
