# I002 — FPGA dataflow that consumes compressed state directly

Status: proposed; block selection and novelty unconfirmed.

Objective: turn compressed representations into meaningful latency or energy improvements for an actual model workload.

Candidate mechanism: unpack and scale only the tile needed by a streaming arithmetic pipeline, keep reusable intermediates on chip, and overlap external-memory transfers with computation. Select the target block only after profiling its fraction of total runtime and traffic.

Review FlightLLM, KIVI, SAW-INT4, FastMamba, and LowRank-SSM. Their existing compression, fusion, dataflow, and memory-system techniques are baselines, not contributions to claim anew. [Reading path](../literature/README.md).

Milestones: traffic/latency model → bit-accurate software reference → RTL simulation with stalls and boundary cases → synthesis/resource/timing evaluation → board trace replay → full-system evaluation including transfer and host overhead. A standalone block throughput number cannot establish model acceleration.

The DE2-115 can validate a bounded RTL/dataflow prototype. Request a newer FPGA/HBM platform if the selected hypothesis needs capacity or bandwidth that this board lacks; do not let the board dictate an unimportant question. [System scope](../architecture/system-scope.md).
