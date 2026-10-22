# Research ideas

Ideas are hypotheses, not measured outcomes or certified unpublished contributions.

| ID | Idea | Status | Closest evidence / next gate |
|---|---|---|---|
| [I001](I001-refinement-reuse.md) | Reuse precision refinements across long-context requests | Proposed; novelty unconfirmed | Establish a strong packed-cache baseline in E004, then quantify oracle headroom beyond fixed compression and ordinary LRU |
| [I002](I002-memory-aware-fpga-dataflow.md) | Streaming FPGA dataflow for compressed model state | Proposed; architecture study | Profile whole-model traffic and compare with FlightLLM / SAW-INT4 / FastMamba before selecting an RTL block |
| [I003](I003-updatable-recurrent-memory.md) | Useful, updatable recurrent memory under a storage budget | Revisit only with a stronger mechanism | E002 did not establish a learned-method advantage over ordinary caches |

Start new ideas with [TEMPLATE.md](TEMPLATE.md), then link an experiment with a predeclared decision rule. Avoid treating “lower precision,” “adaptive bits,” “query awareness,” or “FPGA implementation” alone as novelty.
