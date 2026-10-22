# I003 — Useful and updatable memory under a strict storage budget

Status: earlier synthetic investigation completed; a competitive learned mechanism was not established.

Question: which information should a recurrent system retain, update, and delete when request state has a fixed byte budget?

[E002](../experiments/002-budgeted-memory/README.md) compared explicit caches, delta-memory controls, hybrid variants, and trained MLP-address encoders. Learned models improved on their own initialization but did not beat the strongest ordinary cache controls across the studied conditions. That is evidence against those tested mechanisms, not against every recurrent-memory approach.

Reopen this idea only with a task that requires meaningful generalization beyond explicit lookup and a mechanism with a plausible advantage over a well-tuned cache. Charge shared parameters and per-request state separately. Establish calibration/test separation and matched-byte controls before another training sweep.
