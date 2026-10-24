# Streaming-state distillation — results

**Status: NOT STARTED.** No code, no run, no number. The design is in
`docs/streaming_state_distillation_design.md`.

Why not started: the phase order in the brief places it after closing the
quantization matrix, which only completed at 2026-10-24 23:09 (g64 step 2000).
Writing an untested loss against the trainer and launching it without the
alignment/gradient tests the brief itself requires would have produced an
uninterpretable run.

What is required to complete it:
1. Implement the hook + masked MSE behind `--state-distill-weight` (default 0).
2. Add the seven tests listed in the design doc.
3. Launch the controlled pair at the b12 recipe (2×~6 h GPU).
4. Evaluate both with the official harness (steps 1200/1600/2000).

No claim is made about its effect on WER.
