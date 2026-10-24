# Research decision

_2026-10-25, verified against `docs/experiment_results.csv` and the official
JSONs._

## Current baseline

**Ternary per-tensor, `aligned17b12`/step1600: 15.15 clean / 18.98 other**
(320 ms, 50-utt subsets, official prefix algorithm, 0/4 loop truncations).
Historical reference, not a full-test-set claim. fp16 1.7B anchor 2.24/4.61;
gate 3.74/6.11 remains **unmet**.

## Failed approaches (all measured on the official harness)

- FFN-only quantization (b17): 69–133 clean, 351–789 loops — pathological.
- 2-bit four-level quantizer (b20): 28.18/34.77 — much worse despite identical
  measured storage (2.000 bits/param).
- Group-64 scales (b21): 15.68/21.48 — worse on both splits.
- Lower LR (b18/b19 pilots): offline 51/86 — does not train.
- Mix 0.4 (b14): 18.01/27.73.
- Longer schedule (b15): ties clean, loses `other`.
- Hadamard rotation: 2.1 % reconstruction alone, 7.4 % combined, collapsing all
  35 real matrices to 0.5086 (the ternary noise floor). No ASR evidence; not
  trained because it fails the pre-registered screening threshold.

## Best measured candidate

**b12 (15.15/18.98).** No alternative wins both splits. b16 holds the clean
record (14.94) at a large `other` cost (21.59); b13 holds the `other` record
(17.95) at a large clean cost (21.72).

## Binary QAT

**Do not continue.** Binary was measured earlier at 18.43/25.45 (0.6B) and the
1.7B binary path has no evidence of beating ternary; with ternary itself 11
points from the gate, binary is strictly the harder problem.

## Bonsai techniques

**Do not reopen on current evidence.** Group scaling and rotation are
representation-level levers whose measured effect is ≤7.4 % reconstruction and
negative-to-neutral WER. Reopening requires a new hypothesis (documented in
`docs/bonsai_inspired_quantization_report.md` H).

## Next recommended experiment

**Streaming-state distillation** (design: `docs/streaming_state_distillation_design.md`).
Rationale: every representation and schedule axis is now measured and closed;
the untested axis is the *objective*. The current teacher signal is output-logit
KL only (T=2.0, weight 0.05), which cannot correct a repetitive streaming state
until tokens are already wrong. A per-chunk state loss is the one intervention
that targets the observed failure mode directly.
