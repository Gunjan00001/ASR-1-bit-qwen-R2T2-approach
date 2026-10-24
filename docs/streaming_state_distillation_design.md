# Streaming-state distillation — design

## Verified starting point

`src/asr1bit/train/qat.py::distillation_loss` computes `T²·KL(teacher‖student)`
over **output logits only**, masked to label positions, T=2.0, weight 0.05,
teacher frozen (`--frozen-teacher 1`). Inspection of `train/*` shows **no
hidden-state supervision anywhere**, and `--op-distill` is a different mechanism
(teacher-generated rollout data, not a loss term).

## Hypothesis

The measured failure of every recent arm is **streaming loop explosion** (CER
below WER, hundreds of truncations), while the ordinary transcription path stays
healthy. Output-logit KL only penalises a state once its next tokens are already
wrong. A per-chunk hidden-state target should penalise drift *before* it becomes
repetition.

L = L_ASR + λ_KL·L_KL + λ_state·L_state

## Implementation sketch (not implemented)

1. Hook one mid-depth decoder layer (layer 14 of 28) on both student and frozen
   teacher; capture states on the **same** aligned chunk inputs the streaming
   loss already builds (`train/aligned.py`), so temporal positions match by
   construction.
2. Mask: only positions where the aligned target is non-empty participate; chunk
   boundaries never cross a sample.
3. `L_state = MSE(LN(h_student), LN(h_teacher))` with LayerNorm to remove scale
   mismatch; shapes match (same architecture), so no projection needed.
4. `λ_state` default **0.0**, so the existing path is untouched.
5. Teacher forward under `no_grad`, cached per batch (it already runs for KL).

## Risks

- **Future leakage**: the aligned builder already enforces causal chunk targets;
  the state loss must use the *same* slices, not full-utterance states.
- **Gradient scale**: state MSE can dominate; requires warmup of λ_state.
- **Teacher drift**: teacher is the fp16 pretrained model, so its states are a
  legitimate reference the student never sees at inference.

## Tests required before training

temporal alignment (same slice indices teacher/student), chunk-mask correctness,
padding invariance, no-future-information assertion, gradient flow only through
the student, teacher parameter freezing, λ=0 equals the baseline loss exactly.

## Acceptance criteria

A controlled pair (A: existing recipe, B: +λ_state) at the b12 recipe, same
seed/init/budget. B wins only if official clean **and** other improve, or clean
holds and loops drop materially at equal WER. Screening threshold: any single
official state beating 15.15/18.98.

## Blocker

Implementation + tests are not yet written; the GPU queue was occupied by the
matrix above. This is the next unit of work, not a completed experiment.
