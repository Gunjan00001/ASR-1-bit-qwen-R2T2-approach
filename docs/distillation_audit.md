# Teacher-anchor audit (Phase 6, first task)

_2026-10-23. Read-only inspection of the existing distillation path._

## What exists today

| element | implementation | notes |
|---|---|---|
| objective | `src/asr1bit/train/qat.py::distillation_loss` | `T^2 · KL(teacher ‖ student)` over **output logits**, temperature 2.0, masked to label positions (transcription tokens only, prompt excluded) |
| teacher | `--frozen-teacher 1` → a frozen fp copy of the pretrained model, loaded once in `run_stage1_qat.py` (params `requires_grad_(False)`, `.eval()`) | same tokenizer/vocab as the student |
| weight | `--distill-weight` (0.05 on the 1.7B recipes) | requires `--frozen-teacher 1` |
| rollout option | `--op-distill` uses the teacher to *drive* the rollout and the student clones continuations | different mechanism: data generation, not a loss term |

## What is missing

1. **No intermediate-state supervision.** Nothing matches hidden states,
   attention outputs, or per-chunk representations. The objective is purely
   behavioural (next-token distribution), so a student that has drifted into a
   repetitive streaming state receives no signal until the tokens are wrong.
2. **No streaming-state targets.** The KL is computed on whole-utterance label
   positions; it never sees the LSP/prefix states the model will actually
   inhabit at inference.
3. **No dimensionality handling** — because there is no state loss yet. Any
   extension needs a projection when teacher/student widths differ (they do not
   here: same architecture, so a direct MSE on selected layers is possible).

## Proposed minimal extension (not yet implemented)

`L_total = L_stream + λ_teacher · L_teacher + λ_state · L_state`, with
`L_state` a per-chunk MSE between student and frozen-teacher hidden states at
one mid-depth decoder layer, computed on the *same* input chunks the streaming
loss uses, so the two terms are aligned in time.

Ordering per the mission: implement behind a flag (`--state-distill-weight`),
unit-test the loss and the gradient path, pilot at small scale, and only then
queue a full run — after the higher-priority evidence (b17's corrected
evaluation, selective precision) is in.
