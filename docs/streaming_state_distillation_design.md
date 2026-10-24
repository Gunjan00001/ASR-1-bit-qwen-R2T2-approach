# Streaming-state distillation — design (implemented, pre-registered)

_Updated 2026-10-25. Status: implementation + tests COMPLETE; GPU experiments
pre-registered below. This document is the pre-registration: the seven arms,
the single variable per arm, the budget, and the acceptance rule are fixed
before any result is observed._

## Verified starting point (code inspection, not assumption)

`src/asr1bit/train/qat.py::distillation_loss` computes `T²·KL(teacher‖student)`
over **output logits only**, masked to label positions, T=2.0, weight 0.05,
teacher frozen (`--frozen-teacher 1`). Before this phase there was **no
intermediate-state supervision anywhere** in `src/asr1bit/train/*`: the only
other distillation mechanism, `--op-distill`, generates teacher *rollout data*
(`train/examples.py::build_distill_examples`) and is not a loss term.

What the existing path already provides, and the new term reuses:

| element | implementation | use in this experiment |
|---|---|---|
| streaming chunk samples | `train/aligned.py` → `data/prefix.py` | each aligned sample is one chunk state: audio truncated at `(k+1)*chunk`, stable prefix text, target = words heard in chunk `k` |
| forced alignment | `data/align.py` (Qwen3-ForcedAligner-0.6B) | unchanged; words cached in `data/aligned_cache/` |
| prefix/stop policy | `qwen/stream.py` (`LSPState`) | evaluation only; unchanged |
| QAT student | `bitlinear.py` + `replace.py` | unchanged; the state loss reads the student's quantized decoder |
| frozen teacher | `run_stage1_qat.py` `--frozen-teacher 1` | same full-precision architecture, frozen, `eval()` |
| official evaluation | `scripts/eval_streaming.py` / `eval_streaming_17b.pbs` | unchanged; guardrails are never promotion metrics |

## Hypothesis

Output-logit KL can only penalise a drifting streaming state once the next
tokens are already wrong. A per-chunk hidden-state target at a mid-depth
decoder layer penalises the drift earlier, while the model is still in the
correct chunk state, and should reduce loop explosions and the
re-align/under-repair phase.

## Loss

    L = (1 - w_KL)·CE + w_KL·KL(teacher‖student) + λ_state·L_state

    L_state = mean over valid positions of
              || LayerNorm(h_student) - LayerNorm(h_teacher) ||²

- **Same layer** for student and teacher (`--state-distill-layer`, -1 = middle,
  14 of 28 for the 1.7B), captured with a forward hook on
  `model.thinker.model.layers[i]`.
- **Same batch**: each aligned row is already a causal chunk state (truncated
  audio + stable prefix), so teacher and student see identical inputs and no
  future audio/text is available.
- **Valid positions**: `labels != -100` (supervised target tokens only),
  `attention_mask != 0` (no padding), and `state_mask == 1` (the row is a
  streaming chunk state; ordinary full-utterance samples are excluded, so the
  loss can never see future audio).
- **LayerNorm, no affine**: removes the fp16/quantized scale mismatch; same
  architecture, so no projection is needed.
- **Teacher**: frozen, forward under `no_grad`, states detached; no teacher
  gradient ever flows.
- **Disabled by default**: weight 0 returns the existing loss tensor unchanged
  (`combine_losses` returns `ce` itself); the compatibility test pins this.

Implementation: `src/asr1bit/train/state_distill.py`
(`streaming_state_loss`, `HiddenStateCapture`, `decoder_layer`,
`resolve_layer`, `combine_losses`); trainer flags in
`scripts/run_stage1_qat.py`; row tags in `train/{aligned,dataset,examples}.py`;
`state_mask` built by `train/dataset.py::collate`.

## Required tests (all implemented in `tests/test_state_distill.py`)

| requirement | test |
|---|---|
| temporal alignment | student/teacher resolve to the same layer; capture returns the layer output |
| chunk masks | `state_mask` marks only streaming rows; non-streaming rows cannot change the loss |
| prefix/future isolation | aligned samples carry truncated audio + stable prefix (existing `test_data_prefix.py`); the loss only sees the row's own chunk |
| padding | padded positions excluded (label/attention mask invariance) |
| loss masking | prompt and padding changes leave the loss unchanged; identical states give 0 |
| gradient propagation | gradient reaches student parameters only |
| teacher freezing | teacher parameters have no gradient; frozen teacher path |
| compatibility | weight 0 returns the CE tensor itself; CLI defaults off |

`pytest -q`: 377 passed, 1 skipped; `ruff check .`: clean (2026-10-25).

## Pre-registered experiment matrix (seven arms)

Fixed for every arm: 1.7B, `decoder_attn_ffn` ternary 1.58, init
`data/checkpoints/qat_17b_recover7/last.pt` (offline 9.85), 6 000 aligned
utterances (cached), mix 0.5, batch 1, fp32 AdamW, lr 2.5e-6 cosine, warmup
100, 2 000 steps, frozen teacher, KL weight 0.05. Only the variable named in
the row changes. State checkpoints every 400 steps; official evaluation of
steps 1 200 / 1 600 / 2 000 on clean and other (50-utt protocol), LSP loop
policy on.

| arm | variable vs control | status |
|---|---|---|
| **A — control** | existing recipe, λ_state = 0 | **verified**: b12/step1600 = **15.15 / 18.98** (0 / 4 loops) |
| **B** | λ_state = **0.05**, layer 14, LN-MSE | queued |
| **C** | λ_state = **0.01**, layer 14, LN-MSE | pre-registered (runs after B) |
| **D** | λ_state = **0.20**, layer 14, LN-MSE | pre-registered |
| **E** | λ_state = 0.05, **layer 8** (early) | pre-registered |
| **F** | λ_state = 0.05, **layer 20** (late) | pre-registered |
| **G** | λ_state = 0.05, layer 14, **no LayerNorm** | pre-registered |

Arm A is the existing verified run, not a re-run: the training path is
deterministic (b7/b9 reproduced to the digit), the flags are byte-identical,
and re-running would reproduce it at the cost of a GPU-day. Every arm's flags
are diffed against `qat_stream_aligned17b12.pbs` before submission.

## Acceptance criteria (fixed before results)

1. **Primary**: an arm improves only if an official state beats **15.15 clean
   AND 18.98 other**. A single-split record is reported as such, not promoted.
2. **Gate**: clean ≤ 3.74 and other ≤ 6.11 (fp16 1.7B + 1.5). Reported
   separately; the project does not redefine it.
3. **Screening**: a state at or below 15.15/18.98 with **fewer loop
   truncations** is a stronger result than the same WER with more.
4. **Stopping**: if B is catastrophically worse (e.g. clean > 30 with loops)
   and the loss terms are finite, C/D/E/F/G are not launched; the failure is
   diagnosed first (implementation vs alignment vs optimisation vs
   hypothesis). If B is within ~2 points of control, the weight sweep
   (C/D) runs before the layer/normalisation ablations.

## Risks and controls

- **Future leakage**: prevented by construction — aligned audio is truncated at
  the chunk boundary, and `state_mask` excludes full-utterance rows. Tested.
- **Misaligned representations**: both sides use the same layer index and the
  same batch; there is no pooling or shifting.
- **Gradient scale**: reported per step (`state` in the training JSON); a
  state term that dominates shows up immediately in the loss components.
- **Teacher recomputation**: the teacher already runs for KL; the hook adds no
  extra forward. When only the state term is on, one teacher forward is added.
- **Memory**: one hidden-state tensor per model per step (batch × seq × 2048);
  no checkpointing change.
- **Precision-switch mismatch**: the state loss never substitutes fp16 shadows
  for a trained quantized model; it compares two forward passes under their
  own precision policies.

## Deliverables

- `docs/streaming_state_distillation_results.md` — controlled results.
- `docs/final_experiment_report.md` / `docs/final_experiment_results.csv` —
  Phase-8 record.
- New code and tests committed on the `state-distill` branch; `stage-1` is not
  modified without review.
