# Quantization recovery research — reducing ternary damage in streaming ASR

_2026-10-25, branch `quant-recovery` (off `stage-1` @ `bdcf58a`). This document
audits the committed evidence, states the hypotheses, and selects the first
controlled experiment. Results are in
[`docs/quantization_recovery_experiment.md`](quantization_recovery_experiment.md)._

## 1. Audit of the committed evidence

### 1.1 The control

| field | value |
|---|---|
| checkpoint | `data/checkpoints/qat_stream_aligned17b12/step1600.pt` |
| recipe | Qwen3-ASR-1.7B; ternary 1.58; `decoder_attn_ffn` (196 projections); per-tensor learnable log-scale; fp32 shadows + STE |
| training | init `qat_17b_recover7/last.pt` (offline 9.85); 6 000 aligned utterances; mix 0.5; 2 000 steps; lr 2.5e-6 cosine; fp32 AdamW; frozen fp teacher with KL 0.05; 0.32 s chunks |
| **50-utt clean / other** | **15.148 / 18.977** (CER 6.67 / 7.60; loops 0 / 4) |
| **100-utt clean / other** | **15.755 / 19.989** (CER 6.78 / 8.12; loops 0 / 4) |
| packed size | 2.23 GB, 2.10× vs bf16, 7.60 effective bits/param, 60.0% of parameters ternary |

All streaming numbers use the official Qwen3-ASR prefix algorithm, 320 ms
chunks, greedy, 32 tokens/chunk, LSP margin 3 + loop truncation. The 50- and
100-utterance sets are fixed diagnostic subsets, **not full LibriSpeech
splits**; the full-split cost was measured at ~110 GPU-hours per split.

### 1.2 Closed experiments (official evaluations only)

| approach | result | conclusion |
|---|---|---|
| binary `{-1,+1}` | 0.6B best 18.43/25.45 (one seed-sensitive state); reruns 26.80/35.23, 36.44/43.18; no competitive 1.7B result | worse than ternary; not continued |
| group-64 scales | 15.68 / 21.48 at step 2000 | worse on both splits |
| 2-bit `{-3,-1,1,3}` | 28.18 / 34.77 | identical measured storage to ternary, much worse WER |
| FFN-only, attention fp16 | 69.28 / 107.27 best state | pathological |
| Hadamard rotation | 2.1% reconstruction alone, 7.4% with g64; collapsed 35 matrices to the ternary noise floor; never trained (screening threshold) | no ASR evidence |
| stochastic rounding | first controlled A/B: 24.15 vs 20.76 at step 500 (worse) | harmful at the 1.7B recovery |
| streaming-state distillation | best arm F 14.62/18.18 at 50 utts, but 100-utt other ties (20.05 vs 19.99) | clean-only gain; no pair win |

**Key fact:** weight-space reconstruction does not predict WER. g64 improved
reconstruction by 5.2% and lost 0.53/2.50 WER; the 2-bit quantizer had lower
reconstruction error and lost 13 WER points. The damage that matters is the
**functional** difference on real activations, not the weight error.

### 1.3 Discrepancy note

The `stage-1` docs (`experiment_audit.md`, `research_decision.md`) predate the
streaming-state-distillation phase. The authoritative records for that phase
are on branch `state-distill` @ `22c0d53`
(`docs/final_experiment_report.md`, `docs/final_research_decision.md`). The
baseline values above are identical on both branches. No experiment is assumed
to have run because a launcher exists; every number above has a committed JSON.

## 2. Hypotheses for reducing ternary damage

All candidates keep ternary weights at inference unless stated otherwise.

### H1 — Output-aware quantization-error compensation (QEC)

**Mechanism.** Add a training-only per-layer regularizer that matches the
ternary layer's response to the fp shadow's response on the actual
(activation-quantized) input:

`L_qec = ||Q(W)x − W x||² / (||W x||² + eps)`.

The fp target is detached; the gradient flows only through the quantized path.
At inference the term disappears and the deployed model is unchanged.

**Evidence.** The project's own failures show weight-space error is the wrong
proxy; QEC optimizes the function on the streaming distribution. It is the
training-side analogue of activation-aware quantization error compensation and
is compatible with the existing frozen-teacher KL.

**Benefits / risks.** Directly targets the measured damage; no new parameters;
no inference cost. Risk: a second objective can compete with CE/KL; the
relative error can be dominated by a few outlier activations (normalization
mitigates); compute roughly doubles the linear work per BitLinear during
training.

**Code changes.** `train/qec.py` (controller + loss), trainer flags
`--qec-weight/-stride/-eps`, tests. Already implemented.

**GPU.** Same model/data as the control; ~20–40% slower than the baseline
training run, plus the standard evaluations.

**Prediction.** At λ=0.5, both clean and other improve vs 15.15/18.98 at the
primary state, and the QEC term falls during training.

**Rejection condition.** Either split is worse at the primary state, the term
destabilizes training (non-finite, max grad > 1000, loss >10× baseline), or a
100-utt confirmation fails to reproduce a both-splits improvement.

### H2 — Soft/annealed ternary quantizer

**Mechanism.** Replace the hard `round` with a temperature-controlled soft
quantizer during training (`tanh(W/τ)` or a straight-through sigmoid),
annealing to the hard ternary grid at the end. Inference is hard ternary.

**Evidence.** LSQ/soft-quantization literature; the project's α-blend ramp was
unsafe, but a soft quantizer changes the gradient, not the forward blend.

**Benefits / risks.** Better-conditioned gradients; risks introducing a
train/eval mismatch and another schedule. Code: a soft quantizer + temperature
schedule + tests.

**GPU.** Same as the baseline.

**Prediction.** Lower training loss variance and better WER than hard-STE at
the same budget. Reject if no both-splits gain.

**Status.** Second-choice alternative; it changes the quantizer's training
behavior and needs more schedule design.

### H3 — Precision co-training schedule

**Mechanism.** Alternate full-precision (α=0) and ternary (α=1) steps, e.g.
25% fp steps, so the fp path is periodically refreshed while the ternary path
is optimized. Inference remains ternary.

**Evidence.** The plan lists co-training as an untested lever; the fp readout
drifts during QAT, and the project's binary floor is partly a sign/rounding
bias.

**Benefits / risks.** Simple, same inference model; risk: fp steps may not
improve the ternary function and can waste budget; the schedule is a new
variable.

**GPU.** Same as the baseline.

**Prediction.** A 25% fp schedule improves both splits; reject if it merely
tracks the control or destabilizes the quantized readout.

**Status.** Third choice; the schedule has several sub-choices (fraction,
placement), so it is less clean as a first experiment than QEC.

### H4 — Per-output-channel scales / selective precision with retraining

**Mechanism.** One scale per output channel (rather than per tensor/group), or
keep the most sensitive projections fp16 and retrain.

**Evidence.** Group scales failed; per-channel is a different granularity, but
the reconstruction-to-WER disconnect warns against expecting a gain. Selective
precision reduces compression and was confounded in the shadow-substitution
screen (the screen measured precision-switch mismatch, not sensitivity).

**Prediction / rejection.** A retrained selective arm would need to beat both
splits while reporting the actual non-ternary fraction; any "win" that only
works by keeping large parts in fp16 is not a ternary result.

**Status.** Not first; requires a representation change and a retraining arm.

### H5 — Stochastic rounding

**Mechanism.** Unbiased rounding during training.

**Evidence.** Already measured at the 1.7B: 24.15 vs 20.76 at step 500 (worse).
**Closed** unless a new mechanism (e.g. variance-aware scheduling) is proposed.

## 3. Selection

**First controlled experiment: H1 (QEC), λ=0.5.**

Why: (a) it is training-side and leaves the ternary representation, packed
size and streaming algorithm untouched; (b) it directly optimizes the
functional error that the project's failures implicate; (c) it is a single
variable on the exact b12 recipe; (d) it is cheap enough to run and evaluate
within the available GPU budget. H2/H3 remain documented alternatives if H1
fails.

## 4. Compute and evaluation policy

- Pilot: 100 steps to detect numerical instability/collapse.
- Full: 2 000 steps, identical to b12 except `--qec-weight`.
- Primary state: step 1600 (the control's best); step 2000 as the late check.
- Official evaluation: 50 utts clean/other first; if both splits improve, a
  100-utt paired confirmation. Paired bootstrap CIs are reported; no promotion
  from a guardrail or a single split.
- Full splits are not runnable (~110 GPU-hours/split); this is stated wherever
  a number is quoted.
