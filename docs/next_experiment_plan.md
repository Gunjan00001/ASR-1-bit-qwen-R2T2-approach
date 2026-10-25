# Next experiments — plan and pre-registration

_2026-10-25, branch `next-experiments` (off `quant-recovery` @ `742b5d1`;
`stage-1` remains `bdcf58a`). This plan was written before the 12.5%
co-training and mixed-precision results were observed. The sensitivity screen
(Experiment B) was run first because Experiment C depends on it._

## 1. Authoritative baseline

| field | value |
|---|---|
| checkpoint | `data/checkpoints/qat_stream_aligned17b12/step1600.pt` |
| checkpoint SHA-256 | `0728fe2e09f0679b3043bb7c301f4c8c7f6f86c08859c7b6cd7dc9fe78b1e345` |
| training launcher SHA-256 | `b288470f99c12f78ced18d16b7b987e4ba8a7d8a7b7257e16f31de7a8a65ba79` (`scripts/qat_stream_aligned17b12.pbs`) |
| results-JSON SHA-256 | `8a1c08599bc19b94311a25a847bd2feb9304ecd17cfd47ddee5af2a3f9544d45` |
| architecture | Qwen3-ASR-1.7B: text decoder 28 layers, hidden 2048, intermediate 6144, 16 heads / 8 KV heads, vocab 151 936; audio encoder 24 layers, d_model 1024, output 2048; 2.349B bf16-equivalent params total |
| quantization | ternary 1.58, `decoder_attn_ffn`, per-tensor learnable log-scale; 196 projections, **1.409B params = 60.0%** of the model |
| packed size | **2.232 GB**, 2.10× vs bf16, **7.60 effective bits/param** |
| evaluation | official Qwen3 prefix algorithm, 320 ms, greedy, 32 tokens/chunk, LSP margin 3 + loop truncation; 50-utt **15.1483 / 18.9773** (CER 6.668 / 7.6005; loops 0 / 4); 100-utt **15.7546 / 19.9887** |
| exact subset | `docs/eval_utterance_ids.json`: deterministic `sample_n` seed 0; 50-utt and 100-utt indices, LibriSpeech utterance IDs and texts |
| tests | **379 passed, 1 skipped**; `ruff check .` clean |

The 50/100-utterance sets are diagnostic subsets, not full LibriSpeech. Full
split cost remains ~110 GPU-hours/split at the measured streaming RTF.

## 2. Experiment A — 12.5% full-precision co-training

**Hypothesis.** The 25% fp schedule improved `other` (17.386) but degraded
`clean` (16.314 vs 15.148). A lower 12.5% dose may retain part of the `other`
gain while shrinking the clean regression.

**Single variable.** `--cotrain-fp-every 8`: after the baseline's 2% alpha
delay, every 8th step runs at `alpha=0`; the other 87.5% run at ternary
`alpha=1`. Everything else is the b12 recipe (init, data, 2 000 steps, lr
2.5e-6 cosine, fp32 AdamW, frozen teacher KL 0.05, 6 000 aligned, mix 0.5).

**Actual schedule accounting.** The trainer logs `cotrain_fp` on every
5-step record; the pilot's counts confirm the 1-in-8 pattern. The full run's
log records the same and the final results JSON records the config.

**Pilot safety.** 100 steps, no in-loop evals. Abort on non-finite loss, max
grad >1 000 for two consecutive logs, loss >10× the control for 50 steps, or
OOM.

**Evaluation.** Official prefix algorithm, 50 utts clean/other at step 1600
(primary) and step 2000 (secondary if the primary is within 2 points or
better). If both splits improve, a 100-utt paired confirmation.

**Acceptance.** Clean **and** other below 15.1483 / 18.9773; loops not
increased; stable training; paired bootstrap CI reported. **Reject** if either
split is worse at the primary state or the pilot is unsafe. Do not promote a
single-split winner.

## 3. Experiment B — task-aware layer sensitivity

**Method.** First-order loss impact of each ternary projection's output error
on real streaming chunk batches:

`impact_l = mean_tokens | grad_l · (Q(W_l)x_l − W_l x_l) |`

The gradient is taken to the layer output; the error is the ternary-vs-fp
output on the same activation-quantized input. This is a **screening** metric,
not WER. It is preferable to the invalid shadow-substitution screen (which
measured precision-switch mismatch) and to local reconstruction alone.

**Calibration.** 8 utterances × up to 16 aligned chunk examples at 320 ms from
the cached 8 000-utterance aligned set; baseline checkpoint; read-only.

**Staging.** Cheap per-layer screen → component ranking → one mixed-precision
candidate → full training + official WER. No hundreds of full evaluations.

## 4. Experiment C — sensitivity-guided mixed precision

**Candidate (fixed after the screen, before training).** Keep **all 28
`k_proj` layers at full precision** and all other decoder projections ternary.
Rationale: `k_proj` has the largest component task-impact share (28.0%) with
only 58.7M params (4.2% of the quantized parameters); keeping it fp16 adds
~103 MB to the packed model, a small compression cost. The alternative
`ffn_in` is 23.3% of impact but 56 layers / ~704M params, far more expensive.

**Allocation.** 28 × (1024 × 2048) = **58,720,256 fp16 params**; the other 168
projections (1.350B params) stay ternary; all non-decoder weights stay bf16.
Expected packed size ≈ **2.335 GB**, ≈2.01× vs bf16, ≈7.95 effective
bits/param (measured after training, not assumed).

**Training.** Initialize from the baseline `step1600` checkpoint: the 168
ternary shadows are restored; the 28 k layers keep their pretrained fp16
weights and are frozen (QAT-only protocol). Same recipe/data/teacher/steps as
b12. The mixed model is the only representation change.

**Evaluation.** Official protocol, 50 utts clean/other at step 1600 and 2000;
100-utt confirmation if both splits improve; packed size measured; RTF/peak
memory recorded where available.

**Acceptance.** Both splits below 15.1483 / 18.9773; loops not increased;
stable; compression overhead reported; paired uncertainty. **Reject** if
either split is worse or loops rise. Do not promote on a reconstruction metric.

## 5. Stopping rules

- Experiment A: if 12.5% fails the both-splits rule, close the co-training
  direction (no further dose sweep) unless the result shows a new,
  well-defined mechanism.
- Experiment B: if the impact distribution is flat and no component
  concentrates, do not build an elaborate mixed system; report the screen and
  stop.
- Experiment C: one candidate only. If it fails, do not launch a broad
  mixed-precision search; report the negative result and propose a materially
  different approach.
- `stage-1` is never modified; every run is isolated on `next-experiments`.
