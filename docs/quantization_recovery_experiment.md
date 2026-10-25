# QEC experiment — pre-registered plan and results

_2026-10-25, branch `quant-recovery` (off `stage-1` @ `bdcf58a`). The plan
(Sections 1–7) was written **before any QEC result was observed**; the results
section is appended only from committed JSONs._

## 1. Hypothesis

Adding a training-only output-aware quantization-error compensation (QEC)
term to the b12 recipe will reduce ternary quantization damage and improve
**both** clean and other streaming WER at the same ternary inference cost.

`L = (1−w_KL)·CE + w_KL·KL + λ_qec·L_qec`,
`L_qec = mean_layers ||Q(W)x − Wx||² / (||Wx||² + eps)` with the fp target
detached.

**Rejection condition:** either split is worse at the primary state; training
violates the numerical safety criteria; or a 100-utt confirmation does not
reproduce a both-splits improvement.

## 2. Control

`aligned17b12/step1600` (ternary per-tensor, `decoder_attn_ffn`):
**15.1483 clean / 18.9773 other** on 50 utts; **15.7546 / 19.9887** on 100
utts. Artifacts:
`streaming_qat_stream_aligned17b12_step1600_{clean,other}.json`,
`streaming_b12_step1600_n100_{clean,other}.json`.

## 3. Configuration (single variable: `--qec-weight`)

| field | value |
|---|---|
| model | Qwen/Qwen3-ASR-1.7B |
| policy / width / scale | `decoder_attn_ffn` / 1.58 / per-tensor learnable log-scale |
| init | `data/checkpoints/qat_17b_recover7/last.pt` (offline 9.85) |
| data | `qat_clean_train.100_n8000.pt`; 6 000 aligned (cached), stop-stride 2; mix 0.5; train-n 2 400; op-max-samples 2 400 |
| steps / batch | 2 000 / 1 |
| optimizer / lr | fp32 AdamW / 2.5e-6 cosine, 100 warmup |
| alpha | delay, 2% fp warmup |
| teacher | frozen fp copy, KL 0.05, T=2.0 |
| streaming eval (guardrail) | 20 utts every 250 steps (not a promotion metric) |
| **experimental variable** | **`--qec-weight 0.5`**, stride 1, eps 1e-6 |
| checkpoints | every 400 steps + `last.pt`; `save-every 250` |
| primary state | step 1600; secondary step 2000 |

The launcher is `scripts/qat_qec_b12.pbs`; the control's launcher is
`scripts/qat_stream_aligned17b12.pbs`. The only flag difference is the three
QEC flags (plus the save/log names).

## 4. Evaluation protocol

Official Qwen3-ASR prefix algorithm, 320 ms chunks, greedy, 32 tokens/chunk,
LSP margin 3 + loop truncation, 50 fixed utterances of `test-clean` and
`test-other` (seed 0). If both splits improve, a 100-utterance paired
confirmation is run with the same `sample_n=100, seed=0` for control and
candidate. WER/CER, loop truncations, raw-vs-policy WER, RTF and wall time are
recorded; paired bootstrap CIs (10 000 resamples) are computed.

## 5. Numerical / resource safety (pilot)

The pilot runs 100 steps with `--eval-n 2`, no stream/offline evals, and the
same QEC settings. Abort if any of:

- non-finite loss, CE, KL or QEC term;
- max pre-clip grad norm > 1 000 (control max ≈ 353) for two consecutive logs;
- loss > 10× the control's running mean for 50 consecutive steps;
- CUDA OOM or host OOM.

If the pilot passes, the full 2 000-step run is launched.

## 6. GPU budget and checkpoints

- Pilot: ~10 min (model load + 100 steps).
- Full training: ~2 h at the observed 1.7B rate (QEC adds one fp linear per
  BitLinear per forward).
- Evaluations: step 1600 clean+other ~0.5–4 h depending on output length;
  step 2000 similarly; 100-utt confirmation ~2–8 h per system.
- Checkpoints: `data/checkpoints/qat_qec_b12/step{400,800,1200,1600,2000}.pt`
  plus `last.pt`; the baseline checkpoint is never overwritten.

## 7. Criteria (fixed)

- **Improvement:** step-1600 official clean **and** other both below
  15.1483 / 18.9773; a 100-utt confirmation (if triggered) both below
  15.7546 / 19.9887.
- **Not promoted:** a clean-only gain, a single-split gain, a guardrail gain,
  or a difference whose paired CI is centred on zero without a larger
  confirmation.
- **Gate:** clean ≤ 3.74 and other ≤ 6.11; reported separately, never
  redefined.

## 8. Results

_Appended after the run; nothing in this section is a pre-registration._

### 8.1 Pilot (job 12705)

100 steps, no in-loop evals, QEC λ=0.5. **Passed** the safety criteria:
QEC loss **0.2948** at step 6 and **0.2948** at step 100, finite throughout;
max pre-clip grad 69.6 (control max ≈353); no NaN/Inf/OOM. The pilot's
`qat_seconds` was 62 s for 100 steps.

### 8.2 Full run (job 12706)

2 000 steps completed; all step checkpoints on disk. The QEC term was
**essentially flat**:

| steps | mean QEC |
|---|---:|
| 0–250 | 0.2955 |
| 250–500 | 0.2929 |
| 500–750 | 0.2920 |

Training was stable (no non-finite values; max grad 200.5, in-family with the
control's 352.8). The flat term means the shadow weights could not reduce the
per-layer output error further; the regularizer provided a nearly constant
gradient constraint rather than a shrinking objective.

### 8.3 Official evaluation (50 utts, official prefix algorithm, 320 ms)

| state | clean | other | CER c/o | loops c/o | RTF c/o | raw c/o |
|---|---:|---:|---:|---:|---:|---:|
| **control b12/step1600** | **15.148** | **18.977** | 6.668 / 7.601 | 0 / 4 | 21.96 / 21.73* | 15.042 / 18.977 |
| **QEC step1600** | 15.466 | 19.091 | 7.022 / 7.578 | 0 / 0 | 3.07 / 2.70 | 15.466 / 18.750 |
| QEC step2000 | 73.941 | 18.409 | 57.317 / 7.888 | 22 / 1 | 4.68 / 4.10 | 73.941 / 18.068 |

\* the control's RTF was measured under heavy contention; RTF is not a
controlled comparison. The QEC step-2000 clean state collapsed (22 loop
truncations); step 1600 is its best state.

**Deltas vs control:** step 1600 **+0.318 clean / +0.114 other** (both worse);
step 2000 +58.79 / −0.57 (collapse on clean). Paired bootstrap (10 000
resamples, same references): clean **+0.318 [−1.986, +2.515]**, other
**+0.114 [−1.247, +1.332]**; both intervals include zero, and the
utterance-level counts are 11/9/30 and 8/7/35 (better/worse/ties). No 100-utt
confirmation was triggered because the primary state did not improve both
splits.

### 8.4 Verdict

**Reject H1 (QEC).** The implementation is correct — the term was active,
finite and gradient-carrying — but it did not reduce the functional error or
the WER. The primary state is worse on both splits, and the late state
collapses on clean. A higher λ is not justified: the term itself did not
decrease, so increasing its weight would add a constant constraint without a
new mechanism.

### 8.5 Why it failed (hypothesis-level, not implementation)

- The target `Wx` is the shadow's own fp response; the ternary grid is already
  the nearest representable function, so the output error sits at a local
  minimum (0.29) that the regularizer cannot shrink.
- The CE/KL objectives already move the shadow toward task-useful regions; the
  QEC gradient is largely orthogonal and can over-constrain that movement,
  which is consistent with the small clean/other regression.
- This is another case where a *weight/response-space* objective fails to
  transfer: the project's reconstruction-to-WER disconnect is confirmed at the
  layer-output level.

### 8.6 Compression and runtime

The QEC checkpoint has exactly the baseline layout (196 ternary projections,
2-bit slots + fp16 per-tensor scales): **2 232 271 152 bytes (2.23 GB)**,
**2.10× vs bf16**, **7.60 effective bits/param**, 60.0% of parameters ternary.
QEC adds no inference parameters. The measured streaming RTF was 3.07 (clean)
and 2.70 (other) on the evaluation GPU; the control's 21.96/21.73 was measured
under heavier contention, so RTF is not a controlled comparison and no
speedup is claimed. There is no packed inference runtime; latency, throughput
and memory for a packed deployment were not measured.

Artifact: `artifacts/stage1/packed_size_qec_step1600.json`.

### 8.7 Next justified step (not run here)

A **quantized-vs-fp self-consistency objective on the task output** (match the
quantized student's logits to its own fp logits) is a different mechanism from
layer-output QEC and should be pre-registered separately. Do not repeat QEC at
a higher λ without a new hypothesis.

## 9. Experiment 2 — precision co-training (pre-registered)

_Written before any CT result was observed._

**Single variable:** `--cotrain-fp-every 4` — after the baseline's 2% alpha
delay, every 4th step (1-indexed) runs the forward/backward at `alpha=0`; the
other steps use the scheduled ternary alpha. The deployed model remains
ternary; no quantizer, streaming algorithm or data change.

| field | value |
|---|---|
| control | b12/step1600: 15.148 / 18.977 (50 utts) |
| model / policy | Qwen3-ASR-1.7B / `decoder_attn_ffn`, ternary 1.58, per-tensor learnable scale |
| init | `data/checkpoints/qat_17b_recover7/last.pt` |
| data / mix | 6 000 aligned (cached), stop-stride 2, mix 0.5, train-n 2 400 |
| steps / batch | 2 000 / 1 |
| optimizer / lr | fp32 AdamW / 2.5e-6 cosine, 100 warmup |
| alpha | delay 2%; CT overrides every 4th step after the delay |
| teacher | frozen fp copy, KL 0.05, T=2.0 |
| checkpoints | every 400 + `last.pt` |
| primary state | step 1600; secondary step 2000 |
| launcher | `scripts/qat_cotrain_b12.pbs`; pilot `scripts/qat_cotrain_pilot.pbs` |

**Pilot safety:** 100 steps, no in-loop evals; abort on non-finite loss, max
grad >1 000 for two consecutive logs, loss >10× the control for 50 steps, or
OOM.

**Acceptance:** step-1600 official clean **and** other both below
15.1483 / 18.9773; if both improve, a 100-utt paired confirmation (same
`sample_n=100, seed=0`) must also improve both. **Reject** if either split is
worse at the primary state or the pilot violates safety. No promotion from a
guardrail or a single split.

**Budget:** pilot ~10 min; full run ~2 h; step-1600 evaluations ~0.5–4 h.

