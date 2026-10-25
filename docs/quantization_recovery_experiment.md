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
