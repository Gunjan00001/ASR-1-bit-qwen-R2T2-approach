# b17 diagnostic report — FFN-only quantization (attention fp16)

_2026-10-23. Job 12342, `--policy decoder_ffn`, everything else on the b12
recipe (deep 9.85 base, lr 2.5e-6, 6 000 aligned from cache, mix 0.5,
2 000 steps). All numbers below are from
`artifacts/stage3/qat_stream_aligned17b17.log` and the live job record._

## 1. What is actually broken

| measurement | step 250 | step 500 | interpretation |
|---|---:|---:|---|
| streaming guardrail WER | 1 349.65 | 902.12 | catastrophically high |
| streaming guardrail CER | 784.05 | 455.81 | **CER < WER** |
| offline ternary readout (`EVAL`) | — | **19.39** | normal-ish |
| offline fp16 readout (`EVAL`) | — | 21.19 | shadow readable |

**CER < WER is the decisive shape.** When CER is *below* WER, the hypotheses are
not merely wrong — they are **long and repetitive**: a few correct characters
repeated many times inflate the word-level errors far more than the
character-level ones. This is the classic loop explosion, not a decode that
produced garbage characters.

**Offline is fine.** The step-500 offline ternary readout is 19.39, i.e. the
model still transcribes normally when it sees the whole utterance. The failure
is **specific to the streaming path**.

## 2. What is not broken

| check | evidence | verdict |
|---|---|---|
| numeric stability | no NaN/Inf (the only `nan` match in 2 800 lines is the word "information" in a transformers notice) | clean |
| gradient explosion | max-grad-norm median **8.06**, max **224.0**, min 1.03 (n=121) vs b12's median 3.82, max 352.8 (n=401) | **no explosion**; b17's median is ~2× b12's but its maximum is *lower* |
| loss divergence | CE swings 0.37 → 3.00 → 1.56, KL 2.1–4.3, both bounded and recoverable | spiky but healthy |
| quantizer scales | gamma 0.0191 / 0.0237 / 0.0334 (min/mean/max), stable across steps 531–556 | no scale collapse |
| LR schedule | 2.17e-6 at step 556, decaying on the cosine | as configured |
| policy applied | run config records `"policy": "decoder_ffn"` | confirmed |
| init | `initialized shadows from data/checkpoints/qat_17b_recover7/last.pt (checkpoint step 4000)`, no missing-key warnings | confirmed |

The attention has **no shadow parameters under this policy**, which is correct —
there is nothing to quantize. Its weights are fp16 and trainable, as intended.

## 3. Classification

**Streaming-state instability confined to the unquantized-attention
configuration.** Not numeric failure, not optimization divergence, and not
(established) quantization damage — the ordinary transcription path is intact
and the optimizer is behaving.

**Most likely mechanism (hypothesis, to be tested):** the learning rate was
calibrated for a *fully quantized* model, where BitLinear's weight
quantization regularizes updates and damps the rollout's effect on the
streaming head. With fp16 attention the same LR produces ~2× the median gradient
norm, and the LSP rollout feeds back unquantized attention states, so the
self-conditioning loop amplifies a small drift into repetition. That predicts
the two lower-LR pilots (5e-7, 1e-7) restore stability.

**Alternative hypothesis (also testable):** the fp16 attention is simply the
wrong place to spend precision for streaming, and the damage is in the FFN.
Phase 4's projection ablations distinguish these.

## 4. Actions taken

1. **Run continues** — no numerical, resource, or checkpoint corruption found;
   the job is at step ~560 of 2 000 with 10.5 h of walltime left.
2. Checkpoints and logs preserved as-is.
3. Phase 3 pilots (`b18` lr 5e-7, `b19` lr 1e-7) prepared from the same init with
   the same policy, differing only in LR.
4. Phase 2's official sweep (1 200 / 1 600 / 2 000) is queued to run after the
   arm ends, never before: the guardrail will not select the state.
