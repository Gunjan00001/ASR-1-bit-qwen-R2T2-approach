# Stage 0 results

Gate: reproduce published streaming WER within tolerance; lock `X`. Awaiting the
Kaggle T4 run (T0.0 spike + throughput probe + matrix).

Published reference (LibriSpeech **clean | other**):
- offline: 0.6B `2.11 | 4.55`, 1.7B `1.63 | 3.38`
- streaming (2.0 s): 0.6B `2.54 | 6.27`, 1.7B `1.95 | 4.51`

## Local CPU normalizer validation (transformers backend, offline)

Isolated venv: torch `2.14.0+cpu`, transformers `4.57.6`, `qwen_asr` from PyPI.
Fixed test-clean subsets (`sample_n`, seed 0). Portable transformers backend,
`language=None`, greedy. Scores via `asr1bit.eval.wer.corpus_wer`.

| Model | n | normalizer | WER % | vs published offline |
|---|---|---|---|---|
| Qwen3-ASR-0.6B | 100 | librispeech | 3.26 | +1.15 |
| Qwen3-ASR-0.6B | 100 | **librispeech_hyph** | **2.93** | **+0.82** |
| Qwen3-ASR-0.6B | 300 | librispeech | 3.31 | +1.20 |
| Qwen3-ASR-0.6B | 300 | **librispeech_hyph** | **2.93** | **+0.82** |
| Qwen3-ASR-1.7B | 100 | librispeech | 1.93 | +0.30 |
| Qwen3-ASR-1.7B | 100 | **librispeech_hyph** | **1.71** | **+0.08** |

- Hyphen-splitting (``well-known`` -> ``well known``) beats plain punctuation
  removal consistently and is now ``asr1bit.text.DEFAULT_MODE``
  (``librispeech_hyph``).
- 1.7B is within **+0.08** absolute of the published offline number on 100
  utts — loader, decoding, and normalizer are validated.
- 0.6B is noisier and its subset differs from the published full-split mix; the
  Kaggle full-split run is the arbiter.

CER (librispeech mode): 0.6B 0.75-0.81, 1.7B 0.56.

`ChunkedStreamer` plumbing smoke (portable path, same utterances) produced
non-empty text for all probed samples; RTF on CPU was ~1.7-2.0 (expected, CPU
only, no target).

## Stage 0 gate results

**Gate: PASS.** Published streaming WER reproduced within tolerance.

Environment (both hosts matched): **Tesla T4, sm75, fp16**, python 3.12.13
(Kaggle) / 3.13.15 (Colab), **torch 2.9.1+cu128, transformers 4.57.6,
vllm 0.14.0**, CUDA 12.8. Pinned protocol: greedy, `language=None`,
`max_new_tokens` 1024 offline / 32 streaming, `max_model_len=16384`,
`gpu_memory_utilization=0.85`, `VLLM_ATTENTION_BACKEND=TRITON_ATTN`
(FlashInfer JIT fails on sm75). See `artifacts/stage0/environment_gate.txt`.

**T0.0 spike: PASS** (vLLM streaming on T4; 15 s sample: 2.0 s chunk RTF 0.078,
0.32 s chunk RTF 0.291).

Results (`artifacts/stage0/gate_results.csv`; subsampled rows use a fixed
500-utterance set, seed 0):

| Model | Mode | Chunk | Config | n | sub | WER | CER | RTF | retro lat | Published | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-ASR-1.7B | streaming | 2.0 s | clean | 2620 | full | **1.96** | 0.61 | 0.199 | 3.47 | 1.95 (full) | **+0.01** |
| Qwen3-ASR-1.7B | streaming | 2.0 s | other | 500 | sub | **3.47** | 1.25 | 0.239 | 3.62 | 4.51 (full) | −1.04 |
| Qwen3-ASR-1.7B | streaming | 320 ms | other | 500 | sub | **4.61** | 2.64 | 0.728 | 5.26 | — | — |
| Qwen3-ASR-1.7B | offline | — | clean | 500 | sub | **2.10** | 0.63 | 0.0 | — | 1.63 (full) | +0.47 |
| Qwen3-ASR-1.7B | offline | — | other | 500 | sub | **3.47** | 1.25 | 0.0 | — | 3.38 (full) | +0.09 |
| Qwen3-ASR-1.7B | streaming | 320 ms | clean | 500 | sub | **2.24** | 0.67 | 0.547 | 4.20 | — | — |

The full-split streaming clean number matches published almost exactly (+0.01);
offline/test-other subsamples are within small deltas. Subsampled rows are
labelled and are not directly comparable to the published full-split numbers.

Deferred (opportunistic, not required): full-split offline; 0.6B confirmation.

### Primary metric baselines (fp16) and locked `X`

Primary metric = streaming **320 ms**, clean/other. fp16 baselines measured here:

- streaming 320 ms **clean = 2.24** WER (500-utt subsample)
- streaming 320 ms **other = 4.61** WER (500-utt subsample)

**`X = +1.5` absolute**: the Stage 1+ quantized model must stay within +1.5 WER
of these fp16 baselines at the primary metric. Realism note: naive 1-bit QAT is
reported at ~+52% relative WER, so hitting +1.5 absolute depends on the Stage 2
recovery stack (on-policy distillation, co-training, learnable scales); the
ternary fallback remains an acceptable ship per the plan.
