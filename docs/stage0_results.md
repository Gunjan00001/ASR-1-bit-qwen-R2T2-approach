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

## Kaggle gate (pending)

To be filled from `outputs/stage0/reports.csv`:
- T0.0 spike environment block and PASS/FAIL.
- Throughput probe extrapolation (full-split streaming feasibility).
- Matrix: fp16 offline/streaming WER, latency, RTF vs published.
- Locked `X`.
