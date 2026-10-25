# Layer-wise quantization sensitivity — screening report

_2026-10-25, branch `next-experiments`. Baseline: `qat_stream_aligned17b12/step1600.pt` (SHA-256 `0728fe2e…b1e345`). The screen is a first-order task-loss
proxy, **not** a WER measurement; its only purpose is to justify one
mixed-precision candidate for training and official evaluation._

## 1. Method

For every BlockLinear (BitLinear) projection, on real streaming chunk batches:

    impact_l = mean_tokens | grad_l · (Q(W_l) x_l − W_l x_l) |

- `x_l` is the activation-quantized input the layer actually received;
- `Q(W_l)x_l` is the ternary output; `W_l x_l` is the fp shadow output;
- `grad_l` is the gradient of the training loss (teacher-forced CE) with
  respect to that layer's output;
- the absolute dot product is the first-order loss change if the layer's
  quantization error were removed, averaged over tokens and batches.

The analyzer is read-only: forward hooks only, no parameter gradients, hooks
removed on close. It is implemented in
`src/asr1bit/analysis/sensitivity.py` and tested in
`tests/test_sensitivity.py` (10 tests).

**Calibration set.** 8 training utterances × up to 16 aligned 320 ms chunk
examples from the cached 8 000-utterance aligned set (the first 8 utterances
hit the cache; no aligner run). Source: `artifacts/stage1/layer_sensitivity.json`.

**Cost.** The scoring run itself took ~1 minute on an A100 (start 16:46:27 CET,
artifacts written 16:47:13 CET; the PBS job waited for a slot first). This is
the cheap screening stage; no WER was run for all 196 layers.

**Provenance note.** The first screen (job 12738) wrote per-layer shares with a
normalization bug (dividing by the raw sum instead of the per-batch mean); the
ranking was unchanged because every layer had the same batch count. That
artifact is preserved as `*_pre_fix`, and the authoritative fixed run is job
12740 (`artifacts/stage1/layer_sensitivity.json`).

## 2. Component ranking

| component | layers | params | param share | task-impact share | impact per param-share | local relative error |
|---|---:|---:|---:|---:|---:|---:|
| **k** | 28 | 58,720,256 | 4.17% | **27.95%** | **6.71×** | 0.290 |
| ffn_in (gate+up) | 56 | 704,643,072 | 50.0% | 23.28% | 0.47× | 0.250 |
| o | 28 | 117,440,512 | 8.33% | 15.84% | 1.90× | 0.367 |
| v | 28 | 58,720,256 | 4.17% | 15.33% | 3.68× | 0.294 |
| ffn_out (down) | 28 | 352,321,536 | 25.0% | 11.05% | 0.44× | 0.292 |
| q | 28 | 117,440,512 | 8.33% | 6.54% | 0.78× | 0.332 |

**The local-reconstruction ranking is not the task ranking.** `o` and `q` have
the largest local errors (0.367 / 0.332) but only 15.8% / 6.5% of the task
impact; `k` has a lower local error (0.290) but the largest task share
(28.0%). This is the same disconnect seen with g64/2-bit reconstruction and is
why the screen uses gradients, not weight/response error alone.

## 3. Individual layer ranking (top 15, per-layer share of total impact)

| rank | layer | component | share | local error |
|---:|---|---|---:|---:|
| 1 | `layers.1.mlp.gate_proj` | ffn_in | 3.59% | 0.228 |
| 2 | `layers.0.self_attn.k_proj` | k | 2.68% | 0.442 |
| 3 | `layers.0.self_attn.o_proj` | o | 2.17% | 0.450 |
| 4 | `layers.0.self_attn.v_proj` | v | 2.08% | 0.305 |
| 5 | `layers.19.self_attn.k_proj` | k | 1.95% | 0.298 |
| 6 | `layers.16.self_attn.k_proj` | k | 1.85% | 0.324 |
| 7 | `layers.17.self_attn.k_proj` | k | 1.84% | 0.329 |
| 8 | `layers.18.self_attn.k_proj` | k | 1.70% | 0.276 |
| 9 | `layers.20.self_attn.k_proj` | k | 1.64% | 0.274 |
| 10 | `layers.21.self_attn.k_proj` | k | 1.57% | 0.288 |
| 11 | `layers.1.self_attn.k_proj` | k | 1.45% | 0.315 |
| 12 | `layers.25.self_attn.k_proj` | k | 1.40% | 0.327 |
| 13 | `layers.22.self_attn.k_proj` | k | 1.29% | 0.308 |
| 14 | `layers.2.mlp.gate_proj` | ffn_in | 1.27% | 0.228 |
| 15 | `layers.2.mlp.down_proj` | ffn_out | 1.26% | 0.281 |

`k` is both the largest component and the only component with a dense
mid/late-layer cluster (layers 16–25 plus layer 0). The per-layer impact is
otherwise diffuse: no single layer exceeds 3.6%.

## 4. Precision recommendation

**Keep all 28 `k_proj` layers at fp16; keep the other 168 decoder projections
ternary.** Justification:

- `k` has the largest task-impact share (28.0%) at only 4.2% of the quantized
  parameters; per parameter it is 6.7× more sensitive than average.
- The cost is small: 58.72M fp16 params instead of ternary adds ~103 MB to the
  packed model; the predicted size is **2.335 GB (2.01× vs bf16, ~7.95
  effective bits/param)**, measured after training.
- Keeping `ffn_in` fp16 would cover 23.3% of the impact but 50% of the
  quantized params — a far worse accuracy/compression trade.
- `q` is the least sensitive component (6.5%) despite a high local error, so
  it stays ternary.

This is a **hypothesis for Experiment C**, not a promoted result. The candidate
is trained from the baseline checkpoint with the 168 ternary shadows restored
and the 28 k layers frozen at their pretrained fp16 weights; it is evaluated
with the official streaming protocol against the unchanged baseline.

## 5. Limitations

- First-order linearization on 16 chunk examples; it ranks local loss
  sensitivity, not WER. A layer with low score can still matter through
  non-linear interactions.
- One calibration distribution (LibriSpeech clean training chunks); `other` may
  rank differently.
- The gradient is teacher-forced; streaming rollout states are not sampled.
- No confidence interval on the ranking; the component-level gap (k 28.0% vs
  ffn_in 23.3%) is not large, and the choice is additionally justified by the
  parameter cost.
- The candidate's actual WER and packed size are the deciding measurements.
