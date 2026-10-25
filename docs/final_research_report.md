# Final research report — accuracy-compression recovery experiments

_2026-10-25, branch `next-experiments` (off `quant-recovery` @ `742b5d1`;
`stage-1` untouched at `bdcf58a`). Every metric below is from a committed
official-evaluation JSON; diagnostic subsets are labelled._

## 1. Executive summary

- **Baseline retained.** `aligned17b12/step1600`, ternary 1.58,
  `decoder_attn_ffn`: **15.1483 clean / 18.9773 other** on the 50-utt protocol
  (100-utt reference 15.7546 / 19.9887); packed 2.232 GB (2.10× vs bf16,
  7.60 effective bits/param).
- **Experiment A (12.5% co-training) — REJECT.** 16.525 / 18.296; clean
  +1.377, other −0.682. The lower dose did not reduce the clean regression;
  the 25% arm (16.314 / 17.386) strictly dominates it. Co-training is closed.
- **Experiment B (sensitivity screen) — COMPLETE.** `k_proj` carries 27.95% of
  the first-order task impact at 4.17% of the quantized parameters; `v` 15.33%
  at 4.17%; `ffn_in` 23.28% at 50%. Local reconstruction ranks `o`/`q` higher,
  so reconstruction alone is not a valid allocation signal.
- **Experiment C (mixed k-fp16) — REJECT.** The candidate trained stably
  (168 ternary projections, 28 k layers fp16, 1.350B trainable) but its
  official clean WER is **37.712** (CER 29.445, 5 loops, raw 48.09) vs 15.148.
  The 2 000-step ternary-only adaptation could not absorb the frozen pretrained
  k layers. Packed size 2.335 GB (2.012×, 7.952 bits/param). Its `other` split
  was stopped as non-informative after the clean rejection and is recorded as
  **NOT RUN**, not as a result.
- **No candidate is promoted.** The baseline remains the best pair; the gate
  (≤3.74 clean / ≤6.11 other) is unmet.

## 2. Baseline audit

| field | value |
|---|---|
| checkpoint | `data/checkpoints/qat_stream_aligned17b12/step1600.pt` |
| SHA-256 | `0728fe2e09f0679b3043bb7c301f4c8c7f6f86c08859c7b6cd7dc9fe78b1e345` |
| architecture | Qwen3-ASR-1.7B; text decoder 28×2048 (intermediate 6144, 16/8 heads), audio encoder 24×1024; 2.349B bf16-equivalent params |
| quantized | 196 decoder projections, 1.409B params (60.0%), ternary 1.58 per-tensor learnable scale |
| packed | 2.232 GB, 2.10× vs bf16, 7.60 effective bits/param |
| streaming | official prefix algorithm, 320 ms, greedy, 32 tokens/chunk, LSP margin 3 + truncation |
| 50-utt | **15.1483 / 18.9773** (CER 6.668 / 7.6005; loops 0 / 4) |
| 100-utt | 15.7546 / 19.9887 (CER 6.7757 / 8.1157; loops 0 / 4) |
| exact subset | `docs/eval_utterance_ids.json` (deterministic seed 0; indices, IDs, texts) |
| tests | **396 passed, 1 skipped**; ruff clean |

## 3. Experiment results

| experiment | variable | clean | other | Δ clean / other | loops c/o | verdict |
|---|---|---:|---:|---:|---|---|
| Baseline b12 | — | **15.148** | **18.977** | — | 0 / 4 | control |
| QEC λ=0.5 | output-error regularizer | 15.466 | 19.091 | +0.318 / +0.114 | 0 / 0 | REJECT |
| CT 25% | fp every 4th step | 16.314 | 17.386 | +1.165 / −1.591 | 0 / 0 | REJECT (pair) |
| CT 12.5% | fp every 8th step | 16.525 | 18.296 | +1.377 / −0.682 | 0 / 0 | **REJECT** |
| Mixed k-fp16 | 28 k fp16, 168 ternary | 37.712 | _not run_ | +22.56 / — | 5 / — | **REJECT** |

All 50-utt comparisons use the same utterances and protocol; paired bootstrap
CIs for CT12.5 are clean [−3.093, +7.582], other [−2.545, +1.040] (both include
zero). QEC and CT25 CIs also include zero. No candidate meets the both-splits
rule; the CT arms trade clean for `other`.

## 4. Sensitivity findings

Full method and ranking: [`docs/layer_sensitivity_report.md`](layer_sensitivity_report.md).

- Component task-impact shares: **k 27.95%** (4.17% of params), **ffn_in
  23.28%** (50%), o 15.84% (8.33%), v 15.33% (4.17%), ffn_out 11.05% (25%),
  q 6.54% (8.33%).
- `k` is 6.7× more sensitive per parameter than average; `ffn_in` is 0.47×.
- Local reconstruction ranks o (0.367) and q (0.332) above k (0.290); the
  task-aware screen reverses that ranking. Reconstruction is not a substitute.
- The screen cost ~1 minute on 16 streaming chunks; it is a screening tool,
  not a WER measurement.

## 5. Why the mixed candidate failed

The 28 k layers kept their **pretrained** fp16 weights while the 168 remaining
projections were initialized from the baseline's ternary shadows. The ternary
layers were then trained at lr 2.5e-6 for 2 000 steps with the k layers
frozen. That is a large initial function mismatch, not a controlled
mixed-precision QAT: the k layers were never adapted to the quantized
attention context, and the ternary layers had no capacity/budget to compensate.
The result (clean 37.7) rejects the allocation **as trained**, not the general
idea of mixed precision. A valid follow-up would require training the k layers
(or a 4-bit k) from a matched initialization, which is a different experiment.

## 6. Compression and efficiency

| model | packed bytes | vs bf16 | effective bits/param | RTF (50-utt clean) |
|---|---:|---:|---:|---:|
| baseline ternary | 2,232,271,152 | 2.105× | 7.602 | 21.96* |
| mixed k-fp16 | 2,335,031,544 | 2.012× | 7.952 | 23.75* |

\* measured under different contention; RTF is not a controlled comparison.
No packed inference runtime exists; latency/throughput/peak-memory deployment
claims are not made. The mixed candidate costs ~103 MB more and, in this
training, loses ~22.6 clean WER — a clearly bad trade.

## 7. Literature review (applicability, not assumed transfer)

| paper | method / target | applicability here | what we have tested |
|---|---|---|---|
| **GenPTQ** (Kang & Kim, Findings EMNLP 2025, `aclanthology.org/2025.findings-emnlp.566`) | mixed-precision **weight-only PTQ**, gradient-based sensitivity + continuous search; 2.5-bit average, +0.8% WER on Whisper/Conformer, 15 s search | **High for the sensitivity/allocation idea, medium for the full pipeline**: our first-order screen is the same family; Qwen3-ASR PTQ/calibration is untested and naive PTQ collapsed earlier | first-order task-aware screen (this work); no calibrated PTQ yet |
| **4-bit Conformer native QAT** (Ding et al., Interspeech 2022, `isca-archive.org/interspeech_2022/ding22c_interspeech.html`) | native 4-bit weight/activation QAT; lossless 4-bit at 7.7× vs fp32, mixed 4/8-bit at 5× | **Applicable to the QAT machinery**: 16-level weights are a materially different accuracy/compression point; our 2-bit (4-level) loss does not predict 4-bit behavior | ternary, binary, 2-bit trained; no 4-bit quantizer yet |
| **Heuristic-free KD for streaming ASR** (Yoon, AAAI 2025, `ojs.aaai.org/index.php/AAAI/article/view/34765`) | self-distillation with full-context text, no τ time shift; streaming teacher/student share the backbone | **Applicable in principle**, but our teacher is the same-architecture fp model and our KL/state-distillation attempts gave clean-only gains; the multi-modal text-teacher variant is untested | frozen-teacher output KL; hidden-state distillation; aligned targets |
| **Learning from Back Chunks** (Yang et al., Interspeech 2024, `isca-archive.org/interspeech_2024/yang24m_interspeech.html`) | look-ahead windows + future-aware self-distillation between later/earlier chunks | **Applicable conceptually**; our training already uses future text via the forced aligner, and the official streaming algorithm has no lookahead; a direct implementation would change the streaming state policy | aligned chunk targets + LSP; no back-chunk distillation |
| Smaller student models | compression by model size, not weight precision | **Not the current bottleneck**; fp16 0.6B streams at 2.54/6.27, close to the 1.7B, so size is not what limits the fp16 side | 0.6B and 1.7B baselines |

The published results are **not** evidence that any method works on Qwen3-ASR;
they motivate the next controlled test only.

## 8. Tests and reproducibility

- Full suite: **396 passed, 1 skipped**; `ruff check .` clean.
- New tests: 10 sensitivity, 7 mixed-precision, 8 co-training (plus the QEC
  tests from the previous branch).
- Baseline and all candidates have committed launchers, logs, checkpoints
  (data/ ignored by policy), evaluation JSONs and configs.
- Invalid/preserved: the first sensitivity run (share-normalization bug) is
  kept as `*_pre_fix`; the 25% CT/QEC results remain on `quant-recovery`.
- Exact 50/100 subsets: `docs/eval_utterance_ids.json`.

## 9. Final decision

| experiment | classification |
|---|---|
| A — 12.5% co-training | **REJECT** (clean worse; 25% dominates; direction closed) |
| B — sensitivity screen | **COMPLETE** (analysis, not a WER candidate) |
| C — mixed k-fp16 | **REJECT** (clean 37.7; allocation as trained is not viable) |
| Baseline | **RETAINED**; not modified |

**No promotion.** The ternary baseline remains the best measured
accuracy-compression pair. The gate is still 3.74 / 6.11.

## 10. Recommended next step (materially different)

Stop the ternary-only objective/mixed-transplant line. The evidence points to
a **different quantization granularity**, not another schedule:

1. **4-bit native QAT (16 levels) on the same b12 recipe**, pre-registered as a
   single representation change with the existing QAT stack. The 4-bit
   Conformer result and our 2-bit failure make 4-bit the smallest untested
   point on the accuracy/compression curve; expected effective size ~5.5–6
   bits/param, with a plausible WER between ternary and fp16.
2. Alternatively, **GenPTQ-style calibrated post-training quantization** on
   the fp16 1.7B (weight-only, gradient sensitivity, mixed bit allocation) if
   avoiding another QAT campaign is preferred. Our first-order screen is a
   direct precursor; the missing piece is calibration/bit allocation.

Do not rerun QEC, co-training doses, or the frozen-k mixed transplant without a
new mechanism. Any next candidate must pass the same both-splits official
evaluation and a larger paired confirmation before promotion.
