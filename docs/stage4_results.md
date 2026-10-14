# Stage 4 — streaming results (official Qwen3-ASR prefix algorithm)

**Protocol.** `Qwen3-ASR-0.6B` read at bit-width 1.0 (unless stated), *official*
prefix algorithm (not a teacher-forced approximation), 50 utterances of
`test-clean` (`config clean`) or `test-other` (`config other`), greedy decode
with repetition guard `unfixed_token_num=5` and a 32-token cap per chunk, LSP
loop-emission policy on (`margin_words=3`, `truncate_loops=1`), chunk sizes
160 / 320 / 1 000 ms.  WER/CER are lower-is-better; `truncations` counts loops
cut by the LSP policy; `revisions` counts committed-text replacements; `WER raw`
is the policy-free number (measured on the same generations before the LSP
filter — the honest "what the model actually emitted" figure).

## Headline: the promoted recipe

| state | chunk | clean | other |
|---|---|---|---|
| **ternary, aligned targets (T4.1/T4.2) + LSP** | 160 ms | **13.24** | 20.80 |
| **ternary, aligned targets + LSP** | 320 ms | **12.92** | **20.00** |
| **ternary, aligned targets + LSP** | 1000 ms | **10.59** | **16.59** |
| binary, same recipe (best measured) | 320 ms | 18.43 | 25.45 |
| binary, same recipe (final weights, resumed) | 320 ms | 27.54 | 26.93 |

Zero repetition loops at every promoted **ternary** cell: 0 truncations, and
`WER raw ≈ WER` in the table below — the loop filter had almost nothing to cut,
so the remaining error is content, not collapse.  (The `revisions` column counts
pre-commit text corrections, i.e. the append-only policy stabilizing what it has
already emitted; it is not a loop metric.)  The binary arm's 320 ms cells carry
9 / 40 truncations, where truncation costs a little WER (26.91 raw vs 27.54
policy) — its loop discipline rests on the policy rather than on the weights.

## Full trajectory (every quantized streaming evaluation on disk)

Every row is a real `artifacts/stage1/streaming_*.json`.  This is the deliberate
record: the failures are the evidence that the recipe, not the metric, was the
problem.

| name | config | chunk_ms | WER | CER | truncations | revisions | WER raw |
|---|---|---|---|---|---|---|---|
| `opd_aligned2_c160` | clean | 160 | **13.24** | 5.11 | 0 | 372 | 13.03 |
| `ternary_n` | clean | 320 | **9.43** | 3.86 | — | — | — |
| `binary_n` | clean | 320 | **10.28** | 4.52 | — | — | — |
| `opd_aligned` | clean | 320 | **12.61** | 5.68 | 0 | 239 | 12.71 |
| `opd_aligned2` | clean | 320 | **12.92** | 5.00 | 0 | 184 | 12.92 |
| `opd_aligned3` | clean | 320 | **14.41** | 5.49 | 0 | 243 | 14.19 |
| `opd_aligned2bin` | clean | 320 | **18.43** | 11.84 | 15 | 297 | 17.90 |
| `opd_aligned2bin_final` | clean | 320 | **27.54** | 22.11 | 29 | 313 | 26.91 |
| `binary_long3` | clean | 320 | **33.26** | 29.92 | — | — | — |
| `ternary_long2` | clean | 320 | **33.58** | 30.00 | — | — | — |
| `opd_lspdistill3` | clean | 320 | **144.60** | 160.46 | 0 | 348 | 278.18 |
| `opd_adapt5` | clean | 320 | **175.00** | 200.65 | 233 | 473 | 215.04 |
| `opd_content3` | clean | 320 | **190.78** | 148.88 | 228 | 448 | 532.20 |
| `ternary_sft` | clean | 320 | **203.50** | 865.46 | — | — | — |
| `` | clean | 320 | **264.51** | 755.37 | — | — | — |
| `opd_distill3` | clean | 320 | **278.18** | 339.10 | — | 337 | — |
| `opd_distill2` | clean | 320 | **349.89** | 287.14 | — | 438 | — |
| `opd_anneal3` | clean | 320 | **358.90** | 308.32 | — | 512 | — |
| `sf` | clean | 320 | **438.24** | 960.72 | — | — | — |
| `opd_distill` | clean | 320 | **448.31** | 432.32 | — | 331 | — |
| `opd` | clean | 320 | **920.55** | 1052.73 | — | 64 | — |
| `ternary_prefix` | clean | 320 | **951.80** | 1192.21 | — | — | — |
| `opd_aligned2_c1000` | clean | 1000 | **10.59** | 4.07 | 0 | 77 | 10.49 |
| `opd_aligned2_other_c160` | other | 160 | **20.80** | 10.83 | 0 | 491 | 20.57 |
| `opd_aligned2_other` | other | 320 | **20.00** | 10.10 | 0 | 250 | 19.09 |
| `opd_aligned_other` | other | 320 | **22.50** | 12.31 | 0 | 296 | 22.27 |
| `opd_aligned2bin_other` | other | 320 | **25.45** | 15.31 | 9 | 312 | 25.23 |
| `opd_aligned2bin_final_other` | other | 320 | **26.93** | 17.37 | 40 | 306 | 26.70 |
| `opd_aligned2_other_c1000` | other | 1000 | **16.59** | 7.27 | 0 | 93 | 16.59 |

## What moved the number

1. **Aligned targets (T4.1/T4.2)** — the whole game.  The forced aligner
   (`Qwen3-ForcedAligner-0.6B`) gives word timestamps; chunk *k*'s target is the
   words whose end time falls in chunk *k*, and an empty target is a stop
   sample.  The model no longer has to infer the chunk-to-text mapping from the
   audio, which is what it was failing at (203 → 12.6/12.9).
2. **The LSP loop-emission policy (T4.4)** — cut the residual collapse
   (278 → 144.6 → ~12) and made it *monotone*: committed text never changes
   (append-only), loops are truncated once, keeping one occurrence.
3. **Rejected, with numbers** — DAgger on the student's own prefixes (920.55:
   it trains its own loops back), inference repetition controls (438/264),
   streaming-state distillation (448 → 350 → 278: an under-emit/over-emit
   oscillation), a low-share anneal (offline 15.3 but the loops return).
4. **The 20-utt in-training guardrail is not a promotion metric** — aligned3's
   guardrail looked best (8.7–9.2) and its full-sample clean was worst (14.41).
   Only the 50-utt rows above decide.

## Reproduction

```bash
# promoted ternary state
CKPT=data/checkpoints/qat_stream_aligned2/step2500.pt
qsub -v CKPT=$CKPT,TAG=_repro,TRUNC=1 scripts/eval_streaming_opd.pbs            # clean
qsub -v CKPT=$CKPT,CFG=other,TAG=_repro,TRUNC=1 scripts/eval_streaming_opd.pbs  # other
qsub -v CKPT=$CKPT,CHUNK=160,TAG=_repro,TRUNC=1 scripts/eval_streaming_opd.pbs  # 160 ms
qsub -v CKPT=$CKPT,CHUNK=1000,TAG=_repro,TRUNC=1 scripts/eval_streaming_opd.pbs # 1000 ms
# training (the recipe): aligned targets from the full-cache offline state
#   scripts/qat_stream_aligned2.pbs  (2400 aligned utts pooled, mix 50/50,
#   --aligned-targets 1 --aligned-stop-stride 2, 2500 steps at lr 5e-6)
```
