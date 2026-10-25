# Next experiments — results

_2026-10-25, branch `next-experiments` (off `quant-recovery` @ `742b5d1`).
Only committed artifacts are used; pending cells are not guessed._

## 1. Status

| experiment | job(s) | status |
|---|---|---|
| Baseline verification | — | verified (checkpoint SHA `0728fe2e…b1e345`) |
| Experiment B — sensitivity screen | 12738, 12740 | **complete** |
| Experiment A — 12.5% co-training pilot | 12735 | **passed** |
| Experiment A — 12.5% full run | 12737 | running |
| Experiment C — mixed k-fp16 pilot | 12739 | **passed** (168 ternary + 28 fp16 layers, 1.350B trainable) |
| Experiment C — mixed k-fp16 full run | 12741 | queued |

## 2. Baseline (unchanged)

`aligned17b12/step1600`, official 320 ms streaming protocol, fixed diagnostic
subsets:

| split | n | WER | CER | loops |
|---|---:|---:|---:|---:|
| clean | 50 | **15.1483** | 6.668 | 0 |
| other | 50 | **18.9773** | 7.6005 | 4 |
| clean | 100 | 15.7546 | 6.7757 | 0 |
| other | 100 | 19.9887 | 8.1157 | 4 |

Packed size 2.232 GB (2.10× vs bf16, 7.60 effective bits/param). Exact subset
IDs: `docs/eval_utterance_ids.json`.

## 3. Experiment B — sensitivity screen

Complete; full methodology/ranking in
[`docs/layer_sensitivity_report.md`](layer_sensitivity_report.md). Summary:
**k_proj is the largest task-impact component (28.0% of first-order impact at
4.2% of the quantized parameters)**; local reconstruction ranks `o`/`q` higher,
so it is not a substitute for the task-aware screen. This justifies the mixed
k-fp16 candidate.

## 4. Historical context (previous branch, unchanged)

- QEC λ=0.5: **15.466 / 19.091** at 50 utts; rejected (both worse).
- 25% co-training: **16.314 / 17.386** at 50 utts; clean worse, other better;
  rejected as a pair improvement.

## 5. Experiment A — 12.5% co-training

Pilot (12735) passed: the schedule alternates every 8th step, losses finite,
no NaN/OOM. Full run (12737) is still training. Results will be appended from
the official JSONs at step 1600 (primary) and, if warranted, step 2000.

## 6. Experiment C — mixed k-fp16

Pilot (12739) passed: 28 k layers kept fp16, 168 projections ternary,
1,350,566,056 trainable parameters, 336 shadow tensors, finite loss. The full
run (12741) is queued. Results, packed size and RTF will be appended from the
official evaluation.

## 7. Job ledger

| job | what | status |
|---|---|---|
| 12735 | 12.5% CT pilot | complete, passed |
| 12737 | 12.5% CT full | running |
| 12738 | sensitivity screen (initial; share-normalization bug) | complete; superseded by 12740 |
| 12739 | mixed k-fp16 pilot | complete, passed |
| 12740 | sensitivity screen (fixed shares) | complete |
| 12741 | mixed k-fp16 full | queued |

## 8. Pending

- Official step-1600 clean/other for 12737 and 12741.
- Step-2000 checks where the primary is within 2 points or better.
- 100-utt paired confirmation only if both splits improve.
- Packed-size measurement of the mixed checkpoint.
- Final CSV and research report.
