# Next experiments — results

_2026-10-25, branch `next-experiments` (off `quant-recovery` @ `742b5d1`).
Only committed artifacts are used; pending cells are not guessed._

## 1. Status

| experiment | job(s) | status |
|---|---|---|
| Baseline verification | — | verified (checkpoint SHA `0728fe2e…b1e345`) |
| Experiment B — sensitivity screen | 12738, 12740 | **complete** |
| Experiment A — 12.5% co-training pilot | 12735 | **passed** |
| Experiment A — 12.5% full run | 12737 | **complete** (killed in end-of-run ablation after step 2000; step-1600 safe) |
| Experiment A — 12.5% evaluation | 12749 | **complete** |
| Experiment C — mixed k-fp16 pilot | 12739 | **passed** (168 ternary + 28 fp16 layers, 1.350B trainable) |
| Experiment C — mixed k-fp16 full run | 12741 | complete to step 2000; step-1600 saved |
| Experiment C — mixed k-fp16 evaluation | 12758 | clean complete (rejected); other stopped as non-informative (NOT RUN) |

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

## 5. Experiment A — 12.5% co-training — REJECT

Pilot (12735) passed: the schedule alternates every 8th step, losses finite,
no NaN/OOM. Full run (12737) reached step 2000; the step-1600 checkpoint is
on disk. Official 50-utt evaluation (job 12749):

| state | clean | other | CER c/o | loops c/o |
|---|---:|---:|---:|---:|
| **control b12/step1600** | **15.148** | **18.977** | 6.668 / 7.601 | 0 / 4 |
| 12.5% CT / step1600 | 16.525 | 18.296 | 9.127 / 7.932 | 0 / 0 |

Deltas: **+1.377 clean / −0.682 other**. Paired bootstrap: clean
[−3.093, +7.582], other [−2.545, +1.040]; both include zero.
**The lower dose did not reduce the clean regression** — it is slightly worse
than the 25% arm on both splits (25%: 16.314 / 17.386). The dose-response is
therefore not useful: the 25% arm strictly dominates the 12.5% arm. **Close
the co-training direction**; do not run another dose.

## 6. Experiment C — mixed k-fp16 — REJECT

Pilot (12739) passed: 28 k layers kept fp16, 168 projections ternary,
1,350,566,056 trainable parameters, 336 shadow tensors, finite loss. The full
run (12741) trained 2 000 steps (max grad 415, no non-finite) and saved
step 1600. Official evaluation (job 12758), 50 utts at 320 ms:

| state | clean | other | CER c/o | loops c/o | RTF c/o |
|---|---:|---:|---:|---:|---:|
| **control b12/step1600** | **15.148** | **18.977** | 6.668 / 7.601 | 0 / 4 | 21.96 / 21.73* |
| mixed k-fp16 / step1600 | **37.712** | _not run_ | 29.445 / — | 5 / — | 23.75 / — |

The mixed candidate is **much worse on clean** (+22.56 WER; raw 48.09 with 5
loop truncations). The k-fp16 transplant + 2 000-step ternary-only fine-tune
did not recover the function: the 28 k layers kept their pretrained fp16
weights while the rest were the baseline's ternary shadows, and the frozen
pretrained attention keys created a mismatch the 168 ternary layers could not
adapt to. The first-order sensitivity ranking did **not** translate into a
WER improvement.

Packed size (measured, 168 ternary + 28 base-bf16 k layers):
**2 335 031 544 bytes = 2.335 GB, 2.012× vs bf16, 7.952 effective bits/param**.
The compression cost is small; the accuracy cost is not.

**Reject C.** Do not launch a broader mixed-precision search on this evidence.
The `other` split was stopped after the clean failure: the candidate was
already rejected by +22.56 WER on the primary split, and the other split could
not change that verdict. It is recorded as **NOT RUN**, not as a result.

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
