# Experiment audit — authoritative record

> **Table note (2026-10-25):** the stage-1 machine-readable table is now
> `docs/experiment_results_stage1.csv`; `docs/experiment_results.csv` is the
> next-experiments matrix and `docs/next_experiment_results.md` is its
> narrative record.

_2026-10-25. Branches `stage-1` @ `687fa70`, `bonsai-quant` @ `8cc5b4c`. Working
trees clean. Every number below comes from an official-harness JSON under
`artifacts/stage1/`; guardrail and offline readouts are excluded from verdicts._

## Verified status

| arm | change vs b12 | job(s) | eval status | best clean / other | verdict |
|---|---|---|---|---|---|
| b12 | per-tensor ternary (baseline) | 11871/72 → evals | COMPLETE | **15.15 / 18.98** (step1600) | **best measured** |
| b16 | lr 1.25e-6 | 12204 | COMPLETE | 14.94 / 21.59 (1600) | clean record, worse pair |
| b13 | 8 000 aligned | 11896 | COMPLETE | 21.72 / 17.95 (final) | other record, worse clean |
| b15 | 6 000 steps | 12072 | COMPLETE | 15.15 / 19.55 (2400) | tie/clean, worse other |
| b14 | mix 0.4 | 12037 | COMPLETE | 18.01 / 27.73 | rejected |
| b17 | FFN-only quantization | 12342 | COMPLETE (6/6) | 69.28 / 107.27 @2000; 133.05/111.82 @1200 | **pathological, rejected** |
| b20 | 2-bit `{−3,−1,+1,+3}` | 12478 | COMPLETE (6/6) | 28.18 / 34.77 @2000 | rejected |
| b21 | g64 group scales | 12479 → 12582–12584 | COMPLETE (6/6) | **15.68 / 21.48** @2000 | worse on both, rejected |

### Invalid / superseded

- b17's first evaluations (`*_WRONGPOLICY.json`, 90–93 clean): evaluated with
  `decoder_attn_ffn` against an FFN-only checkpoint. Superseded.
- b21's first batch (45.34/35.91 etc.): evaluator lacked `--group-size`; the
  model was built per-tensor for a grouped checkpoint. Superseded by
  `12582`–`12584` (the fix moved step 1200 from 45.34 → 26.27).

## Discrepancies worth recording

1. **Guardrail vs official**: b17 guardrail 18.12 at step 2000, official 69.28.
   b21 guardrail 15.06, official 15.68/21.48 (close). The in-loop number is not
   a predictor and is never used for selection.
2. **b12's own step 1200** reads 131.57 (121 loops) while step 1600 reads 15.15 —
   state selection is load-bearing for every arm.
3. **Reconstruction does not transfer**: g64 improved reconstruction 5.2 % and
   lost 0.53/2.85 WER; rotation+g64 improved 7.4 % and was never trained.

## Pending

None. The queue is empty; all submitted evaluations have terminal JSONs.
