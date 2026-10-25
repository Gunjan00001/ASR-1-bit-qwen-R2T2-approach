# Experiment audit — authoritative record

_Audit performed 2026-10-25 on `stage-1` @ `bdcf58a` and `bonsai-quant` @
`4807d77`. Both working trees were clean; no jobs of ours were running or
queued at audit start (PBS `gpu`: 22 running / 24 queued cluster-wide, none
ours). Every number below comes from an official-harness JSON under
`artifacts/stage1/`; in-training guardrails, offline readouts and reconstruction
errors are excluded from verdicts._

## 1. Protocol of record

- **Model**: Qwen3-ASR-1.7B (0.6B rows marked).
- **Quantization**: ternary 1.58 (per-tensor unless stated), `decoder_attn_ffn`
  policy unless stated; fp32 shadow weights + STE; learnable log-scales.
- **Training base**: `data/checkpoints/qat_17b_recover7/last.pt` (offline 9.85)
  for every Stage-B arm from b11 onward; earlier arms use their own recorded
  base.
- **Streaming evaluation**: official Qwen3-ASR prefix algorithm through
  `scripts/eval_streaming.py`, 320 ms chunks, greedy, `unfixed_token_num=5`,
  32-token cap per chunk, LSP policy `margin_words=3`,
  `truncate_loops=True`, **50 fixed utterances** of `test-clean` and
  `test-other` (seed 0).
- **Not a full-test-set claim**: every 15.15 / 18.98 / 14.94 / 17.95 number is
  a 50-utterance diagnostic subset, not the full LibriSpeech split.
- **Command**:
  `qsub -v CKPT=<state>.pt,POLICY=<policy>,BIT_WIDTH=<bw>,GROUP_SIZE=<g> scripts/eval_streaming_17b.pbs`

## 2. Verified baseline (the control for the new experiments)

| field | value |
|---|---|
| arm / checkpoint | `aligned17b12` / `step1600.pt` |
| run | PBS training `11871`/`11872`; evals `11891`–`11893` |
| branch / commit at run | `stage-1` (pre-`bdcf58a`; artifacts committed in `687fa70`) |
| policy / width / scale | `decoder_attn_ffn` / 1.58 / per-tensor |
| init | `qat_17b_recover7/last.pt` (offline 9.85) |
| aligned set / mix | 6 000 / 0.5 |
| lr / schedule / steps | 2.5e-6 / cosine, warmup 100 / 2 000 |
| checkpoint | `data/checkpoints/qat_stream_aligned17b12/step1600.pt` |
| **clean** | **15.1483** (CER 6.668, 0 loops, raw 15.0424) |
| **other** | **18.9773** (CER 7.6005, 4 loops, raw 18.9773) |
| artifacts | `artifacts/stage1/streaming_qat_stream_aligned17b12_step1600_{clean,other}.json` |
| storage | ternary, per-tensor: **2.000 bits/param measured** (Bonsai report §F); fp16 baseline 16.000 |

The checkpoint path and the `checkpoint` field inside both JSONs match the
state. Nothing in this audit promotes a guardrail or an intermediate state.

## 3. Complete arm status (official results only)

| arm | change vs b12 | job(s) | official status | best clean / other | verdict |
|---|---|---|---|---|---|
| b5 | recovered 16.53 base, 1 200 aligned | 11592/11666 | complete 2 splits | 19.39 / 24.09 | historical |
| b6 | 11.76 base, 2 400 aligned | 11754/11773 | complete | 21.08 / 28.30 | rejected (two variables) |
| b7 | 10.81 base, 1 200 aligned | 11775/11777 | complete | 16.63 / 23.41 | isolation step |
| b8 | 9.85 base, 1 200 aligned | 11781/11796 | complete | 18.43 / 25.34 | dose reversal |
| b9 | b7 replicate | — | training deterministic; run killed after guardrails matched b7 to the digit | — | closed (reproducibility) |
| b10 | warmup 500 | 11783/11864 | complete | 19.17 / 22.39 | rejected (neutral) |
| b11 | lr 2.5e-6, 1 200 aligned | 11581/11849–11851 | complete | 16.53 / 22.05 | previous line-best |
| **b12** | 6 000 aligned, lr 2.5e-6 | 11871/11891–11893 | complete | **15.15 / 18.98** | **baseline** |
| b13 | 8 000 aligned (data-capped from 12 000) | 11896/11967–11969 | complete | 21.72 / **17.95** | `other` record; pair rejected |
| b14 | mix 0.4 | 12037/12074–12076 | complete | 18.01 / 27.73 | rejected |
| b15 | 6 000 steps | 12072/12085–12088 | complete | 15.15 / 19.55 | ties clean, loses other |
| b16 | lr 1.25e-6 | 12204/12215–12217 | complete | **14.94** / 21.59 | clean record; pair rejected |
| b17 | FFN-only quantization | 12342/12448–12450 + corrected 12466–12468 | complete 6/6 | 69.28 / 107.27 at best state | **rejected, pathological** |
| b18 | lr 5e-7 pilot | training stopped at ~500 | no official eval (closed) | — | closed (offline 51.06) |
| b19 | lr 1e-7 pilot | training stopped at ~500 | no official eval (closed) | — | closed (offline 86.55) |
| b20 | 2-bit `{-3,-1,1,3}` | 12478/12507–12509 | complete 6/6 | 28.18 / 34.77 | **rejected** |
| b21 | g64 group scales | 12479/12582–12584 | complete 6/6 | 15.68 / 21.48 | **rejected (worse both splits)** |
| b17 guards | wrong-policy evaluations | 12448–12450 first batch | invalid | 90–93 clean | **quarantined `*_WRONGPOLICY.json`** |

### b21 final verification (the last outstanding number)

`streaming_qat_stream_aligned17b21_step2000_clean.json` and `..._other.json`,
written 2026-10-24 22:54 / 23:09, both name
`data/checkpoints/qat_stream_aligned17b21/step2000.pt`:

| split | WER | CER | loops | artifact |
|---|---:|---:|---:|---|
| clean | **15.68** | 6.27 | 1 | `artifacts/stage1/streaming_qat_stream_aligned17b21_step2000_clean.json` |
| other | **21.48** | 9.26 | 0 | `artifacts/stage1/streaming_qat_stream_aligned17b21_step2000_other.json` |

Verdict: g64 is worse than b12 on **both** splits (15.68 > 15.15, 21.48 >
18.98); the reconstruction gain did not transfer. Tracker #6 is **COMPLETE**.

### b17 final verification

| state | clean | other | loops (c/o) | valid? |
|---|---:|---:|---|---|
| step 1200 | 133.05 | 111.82 | 740 / 789 | yes (`POLICY=decoder_ffn`) |
| step 1600 | 119.17 | 126.25 | 392 / 514 | yes |
| step 2000 | 69.28 | 107.27 | 351 / 421 | yes |
| wrong-policy 1200/1600/2000 (clean) | 90.36 / 91.63 / 93.01 | — | 13 / 11 / 22 | **no** — attention policy mismatch |

All six corrected artifacts exist and match their checkpoints. The
wrong-policy files remain on disk for the record and are excluded from every
table and CSV.

## 4. Invalid, superseded, incomplete

1. **b17 first evaluation batch** (`*_WRONGPOLICY.json`): evaluator hard-coded
   `decoder_attn_ffn` while the arm trained with `decoder_ffn`. Fixed by
   `POLICY`/`BIT_WIDTH`/`GROUP_SIZE` launcher variables; evidence preserved.
2. **b21 first evaluation batch**: evaluator lacked `--group-size`, so a
   per-tensor model was built for a grouped checkpoint (step 1200 read 45.34).
   Superseded by `12582`–`12584` (26.27).
3. **b13's `--aligned-n 12000`** is data-capped at 8 000 by the training cache;
   the launcher is correct, the corpus is the ceiling. Recorded, not a bug.
4. **b12's step 1200** (131.57, 121 loops) is a valid but broken state; it is
   excluded from "best state" claims, not from the record.
5. **b18/b19** have checkpoints but no official evaluation; closed after the
   training readouts showed they under-train. They must not be cited as
   official results.
6. **No duplicate completed jobs** were found. The only duplicate artifacts are
   the quarantined wrong-policy/wrong-group-size evaluations above.

## 5. Tracker reconciliation (Bonsai)

| tracker item | actual evidence | status |
|---|---|---|
| #6 g64 pilot | step 1200 26.27/27.95; step 1600 15.68/21.82; **step 2000 15.68/21.48** (corrected evaluator) | COMPLETE (6/6), rejected |
| #10 2-bit | 36.86/44.09 → 29.24/33.75 → 28.18/34.77 | COMPLETE (6/6), rejected |
| #11 streaming-state distillation | design existed, **no implementation** at audit time | **implemented this session**; see design/results |
| #7 rotation, #8 rotation×group | reconstruction only; 2.1 % / 7.4 %, never trained | NOT STARTED, closed on evidence |
| #3/#4 rotation prototype + real-weight diagnostic | `rotate.py`, `tests/test_rotate.py`, `bonsai_17b/*` | COMPLETE, isolated |
| group-size reconstruction (0.6B) | 0.5654 → 0.5220/0.5190/0.5138 (g128/g64/g32) | COMPLETE, diagnostic only |

## 6. Discrepancies and standing lessons

- **Guardrail ≠ metric** (eight documented divergences). The 20-utt in-training
  stream WER is never a promotion criterion.
- **State selection is load-bearing**: b12's 131.57 at step 1200 vs 15.15 at
  1600; b16's 14.94 at 1600 vs 16.31 at 2000; b13's early states loop.
- **Reconstruction does not transfer**: g64 −5.2 % relative reconstruction,
  +0.53 clean / +2.50 other; rotation+g64 −7.4 %, never trained because it
  fails the pre-registered threshold.
- **Precision-switch mismatch invalidates shadow-substitution screens**: the
  screen in `docs/b17_diagnostic.md` flips trained ternary modules to their
  fp16 shadows and destroys the model (15.92 → 26–394), so it measures the
  switch, not quantization sensitivity. Not used for any decision.
- **The 15.15/18.98 pair is a 50-utt diagnostic**, not a full-split number.

## 7. State-distillation phase (2026-10-25, added after §1–§6)

Seven pre-registered arms (A control + B–G) trained to 2 000 steps on the b12
recipe; one variable each. All official evaluations are terminal; no job of
ours is running.

| arm | variable | clean (50) | other (50) | status |
|---|---|---:|---:|---|
| A | control (`aligned17b12/step1600`) | 15.15 | 18.98 | verified baseline |
| B | λ=0.05, layer 14 | 15.25 | 21.14 | complete, pair fails |
| C | λ=0.01, layer 14 | 14.94 | 21.36 | complete, pair fails |
| D | λ=0.20, layer 14 | 15.04 | 19.43 | complete, pair fails (near-tie) |
| E | λ=0.05, layer 8 | 14.83 | 20.57 | complete, pair fails |
| F | λ=0.05, layer 20 | **14.62** | **18.18** | **best pair at 50 utts; 100-utt other ties** |
| G | λ=0.05, layer 14, no LN | 14.94 | 19.09 | complete, pair fails by 0.11 |

**100-utt paired confirmation (A vs F):** clean 15.75 vs **14.98** (−0.77);
other **19.99** vs 20.05 (+0.06). Every paired bootstrap CI includes zero.
Verdict: clean-only gain, **no robust both-splits improvement**.

**Step-2000 checks:** D 15.89/26.59; E 20.76/25.34; F 17.80/23.07;
G 15.15/22.73 — all worse than their step-1600 states. State selection remains
load-bearing.

**Job ledger:** training `12642` (B), `12649` (C after OOM retry), `12651` (D
after OOM retry), `12645` (E), `12654` (F), `12657` (G); evaluations `12652`,
`12653`, `12655`, `12656`, `12693`, `12697`, `12695`, `12700`, `12703`,
`12704`, `12698`, `12699`. `12643`/`12644` are the preserved GPU-race OOM
attempts; `12694` skipped a not-yet-written checkpoint. Full detail:
`docs/final_experiment_report.md`.

**Invalidity audit:** none of the state-distillation evaluations used a
mismatched policy or width; the eval JSONs record policy/width from this phase
onward, and the earlier SD JSONs are backed by their committed launchers.

## 8. Machine-readable record

`docs/experiment_results.csv` (regenerated from the artifacts by
`scripts/collect_results.py`) contains 127 official rows with branch, commit,
config, seed, checkpoint, split, WER/CER, loops, raw WER, artifact path and
timestamp. The curated arm-level summary is §3 above. `docs/final_experiment_results.csv`
is the Phase-8 copy; both exclude quarantined files by construction.
