# Final research decision

_2026-10-25. Based on the verified artifacts in `docs/final_experiment_results.csv`
and `docs/final_experiment_report.md`. Measured facts, hypotheses and
recommendations are kept separate._

## 1. Retain ternary quantization?

**Yes.** Ternary per-tensor `decoder_attn_ffn` remains the best measured
representation:

| alternative | clean / other | verdict |
|---|---|---|
| ternary per-tensor (b12) | **15.15 / 18.98** (50), 15.75 / 19.99 (100) | baseline |
| 2-bit four-level (b20) | 28.18 / 34.77 | rejected (identical storage, much worse WER) |
| group-64 scales (b21) | 15.68 / 21.48 | rejected (worse both splits) |
| FFN-only, attention fp16 (b17) | 69.28 / 107.27 best | rejected (pathological) |
| Hadamard rotation | never trained (7.4% reconstruction, below pre-registered threshold) | closed on evidence |
| binary | 18.43 / 25.45 best-ever (0.6B, one seed-sensitive state); 1.7B reruns 26.8–36.4 | not competitive |

The decoder-only ternary model is 2.10× smaller than bf16 at 7.60 effective
bits/param; it is not a fully 1-bit model. No packed runtime exists.

## 2. Does streaming-state distillation improve the result?

**No robust both-splits improvement.** The late-layer arm F (λ=0.05, layer 20)
is the best measured candidate:

- 50-utt: **14.62 clean / 18.18 other** vs 15.15 / 18.98 (Δ −0.53 / −0.80).
- 100-utt paired confirmation: clean **14.98 vs 15.75** (Δ −0.77), other
  **20.05 vs 19.99** (Δ +0.06).
- Every paired bootstrap CI includes zero.

**Conclusion:** the objective produces a small, reproducible clean-side gain
but does not improve the hard split; the 50-utt pair win was sample-specific.
It is not a promotion under the project's both-splits rule at the larger
sample. The implementation and tests are retained; the default remains off.

## 3. Should Bonsai remain an active research direction?

**No.** Group-wise scaling and rotation were tested against actual ASR WER:

- g64 reconstruction −5.2% → WER +0.53 clean / +2.50 other;
- rotation 2.1% alone, 7.4% with g64, collapsing all 35 real matrices to the
  ternary noise floor (0.5086); never trained because it failed the
  pre-registered screening threshold;
- 2-bit storage is identical to ternary and loses 13 WER points.

The branch and its evidence are retained (`bonsai-quant`), but no further
sweeps are justified without a new hypothesis. The state-distillation result
does not supply one.

## 4. Does binary QAT merit further work?

**Not now.** The conditional trigger for Phase 5 was a promising improved
ternary objective. The state objective did not establish a robust pair win, and
the existing binary evidence is weak:

- 0.6B binary best-ever 18.43/25.45 came from one seed-sensitive state whose
  weights were lost to a resume; reruns gave 26.80/35.23 and 36.44/43.18;
- 1.7B binary has no competitive result;
- binary is strictly harder than ternary (sign-only weights), while ternary is
  already ~11 points from the gate.

Changing binary variables without a new mechanism would repeat a closed
experiment.

## 5. What is the exact next action?

**Retain the b12/step1600 baseline and close the objective axis.** The
remaining ~11 WER points are quantization damage, not a streaming-policy
deficit: the fp16 1.7B streams at 2.24/4.61, while every quantized state
clusters at 14–22.

If the project continues, the highest-value pre-registered experiment is a
**quantizer-side co-training arm** (alternate full-precision and ternary steps,
or a mixed ternary/binary precision schedule) at the b12 recipe, with:

1. one variable changed from b12;
2. the same 2 000-step budget, data, optimizer and evaluation protocol;
3. a pre-declared acceptance rule (both splits better; promotion only on full
   official evaluation);
4. a **higher-power evaluation** (≥500 utterances) before any promotion,
   because the 50/100-utt subsets cannot resolve sub-1-point effects.

A 500-utt confirmation of F is the alternative if the clean-only gain is
considered valuable; it will not close the gate. Neither direction should be
started without a new hypothesis and the pre-registered protocol above.

## 6. What not to do

- Do not promote F from the 50-utt pair: the 100-utt `other` split ties.
- Do not rerun binary with the same recipe.
- Do not reopen group-size/rotation sweeps.
- Do not use fp16-shadow substitution as a sensitivity test (it measures the
  precision-switch mismatch, not quantization sensitivity).
- Do not cite 50-utt numbers as full-test-set results.
