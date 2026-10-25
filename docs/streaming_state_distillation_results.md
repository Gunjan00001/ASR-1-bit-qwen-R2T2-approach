# Streaming-state distillation — controlled results

_2026-10-25, branch `state-distill`, commits `2a84979` + `76e3c5d` (code) and
the experiment commit that follows. Every number below comes from an official
`artifacts/stage1/streaming_*.json`; guardrails and offline readouts are
diagnostics only and are excluded from verdicts._

## 1. Setup

- **Control (A)**: the verified `aligned17b12` recipe, checkpoint
  `step1600.pt` — deep 9.85 base, lr 2.5e-6, 6 000 aligned, mix 0.5, 2 000
  steps, ternary `decoder_attn_ffn`, frozen fp teacher, KL 0.05.
- **Arms B–G**: identical to A except one variable each (the state-loss weight,
  layer, or normalization). See `docs/streaming_state_distillation_design.md`.
- **Implementation**: per-chunk hidden-state MSE at the same decoder layer for
  student and frozen teacher, LayerNorm on both sides, masked to supervised
  target tokens of streaming rows only; audio for each row is truncated at the
  chunk boundary, so no future audio/text participates. 23 tests; the disabled
  path returns the existing CE tensor unchanged.
- **Evaluation**: official Qwen3-ASR prefix algorithm, 320 ms, 50 fixed
  utterances per split (seed 0), greedy, 32 tokens/chunk, LSP margin 3 +
  loop truncation. 100-utterance paired confirmation (same deterministic
  subset for control and candidate) for the one arm that beat both splits.
- **Compute note**: a 50-utt clean+other sweep costs ~0.5–4.1 GPU-hours
  depending on output length; the control's own sweep was measured at 7 962 s
  + 6 860 s under contention. A full LibriSpeech split (~110 GPU-hours/split at
  this RTF) was not runnable on the shared 8-GPU node.

## 2. Step-1600 results (primary state, 50 utts)

| arm | variable | clean | other | CER c/o | loops c/o | raw WER c/o |
|---|---|---:|---:|---:|---:|---:|
| **A** | control (no state loss) | **15.15** | **18.98** | 6.67 / 7.60 | 0 / 4 | 15.04 / 18.98 |
| B | λ=0.05, layer 14 | 15.25 | 21.14 | 6.67 / 10.32 | 0 / 2 | 15.04 / 20.91 |
| C | λ=0.01, layer 14 | 14.94 | 21.36 | 6.75 / 10.87 | 0 / 5 | 14.94 / 25.34 |
| D | λ=0.20, layer 14 | 15.04 | 19.43 | 6.86 / 9.21 | 0 / 1 | 14.94 / 19.32 |
| E | λ=0.05, layer 8 | 14.83 | 20.57 | 6.25 / 9.46 | 1 / 2 | 14.83 / 20.45 |
| **F** | **λ=0.05, layer 20** | **14.62** | **18.18** | **5.84 / 7.45** | **0 / 5** | **14.51 / 18.07** |
| G | λ=0.05, layer 14, no LN | 14.94 | 19.09 | 5.00 / 7.25 | 0 / 0 | 14.83 / 18.75 |

Point deltas vs A: B +0.11/+2.16; C −0.21/+2.39; D −0.11/+0.45;
E −0.32/+1.59; **F −0.53/−0.80**; G −0.21/+0.11.

**Only F improves both splits.** Every difference is small; the five other
arms trade a clean gain for an `other` loss. F's `other` still uses the LSP loop
policy (5 truncations, raw 18.07 → policy 18.18); its clean is loop-free.

## 3. Paired bootstrap (50 utts, 10 000 resamples, seed 0)

| arm | clean Δ [95% CI] | other Δ [95% CI] | utterances better/worse/ties (clean; other) |
|---|---|---|---|
| B | +0.11 [−2.36, +2.47] | +2.16 [−0.81, +6.29] | 8/6/36 ; 10/5/35 |
| C | −0.21 [−2.81, +2.49] | +2.39 [−0.86, +6.89] | 9/8/33 ; 10/6/34 |
| D | −0.11 [−2.69, +2.45] | +0.45 [−1.63, +2.74] | 12/10/28 ; 11/8/31 |
| E | −0.32 [−2.89, +2.30] | +1.59 [−0.33, +3.78] | 7/10/33 ; 11/5/34 |
| **F** | **−0.53 [−3.05, +1.58]** | **−0.80 [−2.15, +0.42]** | 8/7/35 ; 6/7/37 |
| G | −0.21 [−2.74, +2.08] | +0.11 [−2.97, +2.75] | 9/8/33 ; 11/9/30 |

Every 95% CI includes zero. At 50 utterances the design cannot distinguish
these arms statistically; F's both-splits point estimate is the strongest
signal, not a proven effect. The utterance-level counts also show the effect is
concentrated (a few utterances with large error reductions), not a broad win.

## 4. 100-utterance paired confirmation (F vs control)

Same deterministic `sample_n=100, seed 0` subset for both systems.

| split | control (b12/step1600) | F (sd_f20/step1600) | Δ | 95% CI |
|---|---:|---:|---:|---|
| clean | 15.75 (CER 6.78, 0 loops) | **14.98** (CER 6.30, 0 loops) | **−0.77** | [−2.16, +0.46] |
| other | **19.99** (CER 8.12, 4 loops) | 20.05 (CER 9.10, 28 loops) | **+0.06** | [−1.45, +1.70] |

**The 50-utt `other` gain does not replicate.** On the 100-utterance set the
two systems tie on `other` (20.05 vs 19.99, +0.06, CI includes zero), while the
clean gain persists and grows (−0.77). F is therefore a **clean-split
improvement, not a both-splits improvement**; the 50-utt pair win was
sample-specific. This is the decisive negative result for the state objective
as implemented.

## 5. State selection is load-bearing (again)

| arm/state | clean | other | loops c/o |
|---|---:|---:|---|
| **F / step 1600** | **14.62** | **18.18** | 0 / 5 |
| F / step 2000 | 17.80 | 23.07 | 0 / 1 |
| **D / step 1600** | **15.04** | **19.43** | 0 / 1 |
| D / step 2000 | 15.89 | 26.59 | 20 / 2 |

Both late states are much worse; the useful state is the mid/late-1600 window.
This is the same pattern as b12 (step 1200: 131.57) and b16 (step 1600 vs
2000), and it is why every arm ships step checkpoints.

## 6. Training stability and loss components

Parsed from each run's JSONL log (2 000 steps; B/C/E/F complete, D complete,
G complete):

| arm | mean CE | mean KL | mean state (non-zero) | max grad norm | non-finite values |
|---|---:|---:|---:|---:|---:|
| B | 1.350 | 3.221 | 1.382 | 355.5 | 0 |
| C | 1.351 | 3.226 | 1.417 | 346.7 | 0 |
| D | 1.354 | 3.227 | 1.267 | 356.5 | 0 |
| E | 1.355 | 3.229 | 1.359 | 339.9 | 0 |
| F | 1.354 | 3.227 | 1.31 | 350.1 | 0 |

The state term is active: its mean falls monotonically with λ
(1.417 → 1.382 → 1.267 at λ 0.01/0.05/0.20). No NaN/Inf; gradient spikes are
in-family with the control (b12 max 352.8). The loss does not diverge.

## 7. Interpretation

- **Hypothesis**: a per-chunk hidden-state target at a mid/late decoder layer
  should stabilise the streaming state earlier than output-logit KL.
- **Result**: the late-layer arm (F) improves **clean** consistently
  (50 utts −0.53, 100 utts −0.77) but does **not** improve `other` robustly
  (50 utts −0.80, 100 utts +0.06). Every other arm trades the splits, and no
  arm approaches the quality gate.
- **Verdict: not a both-splits improvement.** Under the pre-registered rule
  (both splits must improve), F qualifies only on the 50-utt sample; the
  100-utt replication turns `other` into a tie. The state objective as
  implemented (target-token positions, LayerNorm-MSE) therefore does not beat
  the baseline pair.
- **What the pattern suggests**: the state term affects the clean/hard-split
  balance (clean improves while `other` is neutral-to-worse), consistent with
  over-constraining the harder split or shifting the content/stop balance. The
  clean-only gain is not enough for the gate.
- **Not established**: all confidence intervals include zero; the effect is
  small and the 50/100-utt samples are underpowered.
- **Gate**: 14.62/18.18 is far from the original 3.74/6.11; the state objective
  does not close the quantization-damage gap.

## 8. Limitations

- 50-utt (and 100-utt) diagnostic subsets, not full LibriSpeech splits; the
  measured cost makes full splits ~110 GPU-hours each.
- The control and candidate evaluations ran at different times under different
  contention; RTF comparisons are indicative only, and WER is the comparable
  quantity.
- The state loss supervises target-token positions (teacher-forced), not the
  model's own generated prefix states; prefix-state supervision is an untested
  alternative objective.
- Only one seed per arm; the project's runs are deterministic, but seed
  sensitivity of the binary/streaming transition has been observed before.
- The evaluator now records policy/width; the JSONs produced before that
  change carry the policy only in their committed launchers.
