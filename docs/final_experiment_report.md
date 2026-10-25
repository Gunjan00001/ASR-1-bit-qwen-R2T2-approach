# Final experiment report — ASR-1-bit-Qwen

_2026-10-25. Branch `state-distill` (off `stage-1` @ `bdcf58a`); Bonsai branch
`bonsai-quant` @ `4807d77` audited, unchanged. All numbers below are taken from
committed official evaluation JSONs under `artifacts/stage1/`; in-training
guardrails, offline readouts and reconstruction metrics are excluded from every
verdict._

## 1. Executive summary

1. **Audit closed the existing matrix.** The last outstanding g64 step-2000
   evaluation is verified at **15.68 clean / 21.48 other** (worse than the
   15.15/18.98 baseline on both splits); all six corrected b17 states are
   verified (69–133 clean, 351–789 loops — non-viable); the wrong-policy and
   wrong-group-size evaluations remain quarantined. No experiment was left
   without a terminal state.
2. **Streaming-state distillation was implemented from scratch**, with 23 new
   tests (temporal alignment, chunk masks, padding, future isolation, gradient
   propagation, teacher freezing, disabled-path identity). The existing path is
   bit-identical when the new weight is zero.
3. **Seven pre-registered arms (A–G) were trained and officially evaluated**
   on both splits at step 1600, with qualifying step-2000 checks.
4. **The state objective does not robustly improve the baseline pair.**
   The best arm, F (λ=0.05, decoder layer 20), beat both 50-utt splits
   (**14.62 / 18.18** vs 15.15 / 18.98), but the 100-utterance paired
   confirmation reproduced only the clean gain (**14.98 vs 15.75, −0.77**) and
   the `other` gain vanished (**20.05 vs 19.99, +0.06**). All paired bootstrap
   confidence intervals include zero.
5. **The original gate is not met.** 14.62/18.18 is far from **3.74/6.11**.
6. **Compression is real but modest:** the decoder-only ternary model serializes
   to **2.23 GB, 2.10× smaller than bf16, 7.60 effective bits/param**, with
   60.0% of parameters quantized. No packed runtime exists, so deployment
   acceleration is not claimed.
7. **Binary QAT is not continued** (no statistically established improved
   ternary objective to transfer; binary evidence remains poor).
8. **Bonsai is closed** (group scales, rotation, 2-bit all fail on ASR WER;
   the branch is retained, not deleted).

## 2. The verified baseline

| field | value |
|---|---|
| arm / checkpoint | `aligned17b12` / `step1600.pt` |
| commit / run | `stage-1` @ `bdcf58a`; training `11871`/`11872`, evals `11891`–`11893` |
| config | Qwen3-ASR-1.7B, ternary 1.58, `decoder_attn_ffn`, per-tensor learnable scale |
| init | `qat_17b_recover7/last.pt` (offline 9.85) |
| data / mix | 6 000 aligned utterances, mix 0.5, 2 000 steps, lr 2.5e-6 cosine |
| **50-utt clean / other** | **15.1483 / 18.9773** (CER 6.668 / 7.6005; loops 0 / 4) |
| **100-utt clean / other** | **15.7546 / 19.9887** (CER 6.7757 / 8.1157; loops 0 / 4) |
| artifacts | `streaming_qat_stream_aligned17b12_step1600_{clean,other}.json`; `streaming_b12_step1600_n100_{clean,other}.json` |
| storage | 2.23 GB packed (2.10× vs bf16), 7.60 effective bits/param |

These are **fixed 50/100-utterance diagnostic subsets**, not full LibriSpeech
splits. The full-split cost is measured: ~110 GPU-hours per split at this RTF.

## 3. What was implemented

`src/asr1bit/train/state_distill.py`:

- `HiddenStateCapture` (forward hook on `model.thinker.model.layers[i]`);
- `resolve_layer` (middle layer by default);
- `streaming_state_loss` — masked MSE between LayerNorm'd student and frozen
  teacher states at the same layer;
- `combine_losses` — `(1−w)CE + w·KL + λ_state·L_state`, returning the CE
  tensor itself when both weights are zero.

Trainer flags: `--state-distill-weight`, `--state-distill-layer`,
`--state-distill-normalize`, `--seed`. `train/{aligned,dataset,examples}.py`
tag streaming rows; `collate` emits `state_mask`; only rows whose audio is
truncated at the chunk boundary contribute, so full-utterance samples cannot
leak future audio. The teacher is frozen, run under `no_grad`, and its states
are detached. The evaluator now records policy/bit-width/group-size in every
JSON.

Tests: `tests/test_state_distill.py` (23 tests). Whole suite: **377 passed,
1 skipped**; `ruff check .`: clean. A real-model smoke (0.6B, 5 steps) showed
`state = 1.36–1.38` on aligned rows and exactly zero on standard rows, with
finite gradients.

## 4. Experiment matrix and official results

All arms: 1.7B, ternary 1.58, same base/data/mix/optimizer/budget as the
control; one variable changes. Step 1600 is the primary state (the control's
best); step 2000 was evaluated for arms within 2 WER points of the control.

| arm | variable | clean (50) | other (50) | Δ clean / other | loops c/o | step 2000 clean / other |
|---|---|---:|---:|---:|---|---|
| **A** | control | **15.15** | **18.98** | — | 0 / 4 | (15.89 / 22.16 at last; not selected) |
| B | λ=0.05, layer 14 | 15.25 | 21.14 | +0.11 / +2.16 | 0 / 2 | not required (other >2) |
| C | λ=0.01, layer 14 | 14.94 | 21.36 | −0.21 / +2.39 | 0 / 5 | not required (other >2) |
| D | λ=0.20, layer 14 | 15.04 | 19.43 | −0.11 / +0.45 | 0 / 1 | 15.89 / 26.59 |
| E | λ=0.05, layer 8 | 14.83 | 20.57 | −0.32 / +1.59 | 1 / 2 | 20.76 / 25.34 |
| **F** | **λ=0.05, layer 20** | **14.62** | **18.18** | **−0.53 / −0.80** | **0 / 5** | **17.80 / 23.07** |
| G | λ=0.05, layer 14, no LN | 14.94 | 19.09 | −0.21 / +0.11 | 0 / 0 | 15.15 / 22.73 |

F is the only arm better on both splits at the primary state. Its step-2000
state is much worse, so **step 1600 is selected**. D and E/G show the same
pattern: later states degrade, and state selection is load-bearing.

### 100-utterance paired confirmation (control vs F)

| split | control | F | Δ | 95% CI |
|---|---:|---:|---:|---|
| clean | 15.75 | **14.98** | **−0.77** | [−2.16, +0.46] |
| other | **19.99** | 20.05 | **+0.06** | [−1.45, +1.70] |

The clean gain replicates; the `other` gain does not. **F is a clean-split
improvement, not a both-splits improvement.**

## 5. Statistical uncertainty

Paired bootstrap over utterances (10 000 resamples, same references) for every
arm at step 1600:

| arm | clean Δ [95% CI] | other Δ [95% CI] |
|---|---|---|
| B | +0.11 [−2.36, +2.47] | +2.16 [−0.81, +6.29] |
| C | −0.21 [−2.81, +2.49] | +2.39 [−0.86, +6.89] |
| D | −0.11 [−2.69, +2.45] | +0.45 [−1.63, +2.74] |
| E | −0.32 [−2.89, +2.30] | +1.59 [−0.33, +3.78] |
| F | −0.53 [−3.05, +1.58] | −0.80 [−2.15, +0.42] |
| G | −0.21 [−2.74, +2.08] | +0.11 [−2.97, +2.75] |

Every interval includes zero. The 50-utt sample cannot establish effects of
this size; the 100-utt confirmation narrows but does not resolve them.

## 6. Training stability and loss components

| arm | mean CE | mean KL | mean state (non-zero) | max grad | non-finite |
|---|---:|---:|---:|---:|---:|
| B | 1.350 | 3.221 | 1.382 | 355.5 | 0 |
| C | 1.351 | 3.226 | 1.417 | 346.7 | 0 |
| D | 1.354 | 3.227 | 1.267 | 356.5 | 0 |
| E | 1.355 | 3.229 | 1.359 | 339.9 | 0 |
| F | 1.354 | 3.227 | 1.31 | 350.1 | 0 |

The state term is active (its mean falls with λ) and never destabilized
training; gradient spikes are in-family with the control (max 352.8).

## 7. Invalid, superseded and failed work (kept)

- **b17** first evaluations: wrong policy (`decoder_attn_ffn` vs `decoder_ffn`);
  quarantined as `*_WRONGPOLICY.json`; corrected results 69–133 clean.
- **b21** first evaluations: missing `--group-size`; superseded by the corrected
  26.27/27.95/15.68-21.82/15.68-21.48 series.
- **2-bit b20**: 28.18/34.77, rejected; identical measured storage to ternary.
- **g64 b21**: 15.68/21.48, rejected (worse both splits).
- **FFN-only b17**: 69.28/107.27 best state, rejected.
- **C/D first training attempts**: two jobs raced onto GPU 3 and OOM'd; logs
  preserved, resubmitted on pinned GPUs (`12649`, `12651`) with 1–2 cores.
- **G's first evaluation job** (`12694`): skipped because step1600 did not yet
  exist; resubmitted as `12697`.
- **E/G step-2000 states**: 20.76/25.34 and 15.15/22.73 — later states degrade.
- **F step-2000**: 17.80/23.07 — worse than step1600.

## 8. Compression and deployment measurements

- Packed representation of F/step1600 (and the identical b12 layout):
  **2 232 271 152 bytes = 2.13 GiB**, **2.10×** vs the 4.70 GB bf16 base,
  **7.60 effective bits/param**, 196 ternary projections (1.409 B params,
  60.0% of the model), 2-bit slots + one fp16 scale per tensor.
- This is **decoder-only** ternary quantization: embeddings, LM head, audio
  encoder and norms stay bf16. It is not a fully 1-bit model.
- **RTF**: F 4.08 (50-utt clean) / 5.03 (other) / 4.89 (100-utt clean) on an
  A100 through the PyTorch dequantize-then-matmul path; the control's sweeps
  measured 7.77–21.96 under contention. RTF < 1 is **not met**, and there is no
  packed inference kernel, so **no deployment speedup or constant-memory claim
  is made**.
- **Memory**: no packed-runtime measurement exists. The evaluation process uses
  ~20–40 GB of GPU memory (observed); this is not a deployment figure.

## 9. Compute / job ledger (terminal states)

| job | what | status |
|---|---|---|
| 12640/12641 | real-model smoke (0.6B, state loss) | complete |
| 12642 | arm B training | complete (2 000 steps) |
| 12643/12644 | arm C/D first attempts | failed (GPU race OOM; evidence kept) |
| 12649/12651 | arm C/D resubmitted training | complete (2 000 steps) |
| 12645 | arm E training | complete |
| 12654 | arm F training | complete |
| 12657 | arm G training | complete |
| 12652 | B step1600 eval | complete |
| 12653 | E step1600 eval | complete |
| 12655 | C step1600 eval | complete |
| 12656 | D step1600 eval | complete |
| 12693 | F step1600 eval | complete |
| 12694 | G step1600 eval (premature) | exited: checkpoint absent |
| 12697 | G step1600 eval | complete |
| 12695 | D step2000 eval | complete |
| 12700 | F step2000 eval | complete |
| 12703 | E step2000 eval | complete |
| 12704 | G step2000 eval | complete |
| 12698 | control 100-utt confirmation | complete |
| 12699 | F 100-utt confirmation | complete |

No job of ours remains queued or running.

## 10. Tests and lint

- `pytest -q`: **377 passed, 1 skipped** (23 new state-distillation tests).
- `ruff check .`: **all checks passed**.
- The new code path is disabled by default; `combine_losses` returns the CE
  tensor itself at weight 0 (pinned by a test).
- Real-model smoke verified non-zero state loss on aligned rows and zero on
  standard rows.

## 11. Recommendation

1. **Do not replace the baseline.** The F arm is a clean-split candidate, not a
   robust pair improvement; the 100-utt `other` tie refutes the 50-utt pair win.
2. **Do not continue binary QAT on this evidence.** The improved ternary
   objective itself did not establish a pair win, and the previous binary
   evidence is worse (18.43/25.45 best-ever at 0.6B; 1.7B reruns 26.8/35.2).
3. **Do not reopen Bonsai.** Group scaling, rotation and 2-bit all lose ASR
   WER; reconstruction gains do not transfer; the state objective does not
   change the quantizer-side evidence.
4. **Next experiment, if the project continues:** a higher-power paired
   evaluation (≥500 utterances) of F vs the control is the only way to resolve
   the clean-only effect, but it will not close the gate. The gate's remaining
   ~11 WER points are **quantization damage**; the next high-value arm is a
   pre-registered **quantizer-side** intervention (co-training / mixed
   ternary-binary precision, or selective precision *with retraining*), not
   another objective sweep at this sample size.
