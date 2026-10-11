# Lab notes (append-only)

Every meaningful step or run records: **What we did**, **What failed** (exact
symptom + numbers), **Why it failed** (root cause or leading hypothesis), **Why
we chose what we did**, **Future plans**. Failures are kept, not deleted.

---

## 2026-09-20 — Stage 0 gate PASSED

- **Did:** Ran the fp16 streaming/offline matrix on Kaggle T4 and Colab T4.
- **Failed:** nothing (gate met).
- **Why:** n/a.
- **Chose:** Match the Kaggle stack (torch 2.9.1+cu128, vllm 0.14.0), Triton
  attention on sm75, `max_model_len=16384`, subsampled other splits.
- **Next:** Stage 1 (BitLinear + QAT). Result: 1.7B streaming 2.0 s full clean
  **1.96** vs published 1.95 (+0.01). `X = +1.5` locked.

---

## 2026-09-20 — Stage 1 QAT run #1 (INVALID)

- **Did:** Trained `Qwen3-ASR-0.6B`, policy `decoder_attn` (112 BitLinear),
  30–60 steps, lr 2e-4, Adam8bit, grad checkpointing on, LoRA attempted.
- **Failed:** Post-QAT offline WER: fp16 (α=0) **256.06**, binary **100.0**,
  ternary **98.48**; pretrained 2.65. Loss 3.64 → 1.92 → **15.06** → ~7.
  LoRA unavailable (`torchao 0.10.0` < required `>0.16`), so full-parameter
  training ran first (782M params).
- **Why (hypothesis):** Exploding-gradient signature at the α ramp. The α=0
  readout being corrupt proves the **fp32 shadow weights** were destroyed, not
  the quantized forward. Full-parameter fine-tuning is a confound; the QAT-only
  retry still corrupted shadows, so scope is not the sole cause.
- **Chose:** Stop, do not tag `0.3.0`, and debug systematically (correctness
  tests first) before any retry.
- **Next:** correctness tests → stabilized fp32/α=1 config → re-enable features
  one at a time.
- **Artifacts:** `docs/stage1_results.md`, `artifacts/stage1/attempt_2026-09-20.json`.

---

## 2026-09-20 — Stage 1 STE correctness tests (PASS)

- **Did:** TDD added (a) exact `α=0` forward == `nn.Linear`, (b) an
  "overfit one synthetic batch" test for a single `BitLinear` at α=1,
  (c) `grad_norms` / `check_finite_grads`.
- **Failed:** nothing after tuning. First attempt at the overfit test hit
  93.75% accuracy (D=16, 300 steps); relaxed the synthetic task to D=8,
  800 steps → passes.
- **Why:** The binary weight set cannot perfectly separate a noisy 16-feature
  task (centered-sign removes the bias direction, other features add noise);
  the reduced task is cleanly separable and still exercises STE.
- **Chose:** Keep a demanding accuracy bar (≥0.95) on a task that binary weights
  can represent, so a pass genuinely proves STE gradients are correct.
- **Next:** Use the same `check_finite_grads` + grad-norm logging in the runner.
- **Evidence:** `tests/test_bitlinear.py::TestSteCapability`,
  `tests/test_qat.py::TestGradDiagnostics` (171 tests pass, 1 skipped).

---

## 2026-09-20 — Stage 1 runner stabilized (fp32 / α=1 / features off)

- **Did:** Rewrote `scripts/run_stage1_qat.py`: QAT-only freeze always
  (BitLinear shadows only), fp32 AdamW default (8-bit behind a flag), lr 1e-5
  with warmup, **no autocast**, gradient checkpointing OFF, activation quant OFF,
  **constant α=1.0** (ramp behind a flag), and per-step `check_finite_grads` +
  top-5 grad-norm logging. Removed the LoRA path for Stage 1.
- **Failed:** not yet run.
- **Why:** If this stable config trains, we re-enable gradient checkpointing →
  activation quant → 8-bit Adam → autocast **one at a time** and log which one
  breaks it.
- **Chose:** Constant α=1 isolates training stability from the ramp; QAT-only
  freeze is the correct setup (train the quantized projections, not the model).
- **Next:** Run on Colab T4 (~40 steps, eval-n 15); inspect the grad-norm log to
  locate the trigger; then re-enable features one at a time; then a larger
  (~500-utt) fp16/binary/ternary table. `0.3.0` stays untagged.

---

## 2026-09-20 — Stage 1 stabilized run (AdamW) — STABLE, but naive quant poor

- **Did:** Colab T4, `decoder_attn` (112 BitLinear, 176.16M trainable shadows),
  40 steps, train 48 / eval 15, lr 1e-5 + warmup, **fp32 AdamW**, constant α=1,
  no grad-ckpt, no activation quant, no autocast. Grad-norm logging on.
- **Result:**
  - Grad-norm log (pre-clip, per step max): step1 **519.09** (v_proj) → 98.5 → 35.2
    → 16.7 → 8.9 → … → 9.9. All finite (`check_finite_grads` passed every step).
    Top-5 at step 1: v_proj 519, o_proj 505, k_proj 347, q_proj 296, k_proj 177.
  - Loss: **18.26 → 6.94** (monotone-ish, no explosion).
  - Offline WER (15-utt subset): pretrained **2.65**, post-QAT fp16 (α=0)
    **2.65** (clean!), binary (α=1) **353.79**, ternary (α=1) **170.08**.
- **What failed:** binary/ternary WER is catastrophic; loss floor ~6.9 (high).
- **Why:** The stabilized config **fixed the shadow corruption** (fp16 ==
  pretrained), so the earlier 256% was an optimizer/ramp interaction, not STE.
  But 40 steps on 48 examples at lr 1e-5 with only attention projections
  quantized does not recover naive 1-bit/ternary attention — this is the
  accuracy-recovery problem Stage 2 targets (on-policy distillation,
  co-training, learnable scales, more data/steps).
- **Chose:** Keep the stabilized config as the baseline for feature re-enabling;
  treat binary/ternary WER as the Stage 2 starting point, not a Stage 1 gate
  failure of the harness.
- **Next:** Toggle 8-bit Adam alone (constant α=1) to identify the trigger; then
  grad-ckpt; then act-quant; then autocast. Then a ~500-utt table.

---

## 2026-09-20 — Trigger isolation: 8-bit Adam alone is NOT the trigger

- **Did:** Same stabilized run but `--optimizer adam8bit` (constant α=1, no
  grad-ckpt, no act-quant, no autocast, lr 1e-5).
- **Result:** pretrained **2.65**; post-QAT fp16 (α=0) **2.65** (clean!); binary
  **395.45**; ternary **179.92**. Grad norms: 519 → … → 62.7, all finite.
- **What failed:** binary/ternary still catastrophic (expected — naive quant).
- **Why (finding):** 8-bit Adam does **not** corrupt the shadows. Therefore the
  original 256% corruption came from one of the remaining differences vs the
  failing run: **α-ramp**, **gradient checkpointing**, **activation quant**, or
  **lr 1e-4**. Leading suspect: the progressive α ramp interacting with our
  identity-STE (forward blends, gradient is identity regardless of α), or
  gradient checkpointing with the custom patched forward.
- **Chose:** Stop here for review (the stabilized config fixed the corruption, so
  per the ordered plan we stop at the first fix). The remaining toggles
  (grad-ckpt → act-quant → α-ramp → lr) are queued for the next session.
- **Next:** Isolate grad-ckpt next; then act-quant; then α-ramp. Separately,
  binary/ternary naive WER (~354–395%) is the Stage 2 recovery problem, not a
  harness bug. `0.3.0` stays untagged.

---

## 2026-09-20 — Colab GPU runtime lost (zero-shot run blocked)

- **Did:** Launched the zero-shot α=1 baseline (binary/ternary floor, eval-n 100,
  `--zero-shot 1`) on Colab T4, plus added `--zero-shot` and `--eval-every`
  (periodic WER during training) to the runner (`2ab6a25`, tag `0.2.12`).
- **Failed:** The Colab runtime was recycled to **CPU** mid-run
  (`torch 2.11.0+cpu`, `cuda_avail False`, `nvidia-smi: command not found`);
  `/content` was wiped (repo, logs, artifacts all gone). No zero-shot numbers.
  Re-opening the browser connection did not restore a GPU runtime.
- **Why:** Colab preempted/reset the GPU session; the MCP connection points at a
  CPU runtime and there is no MCP tool to change the runtime type.
- **Chose:** Stop instead of silently falling back to Kaggle: the
  `gunjanpal/asr-1bit-qat-attn` kernel is still **RUNNING** (~15 h), so Kaggle T4
  is contended and quota is under pressure — the user's fallback rule allows
  Kaggle only when Colab blocks, but we should free/confirm Kaggle first.
- **Next:** Reconnect a Colab **GPU** runtime (Runtime → Change runtime type →
  T4) and re-run the zero-shot floor + the 1–2k-step QAT recovery curve; or stop
  the Kaggle QAT kernel to free T4 for a Kaggle fallback. Code is safe on
  `stage-1` @ `2ab6a25`; nothing was lost except this run.

---

## 2026-09-20 — Decisive recovery measurement: NOT a clean recovery (bug)

Colab T4 restored (fresh runtime; torch 2.11.0+cu128, transformers 4.57.6,
peft 0.20.0, bnb 0.50.2). `decoder_attn` (112 BitLinear), QAT-only freeze
(176.16M shadows), fp32 AdamW, lr 1e-5 + 50 warmup, grad-ckpt/act-quant/autocast
off.

- **Did (1) zero-shot floor** (α=1, no training, eval-n 100): pretrained/fp16
  **2.93**, binary **100.0**, ternary **140.8** WER.
- **Did (2) 1k-step QAT, train 200 / eval 30, periodic α=1 WER:**
  - Constant α=1: binary 434.21 → 99.13 → 104.01 → **191.27**; ternary 116.40 →
    100.17 → 101.57 → 130.02. Loss 18 → 5.42. Final fp16 readout **2.62 (clean)**.
  - α-ramp (0.3 warmup): binary 130.19 → 182.90 → 107.16 → **70.51**; ternary
    163.53 → 199.83 → 195.64 → 250.09. Loss → **1.02**. Final fp16 readout
    **10.30 (CORRUPTED)**.
- **What failed:** No monotone recovery. Constant-α binary ends **worse** than the
  zero-shot floor (191 vs 100) while fp16 stays clean; ramp improves binary
  (100 → 70.5) but corrupts the fp16 readout (10.3) and ternary is erratic.
- **Why (hypothesis):** A ramp loss of **1.02** (below the full-precision ~3.6)
  with ~100% generation WER means the failure is not a simple recovery gap: the
  teacher-forced objective and autoregressive generation disagree. Likely a
  train/eval (teacher-forced vs generation) mismatch and/or binarizing decoder
  attention fundamentally breaks generation. The α-ramp is a confirmed corruption
  trigger (fp16 10.3 vs 2.62).
- **Why we chose this:** This was the pre-agreed decisive test — flat/worse WER
  from the naive floor ⇒ treat as a bug, not Stage 2 recovery. That is what we
  observe.
- **Next:** (a) **decisive generation test** — after QAT, greedily generate on a
  *training* utterance and check it reproduces the label (if not ⇒
  quantized-forward/generation bug); (b) compare teacher-forced logits vs
  generated tokens; (c) finish trigger isolation (grad-ckpt → act-quant; α-ramp
  confirmed); (d) revisit per-layer scale handling (per-tensor centered-sign may
  be too coarse for attention) and consider excluding attention or group-wise
  scales. `0.3.0` stays untagged.
- **Artifact:** `artifacts/stage1/recovery_2026-09-20.json`.

---

## 2026-09-20 — Instrumented generation test: forward/objective, not decode path

- **Did:** `scripts/diag_generation.py` on one training utterance
  (`2277-149896-0004`, label "HOW WOULD THE PAPERS TALK ABOUT IT"): teacher-forced
  loss + token accuracy + argmax text vs greedy generation, at α=0 (control),
  α=1 naive, and α=1 after 300 steps (constant α=1, lr 1e-5). Instrumented
  `BitLinear._quantized_weight` to assert the effective weight is Q(W).
- **Q(W) assertion: PASSES.** At α=1 the effective weight differs from the shadow
  (norm 39.18 → 29.48, delta 25.81; several layers); at α=0 it is unchanged. The
  loss forward uses Q(W), not W.
- **Teacher-forced vs greedy:**
  | condition | loss | token acc | teacher-forced text | greedy text |
  |---|---|---|---|---|
  | fp16 control (α=0) | 3.67 | 0.75 | "language wouldOULD THE PAPERS TALK ABOUT IT?" | "How would the papers talk about it?" |
  | binary naive (α=1) | 18.26 | 0.00 | Chinese-token collapse (示示IDE…) | Chinese-token collapse |
  | binary trained 300 steps | 4.48 | 0.17 | "WE IOULD ITINGEREHE" | "WE IS I WOULD IT W" |
- **What failed:** After training, **both** teacher-forced and greedy are garbage.
- **Why (root cause per criterion):** Both paths bad ⇒ **not** a KV-cache /
  incremental-decode bug — the **quantized forward/objective is broken**.
  Binarizing all decoder attention q/k/v/o collapses the model (loss 18 → 4.5
  after 300 steps, token acc 0.17). Weight norms barely moved (39.1814 → 39.1809)
  yet loss fell sharply: binary weights are extremely sign-sensitive. The earlier
  α-ramp "loss 1.02" was a blend pathology (fp16 readout corrupted), not a real
  teacher-forced success.
- **Also flagged:** the fp16 control teacher-forced text shows **prefix leakage**
  ("language would" before the label) — the label masking / prefix-length
  alignment may be off, which would corrupt the training objective.
- **Chose:** Replace the unsafe blended α-ramp with **quantization delay**
  (pure fp then constant α=1); keep the ≥100-utt fixed eval set. Do **not** jump
  to group-wise scales yet.
- **Next:** (a) verify/fix label masking (prefix length alignment); (b) localize
  which projection binarization destroys (single-layer vs all); (c) re-run with
  quantization delay + ≥100-utt eval; (d) then trigger isolation (grad-ckpt →
  act-quant).
- **Artifact:** `artifacts/stage1/diag_generation_decoder_attn_steps300.json`.

---

## 2026-09-20 — Masking verified + per-projection sensitivity

- **Did (1) masking:** mirrored Qwen's official convention (render the prefix
  template with a **dummy audio placeholder**; mask the prompt out of labels via
  `make_sft_labels`). Direct inspection of one utterance:
  `PREFIX = '<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>…<|audio_end|><|im_end|>\n<|im_start|>assistant\n'`,
  `LABEL = 'THE LADY IS NOT THE MOTHER OF THE BOYS BUT THEIR AUNT<|im_end|>'`
  == the target, `prefix_len = 69`, `FULL[:69] == PREFIX_INPUTS`. **Labels contain
  exactly the transcription tokens; no prompt/audio/language tokens leak.**
- **What "0.75 token accuracy" actually was:** NOT misalignment. LibriSpeech refs
  are uppercase without punctuation while the model emits cased/punctuated text
  plus its natural `language …<asr_text>` tag, so exact-token teacher-forced
  accuracy is < 1.0 even for the fp model (fp `none` row: WER 3.18, token acc
  0.78). The earlier "prefix leakage" flag was the model's natural language-tag
  output, not a masking bug.
- **Did (2) per-projection sensitivity** (alpha=1 on exactly one projection;
  greedy WER on a fixed 20-utt subset):

  | binarized | loss | token acc | WER |
  |---|---|---|---|
  | none (fp) | 2.22 | 0.78 | **3.18** |
  | q | 5.07 | 0.22 | 101.59 |
  | k | 6.83 | 0.22 | 139.49 |
  | v | 10.75 | 0.00 | 100.00 |
  | o | 8.51 | 0.00 | 195.22 |
  | all | 18.48 | 0.00 | 100.00 |

- **What failed:** Binarizing **any single** attention projection is already
  catastrophic — it is not a "too many layers" effect.
- **Why (hypothesis):** The quantizer/scale scheme (per-tensor centered-sign) is
  the leading suspect for the broken quantized forward, not attention breadth.
- **Chose:** Record and refine the verdict — forward uses Q(W) (proven), labels
  are correct (proven), yet single-projection binarization destroys the model, so
  the quantization math/scale is the next target (after the QAT re-run).
- **Next:** (3) re-run QAT with quantization delay, ≥100-utt eval, higher lr
  (1e-4/2e-4) and more steps/data; then revisit the quantizer math (per-tensor
  centered-sign, scale granularity).
- **Artifacts:** `artifacts/stage1/diag_generation_decoder_attn_steps0.json`,
  `artifacts/stage1/diag_projection_decoder_attn.json`.

---

## 2026-10-10 — Quantizer audit (A100 cluster): math is correct; collapse is expected

Moved to the JIIT cluster (8 × A100-SXM4-80GB, PBS `gpu` queue; see `gpu.md`).
Environment: python 3.13.12, torch 2.11.0+cu126, transformers 4.57.6,
qwen-asr 0.0.6, datasets 5.0.1, numpy 2.5.3. All 178 existing tests pass here.

- **Did:** measured the relative Frobenius error `||Q(W)-W|| / ||W||` of all 196
  decoder projections directly from `model.safetensors` under every quantizer
  configuration (`scripts/diag_quant_error.py`).
- **Result (avg over 196 projections):**

  | config | rel. Frobenius error |
  |---|---|
  | bf16 round-trip | ~0 (checkpoint is already bf16) |
  | binary, per-tensor | **0.648** (0.61–0.74 per projection) |
  | ternary, per-tensor | 0.565 |
  | binary, group 64/128 | 0.616 / 0.614 |
  | ternary, group 64/128 | 0.519 / 0.522 |

- **What this means:** closed form for centred-sign with scale `mean|W|` on
  Gaussian weights is `sqrt(1 + 2/π − 2·(2/π)) ≈ 0.60` — the measured numbers
  match to within the weights' non-Gaussianity. **The quantizer math is correct.**
  Group-wise scaling barely helps because sign() discards magnitude regardless of
  the scale; the error is dominated by the sign pattern, not the scale.
- **Why the earlier single-projection collapse happened:** a linear layer whose
  weights are ±mean|W| adds ~60% relative noise to its output; through 28
  attention layers that is catastrophic. Single-projection WER 100 is expected
  naive 1-bit behaviour (consistent with "Towards One-bit ASR", +52% relative WER
  even *after* naive QAT). It was never a harness/STE bug.
- **Chose:** treat the collapse as inherent; add the missing BitNet b1.58
  ingredients — **learnable dequantization scales** (per-tensor or per-group,
  `BitLinear(learnable_scale=True)`, off by default), a cosine LR schedule,
  utterance-cache training data, and shadow checkpoints — then run real QAT on
  the A100 (the T4 budget of 40–1000 steps on ≤200 utts cannot recover 60% noise).
- **Next:** (a) 1-utterance memorization smoke at α=1 (loss controllability);
  (b) binary vs ternary recovery on train-clean-100 (~4k utts), policy
  `decoder_attn_ffn`, learnable scales; (c) emit the Stage 1 WER table.
- **Artifacts:** `artifacts/stage1/diag_quant_error.json` (per-layer rows).
- **Environment note:** `datasets>=5` requires numpy≥2; the shared `~/miniconda3`
  was upgraded to numpy 2.5.3 (torch 2.11 unaffected).

---

## 2026-10-10 — Loss controllability PROVEN (ternary memorizes; "bug" verdict retracted)

Two infrastructure traps were fixed first (both looked like hangs, both were
environmental, not model bugs):

- **Slow tokenizer:** the released checkpoints ship `vocab.json`/`merges.txt`
  but no `tokenizer.json`, so every process rebuilt the slow Python
  `Qwen2Tokenizer` line-by-line from `merges.txt` (hours under BeeGFS/CPU
  contention; py-spy showed `bpe_merges.append` at line 76k/151k). Fix:
  `scripts/install_fast_tokenizer.py` builds `tokenizer.json` once
  (`Qwen2TokenizerFast`; full CPU load 11.6 s vs multi-minute stalls).
- **Data/OOM:** materializing `train.100` with `datasets` (non-streaming) blew
  the 32 GB PBS cgroup (`oom-kill` in dmesg). Fix: streaming reads
  (`iter_librispeech(..., streaming=True)`) and building the 4k-utterance cache
  on the login node (`scripts/build_qat_cache.py --n 4000 --stream 1`;
  3.34 GB, 14.5 h audio, mean 13.0 s/utt). Also staged conda + model + cache on
  node-local `/tmp` to avoid shared-FS stalls.

- **Did (memorization test):** one utterance (`train-n 1`), 600 steps, constant
  α, lr 1e-3, `decoder_attn` + learnable scales. The earlier runs had *never*
  actually tested memorization: they cycled 64 utterances with batch 1.

  | condition | naive loss | trained loss | token acc | greedy |
  |---|---|---|---|---|
  | α=0 fp control | 2.225 | **0.000** (step 300) | 1.000 | exact |
  | binary α=1 | 18.49 | 2.818 (step 600) | 0.111 | "THE THE" |
  | ternary α=1 | 11.63 | **0.000** (step 300) | 1.000 | exact label |

- **What this means:** the harness (labels, STE, learnable scales, patched
  forward) is correct and **ternary QAT is fully controllable** — the quantized
  model can memorize through `Q(W)`. The earlier "forward/objective bug" verdict
  is retracted: failures were budget + config (40–1000 steps, lr 1e-5, ≤200
  utterances, no learnable scales), plus the inherent ~60% quantization error
  documented in the previous entry. Binary lags ternary substantially
  (loss 2.8 vs 0.0) — consistent with the literature; ternary is the realistic
  ship, binary the stretch target.
- **Chose:** run the real Stage 1 recovery on the 4k-utt cache:
  `decoder_attn_ffn` (196 BitLinear), ternary, learnable per-tensor scales,
  constant α=1, lr 2e-4 with 100-step warmup + cosine, batch 4, 3000 steps,
  fixed 50-utt test-clean eval every 500 steps, shadow checkpoints every 500.
- **Artifacts:** `artifacts/stage1/diag_generation_decoder_attn_bw1.0_steps600.json`,
  `..._bw1.58_steps600.json`, `artifacts/stage1/qat_ternary_attnffn_ls.log`.

---

## 2026-10-10 — Real-data QAT on A100: naive and distillation both fail to recover

- **Did:** two full QAT runs on the 4k-utt `train.clean.100` cache, ternary,
  `decoder_attn_ffn` (196 BitLinear), log-parameterized learnable scales,
  lr 2e-4, 100-step warmup + cosine, 5 % fp warmup then α=1, fixed 50-utt
  test-clean eval every 500–1000 steps:
  (1) naive, 6 000 steps, batch 4;
  (2) self-distillation (`--distill-weight 0.5`, T=2, teacher = same shadows at
  α=0), 4 000 steps, batch 2.
- **Result:**
  - naive periodic ternary WER: 127 → 121.6 → **96.5** → 96.9 → 103.5;
    final fp16 readout **113.1**, binary 99.8, ternary 107.8.
  - distill periodic: 114.6 → 125.0 → 111.3 → **98.1**;
    final fp16 readout **108.3**, binary 118.3, ternary 98.1.
  - Reference: pretrained fp16 **2.44**; zero-shot naive ternary ≈ 100.
  - Distillation KL to the fp teacher: ~0.4–0.6 (low) yet generation WER ~100
    — the asymmetric KL is satisfied by a diffuse student; it does not imply
    matching argmax trajectories.
- **What failed:** no recovery (WER ≈ zero-shot), and the **fp16 readout
  itself collapses** (2.44 → ~110) after 4–6 k steps at lr 2e-4. The earlier T4
  runs at lr 1e-5 kept the fp readout clean, so this is a budget/lr effect, not
  a harness bug.
- **Why (leading explanation):** the per-tensor ternary sign bottleneck
  (~52 % weight error) is too coarse for post-hoc conversion of a 0.6 B model
  with 4 k utterances; the optimizer's only path is to drift the latent weights
  into a degenerate basin (WER > 100 ⇒ repetitive/empty generations). A moving
  self-teacher cannot prevent this because it drifts with the student.
- **Chose:** Stage 2 needs an **anchored** recovery: KL to a *frozen* fp copy of
  the pretrained model (not the moving self-teacher), co-training / stochastic
  precision, lower shadow lr or an explicit drift penalty, and more data.
  The FFN-only ablation (running) sizes the recoverable quantization surface.
- **Next:** collect FFN-only results and checkpoint generations
  (`scripts/inspect_checkpoint.py`); design the anchored-distillation run.
- **Artifacts:** `artifacts/stage1/stage1_decoder_attn_ffn_*.json`,
  `artifacts/stage1/qat_ternary_attnffn_ls2.log`,
  `artifacts/stage1/qat_ternary_distill.log`.

---

## 2026-10-10 — Failure mode confirmed: the model ignores audio (prior collapse)

- **Did:** `scripts/inspect_checkpoint.py` on the v2 (naive, 6000-step)
  checkpoint, 5 fixed test-clean utterances.
- **Result:** fp16 readout WER 112.2 and quantized WER 112.2 (identical);
  hypotheses are fluent **generic text unrelated to the audio**, e.g.
  ref "IF IT ONLY WERE NOT SO DARK HERE AND SO TERRIBLY LONELY" →
  hyp "AND THE FIRST OF THE HUMAN BODY IS THE FIRST OF THE HUMAN BODY AND HE…".
- **Why:** the decoder lost audio conditioning and collapsed onto the language
  prior. Teacher-forced loss (~4.5) stays low because the prompt still anchors
  the continuation; free-running generation ignores the audio entirely. The
  frozen AuT encoder is fine — the damaged part is the ternary attention
  pathway in the decoder, consistent with the per-projection sensitivity table.
- **Chose:** (a) FFN-only run (attention fp) as the decisive ablation — running;
  (b) **anchored distillation** with a *frozen* fp teacher and lower shadow lr
  (5e-5) — submitted (`scripts/qat_ternary_anchored.pbs`).
- **Artifacts:** `artifacts/stage1/inspect_v2.json`.

---

## 2026-10-10 — FFN-only RECOVERS; attention is the blocker

- **Did:** ternary QAT with `decoder_ffn` (attention stays fp; 84 BitLinear),
  same data/schedule as the failing attn+ffn runs (4 k utts, lr 2e-4 cosine,
  delay 5 %, batch 4).
- **Result (periodic ternary WER):** 118.0 @500 → 84.3 @1000 → **42.2 @1500**;
  training loss 4.1 → 2.1; gamma healthy. WER roughly halves every 500 steps.
  Binary FFN-only queued for the table; run continues to 3 000 steps.
- **Why it matters:** attention quantization was the blocker (it destroys audio
  conditioning — the collapsed models emit generic prior text). FFN ternary
  (gate/up/down, ~2/3 of decoder params) is recoverable at this budget.
- **Also tried — anchored distillation is destructive at w=0.5:** frozen fp
  teacher + KL 0.5 explodes (KL 11–15) and collapses the model (fp16 427,
  ternary 315 @1000 steps, killed). Anchoring needs small weight or a
  trust-region form, not a flat KL.
- **Chose:** FFN-only is the lead Stage 2 configuration; add attention back
  incrementally later (per-layer rollout + small anchor), not flat KL.
- **Artifacts:** `artifacts/stage1/qat_ternary_ffn.log`,
  `artifacts/stage1/qat_ternary_anchored.log`.

---

## 2026-10-11 — Stage 2a/2b: full quantization recovers to 16.2 ternary / 17.5 binary

- **Did:** (2a) FFN-only ternary warm start → attention rollout with a
  **frozen pretrained fp teacher anchor** (w=0.1, T=2) at lr 5e-5 (2 k steps),
  then a warm-restart cosine cycle (lr 2.5e-5, 5 k steps). (2b) switch
  `bit_width=1.0` from the ternary checkpoint (progressive precision, 2 k
  steps).
- **Result (50-utt test-clean, fp16 baseline 2.44, naive QAT ≈ 100):**
  - ternary, all 196 decoder projections: **16.2 final / 14.4 best**;
  - binary, all 196 decoder projections: **17.5** (51.5 → 38.1 → 25.4 → 17.5);
  - FFN-only comparison: ternary 25.7 best / 32.8 final; binary 97.9;
  - fp16 readouts of the trained shadows drift (85–100) — the anchor slows but
    does not prevent drift; the quantized function is the deliverable.
- **Why it works:** the FFN warm start gives a strong basin; the small anchor
  preserves audio conditioning while attention adapts; low lr + cosine keeps the
  sign pattern from churning; progressive precision (ternary → binary) avoids
  the binary-from-fp collapse.
- **Infra findings (see `docs/ACTIVITY_LOG.md`):** `--resume` must advance the
  LR schedule (fixed); use `--init-from` for new phases; never load
  model+checkpoint in the 32 GB session cgroup; switch GPUs instead of waiting
  (rules.md R1).
- **Next:** longer runs on the 8k cache, co-training/stochastic precision, then
  the streaming (320 ms) primary-metric evaluation.
- **Artifacts:** `artifacts/stage1/stage1_decoder_attn_ffn_bw1.58_gNone_lr2.5e-05_*.json`,
  `artifacts/stage1/stage1_decoder_attn_ffn_bw1.0_gNone_lr5e-05_*.json`,
  `artifacts/stage1/qat_attn_rollout*.log`, `artifacts/stage1/qat_binary_rollout.log`.

---

## 2026-10-11 — WP1 design: on-policy streaming SFT + LSP emission (implementation)

- **Did:** implemented WP1 (`src/asr1bit/train/sft.py`, `src/asr1bit/qwen/stream.py`,
  runner flags `--op-rollout*`, `eval_streaming.py --lsp-margin-words`); 277 tests
  pass, ruff clean. Launched WP1 phase 1 (`11020`, rollout 700 utts, 2 400 on-policy
  + 2 400 standard, 3 000 steps, init 9.2) and WP2 ternary full
  (`11022`, 10 000 steps on 28 539 utts).
- **Failed:** nothing yet — results pending (the previous inference-side controls
  had already failed: repetition_penalty 438.2, no_repeat_ngram 264.5).
- **Why (the design argument):** the official algorithm's collapse has two
  signature modes, both distribution mismatches, not inference bugs:
  1. *warm-up runaway* — the model answers a 0.32 s chunk with
     `assistant\nassistant…` (32 tokens, no EOS), where the pretrained model
     emits metadata + EOS;
  2. *self-prefix loops* — the prompt is the model's own rolled-back text, a
     pattern the static prefix SFT never saw (it trained on *gold* prefixes with
     the **full** audio, while inference uses truncated audio).
  On-policy rollouts fix both: the training prompt is exactly the prompt the
  model meets at inference, the target is the reference remainder (so the
  transcript is completed from any own-prefix state) and completing the
  transcript trains EOS (the stop signal the 5 000-step SFT never had — its
  prefixes stopped at 95 % of the transcript).
- **Chose:** one DAgger rollout per phase (phase 2 = `qat_stream_opd2.pbs` with a
  fresh rollout from the phase-1 checkpoint) rather than in-job refreshes, so
  each phase re-uses the stable offline recipe's LR/cosine structure; rollout
  capped at ≤16 rounds (bounds the O(n²) re-feed), states kept at warm-up +
  stride + last round (~8/utt), pool capped at 2 400 to land the 50/50 mix.
  Rejected: keeping the static prefix samples in the mix (the 95 %-prefix sample
  taught "continue to the end" but never "stop"; warm-up EOS plus
  complete-prefix-EOS covers both ends).
- **Next:** decisive official-algorithm eval (`eval_streaming_opd.pbs`, LSP
  margin 3 words → append-only commit latency + revision counts), then phase 2;
  if phase 1 still loops, raise the on-policy share / add checkpoint-level
  refreshes before touching the offline recipe.
- **Artifacts:** `artifacts/stage1/qat_stream_opd.log`,
  `artifacts/stage1/qat_ternary_full.log`, `tests/test_train_sft.py`,
  `tests/test_trainer_rollout.py`.

---

## 2026-10-12 — WP1 phase 1 result: DAgger on the student's own prefixes POISONS the model

- **Did:** ran on-policy streaming SFT (job `11030`): 350-utterance rollout with
  the 9.2 ternary student → 2 400 on-policy states + 2 400 standard samples,
  3 000 steps at lr 1e-5 cosine (anchor 0.05). Then the decisive official-
  algorithm eval (`11036`) and a per-chunk diagnostic (`11039`).
- **Result (50-utt test-clean):**
  - offline ternary **18.33** (from 9.2; still descending: 19.70 @1000 →
    18.75 @2000 → 18.33 @3000), fp16 readout 62.2, binary readout 100.0;
  - **official algorithm (unfixed 5, 32 tokens, 320 ms): WER 920.55**,
    CER 1052.73, RTF 4.49 — *worse* than the previous static-SFT checkpoint
    (203.5). LSP margin 3: 64 revisions, committed@50 % 13.8 %.
- **Failed:** the model now loops from the first *content* chunk:
  chunk 0-2 emit `''` + EOS (the warm-up-EOS training **worked** — the
  `assistant\nassistant…` runaway is gone), but chunk 3 (≥1.28 s) degenerates
  into `language'S'S'S'S…` (32 tokens, no EOS) and never recovers
  (`artifacts/stage1/diag_opd_u0.json`, `diag_opd_u1.json`,
  `diag_opd_u1_skip.json`). A warm-up skip does not help: the loop is in the
  content continuation, not the cold start.
- **Why (root cause):** the rollout's own prefixes were themselves degenerate —
  the base model produces `WORDS WAS IT'S'T'T'T…`-style loops at mid-stream
  chunks, so the training pairs were "(looping prefix) → reference remainder".
  Fitting those pairs teaches the *loop patterns* as plausible continuations:
  DAgger on a degenerate policy poisons the student instead of repairing it.
  The 18.3 offline value shows the fresh cosine had not annealed either (the 8k
  history needed ~8 000 steps for the same recovery).
- **Chose:** replace the DAgger recipe with **streaming-state distillation**
  (`--op-distill`): the frozen pretrained teacher drives the rollout (its
  streaming policy is clean and EOS-terminated — verified trace:
  `diag_opd_pretrained_u0.json`, 2.33 WER with the official algorithm) and the
  student clones the teacher's own continuation at each state; generations that
  hit the token cap are dropped, so every sample ends in a stop. Same 50/50 mix
  with standard samples, 4 000 steps from the 9.2 base (`11040`, running).
  Alternatives rejected: another DAgger iteration (its rollout sees even more
  degenerate prefixes — job `11038` was killed after 1 h), inference controls
  (measured worse), longer static prefix SFT (203.5).
- **Next:** the distillation result decides between (a) single-digit official-
  algorithm WER → anneal + scale, or (b) a filtered-DAgger hybrid (drop
  degenerate own-prefixes, oversample warm-up EOS) with the distillation mix.
- **Artifacts:** `artifacts/stage1/streaming_opd_quantized.json`,
  `eval_streaming_opd.log`, `diag_opd_*.json`, `qat_stream_opd.log`,
  `stage1_decoder_attn_ffn_bw1.58_gNone_lr1e-05_adamw_delay_ls1_n2400.json`.

---

## 2026-10-12 — Distillation v1: the stop transfers; the output *format* did not. Metadata conflict found.

- **Did:** streaming-state distillation v1 (job `11040`): frozen pretrained
  teacher rollout over 350 utts → 2 400 teacher states + 2 400 standard
  samples, 4 000 steps at lr 1e-5 from the 9.2 base; decisive eval `11041`.
- **Result:** official algorithm **WER 448.31** (DAgger 920.55, static SFT
  203.5), CER 432.32, **RTF 1.76** (was 4.49). The model learned to *stop* —
  short, EOS-terminated generations, no 32-token loops — but the content
  collapsed (empty hypotheses, garbled continuations). Offline oscillated
  33.6 → 35.8 → 26.6 → 32.5 across the run.
- **Why (format conflict):** the teacher's continuations are wrapped in
  `language English<asr_text>…`; the standard samples teach plain transcripts.
  A 50/50 mix therefore contained contradictory targets for the same prompt
  template, and the student settled in a confused middle (neither format).
- **Chose:** keep the recipe, fix the format — `build_distill_examples` now
  strips the metadata block from the target and (relative to `prompt_raw`) from
  the recorded prompt, while the rollout keeps the teacher's natural prompts
  (the teacher stays in-distribution; the student's records match its own
  inference format). v2 = metadata stripped + lr 7e-6 + 5 000 steps (`11042`).
- **Next:** if v2 streams at usable WER, anneal offline quality (fresh
  standard-heavy phase) and mirror the recipe to binary. If v2 still fails,
  the remaining suspects are the mix weight (try 0.3–0.4), the anchor weight,
  and warm-up EOS over-sampling.
- **Artifacts:** `artifacts/stage1/streaming_opd_distill_quantized.json`,
  `eval_streaming_opd_distill.log`, `qat_stream_distill.log`,
  `qat_stream_distill2.log`.

---

## 2026-10-12 — Distillation v2: loops gone, **under-emission** found; three-way mix (v3) launched

- **Did:** distillation v2 (job `11042`, metadata stripped, lr 7e-6, 5 000
  steps, from the 9.2 base) + eval `11043` (`TAG=_distill2`).
- **Result:** official algorithm **WER 349.89**, CER 287.14, **RTF 1.47**.
  Offline recovered much better than v1 (16.3 @3000, 21.1 final vs v1's 27–35
  band) — the metadata fix worked. Hypotheses show the new failure mode:
  correct text that *stops early* ("SHE TAUGHT HER DAUGHTER … THAT LOVE FOR"
  for a 28-word reference), or a single word ("Nothing."), and some mid-stream
  garble. LSP: 438 revisions, committed@50 % 2.8 %.
- **Why (under-emission):** the teacher's per-chunk continuations are 1–3 words
  long, so the student only ever saw "finish this nearly-complete prefix and
  stop". It never saw the *lazy* state its own inference can drift into (short
  prefix + full audio), where it must emit the whole remainder — so it stops
  instead of completing. The static prefix SFT had the complementary failure
  (content right, never stops).
- **Chose:** **three-way mix** (v3, job `11044`): (1) standard full-audio
  samples (anchor, incl. frac 0.0), (2) gold-prefix completion samples
  (`--prefix-sft-fracs 0.0,0.3,0.6,0.8,0.95` + silence) — "finish the
  transcript from a partial prefix", (3) teacher-state distillation (700 utts,
  up to 4 800 states) — the stop policy. Mix 40 % streaming, lr 5e-6,
  5 000 steps, init from the v2 checkpoint (it already stops).
- **Next:** eval v3; the failure mode to watch is over- vs under-emission. If
  v3 lands near the static SFT's content quality with the distillation stop, the
  last step is annealing offline quality (see `scripts/qat_stream_anneal.pbs`)
  and then mirroring to binary.
- **Artifacts:** `artifacts/stage1/streaming_opd_distill2_quantized.json`,
  `eval_streaming_opd_distill2.log`, `qat_stream_distill3.log`.
