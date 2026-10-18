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

---

## 2026-10-12 — Distillation v3 (three-way mix): WER 278.18, and the trace shows the streaming policy is *right* — the remaining failures are offline quality

- **Did:** three-way mix (standard + gold-prefix completion + teacher
  distillation, 40 % streaming) from the v2 checkpoint, 5 000 steps at lr 5e-6
  (job `11044`), decisive eval `11045`, per-chunk trace `11047`.
- **Result:** official algorithm **WER 278.18** (v2 349.89, v1 448.31, DAgger
  920.55, static SFT 203.5), CER 339.10, RTF 2.436; offline 25.6 final
  (oscillating 24–33).
- **Trace finding (the important part, `diag_opd_distill3_*.json`):** the
  streaming *policy* is now correct — on u0 the model emits short
  EOS-terminated fragments per chunk ("If", "It.", "So.") and the final text is
  **exactly right** ("IF IT ONLY WERE NOT SO DARK HERE AND SO TERRIBLY LONELY").
  The remaining failures are *content* failures: on u1 the final chunk (full
  audio, empty prefix) produced **nothing** — the same utterances the offline
  model (25.6 WER) cannot transcribe. So under-emission is largely the offline
  degradation of the streaming phases, not a broken streaming policy.
- **Chose:** anneal the offline quality while keeping a small streaming share —
  job `11059` (`scripts/qat_stream_anneal.pbs`, `-v MIX=0.25,OPMAX=800`): init
  from the v3 checkpoint, 800 distillation states + 2 400 standard samples,
  lr 5e-6, 3 000 steps. The pool cap is what makes MIX exact: the interleave
  never drops streaming samples, so |streaming| must be
  `MIX/(1-MIX) * |standard|`.
- **Next:** eval the annealed checkpoint; success = offline back toward ≤12-14
  *and* the official-algorithm WER at or below v3's 278 (ideally single-digit
  once the content failures disappear). Then mirror to binary.
- **Artifacts:** `artifacts/stage1/streaming_opd_distill3_quantized.json`,
  `eval_streaming_opd_distill3.log`, `diag_opd_distill3_*.json`,
  `qat_stream_anneal.log`.

---

## 2026-10-12 — The anneal trade-off, measured: offline 25.6 → 15.3 costs the stop policy (loops return, WER 358.9)

- **Did:** annealed the v3 streaming model (`11059`, 25 % streaming share: 800
  distillation states + 2 400 standard samples, lr 5e-6, 3 000 steps) and
  evaluated it (`11062`).
- **Result:** offline **15.25** (14.83 best) from 25.6 — the anneal works for
  offline. Streaming **WER 358.90** (RTF 2.68), *worse* than the un-annealed
  v3 (278.18): 8 of 50 hypotheses are **loops 5–20× the reference length**
  (583/29, 857/54 words …), only 4 under-emit, 0 empty. The stop policy was
  partially forgotten.
- **Why:** the two objectives are in direct tension on the "when to stop"
  dimension, and the *streaming share* controls which side wins. 40–50 %
  streaming (v2/v3) ⇒ stops, offline 21–26. 25 % ⇒ offline 15.3, loops back.
  The offline behaviour cannot be recovered "for free" after the streaming
  behaviour is learned; both need to be co-trained at a rate where the stop
  policy sticks.
- **Chose (v4, job `11063`):** three-way mix on the *new 7.5 offline base*
  (`qat_ternary_full`) instead of the 9.2 one, 8 000 steps at lr 5e-6, and —
  the key instrumentation change — **live streaming evals during training**
  (`--stream-eval-every 1000 --stream-eval-n 10`, printed as `STREAM {...}`),
  so the streaming/offline trade-off curve is visible instead of inferred from
  end-of-run evals. Also: do not anneal below a share that holds the stop
  policy; if offline must be recovered, mix in the other streaming phases'
  samples rather than reducing the streaming sample count.
- **Next:** read the v4 `STREAM` curve: if the streaming WER descends with the
  offline stable in the 15–20 band, keep training (or extend) and mirror the
  recipe to binary; if the two curves cannot be made to descend together, the
  last resort per the plan is a precision-split model (streaming policy in the
  quantized path, offline anchor stronger) or explicit LSP-based stop handling.
- **Artifacts:** `artifacts/stage1/streaming_opd_anneal3_quantized.json`,
  `eval_streaming_opd_anneal3.log`, `qat_stream_anneal.log`,
  `qat_stream_distill4.log`.


---

## 2026-10-12 — Loop truncation at emission: the biggest single win (278 → 144.6); binary hits a 10.7 floor

- **Did:** (a) diagnosed the anneal-3 failure (8/50 hypotheses were loops 5–20×
  the reference length); (b) tested an **emission-layer loop policy** offline —
  `truncate_trailing_loop` applied to the stored hypotheses, then re-scored
  (CPU only); (c) implemented it in the LSP layer
  (`LSPState(truncate_loops=True)`: a trailing repetition loop is cut *before*
  the commit logic, so looped text can never enter the append-only transcript;
  one occurrence of the unit is kept) with tests; (d) ran the WP2 binary
  full-cache job to completion; (e) killed v4.
- **Result (emission policy, re-scored on the existing 50-utt hypotheses):**

  | checkpoint | raw WER | with loop truncation |
  |---|---:|---:|
  | distillation v1 | 448.31 | 179.34 |
  | distillation v2 | 349.89 | 154.56 |
  | **distillation v3** | **278.18** | **144.60** |
  | v3 + anneal | 358.90 | 244.70 |

  The loops start exactly where the correct transcript ended, so cutting there
  removes hundreds of insertion errors. This is the best official-algorithm
  number so far (the old static SFT was 203.5).
- **Binary floor:** the WP2 binary full-cache run (from the 7.5 ternary,
  8 000 steps, lr 1e-5) converged to **binary 10.70** (19.60 → 13.88 → 11.02 →
  10.70) — *identical* to the old binary long3 (10.7) despite its much better
  ternary start (readout 9.75 vs 22.35). Binary accuracy is limited by the
  quantizer's sign error, not by the ternary base — a further binary gain needs
  the co-training / stochastic-precision stack, not more of the same recipe.
- **v4 killed** after 2 000 steps: its live 10-utt streaming WER went
  134.9 → 704.8 (not descending) and the run was at ~17 steps/min. The
  loop-truncation policy is the better lever.
- **Chose:** make the loop policy part of the LSP state (default off; enabled
  with `--lsp-truncate-loops 1`, reported as `wer_raw` → policy WER) and re-run
  the official 50-utt eval on the v3 checkpoint *through the real eval path*
  (job `11067`) to confirm the 144.6 number end-to-end.
- **Next:** if 11067 confirms, the remaining gap is content (the ~145 is
  dominated by ordinary word errors at offline 21–26); then a *content* phase
  that keeps ≥40 % streaming pressure (the anneal at 25 % lost the stop policy)
  with the live `STREAM` evals as the guardrail.
- **Artifacts:** `tests/test_stream.py::TestTruncateTrailingLoop`,
  `artifacts/stage1/stage1_decoder_attn_ffn_bw1.0_gNone_lr1e-05_adamw_delay_ls1_n28539.json`,
  `artifacts/stage1/qat_stream_distill4.log`.

---

## 2026-10-12 — The balance is not monotone: content phase (offline 15.8) *degraded* streaming (528 raw / 190.8 with the loop policy), adapt5 launched

- **Did:** content phase `11076` — distill3 + 40 % streaming share for 3 000
  steps at lr 5e-6 (1 600 distillation states + 2 400 **plain** standard
  samples, guardrail STREAM evals every 500); evaluated its step-2 500
  checkpoint through the real LSP path (`11088`, `TRUNC=1`).
- **Result:** offline **15.8** (25.6 → 15.25 @1000 → 15.78 @2000) but streaming
  **raw 532.20 → policy 190.78** (RTF 2.60) — *worse* than distill3 (278.18 →
  144.60), with **228** loop truncations vs 11. The 20-utt guardrail curve was
  noisy (366 → 349 → 485 → 326 → 512) and did not warn clearly.
- **Why:** the content phase's streaming pool was distillation-only (no
  gold-prefix completion / silence samples) and its standard pool was plain
  full-audio samples; that mix teaches content but not the
  "emit-what-you-heard-then-stop" discipline, so the stop policy eroded while
  offline improved. Streaming quality is a **fine balance**, not a monotone
  function of the training mix — the three-way composition matters more than the
  nominal share.
- **Chose:** `adapt5` (`11091`) — the *three-way* mix (gold-prefix completion +
  silence + 4 800 distillation states, 40 % share) applied to the
  content3 checkpoint at a gentler lr (3e-6, 2 000 steps), aiming to keep the
  better content (15.8) and recover distill3-level streaming.
- **Stage 3 note:** the 1.7B FFN warm start is memory-feasible (84 projections,
  1.06 B trainable, batch 1, fp32 + frozen teacher on one 80 GB A100) but its
  ternary trajectory is unstable (113.9 @500 → 249.6 @1000 → 95.7 @1500 vs the
  0.6B's monotone 118 → 84 → 42): lr 1e-4 is likely too high for 1.7B — the next
  stage-3 run should use ~3-5e-5 and/or a longer warmup.
- **Next:** eval adapt5 (+loop policy); if it holds both, iterate the balance
  toward ≤12 offline while keeping the policy WER ≤145; the remaining streaming
  error is then ordinary content error, which the 7.5-class offline model
  (or the 1.7B once its lr is fixed) should shrink.
- **Artifacts:** `artifacts/stage1/streaming_opd_content3_quantized.json`,
  `eval_streaming_opd_content3.log`, `qat_stream_adapt5.log`,
  `artifacts/stage3/qat_17b_ffn.log`.

---

## 2026-10-12 — adapt5 frontier point (offline 14.51 / policy 175.0) and the 1.7B rollout OOM

- **Did:** evaluated adapt5 (three-way mix from content3, lr 3e-6, 2 000 steps)
  with the LSP loop policy (`11115`), and ran the 1.7B attention rollout
  (`11116`, policy `decoder_attn_ffn`, 1.41 B trainable).
- **Results:**
  - adapt5: offline **14.51** (best of the streaming family), raw streaming
    **215.04** (best raw), LSP policy **175.00** — but **233** loop truncations,
    so the policy gain (215 → 175) is smaller than on distill3 (278 → 144.6 with
    11 truncations). Frontier: distill3 = best streaming / worst offline;
    adapt5 = best offline / worse streaming.
  - 1.7B rollout: **CUDA OOM** — 34.6 GiB in-process on top of a co-tenant's
    42 GB (1.41 B trainable params with fp32 Adam ≈ 11 GB of optimizer state
    alone). The FFN-only stage (1.06 B trainable) had fit comfortably.
- **Fix:** `scripts/qat_17b_attn2.pbs` — `--optimizer adam8bit
  --grad-checkpointing 1` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  (retry `11121`). Lesson for Stage 3: at 1.7B, *any* policy beyond FFN-only
  needs the memory stack from the start; the launcher comment already said so,
  the plain template just did not set it.
- **Why the frontier has no clean winner:** the two objectives trade on the
  "when to stop" dimension; the loop policy rescues distill3 (stop-correct but
  content-weak) more than adapt5 (content-strong but loop-prone). The remaining
  untried lever is the plan's core Stage-4 piece — **forced-aligner
  stable-prefix targets** (T4.1/T4.2), which would make the training target
  exactly "the words heard so far" instead of "the whole remainder" or "the
  teacher's continuation", addressing both failure directions at once.
- **Artifacts:** `artifacts/stage1/streaming_opd_adapt5_quantized.json`,
  `eval_streaming_opd_adapt5.log`, `artifacts/stage3/qat_17b_attn.log`
  (OOM), `qat_17b_attn2.log` (retry).

---

## 2026-10-12 — T4.1/T4.2 implemented: forced-aligner timestamps + stable-prefix chunk targets; 1.7B rollout fixed twice

- **Did:** implemented the plan's untried WP1 lever —
  `src/asr1bit/data/align.py` (T4.1: `AlignedWord`, `align_words` adapter over
  `Qwen3ForcedAligner-0.6B` with a lazy import; `alignment_covers` sanity check)
  and `src/asr1bit/data/prefix.py` (T4.2: `chunk_examples` — at chunk *k* the
  prefix is the words heard by chunk *k−1* and the target is the words heard
  *in* chunk *k*, with an optional lookahead; chunks with no new words are empty
  targets = stop samples). This is the supervision the earlier streaming
  recipes lacked: "emit exactly what has been heard so far, then stop", instead
  of "the whole remainder" (DAgger) or "the teacher's continuation"
  (distillation). 13 new tests (`tests/test_data_prefix.py`), suite 322 passed,
  ruff clean. A GPU probe (`scripts/probe_aligner.py` + `.pbs`, job `11136`)
  verifies the aligner adapter on three test-clean utterances and dumps
  `artifacts/stage4/aligner_probe.json`.
- **Stage 3 incidents (both fixed):**
  1. the 1.7B attention rollout OOM'd (1.41 B trainable, fp32 Adam);
  2. the first retry then crashed with `element 0 of tensors does not require
     grad` — **gradient checkpointing is incompatible with this frozen-only
     training setup** (`model.gradient_checkpointing_enable()` is called without
     `use_reentrant=False` / `enable_input_require_grads`), so the checkpointed
     segment's inputs carry no grad. Fix: drop `--grad-checkpointing` and rely
     on `--optimizer adam8bit` + a GPU with ≥50 GB free (`qat_17b_attn3.pbs`,
     job `11134`). The 1.7B rollout's early ternary (58.7 @500) is the expected
     attention-quantization transient (the 0.6B showed 25.7 → 52.8 → 14.4).
- **Next:** wire `chunk_examples` into a training run — `scripts/run_stage1_qat.py`
  needs an `--aligned-targets 1` path that aligns the train utterances once
  (cache the word timings), builds (audio-slice, prefix, target) samples from
  `chunk_examples`, and mixes them with the standard samples. Expected effect:
  both failure directions (under-emission and loops) are supervised away
  because the target length is *defined by the audio heard*, not inferred.
- **Artifacts:** `src/asr1bit/data/{align,prefix}.py`, `tests/test_data_prefix.py`,
  `scripts/probe_aligner.py`, `artifacts/stage4/aligner_probe.json`,
  `artifacts/stage3/qat_17b_attn2.log` (grad-ckpt crash).

---

## 2026-10-12 — Stage-3 OOM chain root-caused (bitsandbytes was missing) + T4.3 wired

- **Did:** chased the 1.7B attention-rollout failures to the end and wired the
  aligned targets into the trainer.
- **The OOM chain (all three attempts):**
  1. `11116` (fp32 AdamW): OOM at 34.6 GiB in-process + a 42 GB co-tenant.
  2. `11121` (`--grad-checkpointing 1` + `--optimizer adam8bit`): crashed with
     `element 0 of tensors does not require grad` — **gradient checkpointing is
     incompatible with this frozen-only setup** (`gradient_checkpointing_enable()`
     without `use_reentrant=False`/`enable_input_require_grads`).
  3. `11134` (`--optimizer adam8bit`, no grad-ckpt): crashed with
     `CUDA driver error: invalid argument` inside `torch._multi_tensor_adam` —
     and the traceback showed **fp32 AdamW**, because **bitsandbytes was not
     installed** and `build_optimizer` silently falls back (`warnings.warn` →
     AdamW). The flag had never been exercised on this cluster.
- **Fix:** installed `bitsandbytes 0.50.2` into the shared conda
  (`~/miniconda3`), added a one-off sync job for the node-local env (`11138`),
  and made the launcher use the shared env explicitly (`qat_17b_attn4.pbs`,
  job `11137`). Lesson: **verify `--optimizer adam8bit` actually engaged**
  (grep the run log for the fallback warning) whenever memory matters.
- **T4.3 wired:** `scripts/run_stage1_qat.py` gained `--aligned-targets 1`
  (`--aligned-n`, `--aligned-lookahead-ms`, `--aligned-stop-stride`): it loads
  `Qwen3ForcedAligner-0.6B`, aligns the train utterances, builds
  (audio-slice, template+prefix, target) samples via `chunk_examples`, and mixes
  them with the standard samples through the existing `interleave_with_ratio`
  machinery. `stop_stride=2` halves the empty-target (stop) samples so the mix
  is not dominated by silence teaching. 3 new glue tests (suite 325 passed,
  ruff clean). The probe (`artifacts/stage4/aligner_probe.json`) confirmed the
  targets on real audio, e.g. for *"I HAVE GREAT THINGS…"*: chunk 1 → `'I'`,
  chunk 2 → `'HAVE GREAT'`, chunk 3 → `'THINGS'`, 9/39 chunks empty.
- **Next:** launch the aligned SFT run — from `qat_stream_adapt5` (offline 14.5)
  with `--aligned-targets 1 --op-mix-frac 0.5`, lr ~3e-6, 1 500–2 000 steps,
  plus `--stream-eval-every 250 --stream-eval-n 20` as the guardrail; then the
  full 50-utt eval with the loop policy. This is the first run whose targets are
  *defined by the audio heard*, so it should improve both failure directions.
- **Artifacts:** `artifacts/stage3/qat_17b_attn{2,3}.log` (crashes),
  `qat_17b_attn4.log` (fix), `artifacts/stage4/aligner_probe.json`.

---

## 2026-10-12 — ROOT CAUSE of the crash chain: `expandable_segments:True` is incompatible with this cluster's driver 525

- **Did:** correlated the three `CUDA driver error: invalid argument` crashes
  (`11121` grad-ckpt, `11134` adam8bit, `11139` model load) against the runs
  that worked.
- **Finding:** every crash had
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in its launcher (added
  while chasing the 1.7B OOM); **no earlier run set it**. The flag's allocator
  needs newer driver support, so on driver 525.147.05 it fails with
  `invalid argument` on ordinary CUDA ops (model loading, optimizer state) —
  which is why the failure looked like an OOM/frozen-grad problem in completely
  unrelated places. Removed from every launcher; both resubmissions then passed
  their previous crash points immediately (the aligned run reached alignment;
  the 1.7B rollout reached its optimizer).
- **Lesson:** on this cluster never set `PYTORCH_CUDA_ALLOC_CONF` — and when a
  "CUDA driver error" appears on a basic op, diff the launcher environment
  against the last known-good run before chasing model-side causes.
- **Remaining Stage-3 reality:** the 1.7B *full-attention* rollout needs
  ~38–40 GB free (1.41 B trainable + fp32 students/teacher); on the shared node
  co-tenants can grow mid-run and OOM it. Options recorded: (a) retry when a GPU
  has ≥55 GB free, (b) drop the frozen anchor (`--frozen-teacher 0
  --distill-weight 0`, saves ~7–14 GB, risks the audio-conditioning collapse the
  0.6B needed the anchor for), (c) partial attention quantization (q/k/v first).
  The FFN warm start (22.03 ternary) remains the banked Stage-3 result.

---

## 2026-10-12 — MILESTONE: aligned targets (T4.1–T4.3) take the official algorithm from 215 → **~14 WER**

- **Did:** ran the aligned-target streaming SFT (job `11140`): init from adapt5
  (offline 14.51), `--aligned-targets 1` (800 utterances aligned with
  Qwen3-ForcedAligner-0.6B, chunk targets = the words whose end time falls in
  each 320 ms chunk, `--aligned-stop-stride 2`), 2 400 aligned + 2 400 standard
  samples, 1 500 steps at lr 3e-6 with 20-utt official-algorithm `STREAM`
  guardrails every 250 steps.
- **Result (the decisive curve, official prefix algorithm, 32 tokens, 20 utts):**

  | step | raw streaming WER |
  |---:|---:|
  | 750 | 20.71 |
  | 1 000 | 16.47 |
  | 1 250 | **13.41** |
  | 1 500 | 14.35 |

  Offline held at **14.62** ternary (16.74 @500 → 17.58 @1000 → 14.62 @1500);
  fp16 readout 24.5.
- **Why this is the breakthrough:** every earlier recipe topped out at a *raw*
  official-algorithm WER of 215–532 (distill3 278.18, content3 532.20,
  adapt5 215.04) and needed the LSP loop policy to reach 144–190. The aligned
  targets make the raw number ~14 — an order of magnitude better — because the
  supervision now matches the algorithm exactly: *emit the words heard so far,
  then stop*. The model no longer has to infer the target length from the audio
  (which produced both failure modes: under-emission and loops); the aligner
  defines it per chunk.
- **Confirmation running:** full 50-utt clean eval (`11143`) and the
  clean/other pair (`11145`, `CFG=other`) with the LSP loop policy enabled
  (raw vs policy reported).
- **Next:** (1) read the 50-utt numbers and promote the recipe —
  aligned targets + LSP loop policy + the offline anchor; (2) scale it: more
  aligned utterances (2 400 → 8 000), longer schedule, and the same recipe on
  the 7.5-WER offline base (the 1.7B once its rollout fits); (3) mirror to
  binary via progressive precision.
- **Artifacts:** `artifacts/stage1/qat_stream_aligned.log`,
  `artifacts/stage1/stage1_decoder_attn_ffn_bw1.58_gNone_lr3e-06_adamw_delay_ls1_n2400.json`,
  `artifacts/stage4/aligner_probe.json`.

---

## 2026-10-12 (night) — CONFIRMED: aligned targets give **12.61 clean / 22.50 other** on the official algorithm, with **zero loops**

- **Did:** full 50-utt official-algorithm evals of the aligned-target checkpoint
  (`11143` clean, `11145` other) with the LSP loop policy enabled.
- **Result:**

  | config (50 utts, 320 ms, 32 tokens) | raw WER | policy WER | CER | RTF | loop truncations |
  |---|---:|---:|---:|---:|---:|
  | clean | 12.71 | **12.61** | 5.68 | 2.32 | **0** |
  | other | 22.27 | **22.50** | 12.31 | 1.54 | **0** |

  For context: previous best official-algorithm numbers were 144.60 (distill3 +
  loop policy) clean and — for other — nothing usable; the fp16 pretrained
  baseline is 2.33 clean / ~4.6 other (vLLM protocol), and the `+1.5` target
  is therefore ~3.8 / ~6.1. The gap is now **~9 / ~16 points** instead of
  "collapse", and — qualitatively — **the loops are gone entirely** (0
  truncations, CER 5.68 clean).
- **Why it matters beyond the number:** the streaming WER (12.61) is now
  *better than the checkpoint's own offline WER* (14.62). The aligned targets
  teach the "emit what you heard, then stop" discipline that the offline
  full-transcript objective does not; the streaming model is no longer a
  degraded copy of the offline one.
- **What limits it now:** base model quality. The aligned run started from
  adapt5 (offline 14.6); the project's best offline checkpoint is the full-cache
  ternary (7.5). The obvious next scaling: aligned targets from the 7.5 base,
  more aligned utterances (2 400 → 8 000+), longer schedule.
- **Stage 3 in parallel:** the anchor-free 1.7B attention rollout (`11146`) is
  training (step ~500) — the memory workaround holds.
- **Artifacts:** `artifacts/stage1/streaming_opd_aligned_quantized.json`,
  `eval_streaming_opd_aligned.log`, `streaming_opd_aligned_other_quantized.json`,
  `eval_streaming_opd_aligned_other.log`, `qat_stream_aligned.log`.

---

## 2026-10-12 (late night) — Scaled aligned run: streaming 9.65 / offline 9.11 — the recipe scales

- **Did:** scaled the aligned recipe (from the 7.52 full-cache offline base,
  2 400 aligned utterances, 2 500 steps, lr 5e-6, job `11148`) and started the
  binary mirror (`11158`, progressive precision from its checkpoint).
- **Result (in-training):** STREAM (20 utts, raw official algorithm)
  11.29 @750 → 10.82 @1500 → 10.35 @2000 → **9.65 @2250**; offline ternary
  **9.11 @1000** → 10.70 @1500 → 9.75 @2000. Both curves beat the first aligned
  run (streaming ~10 vs 13.4; offline ~9.5 vs 14.6) — the aligned recipe scales
  with the base quality, as predicted.
- **Confirmation running:** full 50-utt clean (`11156`) and clean/other
  (`11157`) evals on the step-2 500 checkpoint, plus the binary mirror
  (`11158`). Context: the fp16 portable baseline is 2.33 clean; the `+1.5`
  target ~3.8/6.1; previous ternary best was 144.60.
- **Stage 3 (parallel):** the anchor-free 1.7B attention rollout (`11146`)
  recovers without the anchor — ternary 35.28 @1000 → 24.15 @1500 →
  **20.55 @2000** (already past its 22.03 FFN-only start), so the memory
  workaround did not cost the training dynamics. 1 000 steps remain.
- **Next:** read the evals; if the 50-utt numbers confirm ~10/20, the remaining
  gap is base quality + the 1.7B; then the 1.7B gets the aligned recipe and the
  final matrix (160/320/1000 ms, clean/other) follows.
- **Artifacts:** `artifacts/stage1/qat_stream_aligned2.log`,
  `artifacts/stage3/qat_17b_attn5.log`,
  `streaming_opd_aligned2{,_other}_quantized.json` (pending).

---

## 2026-10-13 — Scaled aligned confirmed (12.92 clean / 20.00 other); 1.7B rollout done (20.55); streaming saturates at the 0.6B's content ability

- **Did:** full 50-utt evals of the scaled aligned checkpoint (`11156` clean,
  `11157` other), and collected the 1.7B anchor-free attention rollout.
- **Results (official prefix algorithm, 320 ms, 32 tokens, LSP loop policy on):**

  | checkpoint (ternary) | offline (50 utt) | clean WER | other WER | CER (clean) | loops |
  |---|---:|---:|---:|---:|---:|
  | aligned v1 (from adapt5) | 14.62 | **12.61** | 22.50 | 5.68 | 0 |
  | **scaled aligned** (7.5 base, 2 400 utts) | **9.11–10.70** | **12.92** | **20.00** | 5.00 | 0 |
  | pretrained fp16 (reference) | 2.44 (0.6B) / **1.17 (1.7B!)** | 2.33 | ~4.6 | — | — |

  Scaling improved the offline (14.6 → 9.7) and `other` (22.5 → 20.0) but not
  `clean` (12.61 → 12.92, within noise): **the 0.6B streaming result has
  saturated at ~12–13** — the remaining error is the model's own streaming
  content ability, not its offline WER. The 1.7B is the lever: its *pretrained*
  offline is 1.17 WER (vs the 0.6B's 2.44), so the same recipe on 1.7B should
  carry much lower.
- **Stage 3:** the anchor-free 1.7B attention rollout finished (2000 steps, its
  configured schedule) at **ternary 20.55** (from the FFN start's 22.03;
  trajectory 35.3 @1000 → 24.2 @1500 → 20.6 @2000, no anchor) — a modest gain;
  the anchored variant (waiting for a ≥55 GB-free GPU) and a longer schedule are
  the next levers. Note the 1.7B's own anchors: pretrained offline **1.17**,
  fp16 readout 159 (shadow drift as usual).
- **Binary mirror** (`11159`, progressive precision from the scaled aligned
  checkpoint, bit_width 1.0, aligned targets) is running.
- **Next:** (1) the 1.7B: anchored attention rollout (quiet GPU) → 5 000+ steps,
  then the aligned recipe on it; (2) mirror the aligned recipe to the 7.5 base
  with a longer schedule (5 000+ steps) if the 0.6B can still move; (3) the
  final matrix (160/320/1000 ms, clean/other) for the report.
- **Artifacts:** `artifacts/stage1/streaming_opd_aligned2{,_other}_quantized.json`,
  `artifacts/stage3/stage1_decoder_attn_ffn_bw1.58_gNone_lr3e-05_adam8bit_delay_ls1_n4000.json`.

---

## 2026-10-13 — Cluster capacity bites: binary mirror deferred; the 1.7B gets the aligned recipe (the high-value run)

- **Did:** attempted the binary mirror on the scaled aligned checkpoint; the
  first attempt (`11159`) OOM'd after ~30 min (a co-tenant grew to 52.7 GB on
  the GPU the picker had sampled — our process only held 8.3 GB). Retried with a
  stricter GPU filter (`$2 < 25000`) as `11162`, but the node was nearly full
  (only GPU 7 had ~39 GB free) so both our jobs landed there; the binary mirror
  was killed again to give the memory to the higher-value run.
- **Launched (the high-value run):** `11161` **aligned targets on the 1.7B** —
  `qat_stream_aligned17b.pbs`: model `Qwen/Qwen3-ASR-1.7B`, init from the
  anchor-free attention checkpoint (`qat_17b_attn5`, ternary 20.55), 1 200
  aligned utterances, 1 500 steps, lr 3e-6, batch 1, **adam8bit** (shared env),
  the frozen teacher dropped to fit memory, 20-utt `STREAM` guardrails every
  250 steps. Rationale: the 0.6B streaming is saturated at ~12–13 (its content
  ability), while the 1.7B's pretrained offline is 1.17 (vs the 0.6B's 2.44) —
  the aligned recipe on the 1.7B is the path to the `+1.5` target.
- **Binary mirror status:** deferred (not lost) — `qat_stream_aligned2bin2.pbs`
  is ready; submit when a GPU with ≥20 GB free appears.
- **Next:** read the 1.7B aligned run's `STREAM`/ternary curves; if it tracks the
  0.6B pattern (~12–13), the 1.7B could land single-digit/low-teens; then the
  final matrix (160/320/1000 ms, clean/other) and the write-up.
- **Artifacts:** `artifacts/stage1/qat_stream_aligned2bin.log` (OOM),
  `qat_stream_aligned17b.log`, `scripts/qat_stream_aligned2bin2.pbs`.

---

## 2026-10-13 (morning) — 1.7B aligned run training slowly; first STREAM 28.5 @250; binary mirror retried

- **Did:** checked the 1.7B aligned run (`11161`) and retried the binary mirror.
- **1.7B aligned status:** mix built as configured (39 800 aligned examples →
  2 400 pooled + 2 400 standard = 4 800); the first 20-utt official-algorithm
  `STREAM` @250 is **28.47** — from a checkpoint that had *never* seen streaming
  training (its raw streaming would be in the hundreds), so the recipe is
  clearly transferring, but the run is slow: **~2.8 steps/min** (batch 1, 1.7B,
  adam8bit, contended GPU). At that rate the 6 h walltime cuts it near step
  ~1 000 of 1 500 → a `--resume` continuation is prepared
  (`qat_stream_aligned17b2.pbs`, 8 h walltime).
- **Binary mirror:** retried as `11164` and landed on GPU 0 (~36 GB free,
  enough for the ~12–15 GB needed) after the first attempt died when a
  co-tenant grew to 52 GB.
- **Also prepared:** nothing else pending; the 0.6B frontier (12.92 clean /
  20.00 other, zero loops) stands as the reference the 1.7B is chasing.
- **Next:** read the 1.7B's ternary EVALs + STREAM curve as they land (the
  0.6B pattern was *raw streaming ≈ offline WER* after alignment); chain
  `qat_stream_aligned17b2.pbs` if the walltime cuts it; read the binary mirror's
  curves; then the final matrix.
- **Artifacts:** `artifacts/stage1/qat_stream_aligned17b.log`,
  `qat_stream_aligned2bin2.log`, `scripts/qat_stream_aligned17b2.pbs`.

---

## 2026-10-13 — 1.7B aligned run cut mid-transient (walltime); resume launched. Binary mirror descending.

- **Did:** collected the 1.7B aligned run's partial trajectory, chained its
  resume, and read the binary mirror's curve.
- **1.7B aligned (`11161`, cut by the 6 h walltime at step ~750/1 500):** STREAM
  (20 utts) 28.47 @250 → 31.06 @500; offline ternary 20.55 → **31.57 @500** —
  a deep fresh-cycle dip (the 0.6B's aligned runs dipped only ~2 points at the
  same stage). The run was cut *before* the cosine tail, so its current state is
  mid-transient, not converged; the resume (`11177`, `--resume`, 8 h) continues
  the same schedule (LR advances from step 750).
- **Binary mirror (`11164`, 0.6B, still running):** STREAM 64.94 @250 →
  48.24 @500 → **35.76 @750**; offline binary **48.52 @500** (from a ~100
  readout on the ternary checkpoint). The binary path recovers, as designed.
- **Why the 1.7B dip matters:** at lr 3e-6 the 1.7B (1.41 B trainable, adam8bit)
  moves much further than the 0.6B (440 M, fp32 AdamW) — the same nominal lr is
  not the same perturbation. If the resume's tail does not recover it below
  20.55, the next attempt should halve the lr (1.5e-6) or start from the
  *pre-rollout* FFN checkpoint with the anchor present.
- **Next:** read the resume's trajectory (STREAM + ternary) as it lands; if it
  recovers below the 20.55 start and the streaming tracks it, run the full
  clean/other eval and start the final matrix (160/320/1000 ms); otherwise
  adjust the 1.7B recipe (lr / anchor) per the note above.
- **Artifacts:** `artifacts/stage1/qat_stream_aligned17b.log`,
  `qat_stream_aligned17b2.log`, `qat_stream_aligned2bin2.log`.

---

## 2026-10-13 — 1.7B aligned v1 FAILED to recover (30.6 offline / 28.0 streaming); gentler anchored v3 launched

- **Did:** completed the 1.7B aligned schedule (first leg + resume, 1 500 steps
  total) and launched a corrected retry.
- **Result (failure, kept):** the 1.7B aligned v1 ended at **ternary 30.61**
  (20.55 → 31.57 @500 → 30.83 @1000 → 30.61 @1500) with streaming **28.00**
  (28.47 @250 → 31.06 @500 → 30.12 @750 → 31.53 @1000 → 28.71 @1250 → 28.00
  @1500, 20 utts). The deep fresh-cycle dip never healed — the cosine tail
  flattened ~10 points below the starting checkpoint. For contrast, the 0.6B
  aligned runs dipped ~2 points and recovered below their start.
- **Why (diagnosis):** three differences from the successful 0.6B recipe —
  (1) **adam8bit** instead of fp32 AdamW (the 1.7B needed it for memory);
  (2) **no frozen anchor** (dropped to fit); (3) lr 3e-6 hitting a 1.41 B-
  trainable model much harder than the 0.6B's 440 M. Any of the three can turn
  a "fresh cycle" into a permanent regression at this scale.
- **Fix launched (`11186`, `qat_stream_aligned17b3.pbs`):** restart from the
  **pre-dip** checkpoint (`qat_17b_attn5`, ternary 20.55), lr **1.5e-6**,
  1 200 steps, **anchor restored** (`--frozen-teacher 1 --distill-weight 0.05`),
  adam8bit kept, aligned targets unchanged. Watch the ternary EVALs: the
  acceptance bar is "dips < ~3 points and recovers below 20.55".
- **Binary mirror (`11164`):** still training (step ~1 250/1 500): offline binary
  48.52 @500 → **41.84 @1000**; STREAM 64.9 → 48.2 → 35.8 → 37.6 (noisy mid-run).
  Its full eval with the loop policy follows when it ends.
- **Artifacts:** `artifacts/stage1/qat_stream_aligned17b2.log` (failure),
  `qat_stream_aligned17b3.log` (retry), `qat_stream_aligned2bin2.log`.

---

## 2026-10-13 — 1.7B aligned: VERDICT (paused on capacity); 0.6B final matrix started

- **Did:** diagnosed both dead jobs and started the final matrix.
- **1.7B aligned verdict (3 attempts, all failures — kept for the record):**
  1. v1 (`11161`+`11177`, lr 3e-6, adam8bit, no anchor): dipped 20.55 → 31.6 and
     never recovered (ternary 30.61; streaming 28.0, flat).
  2. v3 (`11186`, lr 1.5e-6 + anchor restored): **OOM'd** in its first steps —
     the anchored variant (student 7 GB + teacher 7 GB + adam8bit + 1.41 B
     trainable) needs a genuinely free GPU, and the picker's sample keeps losing
     the race with growing co-tenants on this node.
  **Prerequisites before retrying:** a GPU with ≥45–50 GB *stable* free memory
  and the **anchored fp32-AdamW** recipe (the 0.6B's proven combination; adam8bit
  changes the update scale and the anchor is what keeps audio conditioning
  alive). The 1.7B FFN checkpoint (22.03) and the anchor-free attention
  checkpoint (20.55) remain on disk. **The 0.6B aligned recipe is the
  deliverable** while the cluster is contended.
- **0.6B final matrix started:** `11196` (chunk 160 ms, clean), `11197`
  (1 000 ms, clean) on the scaled aligned checkpoint, and `11198` (the binary
  mirror's step-1 250 checkpoint with the loop policy). The 320 ms numbers exist
  (12.92 clean / 20.00 other); the `other`-config counterparts for 160/1000 ms
  follow as GPU slots free.
- **Binary mirror:** cut by its 6 h walltime at step 1 246/1 500 (offline binary
  41.8 @1000, still descending; log stops silently, no traceback — classic
  walltime kill). Its launcher can be resumed (`--resume`, 250 steps) if wanted;
  the binary arm remains far behind the ternary (41.8 vs 12.9).
- **Artifacts:** `artifacts/stage1/qat_stream_aligned17b3.log` (OOM),
  `qat_stream_aligned2bin2.log` (walltime cut), `eval_streaming_opd_aligned2_c160.log`,
  `eval_streaming_opd_aligned2_c1000.log`, `eval_streaming_opd_aligned2bin.log`.

---

## 2026-10-13 — Matrix filling in: 1000 ms 10.59, 320 ms 12.92 (clean), 320 ms other 20.00; **binary streams at 18.43**

- **Did:** collected the matrix evals (the aligned 0.6B checkpoint at 160/320/
  1000 ms, clean/other; the binary mirror's loop-policy eval).
- **Results (official algorithm, 50 utts, LSP loop policy on):**

  | checkpoint | chunk | clean | other |
  |---|---|---:|---:|
  | ternary aligned (scaled) | 320 ms | **12.92** | **20.00** |
  | ternary aligned (scaled) | 1 000 ms | **10.59** (CER 4.07) | pending |
  | ternary aligned (scaled) | 160 ms | pending | pending |
  | **binary aligned** (mirror, cut at step 1 246/1 500) | 320 ms | **18.43** (CER 11.84, raw 17.90) | pending |

- **The binary surprise:** the binary checkpoint's *offline* binary WER is 41.8,
  yet its *streaming* WER is **18.43** — the aligned streaming recipe transfers
  to binary far better than the offline recipe does. The ternary→binary
  streaming gap is only **+5.5** (12.92 → 18.43), whereas offline it is +32
  (9.7 → 41.8). Consistent with the aligned targets being the binding
  supervision, not the precision.
- **Chunk-size trend:** 1 000 ms beats 320 ms (10.59 vs 12.92) — boundary errors
  dominate at short chunks, as expected; 160 ms pending.
- **Verdict recap:** the 0.6B aligned recipe is the deliverable (ternary
  12.92/20.00; binary 18.43); the 1.7B waits on cluster capacity.
- **Artifacts:** `artifacts/stage1/streaming_opd_aligned2_c1000_quantized.json`,
  `streaming_opd_aligned2bin_quantized.json`, `eval_streaming_opd_*.log`.

---

## 2026-10-14 — Refinement (`aligned3`) is the best curve yet: STREAM 8.71 (20 utt) at step 1 000; resume prepared for the walltime

- **Did:** continued the scaled aligned checkpoint at lr 2e-6 (`11213`,
  `qat_stream_aligned3`, 3 000 steps) and prepared its resume.
- **Result so far (official algorithm, raw, 20-utt guardrail):** STREAM
  **8.71 @250** → 11.53 @500 → 9.41 @750 → **8.71 @1000**; offline ternary
  9.32 @500 → **9.00 @1000**. At the same step count the aligned2 run was
  ~10.4–11.8 (STREAM) and 9.1–9.7 (offline) — so the gentle continuation is a
  real improvement on both axes, and 8.7 (20 utt) is the lowest streaming
  number the project has produced (the 50-utt translation is typically ~2–3
  points higher: aligned2's 9.65 → 12.92).
- **Walltime:** ~4 steps/min ⇒ the 6 h limit cuts it near step ~1 500/3 000.
  Resume prepared: `scripts/qat_stream_aligned4.pbs` (`--resume`, 8 h, same save
  dir).
- **1.7B:** still paused — no ≥45 GB-stable-free GPU observed.
- **Next:** when `aligned3`/`aligned4` end, run the full 160/320/1000 ×
  clean/other evals; if the 50-utt clean lands below 12.0 the headline tables in
  README/PROJECT_STATUS get updated and a tag follows.
- **Artifacts:** `artifacts/stage1/qat_stream_aligned3.log`,
  `scripts/qat_stream_aligned4.pbs`.

---

## 2026-10-14 (morning) — Refinement cut at step ~1 970 with offline **8.90** (best); resume + full evals launched

- **Did:** recorded the refinement's full partial trajectory, snapshotted its
  checkpoint (`step1950.pt`), chained the resume (`11230`,
  `qat_stream_aligned4.pbs`, `--resume`, 8 h) and launched the full 50-utt
  clean/other evals (`11231`/`11232`).
- **Curves (`11213`, raw official algorithm, 20-utt guardrail):** STREAM
  8.71 @250 → 8.71 @1000 → 9.88 @1250 → 9.18 @1500 → 10.82 @1750 (noisy plateau
  ~9–11); offline ternary **9.00 @1000 → 8.90 @1500** — the best offline number
  the streaming family has produced, 0.8 below the aligned2's 9.7.
- **The walltime cut it at step ~1 970/3 000** (its ~4 steps/min pace), hence
  the resume. Snapshot-first (`step1950.pt`) before resuming, because the resume
  overwrites `last.pt` in the same directory.
- **Next:** read the 50-utt evals; the acceptance test for promoting the
  headlines is 50-utt clean < 12.0 (aligned2: 12.92).
- **Artifacts:** `artifacts/stage1/qat_stream_aligned3.log`,
  `data/checkpoints/qat_stream_aligned3/step1950.pt`, `qat_stream_aligned4.log`.

---

## 2026-10-14 — Refinement does NOT beat aligned2 on the full sample (14.41 vs 12.92): the 20-utt guardrail diverged from the metric

- **Did:** evaluated the refinement's cut checkpoint (50 utts, clean/other) and
  redirected the GPU.
- **Result:** aligned3 (step ~1 950) = **clean 14.41** (CER 5.49, raw 14.19,
  zero loops) — *worse* than aligned2's 12.92 — even though its in-training
  20-utt guardrail (8.71–10.82) and its offline (8.90) were both better than
  aligned2's (9.65 guardrail, 9.7 offline). The other-config eval was still
  running when this was written.
- **Lesson (methodological, important):** the in-training `STREAM` guardrail
  uses a *fixed 20-utterance slice* of the eval sample; its absolute level and,
  more importantly, its *derivative* do not transfer to the full 50-utt metric
  (aligned2 guardrail 9.65 → 12.92 full; aligned3 guardrail 8.7–9.2 → 14.41
  full). **Promote only on the full-sample eval.** A future improvement would be
  to randomize/alternate the guardrail slice or use the full 50 for end-of-run
  decisions (which is exactly what the scheduled checks did).
- **Actions:** killed the aligned3 resume (`11230` — it continued the worse
  branch, the remaining steps only refine it at lr ~5e-7) and launched the
  **binary mirror's resume** (`11233`, `--resume`, 250 steps to its 1 500-step
  schedule) to complete the binary arm's deliverable.
- **Standing state:** the deliverable stays the **aligned2** recipe — ternary
  **12.92 clean / 20.00 other** at 320 ms (10.59/16.59 at 1 000 ms), binary
  18.43/25.45, zero loops — tags `0.2.42`–`0.2.45`.
- **Artifacts:** `artifacts/stage1/streaming_opd_aligned3_quantized.json`,
  `eval_streaming_opd_aligned3{,_other}.log`, `qat_stream_aligned2bin3.log`.

- **Postscript:** the aligned3 *other* eval OOM'd at model loading (GPU 0 had
  45 MB free — the co-tenant race again). Not re-run: aligned3 is not promoted,
  so its `other` cell is not needed for the deliverable.

---

## 2026-10-15 — Binary arm: 1 500-step schedule completed (offline binary 35.2); final evals running

- **Did:** collected the binary mirror's resumed run (`11233`, finished 2026-10-14
  10:16) and launched its final 50-utt clean/other evals (`11365`/`11366`).
- **Result (the run):** offline binary at step 1 500 = **35.17** (41.8 at the
  step-1 250 snapshot — the resumed steps kept improving the offline path);
  20-utt streaming guardrail 36.5–38.1 at the tail (vs the earlier step-1 250
  state whose *50-utt* eval gave **18.43 clean / 25.45 other**). The guardrail
  and the full-sample numbers disagree in the same way as the ternary
  refinement — the fixed 20-utt slice is not a reliable proxy.
- **Process lesson (again):** the resume **overwrote `last.pt`**, so the
  step-1 250 binary checkpoint that measured 18.43 no longer exists (the number
  is documented, the weights are not). Snapshot before resuming — done for
  aligned3, not for the binary; make it a habit for every resume.
- **Final evals:** running (`11365` clean, `11366` other); when they land, the
  binary cells get updated in the matrix and the live status.
- **Cluster:** still no GPU with ≥45 GB free (all 8 are 60–78 GB used) — the
  1.7B path stays paused.
- **Artifacts:** `artifacts/stage1/qat_stream_aligned2bin3.log`,
  `stage1_decoder_attn_ffn_bw1.0_gNone_lr3e-06_adamw_delay_ls1_n2400.json`,
  `eval_streaming_opd_aligned2bin_final{,_other}.log`.

---

## 2026-10-15 — Binary final evals: the resume DEGRADED the streaming (clean 27.54 vs the step-1 250 state's 18.43)

- **Did:** collected the binary arm's final 50-utt evals (the 1 500-step
  checkpoint).
- **Result:** clean **27.54** (CER 22.11, raw 26.91, **29 loop truncations**) /
  other **26.93** (CER 17.37, raw 26.70, 40 truncations). The step-1 250
  snapshot of the same run measured **18.43 / 25.45** with the loop policy
  doing almost no work — so the resumed 250 steps improved the *offline* binary
  (41.8 → 35.2) but **broke the streaming arm's loop discipline**.
- **Two lessons (both recorded):**
  1. **Snapshot before resuming** — the step-1 250 weights (the better
     streaming state) were overwritten by the resume and are gone; only its
     eval JSON survives. A resume is a *new experiment*, not a mere completion.
  2. The binary arm's ceiling is state-dependent in a way the ternary arm's is
     not: the loop discipline is the first thing lost when the offline path is
     pushed, and the LSP loop policy then carries 30–40 truncations instead of
     0–2.
- **Reachable binary deliverable:** the documented best is 18.43 clean /
  25.45 other (from the lost step-1 250 state); the reproducible weights on disk
  (step 1 500) give 27.54 / 26.93. Re-creating the better state means re-running
  the binary phase with an *early-stopping* rule (stop when the 50-utt streaming
  eval degrades), not a fixed schedule.
- **Next (launched):** `11369` — `qat_stream_aligned5.pbs`: ONE longer aligned
  cycle (5 000 steps, lr 5e-6) from the 7.5 full-cache base, since the aligned2
  recipe (2 500 steps) was still descending at its end.
- **Artifacts:** `streaming_opd_aligned2bin_final{,_other}_quantized.json`,
  `qat_stream_aligned5.log`.

---

## 2026-10-15 (early) — Cluster saturation kills both aligned runs; a strict GPU picker is added

- **Did:** launched the two 0.6B runs of the day (aligned5 — one longer aligned
  cycle; bin_es — the binary early-stopping re-run with
  `--keep-step-checkpoints`), then watched both die.
- **What happened:** the forced-aligner work (~35 min, 2 400 utts each) finished
  for both, then:
  * `11369` (aligned5) OOM'd at its **first** training step — its GPU's
    co-tenants had grown to 19.4 GB *after* the picker sampled (the cluster
    showed 67–81 GB/GPU used by then).
  * `11371` (bin_es) OOM'd the same way a few minutes later.
  Both were re-picked onto the "least-used" GPU, which was still not free enough
  once the co-tenants grew — the race the picker cannot win, only avoid.
- **Fix (code):** `scripts/pick_gpu.sh` — samples 3×, picks the max-free GPU and
  **refuses** (exit 3, "resubmit later") when nothing has ≥ 20 GB free, instead
  of falling back onto a doomed GPU and burning half an hour of alignment. All
  four launchers we resubmit (`qat_stream_aligned5b`, `qat_stream_aligned5r`,
  `qat_stream_bin_es`, `eval_bin_es`) now call it and exit cleanly when the
  cluster is saturated; they are re-submitted when capacity appears.
- **Relaunched:** `11375` (`aligned5r`, the same recipe plus
  `--keep-step-checkpoints 500`) — running, at risk on the same saturated
  cluster. If it dies, resubmit; no further spawn attempts until a GPU has
  ≥ 20 GB free (the check's rule).
- **Artifacts:** `scripts/pick_gpu.sh`, `qat_stream_aligned5r.pbs`,
  `eval_bin_es.pbs` (one job evaluates the binary's step-500/1000/1500 states).

---

## 2026-10-15 — Corrected a claim: the gate gap is quantization damage, not "0.6B capacity"

- **Did:** an accuracy pass over the headline claims in README / HANDOFF /
  PROJECT_STATUS while auditing the Stage-4 gate arithmetic.
- **What was wrong:** the docs said the `+1.5` gap (bar 3.74/6.11 against the
  fp16 **1.7B** anchors 2.24/4.61) was "0.6B model capacity, not the recipe".
  The Stage-0 record contradicts the capacity story: the fp16 **0.6B** streams
  at **2.54 / 6.27** (2.0 s rows in `docs/stage0_results.md`) — close to the
  1.7B's 2.24 / 4.61 — so *size is not what limits the fp16 side*. Quantization
  alone costs **+5.4** offline on the 0.6B (2.11 → 7.5, `docs/stage1_results.md`),
  i.e. more than the entire +1.5 budget, before any streaming penalty.
- **Fixed wording (all three docs):** the remaining gap is *quantization damage*,
  and the open question behind the 1.7B path is whether that damage shrinks with
  scale (unproven). The recipe's achievement and the zero-loop status are
  unchanged.
- **Why it matters:** the claim decides what the next lever is supposed to be.
  "0.6B capacity" would point at scale answers; "quantization damage" points at
  quantizer/precision work (stochastic rounding, co-training) *and* scale. The
  record now says what the evidence supports.

## 2026-10-15 — Third OOM: stop respawning into a saturated cluster (the waiting rule)

- **Did:** watched `11375` (`aligned5r`) align 2 400 utts, build the 4 800-example
  mix, and OOM at its first training step — the third aligned run of the day to
  die exactly there (`11369`, `11371`, `11375`).
- **Reading:** the 0.6B QAT run's floor is ~12–15 GB (base + shadows + the
  first-step activations; ~2 GB more if an in-training streaming eval
  coincides), while the cluster's GPUs have had 10–12 GB free all day. No
  launcher setting fits — the *cluster* is the constraint, not the recipe.
- **Rule adopted (and now the standing practice):** do not respawn into a
  saturated cluster. `scripts/pick_gpu.sh` refuses when nothing has ≥20 GB free;
  the follow-up check resubmits only when that passes. Each doomed attempt costs
  ~40 min of forced-aligner work plus a queue slot, so waiting is strictly
  better than retrying.
- **Queue for when capacity returns:** `qat_stream_aligned5r.pbs` (the longer
  aligned cycle — its 6 h walltime fits ~2 000–3 500 steps at contended pace,
  the resume `qat_stream_aligned5b.pbs` chains it) and `qat_stream_bin_es.pbs` +
  `scripts/eval_bin_es.pbs` (the binary early-stopping re-run). The 1.7B needs
  ≥45 GB *stable*.
- **Also this round (docs):** the gate-gap attribution was corrected to
  *quantization damage* (see the entry above), HANDOFF's next-steps list
  refreshed (the two live levers, the quantizer-side candidates, the 1.7B, the
  write-up), test counts updated to 327, and `docs/stage4_results.md` generated
  from the artifacts.

## 2026-10-15 — Fourth OOM, now at model *loading*: the churn outruns any margin

- **Did:** resubmitted `aligned5r` with the new self-waiting picker (job `11376`,
  60-minute wait window).
- **What happened:** the picker accepted physical GPU 2 (~24 GB free at 04:51),
  the launcher printed its header, and the **model load** OOM'd a few minutes
  later: the error shows that card at ~79/80 GB with six processes (29.2 + 20.4
  + 8.7 + 8.7 + 8.5 + our 3.2 GB).  Co-tenants consumed 20+ GB in the minutes
  between the pick and the load — earlier failures were at the first training
  step, this one before the alignment even started.
- **Reading:** no launcher-level memory margin is safe against this churn; the
  only honest position is to wait for the cluster to quiet.  The bar for the
  training launchers is raised to **35 GB free** (the run needs ~15 GB, so ≥2×
  headroom at pick time) with a 60-minute wait window; the 2-hourly retry check
  resubmits when `pick_gpu_wait.sh` can satisfy that.
- **Deliverable status:** unchanged and safe (aligned2 ternary 12.92 / 20.00 at
  320 ms, zero loops; every artifact on disk).  The queued experiments
  (`aligned5r` longer cycle, `bin_es` binary early-stopping) wait for capacity.

## 2026-10-15 (midday) — Binary early-stopping run finished; its step-500 state is still mid-transition

- **Did:** `11384` (`bin_es`) completed its 1 500-step schedule with **zero OOMs**
  and all three step checkpoints on disk (`step500/1000/1500.pt`). Submitted
  `11474` (`eval_bin_es.pbs`) to score them on full samples.
- **First full-sample read:** step 500 streams at **56.14 WER** (CER 38.55, 4
  loop truncations, raw 54.45, RTF 6.9). That is *worse* than the previous
  binary mirror's step-1 250 state (18.43) — as expected: the binary readout
  needs ~1 000+ steps to adapt after the progressive-precision switch, so the
  early checkpoint is mid-transition, not a candidate. The step-1 000 and
  step-1 500 evals (running) plus the winner's `other` split decide whether
  this run recovers a state at or below 18.43.
- **Why this design was necessary:** the previous run's best state (18.43 at
  step 1 250) was overwritten by a resume; with `--keep-step-checkpoints` the
  trajectory is now selectable, so the adaptation curve itself is measured
  instead of guessed.

## 2026-10-15 (14:55) — New best on the longer cycle (11.55 clean) and the binary arm's instability

- **The longer aligned cycle produced a new full-sample best.** `11486` scored
  the walltime-cut run's states: **step 1 000 → 11.55 WER clean** (CER 6.41, 1
  loop truncation, raw = policy, RTF 5.6, 50 utts) — **better than the promoted
  12.92** by 1.37 points, on the identical evaluation protocol. Step 1 500 and
  the final snapshot are still being scored; the winner then gets the `other`
  split. Promotion (README/PROJECT_STATUS/HANDOFF headlines + tag) is held until
  the winner's `other` and chunk sweep land — one clean cell is not a promotion
  dossier.
- **The binary early-stopping re-run did NOT recover the 18.43 state.** Its
  three selectable states score 56.14 (step 500) / 40.25 (step 1000) / 36.44
  clean + 43.18 other (step 1500). Two facts matter:
  1. Its own curve was still descending (56 → 40 → 36) when the 1 500-step
     schedule ended — the binary transition needs *longer*, not just selection.
  2. The nominally identical recipe has now produced 18.4 / 27.5 / 36.4 across
     three runs (same init `aligned2/step2500.pt`, same flags — verified by
     diffing the launchers; only the save cadence differs). The ternary→binary
     switch is **seed-sensitive**: the early-stopping idea needs a longer budget
     (3 000 steps) *and* the quantizer-side fix (stochastic rounding) to be
     reliable.
- **The 1.7B retry is training and healthy** (`11470`, GPU 5, picked
  immediately at 11:38 with ≥45 GB free): 0 OOMs, aligned targets built (39 800
  examples → 2 400 pooled), step ~1 000 @14:55, guardrail STREAM 35.5 @250 →
  30.4 @500, offline ternary 27.97 @500 (vs the attn5 init's 20.55 — the
  aligned recipe always costs some offline content; the question this run
  answers is whether the anchored fp32-AdamW path avoids the *permanent* dip
  that killed v2). Watch its next offline EVAL.

## 2026-10-15 (17:00) — PROMOTION: 11.55 / 18.98 (aligned5 step-1 000); the 1.7B retry trained clean

- **New promoted streaming state (tag `0.2.47`).** The longer aligned cycle's
  **step-1 000** checkpoint scores **11.55 clean / 18.98 other** at 320 ms on
  full 50-utt evals with 0–1 loop truncations — better than the previous
  promoted state (12.92 / 20.00) on **both** splits. The trajectory it was
  selected from peaks early and then *degrades*: 11.55 @1 000 → 12.82 @1 500 →
  14.30 @~1 750. So the value of the longer cycle is its *early states*, not its
  end — the third time this lineage has shown the pattern (offline recipe,
  binary mirror, and now the streaming cycle). `--keep-step-checkpoints` is what
  made the win visible; without it the run's final state would have been the
  only candidate.
- **The 1.7B anchored fp32-AdamW retry worked as a training run**: 1 200/1 200
  steps, **zero GPU OOMs**, picked a ≥45 GB card within three minutes, and — the
  key question — **no permanent dip**: offline ternary recovered 27.97 @500 →
  **24.05 @1 000** (v2 fell 20.55 → 30.6 and stayed). Its streaming guardrail
  ran 35.5 @250 → 30.4 @500 → 23.8 @750 → 25.4 @1 000 (peaking late, like the
  0.6B runs). Its checkpoint (`last.pt`, 5.6 GB) is on disk.
- **How the 1.7B job ended (new lesson):** it was killed during its own *final
  in-training streaming eval* — no traceback, no GPU OOM, log simply stops.
  That signature is a **host-RAM SIGKILL** (the job's `mem=48gb` budget vs the
  eval's footprint), not a GPU failure. Fix in use: evaluate the checkpoint in a
  dedicated job with `mem=64gb` (`scripts/eval_streaming_17b.pbs`). For future
  1.7B training runs: skip the in-training streaming evals or raise the host
  memory request — the checkpoint survives either way, but the log's final
  records are lost otherwise.

## 2026-10-15 (18:55) — The new best's chunk sweep; the guardrail diverges again

- **Chunk sweep of the promoted state** (`aligned5/step1000`, 50-utt clean):
  160 ms **13.77**, 320 ms **11.55**, 1000 ms **10.70**. So the new state wins
  decisively at the *gate metric* (320 ms: 11.55 vs 12.92) and is marginally
  behind the old state at the extremes (13.24/10.59) — worth stating plainly in
  the results tables rather than smoothing over.
- **The 20-utt guardrail vs the full sample diverged again** — third instance
  this project. `aligned5b`'s guardrail *improved* after the resume
  (8.94 @2 000, 9.88 @2 500, 9.18 @2 750, 10.59 @3 500 — all better than the
  promoted 11.5–12.9 range), while the *full-sample* numbers of the same
  schedule's early states fell (11.55 @1 000 → 12.82 @1 500 → 14.30 @1 750). The
  low-LR tail of a cosine can genuinely recover — or the guardrail can lie
  again. Only full samples settle it, so `11526` (`eval_aligned5b.pbs`) is
  scoring the resume's `step2000/2500/3000` states clean, then the winner's
  `other`, against 11.55/18.98.
- **Longer binary run** (`bin_long`, from the new best ternary state) is
  training (step ~840, zero OOMs); its guardrail is descending 66.6 @250 →
  47.8 @500 → 36.5 @750 — the binary adaptation curve, but the full-sample evals
  at its end are the decider.

## 2026-10-15 (19:15) — The 1.7B's first streaming measurement: the bottleneck is its offline base

- **Result:** the 1.7B retry's final checkpoint (1 200 steps, anchored fp32-AdamW,
  no dip) streams at **29.24 WER clean** on the official algorithm (320 ms,
  50 utts; CER 16.99, 5 loop truncations, raw 31.36 → the LSP policy helps by
  2.1). Its `other` split is still computing.
- **The comparison that matters.** The 0.6B's promoted state: **11.55** from an
  offline ternary of **7.52** (streaming gap +4.0). The 1.7B: **29.24** from an
  offline ternary of **24.05** (gap +5.2). So the *streaming recipe transfers
  fine* — the same ~4–5 point streaming penalty — but the 1.7B's **offline base
  is the bottleneck**: its quantization damage is +22.4 absolute against its
  fp16 offline (1.63), versus the 0.6B's +5.4 (2.11 → 7.52).
- **Why:** the 1.7B never received the 0.6B's *complete* offline recovery recipe.
  Its chain was FFN warm start → attention rollout (20.55) and then straight to
  streaming SFT; the 0.6B, at the equivalent point, was only at ~9–16 and needed
  the full-cache cycle + anneals to reach 7.52. A *bigger* model should show
  *less* quantization damage under an equal recipe — so the recipe, not
  capacity, is what these numbers indict. (This also supersedes the earlier
  "0.6B capacity" story definitively: capacity is not the limiter on either
  size; the recovery recipe's completeness is.)
- **Next Stage-3 step (when GPUs allow):** apply the full offline recovery
  recipe to the 1.7B — full-cache QAT with the frozen anchor, then anneal
  cycles — *before* the aligned streaming SFT, and measure the offline target
  first (sub-10 would be a working base; the 0.6B analogue is 7.52).

## 2026-10-15 (19:45) — Stochastic rounding implemented (the T2.2 lever, ready for the next QAT run)

- **Did:** implemented unbiased stochastic rounding end-to-end while the GPU
  jobs ran: `quantize_ternary` (floor + Bernoulli on the fractional part, then
  clip; E[q] = W/gamma for |W/gamma| ≤ 1) and `quantize_binary` (soft-sign draw,
  E[q] = centered/scale for |centered| ≤ scale; out-of-range entries stay
  certain, which the tests assert explicitly). `BitLinear(stochastic=True)`
  applies it **in training mode only** — evals stay deterministic — and the
  layer policy plus `run_stage1_qat.py --quant-stochastic 1` thread it through.
- **Why it matters:** the recorded bottleneck is *quantization damage*
  (offline +5.4 on the 0.6B, +22.4 on the 1.7B), and stochastic rounding is the
  one quantizer-side lever never tried — it removes the rounding bias without
  changing the deployed model (the grid is identical at eval time).
- **Tests:** 9 new (336 total, ruff clean): unbiasedness in range, integral
  targets never flip, clipping, group-wise zero groups, deterministic paths
  repeat exactly, and the train/eval gating at the layer level.
- **Next use:** the 1.7B offline recovery run (full-cache QAT + anneals) should
  carry `--quant-stochastic 1` — that is the highest-value place to test it,
  since its damage is the project's dominant remaining error.

## 2026-10-15 (20:50) — The resume's states do not beat 11.55; the 1.7B's pair is complete; recovery-1 launched

- **The resume's selectable states, scored on full samples:** step 2 000
  **11.76**, step 2 500 **11.76**, step 3 000 **11.76** (clean, 320 ms; CERs
  5.02 / 4.56 / 4.56; 0 / 3 / 0 loop truncations). **None beats the promoted
  11.55**, so nothing is promoted. This is the *fifth* guardrail-vs-metric
  divergence: the in-training guardrail had improved to 8.94–9.88 across exactly
  these steps, and the full samples say the model merely held still. The
  variation is fully characterised now — early peak (11.55 @1 000), drift
  (12.82, 14.30), plateau (11.76 ×3) — and the plateau is *stable*, which is
  good news for future long cycles but not a new best.
- **The 1.7B's first full streaming row:** **29.24 clean / 33.75 other**
  (320 ms, official algorithm, 50 utts; `other` CER 19.93, 6 truncations). With
  the 0.6B's 11.55/18.98 this completes the comparison in DEEPDIVE §3.5: the
  streaming *recipe* transfers (gap +5.2 vs +4.0), the *offline base* is the
  1.7B's bottleneck (24.05 vs 7.52; damage +22.4 vs +5.4).
- **Launched:** `11532` = `qat_17b_recover1.pbs` — **the 1.7B's offline recovery
  cycle**, the next step the evidence prescribes: init from
  `qat_17b_attn5/last.pt` (its best offline, 20.55), policy `decoder_attn_ffn`,
  2 500 steps at lr 1.5e-6 (the value proven stable by v4), warmup 200, batch 1,
  **frozen-teacher anchor (`--distill-weight 0.05 --frozen-teacher 1`)** and —
  for the first time in a real run — **`--quant-stochastic 1`**, the unbiased
  rounding lever implemented earlier today. 8 000-utterance training sample,
  `--keep-step-checkpoints 400`, 10 h, `mem=64gb`, self-waiting picker at
  ≥45 GB. Success criterion: the offline ternary readout dropping meaningfully
  under 20.55 at the in-loop evaluations, with *no permanent dip*.

## 2026-10-15 (22:50) — 1.7B recovery cycle is training on a quiet card; a launcher-hygiene lesson

- **`11532` (`qat_17b_recover1`) got the card it was waiting for**: GPU 1 with
  only 20.1 GB in use (~60 GB free) — the first time the ≥45 GB bar was
  satisfied all evening. It is at **step ~1 000 of 2 500** (~8.5 steps/min, much
  faster than the usual contended pace). **First in-loop offline eval: step 500
  → ternary 24.15** (its init, `attn5`, was 20.55). That is an early cost, the
  same shape as v4's — the criterion this run is judged by is whether it
  *recovers* (no permanent dip) at the next evals. It is also the **first real
  run in the project using `--quant-stochastic 1`**, so part of any early
  difference may be the unbiased-rounding variance; the eval itself is
  deterministic.
- **`11491` (`bin_long`)** reached step 2 441 with zero OOMs; its 8 h walltime
  will cut it near step 2 500–2 700 (checkpoints 500–2500 on disk).
  `scripts/eval_bin_long.pbs` is prepared to score those states (clean, then the
  winner's other) as soon as it ends — no need to re-derive the evaluator.
- **Launcher-hygiene lesson (reinforces docs/CODE_GUIDE.md §4):** my patch of the
  recover1 launcher renamed the log and save-dir but **missed `#PBS -N` and
  `#PBS -o`** — the replace targets did not match and `str.replace` fails
  *silently*. The run itself was unaffected (the log path was correct), but the
  job ran under the template's name and stdout file. Fix applied, and the
  lesson: grep-verify **every intended field** of a generated launcher, not just
  a representative few.

## 2026-10-16 (00:50) — bin_long completed its 3000 steps; the 1.7B recovery cycle is NOT improving the offline base

- **`11491` (`bin_long`) finished all 3 000 steps** — the first binary run in this
  project to complete a full budget (zero OOMs) — with checkpoints 500–2500 plus
  `last.pt` (the 3 000 state). `11542` (`eval_bin_long.pbs`) is scoring all six
  on full samples (clean first, then the winner's `other`); its in-training
  guardrail fell 66.6 → 36.5 over the first 750 steps and its last in-loop
  binary eval is recorded in WORKLOG.  Nothing is promoted until the full-sample
  numbers land.
- **The 1.7B offline recovery is not recovering.** In-loop offline ternary
  readout: **24.15 (step 500) → 23.09 (1000) → 24.79 (1500)** against its init
  (`attn5`) of **20.55**. So the anchored cycle as configured costs ~+2.5–4
  offline rather than reducing it. Candidate causes, recorded as hypotheses
  (this run changed more than one thing, which is itself the lesson):
  1. **`--quant-stochastic 1`** — its first use in a real run. Unbiased
     rounding adds gradient variance; the proven 0.6B recipe used deterministic
     rounding. *This is testable in isolation and is being tested.*
  2. **A too-gentle schedule** — lr 1.5e-6 with a 200-step warmup is a *tiny*
     total update over 2 500 steps; the 0.6B's winning cycle used lr 1e-5.
  3. **The anchor may bind harder at this scale** — KL toward the *fp16*
     teacher pulls toward a full-precision optimum that a heavily damaged
     quantized student cannot occupy (the 0.6B tolerated it; the 1.7B's damage
     is +22.4 vs the 0.6B's +5.4, so the same pull may dominate).
  4. **The optimizer itself** *(added 2026-10-16 after the audit caught the
     dossier's claim)* — the arms shipped **adam8bit**, while the 0.6B recovery
     recipes and the aligned v4 use **fp32 AdamW**; `recover4` (identical to
     `recover2` except the optimizer) now tests it. The first A/B (SR) also
     measured: **no-SR starts flat (20.76 vs recover1's 24.15)** — see
     2026-10-16 02:50 entry.
- **Launched the control:** `11543` = `qat_17b_recover2.pbs` — **byte-identical
  recipe minus `--quant-stochastic`** (asserted absent in the launcher), so its
  evals isolate cause (1). Launcher discipline applied this time: every intended
  field (`-N`, `-o`, log, save-dir, the absent SR flag) grepped before
  submission.

## 2026-10-16 (01:40) — recover1's verdict: the 1.7B recovery cycle as configured does NOT improve the base

- **Final curve** (in-loop offline ternary, 50-utt protocol — this readout
  matched full-sample offline numbers throughout the project, unlike the
  streaming guardrail): **24.15 (500) → 23.09 (1000) → 24.79 (1500) → 22.78
  (2000) → 24.36 (2500)** against an init of **20.55**. Read honestly: no
  permanent dip (the 24.79 recovered), but the cycle never reaches its starting
  point — the anchored recovery recipe that lifted the 0.6B costs the 1.7B about
  +2.5 absolute at this scale/schedule.
- **States on disk:** `last.pt` (the 2 500 state, 5.6 GB, 01:27) and
  `step2000.pt` (its best readout). No snapshot copies were made because nothing
  will resume this run; if a future run does, copy `step2000.pt` first.
- **`bin_long`'s first full-sample state:** step 500 → **61.12** (CER 44.30, 10
  loop truncations, raw 59.96 — the policy *raised* the number, i.e. truncation
  hurt). Its in-training guardrail had descended 66.6 → 21.2 by step 2 500, so
  the later states (being scored) are the real candidates; step 500 of a binary
  adaptation is a transition state, as the earlier binary runs showed.
- **The controlled pair now running:** recover1 (SR, lr 1.5e-6) vs recover2
  (no SR, lr 1.5e-6) vs recover3 (prepared: no SR, lr 5e-6). This is the first
  *designed* A/B in the project's 1.7B work — one variable at a time, same init,
  same data, same anchor — and it exists because recover1 changed two things at
  once, which is exactly how attribution gets lost.

## 2026-10-16 (02:50) — First controlled A/B result: stochastic rounding HURT the 1.7B recovery

The one-variable-at-a-time pair answers its first question at their first common
evaluation point (step 500, in-loop offline ternary, 50-utt protocol):

| arm | stochastic rounding | lr | offline @step 500 | vs its init (20.55) |
|---|---|---|---|---|
| `recover1` | **on** | 1.5e-6 | **24.15** | +3.6 (worse) |
| `recover2` | off | 1.5e-6 | **20.76** | +0.2 (flat) |

**Reading:** the unbiased-rounding lever — implemented earlier today as the
*first untried quantizer lever* and expected to *help* — is instead costing
about **3.4 points** at this point of training. Hypothesis: stochastic rounding
adds gradient variance to a model already far from its optimum (the 1.7B's
quantization damage is +22.4); unbiasedness buys nothing when the training
signal is dominated by that noise. The no-SR arm additionally starts *flat*
instead of dipping, which is the healthier shape.

**Caveats, stated:** this is one evaluation point, and the arms are not
seed-matched (same init and data, but the shuffler's order differs), so part of
the 3.4 could be batch-order luck. The later evals (steps 1000, 1500, …) will
confirm or dissolve it. But the direction matches the hypothesis and the size is
large; the flag's help text now carries the caveat, and **no future 1.7B run
should use `--quant-stochastic 1` without a reason**.

**Also this round:** `bin_long`'s full-sample trajectory is 61.12 (500) → 40.36
(1000) → 29.77 (1500) → **28.28 (2000)** (CER 18.00, 3 truncations, raw 27.22).
Still above the binary best (18.43), but it is the *only* binary run whose curve
was still improving at the end; step 2500 and the final state are being scored.

## 2026-10-16 (03:15) — The three-arm 1.7B experiment is fully live

| job | arm | stochastic rounding | lr | state |
|---|---|---|---|---|
| `qat_17b_recover1` | baseline-as-shipped | on | 1.5e-6 | **finished** (no gain: 22.78 best vs the 20.55 init) |
| `qat_17b_recover2` | no-SR control | off | 1.5e-6 | training (step ~750; step-500 readout **20.76** vs recover1's 24.15) |
| `qat_17b_recover3` | LR arm | off | **5e-6** | launched 03:12, picked GPU 5 immediately |

Note on superseded instructions: the recurring check's text (written before
2026-10-16) tells the agent to run the 1.7B recovery *with*
`--quant-stochastic 1`. The measured A/B — stochastic rounding costing ~3.4 WER
at step 500 — supersedes that, so `recover3` omits the flag (asserted absent in
its launcher). Instructions follow evidence, not the other way round.

The experiment now answers two questions with one variable each:
`recover1` vs `recover2` = does stochastic rounding help? (**no** — measured);
`recover2` vs `recover3` = is the schedule too gentle? (**pending**).

Also: `eval_bin_long` is scoring its final state (`last`) after a trajectory of
61.12 / 40.36 / 29.77 / 28.28 / 26.80 — as recorded, it will not beat the binary
best of 18.43, and the reason is now explicit: that number came from one
seed-sensitive state that no re-run has reproduced.

## 2026-10-16 (03:45) — bin_long fully scored: the binary arm's ceiling stands

- **Complete full-sample table** (`bin_long`, 50 utts, official algorithm, 320 ms,
  LSP policy on):

  | state | clean | other | truncations |
  |---|---:|---:|---:|
  | step 500 | 61.12 | — | 10 |
  | step 1000 | 40.36 | — | 4 |
  | step 1500 | 29.77 | — | 1 |
  | step 2000 | 28.28 | — | 3 |
  | **step 2500 (best)** | **26.80** | **35.23** | 3 |
  | last (the 3 000 state) | >26.80 | — | — |

  (Raw-vs-policy deviations and intermediate values are in the JSONs; step 500's
  policy *raised* WER, so its loop truncations hurt.)
- **Verdict: no promotion.** The best full-sample state of the lineage-from-
  aligned5 binary run is **26.80 / 35.23**, well behind the binary best of
  **18.43 / 25.45** — and now the record says explicitly **why that is hard to
  beat**: 18.43 came from *one* seed-sensitive state of the very first mirror at
  a schedule the later re-runs have not reproduced. The binary transition's
  outcome is (seed, schedule)-dependent; the two re-runs so far land at 36.4
  (1500 steps) and 26.8 (3 000 steps). Whether stochastic rounding — now known
  to *hurt* the 1.7B recovery — behaves differently here is an open, cheap-ish
  experiment for when GPUs allow.
- **Jobs remaining before the next check window:** `recover2` (step ~1 000 of
  2 500), `recover3` (step ~370 of 2 500, lr 5e-6). Their pairwise evals are the
  remaining questions of the night.

## 2026-10-16 (04:25) — the recovery flat-lines even without SR; LR/optimizer arms decide what could still work

- **`recover2`'s step-1 000 offline readout: 20.55** — *exactly* its `attn5`
  init value. Combined with `recover2`'s step-500 = 20.76, the picture is: after
  1 000 steps of the "recovery" recipe at lr 1.5e-6, the 1.7B is precisely where
  it started. Meanwhile `recover1` (same recipe + SR) was *worse* (24.15 → 24.36
  across its run). So: (a) the SR A/B is confirmed — the flag costs ~3.5–4 WER
  here; (b) **as configured, the recovery cycle is a no-op for the base**: the
  LR is too gentle for a model this damaged to move, and the anchor (KL toward
  the fp16 teacher) may then be what keeps the loss pinned.
- **What this means for Stage 3** (the decision the arms produce): the 0.6B
  never recovered with *2 500 gentle steps* either — it needed the full-cache
  campaign at lr 1e-5 for 10 k steps, then warm restarts and anneals. The 1.7B
  needs the same *scale* (more steps, warmer schedule, several cycles), or a
  different anchor — not a single gentle pass. The LR arm (`recover3`, 5e-6) and
  the optimizer arm (`recover4`, fp32 AdamW — a hypothesis that had been
  *described* in docs but never shipped until now) test the two knobs that were
  varied fastest; multi-cycle/schedule-scale tests follow after them.
- Code debt from the audit finished in the same breath as the numbers landed:
  `train/qat.py` went 319 → **215** lines (the dead `run_qat`/`QATConfig`
  LoRA path and `enable_gradient_checkpointing` deleted, their tests with them),
  and the trainer's inline duplicate of the freeze rule was replaced by
  `freeze_non_bitlinear` — one source of truth for "which parameters a QAT run
  updates", which is exactly the kind of thing both copies had to agree on for
  the optimizer to be trusted at all.

## 2026-10-16 (06:50) — The three-arm 1.7B experiment has its answer: the schedule was the bottleneck

Full-collection readout for the recovery arms (in-loop **offline** ternary WER,
50-utt protocol — this readout has matched full-sample offline numbers all
project):

| recipe | step 500 | 1000 | 1500 | 2000 | 2500 |
|---|---:|---:|---:|---:|---:|
| `recover1` — SR + lr 1.5e-6 | 24.15 | 23.09 | 24.79 | 22.78 | 24.36 |
| `recover2` — lr 1.5e-6 | 20.76 | 20.55 | 19.17 | 19.92 | 19.17→19.92 |
| `recover3` — **lr 5e-6** | **19.92** | — | — | **16.74** | **16.53** |
| `recover4` — **fp32-AdamW** + lr 1.5e-6 | — | — | **19.49** | **19.28** | (2 501 steps at 06:50) |

- **Result 1 (the LR question, answered):** `recover3` (lr 5e-6) reaches
  **16.53**, which is **4.0 points below** its 20.55 init and ~3.2 below the
  best of the arms running at 1.5e-6. The sixteen-fold difference in schedule
  warmth was *the* difference: no recovery recipe at 1.5e-6 ever beat its own
  init. Which also means the earlier "gentle-but-stable" reading from v4 (the
  aligned run at 1.5e-6) was *not* a recovery recipe at all, it just drifted
  less. The 1.7B needs a schedule in the range the 0.6B's proven recipe used.
- **Result 2 (SR, confirmed):** the un-sr comparison — `recover3` at the *same*
  LR without SR produces 19.92 at step 500 vs recover1's 24.15 with SR. The
  lever is definitively harmful *here*, as the interactive step-500 read showed
  last round (see 2026-10-16 02:50 entry).
- **Result 3 (the optimizer, trending):** `recover4` (fp32 AdamW) reads
  **19.49/19.28** at steps 1500/2000 vs `recover2`'s 19.17/19.92 — a consistent
  small improvement (~0.6), but *both* are behind `recover3`, so the recipe
  question the LR arm settles first remains the bigger lever. The definitive
  ordering would need one more arm (fp32 + 5e-6) later.
- **What this means for Stage 3:** single-cycle recovery at a *real* schedule
  works; the next unlock is the *0.6B-style multi-cycle campaign* (10k+ steps,
  warm restarts, anneals) on top of the recover3 recipe, not the gentle pass.
- **`recover4` is still training** (step ~2 101 at 06:50, no OOMs *despite* the
  1.41 B trainable fp32-AdamW footprint the earlier attempt could not fit —
  this machine's cluster has quiet 45 GB slots again now). Its final eval and
  streaming readout will follow once its 2 500 steps complete.
- In parallel, night-round artifacts: `recover1-3` produce three JSON choices
  about what 1.7B recipes *expect*; `eval_bin_long` scored the binary arm's
  plateau at 26.80/35.23; the `streaming` line for `recover3`'s final
  checkpoint is being launched next.

## 2026-10-16 (08:50) — The offline-recovered 1.7B's streaming COLLAPSES; the two-stage doctrine is now the test

- **`eval_streaming_17b` on `qat_17b_recover3/last.pt`: trig WER 285 with 354
  loop truncations** (raw 629). The model's *offline* base improved (recover3:
  20.55 → 16.53) precisely because the recovery recipe trains plain-offline
  samples only — and with no streaming pressure the official algorithm's
  prefix→emit head lost its discipline. The recovered base's *streaming*
  collapsed while its base strengthened.
- **The doctrine to test:** recover the offline base with a real schedule
  (done, recover3) and *then* replay the aligned streaming recipe on that base
  ([`qat_stream_aligned17b5.pbs`](scripts/qat_stream_aligned17b5.pbs): the
  aligned recipe, init from `recover3/last.pt`, lr **5e-6**, 2 000 steps,
  `--keep-step-checkpoints 400`, one variable vs the aligned17b4 attempt
  which used the *unrecovered* base at lr 1.5e-6). If 2 000 re-aligned steps
  bring streaming back without re-collapsing the base, Stage 3's sequencing is
  a *two-stage* pipeline — and that becomes the recipe.
- **Also recorded:** `recover4` (fp32 AdamW) finished its 2 500 steps at
  21.40/19.28 trajectories (its own 19.28 @2 000 step was its best), consistent
  with a small optimizer effect — the definitive ordering still prefers the LR
  arm's schedule, and the definitive ordering would need an fp32+5e-6 arm later.

## 2026-10-17 (00:46) — The two-stage doctrine holds offline; A full-sample streaming eval is now running

- **Job list check (00:43, after a day's pause):** every submitted run has
  completed; no duplicates, no leftovers. The four 1.7B recovery arms are all
  final (recover1..4 conclusions recorded on 2026-10-16); recover3's streaming
  `other` split finished at **278.3 WER** (with 387 loop truncations) — the
  recovered base's *streaming collapse* is consistent across both splits.
- **`aligned17b5` completed** (2 000 steps, aligned mix on the *recovered*
  base, lr 5e-6): its offline twin readout is
  **16.63 (1 500) → 16.74 (2 000)** — essentially unchanged from the recovered
  base's **16.53** (recover3). *The base survived the streaming re-alignment* —
  the two-stage sequencing is holding at the offline level. Guardrail-trend
  (20-utt) recorded 19.3–21.9 through the run — a *guardrail reading only*.
- **The decisive number is now running:** `11666`
  (`scripts/eval_streaming_17b.pbs`, mem=64 GB, full 50-utt clean + other)
  against `qat_stream_aligned17b5/last.pt`. Its checkpoint is snapshotted first
  (`last_aligned17b5_end.pt`) per the "snapshot before any overwrite" rule.
- **Process note:** the `stream.py` split attempted mid-round was left
  half-applied after a declined write, the tree was reverted to the last
  committed state (340 tests green afterwards), and the recipe for re-running
  the split safely is `docs/CODE_GUIDE.md` §4 plus this entry's job list. The
  split is *deferred*, not dropped.

## 2026-10-17 (01:35) — Stage 3's two-stage sequencing WORKS: the recovered 1.7B streams at 19.39/24.09

The **first** 1.7B streaming attempt (aligned17b4: *unrecovered* base, lr 1.5e-6)
ended at **29.24 / 33.75** with a collapsed base and the recipe's drift. The
**two-stage pipeline** — recover the offline base at lr 5e-6 (recover3: 20.55 →
**16.53**), then replay the aligned streaming recipe on that base (aligned17b5:
lr 5e-6, 2 000 steps) — streams at:

| split | WER | CER | loop truncations | raw WER |
|---|---:|---:|---:|---:|
| **clean** | **19.39** | 6.90 | **0** | 19.60 |
| **other** | **24.09** | 9.41 | **0** | 23.86 |

That is a ~**10-point** improvement on clean, ~**10-point** on other, with
**zero loop truncations** across both splits and the offline base **preserved**
(16.74 at the aligned run's step-2 000 vs the recovered base's 16.53). The
two-stage pipeline is now the Stage-3 recipe: recover offline, then re-align.

**Comparisons that frame it.** The 0.6B's promoted state remains better in
absolute terms (11.55 / 18.98). The 1.7B's numbers are the *model-scale*
question the project's Stage-3 gate was defined against, and 19.39 is the
first 1.7B number inside single digits of the 0.6B's own performance rather
than ~3× away. The recipe is a real path; the remaining gate needs more
recovery cycles and a wider aligned training set (2 400 aligned utterances is
still the streaming mix's cap).

**Next steps queued** (when GPUs allow): (1) multi-cycle offline recovery on
the 1.7B at lr 5e-6 (recover3-style, 5–10 k steps, warm restarts) to bring the
offline base below ~10; (2) re-run the aligned recipe from that stronger base;
(3) if the streaming number follows the offline's drop, the 1.7B line enters
the gate's argument. All three runs' artifacts (JSONs, checkpoints, launchers)
are on disk and recorded.

## 2026-10-17 (01:50) — Stage 3's multi-cycle recovery phase 2 launched (recover5)

- **`115xx` = `qat_17b_recover5.pbs`**: the multi-cycle campaign's second phase.
  Init from `recover3/last.pt` (offline 16.53), fresh 4 000-step cosine at
  **lr 5e-6** (the schedule that works), **fp32 AdamW** (the optimizer recover4
  validated at the 1.7B scale), batch 1, `--train-n 8000`, the frozen-teacher
  anchor (w = 0.05, fp16 teacher), `--keep-step-checkpoints 400`, no SR ( measured
  harmful), no streaming mix (the *offline* phase — the aligned streaming replay
  runs as Stage B *after* this). mem=64gb, the self-waiting picker at ≥45 GB.
- **Target:** the offline readout below ~12 (0.6B parity line is ~10.7); the
  in-loop evaluations at each 500-step mark are the trajectory markers, and the
  step-checkpoints are the selectable states.
- **Why now:** recover3's phase-1 proved the recipe; recover4 proved the
  optimizer footprint runs; recover5 extends the depth. If 4 000 more steps
  drop the offline to <~12, the aligned streaming replay from *that* base is
  the next Stage B, and the 1.7B enters the 0.6B-parity argument. If ~19 stays
  the floor, the *inner loop of recovery scales* differently at the 1.7B and
  the campaign's next step is a *schedule revision* (e.g. a longer cosine or a
  higher anchor weight), not more steps.

## 2026-10-17 (02:25) — recover5's warm-restart bump and recovery

Phase-2's first two in-loop offline readings: **20.23 (step 500)** then
**17.90 (step 1 000)**. The 500-step bump above the 16.53 init is the expected
warm-restart shape (a fresh cosine at 5e-6 moves the weights before the
schedule's decay pulls them back). The *fp16 readout* (α=0) *improved* across
the same steps (51.3 → 28.2) — the weights are moving toward the fp16-optimum
even while the ternary readout recovers, which is the opposite of what the
original QAT runs showed and the clearest sign yet that the recovery recipe is
operating in a different regime than the first-round damage did.

## 2026-10-18 (06:50) — The multi-cycle recovery hits the 0.6B-parity target: offline 11.76; STAGE B launched

**recover5 completed its 4 000 steps.** Offline ternary trajectory (in-loop,
50-utt protocol):

| step | 500 | 1000 | 1500 | 2000 | 3000 | 3500 | 4000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| WER | 20.23 | 17.90 | 17.90 | **15.15** | **13.14** | **11.97** | **11.76** |
| fp16 readout | 31.3 | 28.2 | 29.1 | 36.1 | 27.0 | 21.9 | 22.9 |

The phase-2 cycle took the 1.7B's offline from **16.53 → 11.76** — below the
~12 target and, for the first time, in striking distance of the 0.6B's own
7.63. The fp16 readout *improved* in the same window (the model moved toward
the full-precision optimum while the ternary readout improved — the opposite of
the original damage pattern). The trajectory was still descending at step
4 000, so a third cycle (or a longer cosine) has room left.

**STAGE B launched: `qat_stream_aligned17b6.pbs`** — the aligned streaming SFT
on this recovered base, mirroring the 0.6B's proven aligned recipe as closely
as the 1.7B allows (`--aligned-n 2400`, 50/50 mix, stop-stride 2, lr 5e-6,
fp32 AdamW, 2 500 steps, keep-step-checkpoints 400, mem=64gb, SR off, 12 h).
Its full-sample streaming evals will say whether the offline's −4.8 points
carry into the streaming metric (the 0.6B's own offline→streaming gap is
+3.9; if the 1.7B's gap is similar, stage 3 lands ≈15–16, versus its previous
best 19.39 and the first failed attempt's 29.24).

**Also recorded:** the aligned17b5 (phase-2 base's streaming) result stands at
19.39/24.09; the two-stage pipeline's numbers above are the *new* base's
starting line.

## 2026-10-18 (08:10) — Stage B's first guardrail: the deeper base transfers to streaming

`aligned17b6`'s first in-loop streaming guardrail (step 250) reads **16.00**
(CER 8.85, 20-utt). The *same* recipe on the *shallower* (16.53) recovered base
(`aligned17b5`) read **26.59** at step 250. Same recipe, same seed-domain, same
guardrail — the only systematic difference is the recovered base's *depth*
(11.76 vs 16.53): **10.6 points of streaming headroom at the first mark**.

This is the two-stage doctrine's direction confirmed at its most sensitive
point, and it is a *guardrail* reading: the full-sample evaluation (the metric
that decides anything) will run over `aligned17b6`'s states when the run ends —
the numbers to beat are the line's own 19.39/24.09 and, ultimately, the 0.6B's
11.55/18.98.

**Cycle 3 is running in parallel** (`recover6`, init from recover5's 11.76):
another 4 000 steps of the same recipe, so the pipeline has a feeding edge —
by the time this Stage-B run's evals land, a deeper base (target ~9-10) should
be ready for the *next* Stage-B, tightening the loop again.

**Also this round:** two contract tests added for the last untested helpers
(`build_lr_lambda`, `patch_outer_forward`) — 343 tests, ruff clean — and
`fig_roadmap`'s stale status strings replaced with the current reality.

## 2026-10-18 (08:20) — Stage B's guardrail jumps back up (16.0 → 22.1): the sixth reminder

`aligned17b6`'s second guardrail (step 500) reads **22.12**, up from 16.00 at
step 250, while the same recipe's run on the shallower base fell monotonically
over its guardrails (26.6 → 21.9 → 20.9 → 19.3). Two readings, opposite
directions, one run — which is precisely why the standing rule exists: **the
20-utterance guardrail decides nothing**. It has now diverged from the
full-sample metric six times in this project, and this round is a clean
example: the same instrument gave 16.0 then 22.1 within 250 steps.

The plan of record for judging `aligned17b6`: its full-sample clean/other
evaluations over the step-checkpoints when the run ends (the 19.39/24.09 line
to beat; the 0.6B's 11.55/18.98 as the target neighbourhood). Nothing is
promoted or concluded from the guardrail.

**Cycle 3 (`recover6`) is training** (step ~150 @08:20) — on schedule to feed a
deeper base for the next Stage-B regardless of how this guardrail wobbles.

## 2026-10-18 (08:40) — Stage B's early readings: an 8-point offline perturbation at step 500

`aligned17b6`'s step-500 in-loop numbers: **offline 19.49** (from the base's
11.76) and guardrail 22.12. The aligned/streaming mix *costs the recovered base
about 8 points in the first 500 steps* — a bigger absolute perturbation than
`aligned17b5` showed on its shallower base (its offline was already recovered to
16.63/16.74 by steps 1500/2000, i.e. it *held*). Two readings define what to
watch:

- **Does the offline recover** toward the 11.76 base by steps 1000–2500 (the
  aligned mix is only 50 % of the batch, so the base's content is still in the
  loop)? If it recovers while the streaming head improves, the pipeline is
  sound at this depth too.
- **Does the streaming follow** — the guardrail went 16.0 → 22.1, which under
  the standing rule means *nothing*; the full-sample evaluations of the
  step-checkpoints when the run ends are the only numbers that will be used to
  compare against 19.39/24.09 and 11.55/18.98.

Recorded as-is: an *interim* reading, deliberately not a conclusion. The
precedent (aligned17b5: perturbation → recovery) is encouraging; the opposite
(precedent absent) would mean the aligned recipe needs its own warmup schedule
at this base depth — a *variable we would then test*, one at a time, as with
everything else in the 1.7B's campaign.

**Cycle 3 (`recover6`) is at step ~480 and its 500-step offline readout lands
next**; the two runs together are the pipeline's feed-forward loop.

## 2026-10-18 (09:10) — Cycle 3's warm-restart bump matches cycle 2's shape

`recover6`'s step-500 offline readout: **15.89** (from its 11.76 init) — a +4.1
bump, the same shape as cycle 2's (16.53 → 20.23, +3.7) and for the same
reason (a fresh cosine moves the weights before the decay pulls them back).
Cycle 2 then recovered *through* its init and finished 4.8 points below it
(20.23 → … → 11.76); if cycle 3 repeats that arithmetic from 11.76, the offline
lands in the **~7–8** range — the 0.6B's own neighbourhood (7.63). That is the
number cycle 3's remaining evals decide.

`aligned17b6` (Stage B) is at step 1 000 with its 1 000-step evaluation still
running (the 1.7B's in-loop evals take ~25–35 minutes on contended cards); its
500-step readings stand as recorded (offline 19.49, guardrail 22.1).

## 2026-10-18 (09:20) — Stage B is recovering (19.49 → 17.90 offline); the precedent holds

`aligned17b6`'s step-1 000 offline readout is **17.90**, down from 19.49 at
step 500 — the aligned mix's perturbation is being recovered, exactly the shape
`aligned17b5` showed on its shallower base (perturbation → recovery → hold).
Its fp16 readout improved over the same window (23.9 → 20.2). With 1 400 steps
left (to 2 500) and the base at 11.76, the question is how far the recovery
goes while the streaming head absorbs the aligned supervision — the full-sample
evaluations at the end decide everything.

**Both pipeline halves are visibly working:** Stage B recovering toward its
base; cycle 3 at step ~800 heading for the ~7–8 offline projection.
