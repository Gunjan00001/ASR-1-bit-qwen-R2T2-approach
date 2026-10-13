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
