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
