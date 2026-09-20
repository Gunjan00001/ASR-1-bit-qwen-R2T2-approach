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
