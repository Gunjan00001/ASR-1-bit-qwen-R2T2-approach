# Stage 1 results (T1.4/T1.5) — gate NOT met

Branch `stage-1` (off `main` `67f2a85` / `0.2.0`). Code T1.1–T1.4 is implemented
and unit-tested (164 passed, 1 skipped); the QAT run and WER ablation do **not**
yet pass the gate. `0.3.0` is intentionally **not** tagged.

## What is done

- **T1.1** `quant.py`: `quantize_binary` (centered sign), `quantize_ternary`
  (RoundClip), optional group-wise scaling, `dequantize_weights`.
- **T1.2** `bitlinear.py`: `BitLinear` (fp32 shadow, STE, bit-width switch,
  INT8 absmax activations, `from_linear`/`load_from_linear`, progressive alpha).
- **T1.3** `replace.py`: `apply_layer_policy` (default `decoder_attn`);
  encoder/embeddings/head/norms preserved.
- **T1.4** `train/qat.py`: progressive-alpha schedule, `set_qat_alpha`,
  `freeze_non_bitlinear`, `build_optimizer` (8-bit Adam w/ AdamW fallback),
  `train_step`, LoRA helpers, `run_qat`. `scripts/run_stage1_qat.py` runs QAT and
  emits the fp16/binary/ternary offline WER ablations.

Checkpoints `0.2.1`–`0.2.10` on `stage-1`.

## Colab run (env)

```
Colab Tesla T4 (sm75), python 3.13.15
torch 2.9.1+cu128 | transformers 4.57.6 | vllm 0.14.0
peft 0.20.0 | bitsandbytes 0.50.2 | torchao 0.10.0 (too old for peft LoRA; needs >0.16)
```

## Result (INVALID)

Model `Qwen/Qwen3-ASR-0.6B`, policy `decoder_attn` (112 BitLinear), steps 30,
train_n 48, eval_n 15 (fixed clean subset), lr 1e-4, Adam8bit, grad ckpt.

| Precision | Offline WER | CER |
|---|---|---|
| pretrained (baseline) | **2.65** | 0.76 |
| fp16 after QAT (alpha=0) | 256.06 | 201.17 |
| binary after QAT (alpha=1) | 100.00 | 100.00 |
| ternary after QAT (alpha=1) | 98.48 | 90.66 |

Loss: 3.64 (alpha 0) → 1.92 (alpha 0.44) → **15.06 (alpha 1.0)** → ~7.0.

Diagnosis: the pretrained baseline reproduces, so the harness is sound, but QAT
is unstable — the loss spikes at the quantization transition and the fp16
(alpha=0) readout is corrupted, i.e. the shadow weights themselves, not just the
quantized forward. LoRA was unavailable (torchao too old), so the fallback
trained BitLinear shadows; corruption persists, so the issue is not LoRA scope.
Suspects: gradient checkpointing vs the custom STE/patched forward, lr too high
for full shadow updates, tiny 48-example train set, Adam8bit behaviour.

## Debug phase (2026-09-20) — root cause narrowed

- **STE correctness:** TDD "overfit one synthetic batch" at α=1 passes
  (`tests/test_bitlinear.py::TestSteCapability`), and α=0 forward is *exactly*
  `nn.Linear`. So STE gradients are correct and sufficient.
- **Diagnostics:** `grad_norms` + `check_finite_grads` added and used every step.
- **Stabilized config:** QAT-only freeze (176.16M shadow params), fp32 AdamW,
  lr 1e-5 + warmup, constant α=1, no grad-ckpt, no activation quant, no autocast.

| Run (40 steps, 48 train / 15 eval, α=1, lr 1e-5) | pretrained | fp16 (α=0) | binary (α=1) | ternary (α=1) |
|---|---|---|---|---|
| AdamW, no grad-ckpt, no act-quant | 2.65 | **2.65 (clean)** | 353.79 | 170.08 |
| Adam8bit, no grad-ckpt, no act-quant | 2.65 | **2.65 (clean)** | 395.45 | 179.92 |

Grad-norm log (AdamW run, pre-clip per-step max): 519.09 (v_proj) → 98.5 → 35.2
→ 16.7 → 8.9 → … → 9.9; all finite. Loss 18.26 → 6.94 (no explosion).

**Findings:**
- The stabilized config **fixes the shadow corruption** (post-QAT fp16 ==
  pretrained, exactly).
- **8-bit Adam is not the trigger** (clean fp16 with it).
- The original 256% corruption came from one of the remaining differences vs the
  failing run: **α-ramp**, **gradient checkpointing**, **activation quant**, or
  **lr 1e-4** (leading suspect: α-ramp vs identity-STE, or grad-ckpt with the
  patched forward).
- Naive binary/ternary of decoder attention is catastrophic (~170–395% WER) even
  when training is stable — that is the Stage 2 accuracy-recovery problem.

## Decisive recovery measurement (2026-09-20) — bug, not recovery

Colab T4, `decoder_attn` (112 BitLinear), QAT-only freeze, fp32 AdamW, lr 1e-5.

- **Zero-shot floor** (α=1, no training, eval-n 100): pretrained/fp16 **2.93**,
  binary **100.0**, ternary **140.8** WER.
- **1k-step QAT** (train 200, eval 30), periodic α=1 WER:

| Run | step 250 | step 500 | step 750 | step 1000 | final fp16 | loss |
|---|---|---|---|---|---|---|
| constant α=1, binary | 434.2 | 99.1 | 104.0 | **191.3** | 2.62 (clean) | 5.42 |
| constant α=1, ternary | 116.4 | 100.2 | 101.6 | 130.0 | 2.62 | 5.42 |
| α-ramp, binary | 130.2 | 182.9 | 107.2 | **70.5** | **10.30 (corrupt)** | 1.02 |
| α-ramp, ternary | 163.5 | 199.8 | 195.6 | 250.1 | 10.30 | 1.02 |

**Verdict: not a clean recovery.** Constant-α binary ends *worse* than the floor
(191 vs 100) with a clean fp16 readout; the ramp improves binary (→70.5) but
corrupts fp16 (10.3). A ramp loss of **1.02** (below full precision ~3.6) with
~100% generation WER indicates a **teacher-forced vs autoregressive mismatch**
and/or that binarizing decoder attention fundamentally breaks generation — i.e.
a bug, not merely a Stage 2 recovery gap. The α-ramp is a confirmed corruption
trigger.

Artifact: `artifacts/stage1/recovery_2026-09-20.json`.

## Next steps

1. **Decisive generation test:** after QAT, greedily generate on a *training*
   utterance; if it does not reproduce the label, it is a quantized-forward /
   generation bug.
2. Compare teacher-forced logits vs generated tokens to locate the divergence.
3. Finish trigger isolation: grad-ckpt → activation quant (α-ramp confirmed).
4. Revisit per-layer scale handling (per-tensor centered-sign may be too coarse
   for attention); try group-wise scales or exclude attention.
5. `0.3.0` stays untagged until the Stage 1 gate is genuinely met.
