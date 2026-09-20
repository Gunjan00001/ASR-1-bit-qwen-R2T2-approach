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

## Next steps

1. Add grad-norm/NaN diagnostics; assert finite grads before `optimizer.step()`.
2. TDD an "overfit one batch" BitLinear test at alpha=1 to prove STE correctness.
3. Fix LoRA (torchao >= 0.16 or vendored LoRA) so only adapters + shadows train.
4. bf16/fp16 autocast, lower lr, longer warmup, larger labelled subset.
5. Re-run the fp16/binary/ternary WER table; then evaluate the Stage 1 gate.
