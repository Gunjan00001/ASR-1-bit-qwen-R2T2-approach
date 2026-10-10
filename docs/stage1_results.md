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

## Instrumented generation test (2026-09-20) — forward/objective bug

`scripts/diag_generation.py`, one training utterance, α=1:

| condition | loss | token acc | teacher-forced text | greedy text |
|---|---|---|---|---|
| fp16 control (α=0) | 3.67 | 0.75 | "language wouldOULD THE PAPERS TALK ABOUT IT?" | "How would the papers talk about it?" |
| binary naive (α=1) | 18.26 | 0.00 | Chinese-token collapse | Chinese-token collapse |
| binary trained 300 steps | 4.48 | 0.17 | "WE IOULD ITINGEREHE" | "WE IS I WOULD IT W" |

- **Q(W) assertion PASSES:** at α=1 the effective weight differs from the shadow
  (39.18 → 29.48, delta 25.81); at α=0 unchanged. The loss forward uses Q(W).
- **Both teacher-forced and greedy are bad** ⇒ not a KV-cache/decode-only bug;
  the **quantized forward/objective is broken** (binarizing all decoder attention
  q/k/v/o collapses the model; 300 steps at lr 1e-5 does not recover).
- Flagged: the fp16 control shows **prefix leakage** in the teacher-forced text →
  label masking / prefix-length alignment may be off.

Artifact: `artifacts/stage1/diag_generation_decoder_attn_steps300.json`.

## Masking verified + per-projection sensitivity (2026-09-20)

**Masking is correct** (mirrors Qwen's official `qwen3_asr_sft.py`: prefix
template rendered with a dummy audio placeholder; `make_sft_labels` masks the
prompt). Direct check: `LABEL == TARGET` exactly ("THE LADY IS NOT THE MOTHER OF
THE BOYS BUT THEIR AUNT"), `FULL[:prefix_len] == PREFIX_INPUTS`. The earlier
sub-1.0 teacher-forced token accuracy is a **format artifact** (uppercase
LibriSpeech refs vs the model's cased/punctuated output + language tag), not
misalignment — the fp row has WER 3.18 with token acc 0.78.

**Per-projection sensitivity** (α=1 on one projection; 20-utt WER):

| binarized | loss | token acc | WER |
|---|---|---|---|
| none (fp) | 2.22 | 0.78 | **3.18** |
| q | 5.07 | 0.22 | 101.59 |
| k | 6.83 | 0.22 | 139.49 |
| v | 10.75 | 0.00 | 100.00 |
| o | 8.51 | 0.00 | 195.22 |
| all | 18.48 | 0.00 | 100.00 |

**Binarizing any single attention projection is already catastrophic** — not a
"too many layers" effect. Forward uses Q(W) (proven), labels are correct
(proven); the leading suspect is now the **quantizer/scale math** (per-tensor
centered-sign), not attention breadth.

Artifacts: `artifacts/stage1/diag_generation_decoder_attn_steps0.json`,
`artifacts/stage1/diag_projection_decoder_attn.json`.

## Decisions

- **Drop the blended α-ramp** (confirmed corruption trigger). Replace with
  **quantization delay** (`quantization_delay_alpha`, `--alpha-mode delay`).
- Use a fixed **≥100-utterance** eval set for the floor and every checkpoint.
- Label masking now matches Qwen's official convention (dummy-audio prefix).
- Do **not** pursue attention exclusion yet; the per-projection table points at
  the quantizer/scale, which is the next target after the QAT re-run.

## Next steps

1. Re-run QAT with quantization delay, ≥100-utt eval, higher lr (1e-4/2e-4),
   more steps/data.
2. Revisit the quantizer math (per-tensor centered-sign; scale granularity).
3. Finish trigger isolation (grad-ckpt → activation quant).
4. `0.3.0` stays untagged until the Stage 1 gate is genuinely met.

---

# Stage 1 results — A100 session (2026-10-10)

Environment: JIIT cluster, `jiit-gpu01`, 8 × A100-80GB (shared, PBS `gpu`
queue), python 3.13.12, **torch 2.11.0+cu126, transformers 4.57.6,
qwen-asr 0.0.6**, numpy 2.5.3. Model fp32; all QAT-only (BitLinear shadows +
learnable scales trainable, 440M params for `decoder_attn_ffn`).

## Infrastructure fixes (see `docs/LAB_NOTES.md`)

- **Fast tokenizer** (`scripts/install_fast_tokenizer.py`): checkpoints ship no
  `tokenizer.json`; the slow Python tokenizer rebuild looked like a hang
  (multi-minute stalls under cluster FS/CPU contention). 11.6 s full load after.
- **Streaming data** (`scripts/build_qat_cache.py --stream 1`, run on the login
  node): the non-streaming `datasets` load blew the 32 GB PBS cgroup
  (`oom-kill` in dmesg). Cache: 4 000 utterances `train.clean.100`, 14.5 h
  audio, mean 13.0 s/utt, 3.34 GB.
- **Node-local staging** of conda/model/cache on `/tmp` for fast startup.

## Quantizer audit (`scripts/diag_quant_error.py`)

Relative Frobenius error `||Q(W)-W||/||W||` over all 196 decoder projections
(released checkpoint, fp32):

| config | avg | note |
|---|---|---|
| binary, per-tensor | **0.648** | closed form ≈ 0.60 for Gaussian weights |
| ternary, per-tensor | 0.565 | |
| binary, group 64/128 | 0.616 / 0.614 | sign pattern dominates |
| ternary, group 64/128 | 0.519 / 0.522 | |

**Verdict:** quantizer math is correct; the earlier single-projection collapse
(WER 100) is inherent naive 1-bit behaviour, not a harness bug.

## Loss controllability (single-utterance memorization, 600 steps, lr 1e-3)

| condition | naive loss | trained loss | token acc | greedy |
|---|---|---|---|---|
| α=0 fp control | 2.225 | **0.000** (step 300) | 1.000 | exact |
| binary α=1 | 18.49 | 2.818 (step 600) | 0.111 | "THE THE" |
| ternary α=1 | 11.63 | **0.000** (step 300) | 1.000 | exact label |

**Ternary QAT is fully controllable** — the quantized model can memorize
through `Q(W)`. Binary lags (loss 2.8), consistent with the literature.

## Real-data QAT runs (4 000 train.100 utts, fixed 50-utt test-clean eval)

Config: ternary, `decoder_attn_ffn`, learnable **log-parameterized** scales
(first run showed gamma collapsing to ~0 / going negative with a free
parameter — fixed), lr 2e-4, 100-step warmup, cosine, batch 4 (distill: 2),
5 % fp warmup then α=1, lazy collation, shadow checkpoints.

| run | steps | periodic ternary WER | final fp16 readout | final binary | final ternary |
|---|---|---|---|---|---|
| naive (v1, free gamma) | 3 000 | 93.6 @500 → 232.9 @1000 (killed) | — | — | — |
| naive (v2, log gamma) | 6 000 | 127 → 121.6 → **96.5** → 96.9 → 103.5 | **113.1** | 99.8 | 107.8 |
| self-distillation (w=0.5, T=2) | 4 000 | 114.6 → 125.0 → 111.3 → **98.1** | **108.3** | 118.3 | 98.1 |
| **FFN-only ternary** | 3 000 | 118.0 → 84.3 → 42.2 → 25.7 → 32.8* | 226.5 | 71.0 | **32.8** |
| FFN-only binary | 3 000 | (binary) 97.0 → 387.3 → 100.6 → 93.9 → 91.9 → 97.9 | 223.9 | **97.9** | 106.8 |
| **Stage 2a: attn+FFN ternary, FFN-warm-start + anchor w=0.1** | 2 000 | 52.8 → 38.0 → 23.1 → **22.6** | 97.6 | 99.5 | **22.6** |
| **Stage 2a-ext2 (warm restart, lr 2.5e-5, 5 000)** | 5 000 | 19.4 → 17.5 → 17.7 → **14.4** → 16.2 | 85.8 | 99.9 | **16.2** (best 14.4) |
| **Stage 2b: binary rollout from ternary** | 2 000 | 51.5 → 38.1 → 25.4 → **17.5** | 99.8 | **17.5** | 30.5 |
| anchored distillation (frozen teacher, w=0.5, lr 5e-5) | 4 000 | **315.4 @1000**, fp16 427 (killed) | — | — | — |

\* the 25.7 @2000 is the best FFN checkpoint; the resumed run (fresh
optimizer, LR schedule restarted before the fix) reached 32.8 @3000.

Reference: pretrained fp16 = **2.44** WER (50-utt subsample); zero-shot naive
ternary ≈ 100.

**Stage 2a is the headline result:** all 196 decoder projections
(attention + FFN) ternary at **16.2 WER final / 14.4 best** — the configuration
that collapses to ~100 with naive QAT. Recipe: train FFN-only first (84 layers,
recovers to 25.7), then quantize attention with a **frozen-pretrained-teacher
anchor (w=0.1, T=2)** at lr 5e-5, then a **warm-restart cosine cycle** (lr
2.5e-5 over 5000 steps). Trajectory: 52.8 → 38.0 → 23.1 → 22.6 → 21.5 → 19.4 →
17.5 → 17.7 → 14.4 → 16.2.

**Progressive precision for binary (Stage 2b):** starting from the ternary
rollout checkpoint and switching `bit_width=1.0` gives binary
51.5 → 38.1 → 25.4 → **17.5** (2 000 steps) — far better than binary FFN-only
(best 91.9) and the naive binary collapse (~100). Final table: pretrained 2.44,
fp16 99.8, **binary 17.5**, ternary readout 30.5.

**Shadow-drift pattern (all long runs at lr 2e-4):** the fp16 readout of the
trained shadows degrades to WER ~110–230 even when the quantized readout
recovers (FFN-only ternary: fp16 226.5 vs ternary 32.8). The STE updates move
the latent weights far from the pretrained basin; the *quantized* function is
what recovers. This is acceptable for a quantized deliverable but means an
anchored objective is needed to preserve the fp path (Stage 2a).

**FFN-only (`decoder_ffn`: attention stays fp) recovers** — ternary WER
118 → 84 → **42** → **25.7** at steps 500/1000/1500/2000; training loss 4.1 →
2.1. Keeping the attention projections fp preserves audio conditioning; the FFN
ternary bottleneck is recoverable at this budget. **Binary FFN-only does not
recover** within 3 000 steps (best 91.9, ends 97.9) — binary remains the stretch
target and needs the Stage 2 stack.

**Failure mode (checkpoint inspection):** the collapsed attn+ffn models emit
fluent generic text unrelated to the audio ("AND THE FIRST OF THE HUMAN BODY…")
— the decoder loses audio conditioning and falls back to the language prior
(fp16 and ternary readouts identical at WER 112). Attention quantization is the
cause; the FFN-only result confirms it.

## Findings (2026-10-10)

1. **Naive and self-distillation QAT both fail to recover** at this scale/budget:
   ternary WER plateaus at ~98–108, no better than zero-shot (~100), while the
   teacher-forced loss descends (18.9 → ~4.5–6).
2. **The fp16 readout degrades** (2.44 → ~108–113) after 4–6 k steps at
   lr 2e-4 — the shadows drift far from the pretrained basin (unlike the
   earlier T4 runs at lr 1e-5, which kept the fp readout clean). Training at
   this lr is *destructive*: the optimizer follows the quantized loss into a
   degenerate solution (periodic WER > 100 suggests repetitive/empty
   generations; checkpoint inspection artifact pending).
3. Self-distillation lowers the KL to the fp teacher (~0.4–0.6) and speeds the
   CE descent, but the fp teacher is the same drifting shadows, so it does not
   prevent the collapse.
4. Learnable scales must be **positive by construction**: the first run showed
   a layer's gamma at −0.004 and others collapsed ~10×; log-parameterization
   (`gamma = exp(log_scale)`) fixed the pathology (gamma stays ~0.014–0.036).
5. Gamma stats in the runner show the scales staying healthy in v2 — the
   failure is not a scale collapse.

## Interpretation

Post-hoc ternary conversion of a 0.6 B ASR model by QAT on 4 k utterances is in
the **destructive regime** when attention is quantized: the sign bottleneck
(per-tensor ternary, ~52 % weight error) is too coarse for the attention
pathway, the decoder loses audio conditioning, and the optimizer drifts the
latent weights into the language prior. **FFN-only quantization recovers**
(118 → 42 WER in 1.5 k steps, still descending) — attention must stay
higher-precision at this scale, matching the per-projection sensitivity table.
The Stage 2 recovery stack should be built on the FFN-only surface first, then
attention added incrementally with an anchored/trust-region objective.

## Next steps (Stage 2 priorities)

1. **FFN-only is the lead configuration** — finish the ternary run (expect
   WER ≈ 20–30 by 3 000 steps), then binary FFN-only for the table.
2. **Add attention back incrementally** with an anchored objective: small KL
   weight (≤0.1) to a frozen fp teacher, or per-layer quantization rollout
   (FFN first, attention last), rather than flat w=0.5 KL (destructive, 315 WER).
3. **Co-training / stochastic precision**: alternate fp and quantized steps so
   the fp path remains strong.
4. Only then extend training time and data.
5. `0.3.0` stays untagged: the gate's "WER table emitted" is satisfied and the
   loss is controllable, but recovery is still in progress; keep Stage 1 open.
