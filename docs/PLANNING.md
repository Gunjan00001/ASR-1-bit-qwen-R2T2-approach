# ASR-1-bit-Qwen: 1-Bit Streaming ASR (Qwen-only)

**Status:** ACTIVE  |  **Version:** 0.1.0  |  **Owner:** Gunjan00001
**Last updated:** 2026-09-20

---

## 1. Goal

Build an **English streaming ASR** with near-binary weight quantization
(target `{-1, +1}`, ternary `{-1, 0, +1}` fallback) that runs on **CPU/edge**
and matches the **published streaming quality of R2T2** — using **Qwen models
ONLY**.

R2T2 is used *only as an external benchmark target*. No R2T2 weights or code
are used anywhere in this project.

## 2. Non-goals

- No use of R2T2 weights or code.
- No multilingual target (English only).
- No distribution / commercialization (research use only).

## 3. Success criteria (fixed bars)

- LibriSpeech `test-clean` / `test-other` WER at **320 ms chunk** within `+X`
  absolute of the fp16 Qwen3-ASR-1.7B baseline. `X` is locked after Stage 0
  (initial target: `+1.5` absolute).
- True streaming: **append-only** — committed text is never revised.
- CPU: **RTF < 1** and **constant memory** as audio length grows.
- Ternary fallback is an acceptable outcome if binary misses the bar.

## 4. Constraints

- Python 3.12.
- Dev hardware: Kaggle T4/P100 (16 GB, ~30 h/week); Colab Pro A100 40 GB / L4 22 GB.
- fp32 **shadow weights**; fp16 compute on T4/L4, bf16 on A100.
- 8-bit Adam + LoRA + gradient checkpointing for all QAT runs.
- Allowed models (all Apache-2.0):
  - `Qwen/Qwen3-ASR-0.6B`
  - `Qwen/Qwen3-ASR-1.7B`
  - `Qwen/Qwen3-ForcedAligner-0.6B`

## 5. Background (verified)

- **Qwen3-ASR-1.7B / 0.6B**: Apache-2.0; native streaming inference; official
  finetuning toolkit; architecture = Qwen3 LLM + projector + AuT encoder
  (~300 M for 1.7B, ~180 M for 0.6B).
- **Published streaming WER** (LibriSpeech, avg): 0.6B `2.54`, 1.7B `1.95`.
- **R2T2** is a Youdao fine-tune of Qwen3-ASR-1.7B adding stable-prefix
  ("Longest Stable Prefix", LSP) streaming. Its model weights use a
  restrictive license; only its *published numbers* are referenced here.
- **Prior art**:
  - BitNet b1 (binary) → BitNet b1.58 (ternary): the FP16-vs-low-bit gap
    shrinks as model size grows; ternary matches FP16 from ~3B.
  - "Towards One-bit ASR" (Interspeech 2025, arXiv:2505.21245): naive 1-bit
    QAT gives **+52% relative WER**; recovered via co-training, stochastic
    precision, and tensor-wise learnable scales.
  - VibeVoice-ASR-BitNet (Microsoft): ternary QAT + ggml SIMD, **RTF < 1 on
    3 CPU threads**.
- **DeepSeek-V4.1-Flash tech report**: *not* about weight quantization (it is
  KV-cache compression for a 552B MoE). Transferable ideas:
  1. **Heterogeneous precision** — sensitive components kept higher (their
     SWA KV kept FP8 while main KV went FP4).
  2. **Sliding-window attention** for bounded memory.
  3. **QAT applied in post-training**, warm-started from a full-precision model.
  4. **On-policy distillation** to recover quality.
  5. Low-bit KV cache, cross-layer KV reuse, speculative decoding (latency).

## 6. Design decisions

- **D1 — Model targets:** stage on **0.6B**, deliver on **1.7B**. Never R2T2.
- **D2 — What is binary:** decoder `q/k/v/o` + `gate/up/down` projections only.
  Keep higher precision: embeddings + LM head (~6-bit), LayerNorm/softmax (fp),
  AuT audio encoder + attention KV (FP8/INT8).
- **D3 — Quantizers:** binary `sign(W - mean(W)) * mean|W|`; ternary
  `RoundClip(W / γ, -1, 1)` with `γ = mean|W|`. fp32 shadow weights + STE;
  INT8 absmax per-token activations.
- **D4 — Streaming:** fixed-left-context chunked decode, **no future lookahead**;
  stable-prefix append-only emission; sliding-window attention to bound memory.
- **D5 — Recovery stack:** progressive quantization ramp; on-policy
  distillation from the fp Qwen teacher; multi-precision co-training +
  stochastic precision; tensor-wise learnable scales.
- **D6 — Evaluation:** WER (jiwer), retrospective chunk latency, CPU RTF,
  packed model size.
- **D7 — Deployment:** pack 1-bit/2-bit weights, ggml/bitnet.cpp SIMD kernels,
  low-bit KV; optional tiny drafter for speculative decode.

## 7. Architecture / data flow

Inference:

```
audio -> 16 kHz mono
      -> AuT encoder (higher precision)
      -> projector
      -> Qwen3 decoder (BitLinear weights, quantized)
      -> incremental text
      -> LSP emit
      -> append-only committed transcript
```

Training data flow:

```
LibriSpeech / CommonVoice -> manifests
Qwen3-ForcedAligner-0.6B  -> word/token timestamps
  -> stable-prefix + chunk/lookahead samples
  -> streaming SFT + QAT
```

## 8. File structure

```
src/asr1bit/quant.py          # binary/ternary quantizers, STE, absmean, RoundClip
src/asr1bit/bitlinear.py      # BitLinear drop-in for nn.Linear; bit-width switch
src/asr1bit/replace.py        # layer policy: which linears become BitLinear
src/asr1bit/data/load.py      # LibriSpeech/CommonVoice manifests -> 16k mono
src/asr1bit/data/align.py     # Qwen3-ForcedAligner -> word/token timestamps
src/asr1bit/data/prefix.py    # stable-prefix + chunk/lookahead targets
src/asr1bit/qwen/stream.py    # incremental chunked decoding + LSP emission
src/asr1bit/train/qat.py      # progressive QAT loop
src/asr1bit/train/opd.py      # on-policy distillation from fp teacher
src/asr1bit/train/cotrain.py  # multi-precision co-training / stochastic precision
src/asr1bit/train/sft.py      # streaming stable-prefix SFT
src/asr1bit/eval/wer.py       # WER/CER
src/asr1bit/eval/latency.py   # retrospective chunk latency + RTF
src/asr1bit/eval/harness.py   # config matrix -> result tables
src/asr1bit/deploy/export.py  # pack weights -> GGUF
tests/                        # one test module per src unit
configs/                      # YAML run configs
```

## 9. Staged plan

### Stage 0 — Baselines & harness
- T0.1 scaffold + env
- T0.2 data loader
- T0.3 fp16 streaming baseline
- T0.4 WER/latency/RTF harness

**Gate:** reproduce published streaming WER within tolerance.

### Stage 1 — BitLinear + QAT on 0.6B
- T1.1 quantizers
- T1.2 BitLinear
- T1.3 layer policy
- T1.4 progressive QAT
- T1.5 ablations

**Gate:** binary/ternary loss controllable; WER table emitted.

### Stage 2 — Accuracy recovery
- T2.1 on-policy distillation
- T2.2 co-training / stochastic precision
- T2.3 recovery run + eval

**Gate:** measurable WER recovery vs Stage 1.

### Stage 3 — Scale to 1.7B
- T3.1 memory stack
- T3.2 config + checkpoint/resume
- T3.3 run + eval

**Gate:** fits Kaggle T4 / Colab L4 with documented fallback; WER vs fp16.

### Stage 4 — Streaming parity (R2T2 method replacement)
- T4.1 alignment builder
- T4.2 stable-prefix targets
- T4.3 streaming SFT
- T4.4 LSP emission policy
- T4.5 streaming run + eval

**Gate:** append-only invariant holds; WER/latency vs R2T2 published.

### Stage 5 — Edge deployment
- T5.1 packing
- T5.2 GGUF export
- T5.3 CPU kernels
- T5.4 bounded-memory streaming

**Gate:** RTF < 1 on target CPU; constant memory.

### Stage 6 — Evaluation & writeup
- T6.1 matrix runner
- T6.2 tables/plots
- T6.3 report

## 10. Key interfaces (locked)

- `quantize_binary(W) -> (qW, scale)`
- `quantize_ternary(W) -> (qW, scale)`
- `BitLinear(in_dim, out_dim, bit_width, quantize_activations)`
- `apply_layer_policy(model, policy) -> model`
- `LSPState` + `emit_stable(state) -> (committed, pending)`
- `run_baseline(model_id, mode, chunk_ms) -> WERReport`
- `export_gguf(model, out_path)`

## 11. Evaluation protocol

- LibriSpeech `test-clean` / `test-other` (primary); one noisy set (secondary).
- Chunk sizes **160 / 320 / 1000 ms**; report retrospective chunk latency.
- CPU RTF and packed model size.
- Compare fp16 vs ternary vs binary.
- External target: R2T2 published English numbers (benchmark only).

## 12. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Binary misses bar at 1.7B | Ternary switch (realistic ship) |
| Colab A100 unavailable / OOM | LoRA / partial-layer QAT, smaller batch |
| LSP details unpublished | Our approximate emission policy; accept higher latency |
| OPD compute cost | Bounded recovery phase only |

## 13. Version control

- Remote: `https://github.com/Gunjan00001/ASR-1-bit-qwen-R2T2-approach.git`
- Tag format: **`MAJOR.MINOR.BUGS`** (third component = patch/bug count).
- `0.1.0` = initial scaffold. Bump **MINOR** per completed stage; **BUGS** per
  fix; **MAJOR** on milestone (e.g., `1.0.0` = 1.7B binary streaming pipeline
  works end-to-end).

## 14. Open questions

- Lock the baseline WER delta (`X`) after Stage 0.
- Confirm no R2T2-derived data or weights anywhere.

## 15. References

- Qwen3-ASR technical report — arXiv:2601.21337
- Qwen3-ASR model card / repo — `Qwen/Qwen3-ASR-1.7B`, `QwenLM/Qwen3-ASR`
- R2T2 — `netease-youdao/Confucius4-R2T2` (benchmark target only)
- BitNet — JMLR 2025, "BitNet: 1-bit Pre-training for Large Language Models"
- Towards One-bit ASR — arXiv:2505.21245 (Interspeech 2025)
- VibeVoice-ASR-BitNet — Microsoft Research technical report
- DeepSeek-V4.1-Flash — DeepSeek-AI technical report
