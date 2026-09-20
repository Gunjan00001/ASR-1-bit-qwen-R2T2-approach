# ASR-1-bit-Qwen: 1-Bit Streaming ASR (Qwen-only)

**Status:** ACTIVE  |  **Version:** 0.1.1  |  **Owner:** Gunjan00001
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
- **Our** model (Stage 4): true streaming, **append-only** — committed text is
  never revised. The Qwen3-ASR baseline is *pseudo-streaming* (`※`) and may
  revise its unfixed tail; this bar applies to our LSP policy, not the baseline.
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
  (~300 M for 1.7B, ~180 M for 0.6B). Native `transformers` support exists
  (since 2026-06-26, `*-hf` checkpoints + `torch.compile`), but **streaming is
  vLLM-only** — the transformers backend cannot reproduce streaming.
- **Published streaming WER** (LibriSpeech `clean|other`): 0.6B `2.54|6.27`,
  1.7B `1.95|4.51`. Official streaming defaults: `chunk_size_sec=2.0`,
  `unfixed_chunk_num=2`, `unfixed_token_num=5`, greedy.
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

Decisions: Kaggle **T4** (not P100) for gate runs; local Python 3.12 venv;
gate = reproduce published streaming WER at 2.0 s + lock the 160/320/1000 ms
matrix; eval test-clean and test-other; stage-0 branch, merge to main per task,
tag `0.2.0` at gate.

- **T0.0 vLLM go/no-go spike (blocking).** Prove `qwen-asr[vllm]` streaming
  runs on Kaggle **T4** — vLLM nightly is **cu129** and P100/Pascal is
  unsupported. If it fails, move the streaming gate to Colab Pro (A100/L4) or
  the `qwenllm/qwen3-asr` Docker image.
- **T0.1 scaffold + env.** Python 3.12, `.venv`, `.python-version`, deps
  (`torch`, `transformers`, `qwen-asr`, `datasets`, `jiwer`, `soundfile`,
  `numpy`, `pyyaml`), `requirements-stage0.txt`, `tests/conftest.py`.
- **T0.2 data loader.** `load_manifest` + `iter_librispeech` -> 16 kHz mono
  float32 + normalized text. **Normalization sub-task:** replicate Qwen's text
  normalization exactly and share it between `data/load.py` and `eval/wer.py`
  (jiwer defaults will NOT reproduce published WER).
- **T0.3 fp16 streaming baseline.** `ASRStreamingState` + `ChunkedStreamer`
  replicating the official algorithm: per-chunk re-feed of accumulated audio,
  `chunk_size_sec=2.0`, `unfixed_chunk_num=2`, `unfixed_token_num=5`, greedy.
  Backends: vLLM (official reference) and transformers (offline + future
  BitLinear path). **Do not assert append-only on the Qwen baseline** — it is
  pseudo-streaming (`※`) and may revise text; assert rollback matches the
  official algorithm. Per-utterance results persisted for session resume.
- **T0.4 WER / latency / RTF harness.** jiwer WER/CER + corpus aggregation,
  retrospective chunk latency, CPU RTF, `WERReport`, matrix runner -> CSV/JSON.

**Gate:** reproduce published streaming WER (LibriSpeech `clean|other`:
0.6B `2.54|6.27`, 1.7B `1.95|4.51`) within tolerance; then lock `X` with the
owner and tag `0.2.0`.

**Feasibility:** the full streaming matrix is infeasible (streaming is
sequential, no batching, ~O(n²) re-feed). Offline = full splits (batched).
Streaming = official `2.0 s` on **full test-clean** + a fixed ~500-utterance
random subsample of **test-other**, plus `320 ms` on that same subsample.
Label subsampled results explicitly.

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

- Datasets: LibriSpeech `test-clean` / `test-other` (primary); one noisy set
  (secondary).
- **Pinned official protocol (for reproduction):** `dtype=bfloat16`, greedy
  decoding, `language=None` (no language parameter), `max_new_tokens=1024`
  offline / `32` streaming, vLLM backend. Record `vllm` / `torch` / `qwen-asr`
  versions with every result table.
- Chunk sizes **160 / 320 / 1000 ms**; official baseline at **2.0 s**
  (`unfixed_chunk_num=2`, `unfixed_token_num=5`). Report retrospective chunk
  latency.
- CPU RTF and packed model size.
- Compare fp16 vs ternary vs binary.
- External target: R2T2 published English numbers (benchmark only).
- Streaming is vLLM-only and single-utterance (no batching, no timestamps), so
  streaming evals use the Stage 0 subsampling policy.

## 12. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Binary misses bar at 1.7B | Ternary switch (realistic ship) |
| Colab A100 unavailable / OOM | LoRA / partial-layer QAT, smaller batch |
| LSP details unpublished | Our approximate emission policy; accept higher latency |
| OPD compute cost | Bounded recovery phase only |
| vLLM nightly (cu129) won't run on Kaggle | T0.0 spike; use T4; fall back to Colab Pro / Docker |
| Full streaming matrix infeasible | Subsample test-other; 2 chunk sizes; offline full |
| WER won't reproduce (normalization) | Replicate Qwen's normalizer; pin protocol/versions |

## 13. Version control

- Remote: `https://github.com/Gunjan00001/ASR-1-bit-qwen-R2T2-approach.git`
- Tag format: **`MAJOR.MINOR.BUGS`** (third component = patch/bug count).
  No leading `v`.
- `0.1.0` = initial scaffold; `0.1.1` = Stage 0 plan corrections.
  Bump **MINOR** per completed stage; **BUGS** per fix; **MAJOR** on milestone
  (e.g., `1.0.0` = 1.7B binary streaming pipeline works end-to-end).
- Stage 0 gate tag is `0.2.0`.

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
