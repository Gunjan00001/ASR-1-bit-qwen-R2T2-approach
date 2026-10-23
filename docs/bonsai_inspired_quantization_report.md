# Bonsai-inspired quantization — final report

_2026-10-23. Baseline branch `stage-1` @ `de111d1`; experimental branch
`bonsai-quant` (worktree `../asr1bit-bonsai`). No baseline code or running job
was modified._

## A. Research findings (cited)

From the model card (<https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf>)
and the demo repository (<https://github.com/PrismML-Eng/Bonsai-demo>):

- **Representation:** ternary `{−1, 0, +1}` with **one FP16 scale per group of
  128** weights; ≈1.71 bits/weight for the code + scale, 1.72 for the model once
  26.2 M higher-precision parameters are counted.
- **Basis:** "each matrix is transformed blockwise by an orthogonal Hadamard
  rotation before the ternary assignment, and the runtime applies the matching
  transform to activations"; **block 1024, fixed ±1 signs**, rotation folded
  into stored weights, declared as metadata so a runtime either applies it or
  refuses the file.
- **Coverage:** embeddings, attention projections, MLP projections, LM head; the
  vision tower ships separately at Q8_0.
- **Packing/runtime:** `PTQ1_0` dense trits (1.75 bpw) and `PQ2_0` 2-bit slots
  (2.13 bpw), consumed directly by custom kernels in a llama.cpp fork.

**Not public:** the training/conversion pipeline, the calibration and
scale-initialisation algorithm, how the rotation interacts with training, group
axis, and handling of non-divisible dimensions. The demo repository is
inference/setup only. Full notes and limitations: `docs/bonsai_method_research.md`.

## B. Implementation audit

| item | finding |
|---|---|
| baseline scale granularity | **per-tensor** (`group_size = args.group_size or None`; no launcher passes it) |
| group layout | contiguous rows of the input dimension (`_group_view` reshape `(-1, g)`) |
| non-divisible dims | raises (`quant._group_view`, `BitLinear._scale_shape`) — no padding path |
| gradient | straight-through (`weight + (blended − weight).detach()`) |
| inference | same `BitLinear`, `alpha` 0/1, `bit_width`; equality holds **only if the evaluator builds the same policy** — the b17 incident was exactly that failure, now guarded by `POLICY=` |
| conventions | `(out, in)`, `y = x Wᵀ`, decoder-only projections |

Added this phase: `rotate.py` + 6 gates (branch `bonsai-quant`), a sharded
real-weight diagnostic, and the 2-bit quantizer with 3 tests. The rotation
equivalence test caught a real orientation bug (weights `W Rᵀ` vs activations
`R`); both now use `Rᵀ`.

## C. Real-weight diagnostics — 1.7B, 35 matrices (5 layers × 7 projections)

Weights: HF snapshot `Qwen/Qwen3-ASR-1.7B`, two shards, read with `safe_open`.

| arm | mean rel. L2 | min | max | vs baseline |
|---|---:|---:|---:|---:|
| A per-tensor (baseline) | **0.5490** | 0.5243 | 0.6171 | — |
| B g128 | 0.5231 | 0.5144 | 0.5461 | −4.7 % |
| B g64 | 0.5207 | 0.5121 | 0.5434 | −5.2 % |
| B g32 | 0.5159 | 0.5075 | 0.5381 | −6.0 % |
| C rotation, block 1024 | 0.5373 | 0.5146 | 0.6030 | −2.1 % |
| **D rotation + g64** | **0.5086** | 0.5079 | 0.5089 | **−7.4 %** |
| D rotation + g64, block 512 | 0.5085 | 0.5080 | 0.5088 | −7.4 % |
| D rotation + g64, block 256 | 0.5085 | 0.5079 | 0.5089 | −7.4 % |

Per-projection gains for the best arm (D): **q −12.0 %**, k −9.4 %, o −7.4 %,
gate −6.3 %, v −6.3 %, down −5.5 %, up −4.3 %. Rotation alone is *not* the
lever; only rotation **plus** group scales beats grouping alone, and then only
by ~2 points of relative error.

**Two structural findings:**

1. **D collapses every matrix to ≈0.5086 regardless of projection, layer or
   block size.** That is the ternary noise floor for near-Gaussian weights
   (theory ≈0.517 for Gaussian; observed 0.5086), so there is very little left
   for a representation change to win.
2. The provisional 25 % screening threshold is **not met** (best is 7.4 %), and
   this project has never converted reconstruction gains of this size into WER:
   the 0.6B ternary/g128 difference (8 %) accompanied no measurable WER move,
   while the WER gap to the gate is ~11 points. The threshold therefore looks
   *generous*, not conservative.

## D. Training results

`aligned17b17` (FFN-only, attention fp16) completed 2 000 steps; official
evaluation with the corrected `POLICY=decoder_ffn`:

| state | clean WER | loops |
|---|---:|---:|
| step 1 200 | _running_ (PBS `12466`) | — |
| step 1 600 | **119.17** | 392 |
| step 2 000 | **69.28** | 351 |

No Bonsai configuration was trained: the reconstruction gate was not met, and
GPU capacity was occupied by these official evaluations until 19:00.

## E. 2-bit comparison

Implemented (`quantize_2bit`, levels `{−3,−1,+1,+3}`, threshold 2× group mean;
`BitLinear` dispatch at `bit_width=2.0`), 3 tests passing, **lower reconstruction
error than ternary on synthetic weights**. **No training and no official WER** —
untrained, so no comparison to the trained ternary model is claimed.

## F. Deployment measurements

| quantity | ternary baseline | g64 | rotation+g64 |
|---|---:|---:|---:|
| per-weight code | ~2 bits packed | 2 | 2 |
| scale metadata | 16/N ≈ 0 | 16/64 = 0.25 | 0.25 |
| **effective bits/param (analytic)** | **≈2.00** | **2.25** | **2.25** |
| extra runtime work | — | none | activation Hadamard per block |

These are **analytic estimates**, not measurements: this repository has no packed
ternary runtime, so model size, peak memory and latency cannot be measured yet.
That limitation is explicit, per the brief.

## G. Reproducibility

| item | value |
|---|---|
| branches | `stage-1` @ `de111d1` (docs), `bonsai-quant` (rotation + diagnostics) |
| weights | `/tmp/asr1bit/hf/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e.../model-0000{1,2}-of-00002.safetensors` |
| diagnostic | `PYTHONPATH=src python scripts/bonsai_weight_diag.py --safetensors <snapshot dir> --out artifacts/stage1/bonsai_17b` |
| rotation gates | `cd ../asr1bit-bonsai && PYTHONPATH=src pytest -q tests/test_rotate.py` |
| 0.6B group data | `artifacts/stage1/diag_quant_error.json` |
| b17 evaluations | PBS `12466`–`12468`, `POLICY=decoder_ffn`, JSONs `artifacts/stage1/streaming_qat_stream_aligned17b17_*`; `*_WRONGPOLICY.json` excluded |
| seeds | rotation sign seed 0 (diagnostic); all runs deterministic by construction |
| CPU PBS attempt | `12477` (workq) — FAILED: node had no model weights; superseded by the local run |

## H. Final decision

**Retain the current ternary approach.** Do not adopt group-wise, rotation, or
the combination as a training change on this evidence.

Reasoning, with trade-offs:

- **Accuracy:** the best representation change buys **7.4 %** relative
  reconstruction error and flattens every matrix to the ternary noise floor
  (0.5086). The project's gap is ~11 WER points; no measured link supports
  closing it from here, and the one configuration actually trained with a
  changed precision policy (FFN-only) was catastrophic (69–119 WER).
- **Compression:** group scales *increase* effective bits (2.00 → 2.25 at g64)
  and rotation adds a mandatory activation transform at inference. Neither
  improves the compression objective.
- **Deployment:** no packed runtime exists here, so gains could not be realized
  even if they were large.
- **2-bit fallback:** promising on reconstruction and cheap to try, but
  **untrained**; it stays the single highest-value follow-up experiment once
  GPUs free up, because it changes the noise floor rather than nudging it.

If one representation change must be carried forward, it is **g64 group scales**
(cheapest to adopt, no runtime change, −5.2 % reconstruction), not rotation —
but it should be validated by official streaming WER before adoption, and that
pilot is **not queued**: it fails the pre-registered screening threshold, and
GPU time is better spent on the 2-bit arm and the distillation work already
audited.
