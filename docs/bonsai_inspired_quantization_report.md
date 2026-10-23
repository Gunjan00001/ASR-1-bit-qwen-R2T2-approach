# Bonsai-inspired quantization investigation

_2026-10-23. Main branch `stage-1` @ `df92ae3`; experimental branch `bonsai-quant`
@ `e700b6a` (worktree `../asr1bit-bonsai`, outside the repo). No Bonsai change
touches the baseline quantizer or any running experiment._

## 0. First-response status

| item | value |
|---|---|
| branch / commit | `stage-1` @ `df92ae3` (baseline), `bonsai-quant` @ `e700b6a` (experiment) |
| scheduler | PBS; GPU queue; 3 jobs of ours in flight (`12466`–`12468`, official b17 evals), all `R` for ~1 h 25 m |
| GPU availability | none free ≥45 GB at audit time — **no GPU work queued** |
| b17 evals | steps 1 200 / 1 600 / 2 000, `POLICY=decoder_ffn`; **no JSON yet** (`*_WRONGPOLICY.json` quarantined and excluded) |
| quantizer path | `src/asr1bit/quant.py` (`quantize_ternary`, `dequantize_weights`, `_group_view`) |
| scale grouping | `src/asr1bit/bitlinear.py` (`group_size`, `_scale_shape`, `gamma()`); CLI `--group-size` in `scripts/run_stage1_qat.py:213` |
| isolated output dir | `artifacts/stage1/bonsai_17b/` (diagnostics) and, for any future training, `data/checkpoints/bonsai_*` |
| offline tests added | `tests/test_rotate.py` (6 gates) on `bonsai-quant`; full suite **359 passed** there |
| implementation uncertainty | whether real 1.7B projections are outlier-heavy enough for rotation to help; and whether a rotated model is trainable under the existing recipes (both require real weights) |

## Phase A — audit (read-only; baseline unmodified)

1. **Scale granularity in the baseline: per-tensor.** `group_size = args.group_size or None`
   and no launcher passes `--group-size`, so every BitLinear uses one scale per
   projection (the recovered checkpoints record `gNone` in their run tags).
2. **Group layout.** `_group_view` reshapes the weight to `(-1, group_size)` and
   reduces along the last dimension, so groups are contiguous rows of the input
   dimension. `bitlinear._scale_shape` requires `numel % group_size == 0` and
   raises otherwise — there is no padding path.
3. **Non-divisible dimensions.** `quant._group_view` raises if `group_size` does
   not divide the last dimension; `BitLinear` raises if it does not divide
   `numel`. No silent truncation.
4. **Gradient path: straight-through.** `BitLinear._quantized_weight` computes
   `output = weight + (blended - weight).detach()` — forward uses the quantized
   value, gradients flow to the fp32 shadow.
5. **Inference reproduction.** The same `BitLinear` is used at eval time with
   `alpha` (0 = fp16 shadow, 1 = quantized) and `bit_width`; so simulated and
   deployed maths match **provided the evaluator builds the same policy** — the
   b17 failure was exactly a policy mismatch, now guarded by `POLICY=`.
6. **Conventions.** `nn.Linear` semantics, weight `(out, in)`, `y = x W^T`;
   policies match `self_attn.{q,k,v,o}_proj` and `mlp.{gate,up,down}_proj` in the
   decoder only (encoder excluded by name tokens).
7. **Ternary reconstruction error (measured, 0.6B, `diag_quant_error.json`):**

| projection | per-tensor | g64 | g128 |
|---|---:|---:|---:|
| q | **0.6205** | 0.5176 | 0.5229 |
| k | 0.5810 | 0.5212 | 0.5326 |
| gate | 0.5609 | 0.5159 | 0.5264 |
| o | 0.5562 | 0.5143 | 0.5215 |
| v | 0.5503 | 0.5221 | 0.5257 |
| down | 0.5483 | 0.5250 | 0.5319 |
| up | 0.5407 | 0.5165 | 0.5240 |
| **mean** | **0.5654** | **0.5190** | **0.5220** |

Group scaling helps most where per-tensor error is worst (q: −16.6 %), least
where it is already low (v: −5.1 %).

## Phase B — group-wise ablation

| arm | scale config | mean rel. L2 (0.6B) | vs G0 | scales per projection | approx. bits/param |
|---|---|---:|---:|---:|---:|
| G0 | per-tensor (baseline) | 0.5654 | — | 1 | 2 + 16/N ≈ 2.0 |
| G128 | group 128 | 0.5220 | **−7.7 %** | in/128 | 2.125 |
| G64 | group 64 | 0.5190 | **−8.2 %** | in/64 | 2.25 |

G32 (already in the diagnostic) reaches 0.5138, i.e. the curve is flat beyond
64 — **g64 is the knee**. Bits/param are *estimates* assuming 2-bit packed
ternary storage plus fp16 scales; they are labelled as such, not measured.

Edge cases covered by existing tests: undivided dimensions raise, broadcasting
is exercised by the quantizer's group tests, and checkpoint save/load is
unchanged because scale shape is derived from `group_size` at construction.
**Decision gate:** reconstruction improves, but an 8 % error reduction has never
translated into the ~11 WER points this project needs; no full run is justified
on reconstruction alone.

## Phase C — rotation prototype (implemented, gates passing)

`src/asr1bit/rotate.py` on `bonsai-quant`: normalized Sylvester Hadamard per
block, pre-multiplied by a seeded ±1 diagonal; remainder features unrotated.

**Correctness gates (6 tests, all passing):** orthogonality `R R^T = I`;
deterministic seeded signs; **unquantized forward equivalence** for widths
1024/1000/100 (block 256/64) including the remainder; inverse round-trip;
ternary outputs stay in {−1, 0, +1}; rotated quantization lowers relative error
on outlier-heavy synthetic weights.

The equivalence test **caught a real bug** in the first draft: weights used
`W R^T` while activations used `R`, which silently changes the function. Both
now use `R^T` (`x' W'^T = x R^T R W^T = x W^T`).

**Block size:** 1024 is a hypothesis, not a claim. Real-model rotation error is
**not measured**: the 1.7B safetensors are not on this node and the login node
cannot download; the diagnostic must run on a node holding the weights.

## Phase D / E / F — status

| phase | status | reason |
|---|---|---|
| D (rotation × groups) | **blocked** | needs real-weight rotation measurements (Phase C offline on 1.7B) |
| E (ASR pilot) | **not queued** | no configuration passed the reconstruction gate *and* GPUs are busy with `12466`–`12468` |
| F (deployment) | **not started** | no trained rotated model exists; size/latency numbers are not measurable yet |

**Storage estimate (analytic, not measured):** ternary 2 bits/weight packed;
g64 adds 16/64 = 0.25 bits/weight → **2.25 bits/param**; g128 → 2.125;
per-tensor +16/N ≈ negligible. Rotation adds **no** per-weight storage (its
metadata is one seed and one block size), but it requires `R` to be applied to
activations at inference, which this project's runtime does not yet do.

## Recommendation (evidence-based)

1. **Do not train a Bonsai configuration yet.** Group-wise scaling and rotation
   are reconstruction-error levers, and the one measured reconstruction gain
   (−8 %) is two orders too small relative to the 11-point WER gap — but that
   comparison is a hypothesis, not a measurement, which is why it must not be
   spent on GPU time first.
2. **Next cheap step (no GPU contention):** run the rotation measurement on the
   real 1.7B projections on a node that holds the weights, using the same
   `rotate_weight` + `quantize_ternary` pair; record per-projection error for
   C0–C3. This decides whether any rotation arm is worth a pilot.
3. **If** rotation materially beats group scaling on real weights (say ≥25 %
   relative reconstruction reduction), then a single short pilot (500 steps,
   the b17 recipe shape, one variable) is justified. Otherwise the Bonsai
   direction is closed with a measured negative result, and the project's
   remaining lever stays the quantizer-side distillation work already audited.

## Reproduce

```bash
# baseline reconstruction numbers (checked in): artifacts/stage1/diag_quant_error.json
# 1.7B reconstruction (needs weights on the node):
PYTHONPATH=src python scripts/diag_quant_error.py \
  --safetensors /path/to/Qwen3-ASR-1.7B --out artifacts/stage1/bonsai_17b
# rotation gates (isolated worktree):
cd ../asr1bit-bonsai && git checkout bonsai-quant && PYTHONPATH=src pytest -q tests/test_rotate.py
```
