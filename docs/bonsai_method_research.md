# Bonsai 2 — primary-source research notes

_Compiled 2026-10-23 from the model card, the demo repository, and the model's
public files. Anything not in those sources is marked as unknown._

## Sources

1. Model card: <https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf>
   (read 2026-10-23; quotes below are from that page).
2. Demo / runtime repository: <https://github.com/PrismML-Eng/Bonsai-demo>
   (README read 2026-10-23).
3. Runtime fork: <https://github.com/PrismML-Eng/llama.cpp> (referenced by both).
4. Whitepaper PDF: `bonsai-2-27b-whitepaper.pdf` in the demo repository
   (linked from the card; methodology pointer, not fetched here).
5. Announcement: <https://prismml.com/news/bonsai-2-27b>.

## Documented facts

**Weight representation.** "Ternary g128: {−1, 0, +1} weights with FP16
group-wise scaling" — one shared FP16 scale per group of **128** weights. The
card computes ≈1.71 bits/weight (ternary code + 16-bit scale / 128), and 1.72
bits/weight for the model as a whole once 26.2 M parameters (0.0976 % of the
language model: recurrent-state path of the linear-attention layers plus
normalisation weights) are counted at higher precision.

**Coverage.** "End-to-end ternary language weights across embeddings, attention
projections, MLP projections, and LM head — no high-precision escape hatches."
The vision tower is a separate Q8_0 pack.

**Rotation.** "Each matrix is transformed blockwise by an orthogonal Hadamard
rotation before the ternary assignment, and the runtime applies the matching
transform to activations. The rotation is folded into the stored weights
offline, so it costs no extra bits and no extra weight traffic; the packed model
declares its rotation as metadata, so a runtime either applies the matching
transform or refuses to load the file." The model table states the block size is
**1024** with **fixed ±1 signs**.

**Packing and runtime.** Two GGUF packings: **PTQ1_0** stores trits densely
(1.75 bits/weight, 5.95 GB) and **PQ2_0** stores each trit in a 2-bit slot
(2.13 bits/weight, 7.21 GB); "packed weights are consumed directly, never
expanded back to FP16." Kernels live in the PrismML llama.cpp fork; stock
llama.cpp rejects the types, and loading a plain `Q2_0` file without the
Hadamard activation runtime "produces garbage". PQ2_0 is faster on H100/A100/
Blackwell, PTQ1_0 on Ada and L4.

**Base/architecture.** Derived from Qwen3.8-27B (hybrid attention, ~75 % linear)
with architecture unchanged; 262 K context; 27.36 B parameters.

**Demo repository scope.** The README is explicit that the repo is setup,
download, and inference: `setup.sh`, binary downloads, `run_llama.sh`,
`run_mlx.sh`, servers. It contains **no training code, no conversion script, and
no calibration code** — it references the whitepaper for methodology and pins a
runtime release.

## Answers to the brief's questions

1. **Levels and scales:** documented — `{−1, 0, +1}` with FP16 per-group scale,
   group 128.
2. **Group formation:** documented as "one shared FP16 scale factor for every
   group of 128 weights", but the *axis* (contiguous input rows vs output rows)
   is **not stated** in the public material.
3. **Transforms before quantization:** blockwise Hadamard with fixed ±1 signs,
   folded into stored weights; block 1024.
4. **Transform implementation:** the mathematical contract (rotate weights,
   apply matching transform to activations at runtime, declare it in metadata)
   is documented; the **kernel code** is in the fork, not in the card. Sign
   pattern generation is not documented beyond "fixed ±1".
5. **Which matrices:** language weights: embeddings, attention projections, MLP
   projections, LM head. Whether every projection is rotated is not stated
   explicitly; the card says "each matrix".
6. **Non-divisible dimensions:** **not documented publicly.** (Our prototype
   leaves remainder features unrotated, which is exact; Bonsai's handling is
   unknown.)
7. **Runtime pack/unpack:** documented at the format level — dense trits
   (PTQ1_0) or 2-bit slots (PQ2_0), consumed directly by custom kernels, never
   expanded.
8. **Precision of scales/activations/other components:** scales FP16; the 26.2 M
   non-ternary parameters stay higher precision (type unspecified); activations
   are not described beyond the matching Hadamard transform.
9. **What is unavailable:** the training recipe (whether ternary-aware QAT or
   PTQ after rotation), the calibration/assignment algorithm, scale
   initialisation, per-layer precision allocation, and any conversion code. The
   demo is inference-only.
10. **What transfers to a 1.7B streaming ASR:** the *representation*
    (g128 ternary + FP16 scales), the *rotation contract* (fold into weights,
    match on activations, declare in metadata), and the *packing choices* are
    all directly testable here. What does **not** transfer without independent
    verification: the training pipeline (unpublished), the claim that rotation
    helps at 1.7B scale in a streaming objective, and the runtime speed claims
    (different architecture, different kernels).

## Why this matters for our project

Bonsai's evidence is *language-model benchmark* retention at 1.72 bpw on a 27B
hybrid-attention model, converted with a pipeline we cannot inspect. Our failure
mode is different: the 1.7B ternary ASR model loses streaming stability (loops)
and 11 WER points. A representation that improves *reconstruction* is necessary
but not sufficient; only an official streaming evaluation can decide.

## Limitations of this evidence

- Marketing benchmarks on a different task family; no ASR, no streaming.
- The card does not state group axis, non-divisible handling, or how the
  rotation interacts with quantization error during training.
- Whitepaper methodology not independently reproduced here; the demo repo
  contains no conversion code, so any reimplementation is our own.
