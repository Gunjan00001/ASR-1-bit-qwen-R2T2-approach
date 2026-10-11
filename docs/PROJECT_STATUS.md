# Project status — ASR-1-bit-Qwen

**Date:** 2026-10-11 · **Branch:** `stage-1` · **Base model:** `Qwen/Qwen3-ASR-0.6B`
**Target:** English streaming ASR with near-binary weights, R2T2-class quality on CPU/edge.

> This is the single-page status report. Protocol details: `docs/PLANNING.md`;
> failure history: `docs/LAB_NOTES.md`; step-by-step work log:
> `docs/ACTIVITY_LOG.md`; operating rules: `rules.md`.

---

## TL;DR

- **Stage 0 gate passed** (`0.2.0`): fp16 streaming baselines reproduced on
  T4 (1.7B full clean 1.96 vs published 1.95); `X = +1.5` locked.
- **Stage 1 (BitLinear + QAT) machinery works**: ternary QAT can memorize a
  single utterance exactly (loss 0.000); all quantizer/training code is tested
  (203 tests).
- **Naive QAT collapses** (~100 WER) — this is expected, not a bug (quantizer
  audit: per-tensor binary error 0.648 ≈ the Gaussian closed form 0.60).
- **A working recovery recipe was found (Stage 2a/2b) and scaled up on 8k data:**
  FFN warm start → attention rollout with a **frozen pretrained-teacher anchor**
  → repeated warm-restart cosine cycles; binary via **progressive precision**.
  - **Ternary, all 196 decoder projections: 9.2 final / 9.1 best WER**
  - **Binary, all 196 decoder projections: 10.7 WER**
  - (fp16 pretrained reference on the same 50-utt sample: **2.44**)
  - Offline recovery plateaued after four continuation rounds; **streaming
    robustness is now the critical path** — see the streaming section below.

---

## WER report (fixed 50-utt `test-clean` subsample, seed 0, offline)

Each row is a *trained* model evaluated at the stated precision. A "binary"
number on a ternary-trained row is a readout only (not a trained binary model).

| Configuration (trained) | Steps | fp16 readout | ternary | binary |
|---|---|---:|---:|---:|
| fp16 pretrained (no QAT) | — | **2.44** | — | — |
| Naive attn+FFN ternary | 6 000 | 113.1 | 107.8 | 99.8 |
| Self-distill attn+FFN ternary (w=0.5) | 4 000 | 108.3 | 98.1 | 118.3 |
| FFN-only ternary | 3 000 | 226.5 | 32.8 (best 25.7) | 71.0 |
| FFN-only binary | 3 000 | 223.9 | 106.8 | 97.9 |
| Stage 2a attn+FFN ternary (4k data) | 5 000 | 85.8 | 16.2 (best 14.4) | 99.9 |
| Stage 2b attn+FFN binary (4k data) | 2 000 | 99.8 | 30.5 | 17.5 |
| **Ternary long (8k data)** | 8 000 | 50.7 | **11.4 (best 10.6)** | 100.0 |
| **Binary long (8k data)** | 6 000 | 37.2 | 21.3 | **12.9** |
| **Ternary long2 (8k, lr 1e-5)** | 4 000 | 37.7 | **9.2 (best 9.1)** | 100.4 |
| **Binary long3 (8k, lr 5e-6)** | 3 000 | 31.7 | 22.4 | **10.7** |
| Zero-shot naive (no training) | 0 | — | ~100 | ~100 |

*Primary metric note:* the Stage 0 gate is defined on **streaming 320 ms**
clean/other (fp16: 2.24 / 4.61, `X = +1.5`). The numbers above are **offline**
on a 50-utt clean subsample. Streaming evaluation of quantized models is an
open work package — see the dedicated section below for what was found.

---

## Streaming evaluation (320 ms) — status and findings

Streaming uses the portable `ChunkedStreamer` (vLLM cannot serve BitLinear).
Measured on the same 50-utt sample, 320 ms chunks:

| Variant | Pretrained fp16 | Ternary QAT (offline 9.2) | Binary QAT (offline 10.7) |
|---|---:|---:|---:|
| **Official prefix algorithm** (`unfixed_token_num=5`) | **2.33** | **collapse (1296)** | — |
| Same + short prefix-SFT (1 500 steps) | — | collapse (952) | — |
| Same + **scaled streaming SFT** (5 000 steps, full cache, prefix+silence) | — | **203.5** (still loops) | — |
| Same + repetition_penalty=1.3 | — | 438.2 (worse) | — |
| Same + no_repeat_ngram_size=4 | — | 264.5 (worse) | — |
| No-prefix re-feed (`unfixed_token_num=9999`, 32 tokens) | 19.28 | 33.58 | 33.26 |
| **No-prefix re-feed, 128-token budget** | **2.54** | **9.43** (CER 3.86) | **10.28** (CER 4.52) |

**Root causes (per-chunk traces, `scripts/diag_stream_generation.py`):**

1. **Near-empty first chunks:** the pretrained model answers a 0.32 s chunk with
   `language None<asr_text>` + EOS; the QAT model runaway-generates
   `assistant\nassistant…` (32 tokens, no EOS).
2. **Prefix continuation:** the official algorithm re-feeds a rolled-back copy of
   the model's own text as the prompt. The QAT model was never trained on that
   pattern and falls into token repetition (`'T'T'T…`, `HAVE HAVE…`). The
   pretrained model handles it natively.
3. A short prefix-continuation SFT phase (1 500 steps, 4 transcript fractions,
   2 k utts) **reduced but did not fix** the collapse (1296 → 952): the model
   still degenerates when continuing its own partial text.

**What this means:**

- The quantized models are **offline-strong** (9.2 / 10.7). Under the **valid
  chunked re-feed evaluation** (no-prefix, 128-token budget), ternary scores
  **9.43** and binary **10.28** WER — essentially their offline quality — with
  a **+6.9 / +7.7 gap** to the same-variant pretrained baseline (2.54). This is
  the current streaming result.
- The **official prefix algorithm** remains broken for QAT models: even after a
  scaled streaming SFT (5 000 steps, full cache, prefix + silence samples) the
  WER is 203.5 — the transcripts are largely correct but the model then falls
  into repetition loops and never stops. Inference controls do not fix it
  (repetition_penalty=1.3: 438.2; no_repeat_ngram_size=4: 264.5 — both worse).
- The official algorithm's prefix-continuation requirement is a **research-grade
  Stage 4 item**: it needs on-policy streaming training (train on the model's
  own partial generations) and/or the LSP emission policy, not a simple SFT
  add-on. Until then, the chunked re-feed variant (9.43 / 10.28, RTF ~10) is
  the reported streaming number.

---

## WER trajectories

![WER trajectories](figures/wer_trajectories.png)

Reading the chart (log scale):

- **Naive attn+FFN (red)** starts at ~127 and stays ~100 — quantizing attention
  naively destroys the model and training never recovers.
- **FFN-only (orange)** recovers 118 → 84 → 42 → 25.7 in 2 000 steps: keeping
  attention in fp preserves audio conditioning. (The dotted orange segment is
  the resumed run, affected by the LR-restart bug — see `docs/ACTIVITY_LOG.md`.)
- **Stage 2a (blue)** starts from the FFN-only checkpoint (25.7), pays +27
  points when attention is quantized (52.8 @500), then recovers steadily to
  **14.4** with the frozen-teacher anchor and a warm-restart cosine cycle.
- **Stage 2b (green)** starts from the ternary checkpoint and switches to
  binary: 51.5 → 38.1 → 25.4 → **17.5**.
- **Long runs on 8k data (purple/brown)** continue the best checkpoints with a
  fresh long cycle (ternary lr 2.5e-5, 8 000 steps; binary lr 2e-5, 6 000
  steps). Ternary: 24.8 → … → **10.6 @7000 → 11.4 final**. Binary:
  23.3 → … → **12.9 final**. The fresh LR cycle causes a transient (the model
  is perturbed first), then the descent continues past the previous best as the
  cosine decays.

## Final WER by configuration

![Final WER by configuration](figures/final_wer.png)

## Shadow drift: the fp16 readout degrades during QAT

![fp16 drift](figures/fp16_drift.png)

At lr 2e-4 (long runs) the STE updates move the latent weights far from the
pretrained basin: the fp16 readout rises to WER 80–230 even when the quantized
readout recovers. The anchor (w=0.1) slows but does not prevent this drift. The
**quantized model is the deliverable**; preserving the fp readout is an open
work package (trust-region anchor).

## Quantizer audit (why naive QAT is expected to fail)

![Quantizer error](figures/quantizer_error.png)

Per-tensor binary quantization of real decoder weights has relative Frobenius
error **0.648** — essentially the Gaussian closed form (0.60). Group-wise
scaling barely helps (sign pattern dominates). This is why naive 1-bit
conversion collapses; recovery must come from training, not from the quantizer.

## Roadmap status

![Roadmap](figures/roadmap.png)

| Stage | Status | Notes |
|---|---|---|
| 0 — baselines & harness | **gate passed** (`0.2.0`) | fp16 numbers reproduced on T4 |
| 1 — BitLinear + QAT | **machinery done; gate open** | loss controllable; WER tables emitted; recovery delivered by Stage 2 recipe |
| 2 — accuracy recovery | **in progress** | rollout + anchor + progressive precision → 14.4/16.2 ternary, 17.5 binary |
| 3 — scale to 1.7B | not started | memory stack + checkpointing ready (`--resume`) |
| 4 — streaming parity (LSP) | not started | aligner + stable-prefix targets |
| 5 — edge deployment | not started | packing/GGUF/CPU kernels |
| 6 — evaluation & writeup | not started | full matrix + report |

---

## The recipe that works

1. **FFN warm start**: quantize `gate/up/down` only (84 layers) and train until
   the loss recovers (WER ≈ 25.7). Attention stays fp → audio conditioning is
   preserved.
2. **Attention rollout with a small anchor**: quantize attention too (196
   layers) and train with
   `L = 0.9 · CE + 0.1 · KL(student ‖ frozen-pretrained fp teacher)`
   at lr 5e-5, cosine. The anchor prevents the audio-conditioning collapse.
3. **Warm restarts on more data**: a fresh cosine cycle (lr 2.5e-5, 8 000
   steps, 8k-utterance cache, anchor 0.05) takes ternary from 16.2 → **11.4**
   (best 10.6). The fresh cycle perturbs the model for the first ~2 000 steps,
   then the descent continues past the previous best as the LR decays — do not
   panic at the transient.
4. **Progressive precision for binary**: switch `bit_width=1.0` starting from
   the ternary checkpoint (binary 17.5 at 4k data), then a long continuation
   (lr 2e-5, 6 000 steps, 8k data) reaches **12.9**.
5. **Shadow-drift control**: lower lr + anchor reduce the fp16 readout drift
   (from 85–230 down to ~37–65), though it is not eliminated.

Key infrastructure that made this possible: log-parameterized learnable scales
(always positive), `--resume` with correct LR-schedule advance, `--init-from`
for new phases, frozen-teacher distillation, and the cluster rules in
`rules.md` (GPU switching instead of waiting).

## What failed (kept for the record)

| Attempt | Symptom | Root cause | Fix / lesson |
|---|---|---|---|
| Free-parameter learnable scale | gamma negative / 10× collapse | unconstrained additive parameter | log-parameterization (`gamma = exp(log_scale)`) |
| Blended α-ramp | fp16 readout corruption | blend + identity-STE mismatch | use `--alpha-mode delay` |
| Self-distillation (moving teacher) | WER 98–125, no recovery | teacher drifts with the student | use a **frozen** pretrained teacher |
| Anchored KL w=0.5 from fp start | KL 11–15, WER 315 | KL gradient dominates | small weight (0.1) + FFN warm start |
| Binary FFN-only | stuck ~92–98 | binary sign churn from fp start | progressive precision from ternary |
| `--resume` before the fix | resumed run got fresh high LR, WER 25.7→42.8 | scheduler restarted at step 0 | `scheduler.last_epoch = start_step` |
| Long runs at lr 2e-4 | fp16 readout destroyed | latent-weight drift | anchor + lower lr (open item) |

## Reproduce

```bash
# 1. data cache (login node)
python scripts/build_qat_cache.py --n 4000 --stream 1

# 2. FFN warm start (ternary)
qsub scripts/qat_ternary_ffn.pbs        # → 25.7 @2000

# 3. attention rollout + frozen anchor
qsub scripts/qat_attn_rollout.pbs       # → 22.6 @2000
qsub scripts/qat_attn_rollout_ext.pbs   # warm restart
qsub scripts/qat_attn_rollout_ext2.pbs  # GPU-switch resume if contended

# 4. binary via progressive precision
qsub scripts/qat_binary_rollout.pbs     # → 17.5 @2000

# figures / tables
python scripts/make_plots.py
python scripts/inspect_checkpoint.py --checkpoint <ckpt> --policy decoder_attn_ffn
```

---

## Future plan

**Streaming robustness (the critical path — Stage 4 brought forward)**
1. The quantized models are offline-strong (9.2 / 10.7). The **valid streaming
   number is 9.43** (ternary) under the chunked re-feed variant (no-prefix,
   128-token budget; pretrained 2.54 → **+6.9**).
2. The **official prefix algorithm** remains broken despite a scaled streaming
   SFT (203.5 WER; correct transcripts followed by repetition loops). Inference
   controls do not fix it (repetition_penalty made it worse: 438).
3. Next for the official algorithm — research-grade work:
   - **on-policy streaming SFT**: build prefixes from the model's *own*
     generations (not gold transcripts), so training matches the inference
     distribution;
   - **LSP emission policy** (Stage 4 / T4.4) with explicit stop handling;
   - longer streaming SFT (10–20 k steps) with a balanced mix (keep ≥50 %
     standard samples to limit the offline cost, currently ~3 points).
4. Interim reporting uses the chunked re-feed number (9.43, RTF ~9.7).

**Stage 2 — accuracy recovery (offline, done for now)**
5. Ternary 9.2 / binary 10.7 offline; continuation rounds plateaued on 8k data.
   The full 28.5 k cache is the next scale-up for the offline recipe.

**Stage 3 — scale to 1.7B**
6. Memory stack (LoRA / partial QAT), checkpoint/resume (ready), run + eval.

**Stage 5 — edge deployment**
7. Pack 1-bit/2-bit weights, GGUF export, CPU SIMD kernels, RTF < 1 and
   constant memory.

**Stage 6 — final evaluation & writeup**
8. Full matrix (chunk sizes 160/320/1000 ms, clean/other), tables/plots, report.

**Known open risks**
- Streaming robustness is now the primary gap: offline 9.2 vs the `+1.5`
  target (3.74) is ~5.5 points, but the official streaming algorithm currently
  fails entirely for QAT models.
- The fp16 readout still drifts (31–50); must be solved for the plan's
  "co-train" story.
- All checkpoints and caches are kept on disk; nothing is lost between rounds.
