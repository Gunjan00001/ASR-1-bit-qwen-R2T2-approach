# Research decision

_2026-10-25, verified against `docs/final_experiment_results.csv`, the official
JSONs, and the 100-utterance paired confirmation. Full record:
`docs/final_experiment_report.md`; detailed decisions:
`docs/final_research_decision.md`._

## Current baseline

**b12/step1600 (ternary per-tensor, `decoder_attn_ffn`):**
**15.1483 clean / 18.9773 other** on the 50-utt protocol; **15.7546 / 19.9887**
on the 100-utt confirmation set. These are fixed diagnostic subsets, not full
LibriSpeech splits. The checkpoint is
`data/checkpoints/qat_stream_aligned17b12/step1600.pt`; the artifacts are
`streaming_qat_stream_aligned17b12_step1600_{clean,other}.json`.

## Failed approaches (official evaluations only)

- FFN-only quantization (b17): 69–133 clean, 351–789 loops — non-viable.
- 2-bit four-level quantizer (b20): 28.18/34.77 — identical storage, much worse.
- Group-64 scales (b21): 15.68/21.48 — worse on both splits.
- Mix 0.4 (b14), 6 000 steps (b15), LR 1.25e-6 (b16, clean record 14.94 but
  other 21.59), warmup 500 (b10), 8 000 aligned (b13, other record 17.95 but
  clean 21.72).
- **Streaming-state distillation (B–G):** only F (λ=0.05, layer 20) beat both
  splits at 50 utts (**14.62/18.18**), but the 100-utt confirmation reproduced
  only clean (**14.98 vs 15.75**) and tied `other` (**20.05 vs 19.99**). All
  paired CIs include zero. **No robust both-splits improvement.**
- Rotation/group/2-bit reconstruction gains do not transfer to WER.

## Best measured candidate

**b12/step1600 remains the baseline pair.** F/step1600 is the best
**clean-split** candidate (14.62 at 50 utts; 14.98 at 100 utts) but not a
pair improvement. It is not promoted.

## Binary QAT

**Do not continue.** The improved ternary objective did not establish a pair
win; the binary arm's best-ever 18.43/25.45 (0.6B) was a single seed-sensitive
state and reruns were 26.8–36.4; the 1.7B binary path has no competitive
result. Binary remains the harder problem.

## Bonsai techniques

**Do not reopen.** Group scaling and rotation were measured against ASR WER and
lost; 2-bit is storage-identical to ternary and loses accuracy; rotation never
passed its pre-registered screening threshold. The `bonsai-quant` branch and
its evidence are retained, but the direction is closed without a new hypothesis.

## Next recommended experiment

**Retain the baseline; close the objective axis.** The gate
(≤3.74 clean / ≤6.11 other) is unmet and the remaining gap is **quantization
damage** (fp16 1.7B 2.24/4.61 vs quantized 14–22). The next justified arm is a
**quantizer-side co-training / mixed-precision schedule** at the b12 recipe,
pre-registered with one variable, the same budget, and a **≥500-utterance**
evaluation before any promotion. A 500-utt confirmation of F is the alternative
if the clean-only gain is valued; it will not close the gate.
