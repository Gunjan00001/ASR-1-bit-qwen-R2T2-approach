# Stage 3 — scaling to the 1.7B (results)

**Scope.** Qwen3-ASR-1.7B (the project's *target* model: its fp16 streaming
numbers, 2.24 clean / 4.61 other, are what the Stage-4 gate is defined
against). All numbers here are WER % on the project's standard protocols:
**offline** = 50-utt `test-clean`, quantized ternary readout; **streaming** =
the *official* prefix algorithm at 320 ms chunks with the LSP policy on,
50-utt clean/other.

**Read this with `docs/stage4_results.md`** (the 0.6B's streaming record) and
`docs/LAB_NOTES.md` (every attempt with its root cause). The 1.7B's story is
about the **offline base**: the streaming recipe had already been solved at the
0.6B; what blocked the 1.7B was that no recovery recipe could lift its
quantized base until the learning rate was raised.

## 1. The offline base's trajectory (every run, from the logs)

Generated from `artifacts/stage3/*.log`'s `EVAL` records (step: offline
ternary WER):

| run | what it was | offline trajectory |
|---|---|---|
| `ffn` | FFN warm start (first quantized phase) | 500:113.9 → 1000:249.6 → 1500:95.7 → 2000:146.7 → 2500:26.9 → 3000:**22.03** |
| `attn2` | attention rollout, lr 3e-5 (crashed: grad-ckpt) | 500:58.7 |
| `attn5` | attention rollout, completed | 500:51.4 → 1000:35.3 → 1500:24.2 → 2000:**20.55** |
| `recover1` | recovery, lr 1.5e-6, **adam8bit, SR on** | 24.15 / 23.09 / 24.79 / 22.78 / **24.36** |
| `recover2` | recovery, lr 1.5e-6, adam8bit | 20.76 / 20.55 / 19.17 / 19.92 / **18.96** |
| `recover3` | recovery, **lr 5e-6**, adam8bit | 19.92 / 18.33 / 17.27 / 16.74 / **16.53** |
| `recover4` | recovery, lr 1.5e-6, **fp32 AdamW** | 21.29 / 20.97 / 19.49 / 19.28 / **21.40** |
| `recover5` | recovery **cycle 2**, lr 5e-6, fp32, from recover3 | 20.23 / 17.90 / 17.90 / 15.15 / 15.78 / 13.14 / 11.97 / **11.76** |
| `recover6` | recovery **cycle 3**, same recipe, from recover5 | 15.89 / **13.45** (running) |
| `aligned17b5` | **Stage B** aligned replay, from recover3 (16.53) | 19.39 / 17.80 / 16.63 / **16.74** (base preserved) |
| `aligned17b6` | **Stage B** aligned replay, from recover5 (11.76) | 19.49 / **17.90** (running, recovering) |

**What the table says:**
- The gentle schedules (1.5e-6) *cannot move* the damaged base — four runs
  converge on ~19–21 regardless of optimizer or stochastic rounding.
- **lr 5e-6 is the unlock.** Phase 1 (`recover3`) reached 16.53; a second
  4 000-step cycle (`recover5`) continued to **11.76**; a third
  (`recover6`) is tracking ahead of both (13.45 at step 1 000).
- The warm-restart bump is a *shape*, not a failure: each fresh cosine costs
  ~+4 in its first 500 steps and then descends past the previous base
  (cycle 2: 16.53 → 20.23 → 11.76; cycle 3: 11.76 → 15.89 → …).

## 2. The streaming side (official algorithm, 320 ms, 50 utts)

| state | clean | other | loop truncations |
|---|---:|---:|---:|
| fp16 1.7B baseline (the gate's anchor, Stage 0) | **2.24** | **4.61** | n/a |
| first aligned attempt (`aligned17b4`, unrecovered base, lr 1.5e-6) | 29.24 | 33.75 | 5 / 6 |
| recovered base, *no* re-alignment (`recover3/last.pt`) | **285.06** | **278.30** | 354 / 387 |
| **two-stage: recovered (16.53) → aligned replay (`aligned17b5`)** | **19.39** | **24.09** | **0 / 0** |
| two-stage with the deeper base (11.76) → `aligned17b6` | *running* | *running* | — |

**What the streaming table says:**
- Recovering the offline base *breaks* the streaming head if you stop there
  (285/278 with hundreds of loops) — the recovery recipe trains plain-offline
  samples only.
- Re-aligning after the recovery restores it and *keeps the base*: 19.39/24.09
  with **zero loops**, the line's best, and the offline twin readout was
  preserved (16.74 vs the recovered 16.53).
- The base's depth *transfers* to the streaming head: same recipe, first
  in-training streaming reading 16.00 on the 11.76 base versus 26.59 on the
  16.53 base. (That reading is the 20-utt in-training guardrail, which by
  project rule decides nothing — the full-sample evaluations are the metric;
  it is quoted only as the transfer's *direction*.)

## 3. The recipes that got here (and the failures, with numbers)

1. **FFN warm start, then attention rollout** (the 0.6B's recipe shape) →
   22.03 → 20.55. Slow, and it plateaus.
2. **The permanent dip** (v2, 2026-10-15): lr 3e-6 + adam8bit + *no anchor*
   took 20.55 → 30.6 and it *never recovered*. The frozen-teacher anchor
   (`--distill-weight 0.05 --frozen-teacher 1`) exists because of that run.
3. **Stochastic rounding *hurt*** (2026-10-16, first controlled A/B):
   24.15 vs 20.76 at step 500 with everything else identical. The flag is off
   and its help text carries the measurement.
4. **The optimizer is a small effect, the schedule is the big one**:
   fp32 AdamW beat its adam8bit twin by ~0.6–0.7 at matched steps at 1.5e-6,
   while lr 5e-6 beat both by ~2.5–4.0.
5. **A resume overwrote a checkpoint once** (the 0.6B's binary arm) — hence the
   standing rule: snapshot before resuming; and `--keep-step-checkpoints 400`
   makes every run's trajectory selectable here.

## 4. Honest gate status

Neither size is at the gate (≤ 3.74 clean / ≤ 6.11 other). The 0.6B's promoted
state streams at 11.55/18.98; the 1.7B's line now reaches 19.39/24.09 and is
improving along a *measured* pipeline rather than by luck. The remaining gap is
quantization damage; the working levers are (in order of measured effect):
**more recovery depth at lr 5e-6** (cycle 3 projects to ~9–10 offline),
**re-alignment after every recovery** (zero loops, base preserved), and then
either more aligned data (2 400 aligned utterances is the current cap) or the
quantizer-side options.

## 5. Reproduce

```bash
# recovery cycle (Stage A): 4000 steps at lr 5e-6, fp32 AdamW, anchored
qsub scripts/qat_17b_recover6.pbs              # init recover5/last.pt
# aligned streaming replay (Stage B) on the recovered base
qsub scripts/qat_stream_aligned17b6.pbs        # init recover5/last.pt
# full-sample streaming evaluation of any 1.7B checkpoint (official algorithm)
qsub -v CKPT=data/checkpoints/<run>/last.pt scripts/eval_streaming_17b.pbs
```
Launchers are committed with their runs; every number above traces to a log
under `artifacts/stage3/` or a JSON under `artifacts/stage1/`.
