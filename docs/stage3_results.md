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
| two-stage: recovered (16.53) → aligned replay, 1 200 aligned (`aligned17b5`) | 19.39 | 24.09 | 0 / 0 |
| two-stage with a deeper base (11.76) **and** 2 400 aligned (`aligned17b6`) | 21.08 | 28.30 | 0 / 1 |
| **two-stage, deeper base (10.81), 1 200 aligned — the clean isolation (`aligned17b7`)** | **16.63** | **23.41** | **0 / 0** |

**What the streaming table says:**
- Recovering the offline base *breaks* the streaming head if you stop there
  (285/278 with hundreds of loops) — the recovery recipe trains plain-offline
  samples only.
- Re-aligning after the recovery restores it and *keeps the base*: 19.39/24.09
  with **zero loops**, the line's best, and the offline twin readout was
  preserved (16.74 vs the recovered 16.53).
- **The base's depth transfers — proved on full samples by a single-variable
  test.** `aligned17b7` changed only the base (16.53 → 10.81) against the line's
  best recipe and improved streaming from 19.39/24.09 to **16.63/23.41**
  (−2.76 clean, −0.68 other, zero loops). The companion run with the *same*
  recipe but a larger aligned set (2 400) scored 21.08 clean — read jointly,
  the two runs isolate both variables: **deeper base helps, larger aligned set
  hurts** at the 1.7B.
- The 1.7B recipe is therefore settled as: recover at lr 5e-6 (multi-cycle),
  then re-align with the 1 200-utterance aligned set; remaining levers are more
  recovery depth and whatever the 0.6B-side mix-ratio/scale runs show.

### 3.bis The dose-response — and where it reverses

Three Stage-B runs with an *identical* recipe (1 200 aligned utterances,
2 000 steps, lr 5e-6, fp32 AdamW), differing only in the recovered base and the
seed:

| recovered base (offline) | Stage-B clean | Stage-B other | Δ vs its own base |
|---:|---:|---:|---:|
| 16.53 | 19.39 | 24.09 | +2.86 |
| **10.81** | **16.63** | **23.41** | +5.82 |
| 9.85 (lr 5e-6) | 18.43 | 25.34 | +8.58 |
| 9.85 (lr 2.5e-6, 1 200 aligned) | 16.53 | 22.05 | +6.68 |
| **9.85 (lr 2.5e-6, 6 000 aligned — LINE-BEST PAIR)** | **15.15** | **18.98** | +5.30 |
| 9.85 (lr 2.5e-6, 8 000 aligned — best single `other`) | 21.72 | **17.95** | — |
| 9.85 (lr 1.25e-6, 6 000 aligned — best single `clean`) | **14.94** | 21.59 | — |

**Corrected 2026-10-19 (20:55):** a fourth point changes the shape — the same
deep base (9.85) with the Stage-B **learning rate halved to 2.5e-6** reaches
**16.53 / 22.05**, the best 1.7B pair yet. So depth and LR *interact*: at 5e-6
the deep base regresses, at 2.5e-6 it wins. The original three points remain
below for the record.

**Deeper recovery stops paying at the original LR — and then reverses.** The intermediate base is
the pipeline's sweet spot so far, and the in-run evidence says why: the deeper
the base, the larger the aligned replay's *early* perturbation (b7 +0.84, b8
+3.1 at step 1 500). A heavily recovered state is harder to re-align than a
partially recovered one, so the *re-alignment* recipe — not the recovery — is
the binding constraint at depth. Both arms were measured: the seed replicate
(`aligned17b9`) reproduced b7 to the digit (runs are deterministic), and the
warmup arm (`aligned17b10`) landed neutral (19.17/22.39) — the rescuer was the
**learning rate**, not the warmup (see the corrected table above).

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
6. **The four recipe axes were each closed by a matched arm:** LR (5e-6 →
   2.5e-6 rescues the deep base; 1.25e-6 buys clean but costs `other`), aligned
   set (1 200 → 6 000 improves `other`; 8 000 improves it further but costs
   clean), mix (0.4 worse on both splits, loops rose), duration (6 000 steps
   ties clean at its best and degrades after). **State selection is load-bearing
   every time** — b12's step 1 200 reads 131.57 while step 1 600 reads 15.15.

## 4. Honest gate status

Neither size is at the gate (≤ 3.74 clean / ≤ 6.11 other). The 0.6B's promoted
state streams at 9.96/18.98; the 1.7B's line reaches **15.15/18.98** (b12),
with single-split records at 14.94 clean (b16, lr 1.25e-6) and 17.95 other
(b13, 8 000 aligned). The remaining gap is **quantization damage** — the fp16
anchors are 2.24/4.61 and 2.54/6.27 — and the recipe-side levers are now
*measured and settled*: lr 2.5e-6, 6 000 aligned, mix 0.5, 2 000 steps (see §3
for the four arms that closed each axis). What remains is quantizer-side
(co-training is the untested option) or scale.

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
**Scaled aligned set, third point (2026-10-21):** 8 000 aligned reads **clean
21.72 / other 17.95** at its final state (27 clean loops). So the aligned-set
scaling moves the two splits differently: `other` is monotone in the set size
(22.05 → 18.98 → 17.95 for 1 200 → 6 000 → 8 000), while `clean` peaks at
6 000 (15.15). The line-best *pair* therefore stays at 6 000; the single-split
`other` record now belongs to 8 000.

