# ASR-1-bit-Qwen

English streaming ASR with near-binary weight quantization (`{-1,+1}` target,
ternary fallback), built **only** from Qwen models and targeting R2T2-class
published streaming quality on CPU/edge.

- **Plan of record:** [`docs/PLANNING.md`](docs/PLANNING.md)
- **Base models (Apache-2.0):**
  - `Qwen/Qwen3-ASR-0.6B`
  - `Qwen/Qwen3-ASR-1.7B`
  - `Qwen/Qwen3-ForcedAligner-0.6B`
- **Scope:** English only, research use, no distribution.
- **No R2T2 weights/code are used** — R2T2 is a benchmark target only.

---

## At a glance (for a first-time reader)

**The problem.** Take Qwen3-ASR — a state-of-the-art English speech recogniser —
and force its weights to be *ternary* `{−1, 0, +1}` (with a binary `{−1, +1}`
target). Few-bit weights replace multiplications with additions and
comparisons, which is what makes real-time, low-power, on-device speech
recognition possible. The catch: quantization destroys accuracy, and for
*streaming* recognition it destroyed the model completely.

**The headline.** The official Qwen streaming algorithm used to **collapse** for
quantized weights — WER **203.5 %** with endless repetition loops. It now
streams at **11.55 % clean / 18.98 % other** (320 ms chunks, 50-utt test sets)
with **zero loops**, on ternary weights, while the best offline quantized model
reaches **7.52 %** (ternary) / **10.70 %** (binary) on the full 28,539-utterance
training cache.

**The fix, in one line.** A forced-aligner supplies word-level time-stamps, from
which we build per-chunk training targets ("say exactly the words that end in
this 320 ms window, or say nothing") plus an append-only loop-suppressing
emission policy — so the model is *taught* the streaming question instead of
having to infer it.

**The honest status.** The project's fixed bar is "within `+1.5` WER of the fp16
1.7B model" (≤ 3.74 / 6.11). It is **not met yet**; the measured bottleneck is
*quantization damage*, not capacity or the recipe, and the next two levers are
identified (a complete offline recovery for the 1.7B, and unbiased stochastic
rounding). Everything is reproducible from committed launchers, tags and
per-evaluation JSONs.

**Start here:** [`docs/DEEPDIVE.md`](docs/DEEPDIVE.md) — the whole project
explained from the ground up, including every failed attempt with its number
(also written as viva/interview preparation). Current state:
[`docs/HANDOFF.md`](docs/HANDOFF.md). The plan of record:
[`docs/PLANNING.md`](docs/PLANNING.md).

---

## Where the project stands (2026-10-15)

All numbers are WER % at **320 ms chunks** unless stated; every row is backed by
a JSON under `artifacts/stage1/` (regenerate the summary with
`PYTHONPATH=src python scripts/make_plots.py`).

| What | test-clean | test-other | Source |
|---|---:|---:|---|
| fp16 Qwen3-ASR-**1.7B** streaming (Stage-0 gate anchor) | **2.24** | **4.61** | `docs/stage0_results.md` |
| fp16 Qwen3-ASR-**0.6B** streaming (our portable path) | 2.33–2.54 | 6.27 | Stage 0 / `streaming_pretrained*` |
| **Ternary, all 196 decoder projections — offline** (28 539-utt cache) | **7.52** | — | `qat_ternary_full` (the run's best in-loop reading; its final checkpoint evaluates 7.63) |
| Binary, all 196 decoder projections — offline | 10.70 | — | `qat_binary_full` |
| Streaming, *naive chunked re-feed* variant (not the official algorithm) | 9.43 | — | `streaming_ternary_noprefix128` |
| **STREAMING, official prefix algorithm — aligned targets + LSP (NEW BEST)** | **11.55** | **18.98** | `streaming_al5_step1000` |
| ↳ its 160 ms / 1000 ms sweep | 13.77 / **10.70** | — | `streaming_opd_al5best_*` |
| ↳ previous promoted state — `aligned2`, 2 500-step cycle | 12.92 | 20.00 | `streaming_opd_aligned2` |
| ↳ that state at 160 ms / 1000 ms | 13.24 / 10.59 | 20.80 / 16.59 | `*_c160`, `*_c1000` |
| Streaming, binary arm, same recipe (best measured) | 18.43 | 25.45 | step-1 250 state (weights lost to a resume ✗; final weights give 27.54 / 26.93) |
| Naive QAT, before the recovery recipe (history) | ≈100 | — | `docs/stage1_results.md` |

**The Stage-4 gate** is *within `+1.5` absolute of the fp16 1.7B baseline at
320 ms*, i.e. **clean ≤ 3.74 / other ≤ 6.11**. It is **not met yet**. The
honest attribution of the remaining gap is **quantization damage, not streaming
capacity and not the recipe**: the fp16 *0.6B* streams at 2.54 / 6.27 — close
to the 1.7B's 2.24 / 4.61, so model size is not what limits the fp16 side —
while quantization alone costs **+5.4** on the offline path (2.11 → 7.52),
above the entire +1.5 budget. Whether that damage shrinks with scale is the
open 1.7B question.

**Is progress being made? Yes, and it is measurable:**

1. **The quantization machinery is proven** — BitLinear QAT is implemented,
   audited and unit-tested (337 tests, ruff clean); loss controllability is
   demonstrated (ternary QAT memorizes a batch exactly); the quantizer's error
   on the real decoder weights matches the closed-form expectation.
2. **The offline recovery recipe scales** — from total collapse (WER ≈ 100 for
   naive ternary QAT) to **7.52 ternary / 10.70 binary** on the full
   28 539-utterance cache.
3. **The official streaming algorithm is fixed at the recipe level** — it used
   to collapse outright for quantized models (**203.5 WER, repetition loops**);
   with **forced-aligner chunk targets + the LSP loop-emission policy** it now
   streams at **11.55 clean / 18.98 other** (the longer 5 000-step cycle's
   step-1 000 state; the earlier promoted state was 12.92 / 20.00), zero
   repetition loops, and the binary arm follows (best measured 18.43 / 25.45).

**What is open:** the `+1.5` gate (quantization damage), the 1.7B scale test
(paused on cluster capacity), edge deployment (Stage 5) and the final write-up
(Stage 6).

---

## The story so far — six stages

| Stage | State | What it means |
|---|---|---|
| **0. Baselines & harness** | ✅ gate passed (`0.2.0`) | fp16 baselines reproduced; anchors locked (1.7B streaming 2.24 / 4.61, `X = +1.5`) |
| **1. BitLinear + QAT** | ✅ machinery done | quantized layers, QAT loop, caches, launchers; naive ternary QAT collapses (≈100 WER) |
| **2. Accuracy recovery** | ✅ recipe found & scaled | FFN warm start → attention rollout + frozen anchor → anneals; ternary **7.52** / binary **10.70** offline on the full cache |
| **3. Scale to 1.7B** | ⏸ paused (cluster) | FFN warm start **22.03** → attention rollout **20.55**; the aligned recipe needs ~45 GB *stable* GPU memory |
| **4. Streaming parity (LSP)** | 🟡 recipe delivered, gate open | official algorithm **203.5 → 12.92 / 20.00**, zero loops; append-only LSP policy implemented + tested |
| **5. Edge deployment** | ⬜ not started | CPU/RTF work (RTF < 1 target) |
| **6. Evaluation & write-up** | 🟡 begun | `docs/stage4_results.md`, `docs/PROJECT_STATUS.md`, HANDOFF, this README, figures regenerated from artifacts |

The recipe (all committed and reproduced by launchers in `scripts/`):

```
best offline checkpoint (ternary or binary, full cache)
  → aligned-targets streaming SFT:  Qwen3-ForcedAligner-0.6B word timestamps
      → per-chunk targets ("emit the words whose end time falls in this chunk,
        then stop"; empty chunks are stop samples)
      → 50/50 mix with standard samples, 2 400 aligned utts pooled
  → evaluate with the OFFICIAL prefix algorithm + LSP loop policy
      (margin 3 words, truncate loops keeping one occurrence, append-only)
```

**Promotion rule (learned the hard way):** only *full-sample* evaluations
(50 utts, both splits) decide. The fixed 20-utt in-training guardrail neither
level- nor trend-transfers (aligned3's guardrail looked best at 8.7–9.2, its
full-sample clean was worst at 14.41). Every failed recipe is kept with its
numbers in [`docs/LAB_NOTES.md`](docs/LAB_NOTES.md).

---

## Figures — how to read every component

All figures are regenerated by `PYTHONPATH=src python scripts/make_plots.py`
(reads the `EVAL` lines of the run logs plus the final ablation JSONs; the
extracted numbers land in `artifacts/stage1/wer_summary.json`). Lower WER is
always better. "50-utt" means a fixed 50-utterance `test-clean` slice (seed 0),
"other" means `test-other`.

### 1. `wer_trajectories.png` — offline WER during QAT

![WER trajectories](docs/figures/wer_trajectories.png)

- **Axes:** x = QAT step; y = WER % on a **logarithmic** scale (so a collapse
  like 100 % and a good state like 9 % are visible together).
- **Markers:** every marker is one in-training evaluation (`EVAL` line) of the
  *quantized* readout (α = 1). Historical runs use circles; the recovery-era
  and streaming-era runs reuse circles but are the five newest colors below.
- **Series** (color · style · meaning):
  - `Naive attn+FFN (ternary)` — **red, solid** — the first QAT attempt on all
    attention+FFN projections: collapses to ≈100 % (history).
  - `FFN-only (ternary)` — **orange, solid** — quantizing only the FFN
    projections: the first sign of a workable state (≈33 %).
  - `FFN-only resume (LR-restart artifact)` — **orange, dotted** — kept
    *separate* on purpose: this jump is a learning-rate-restart artifact, not
    real progress; it is drawn dotted so it is never read as a continuation.
  - `Stage 2a attn+FFN rollout (ternary)` — **blue, solid** — attention
    rollout with frozen anchor (four concatenated runs: base + resume + two
    extensions); the recovery recipe's origin (≈16 %).
  - `Stage 2b attn+FFN rollout (binary)` — **green, solid** — the same for
    binary weights (≈17 %).
  - `Ternary long (8k data, 8000 steps)` — **purple** — scaling the recipe to
    a 8k-utterance cache (≈11.4 %).
  - `Binary long (8k data, 6000 steps)` — **brown** — binary at 8k (≈12.9 %).
  - `Ternary long2 (8k, lr 1e-5)` — **pink** — higher LR, best of the 8k era
    (≈9.2 %).
  - `Binary long3 (8k, lr 5e-6)` — **grey** — lower LR, the binary 8k best
    (≈10.7 %).
  - `Recovery recipe — ternary (full cache)` — **black, solid** — the recipe
    scaled to the full 28 539-utterance cache: the **7.52 %** offline best.
  - `Full-cache anneal — ternary (2nd cycle)` — **cyan, dashed** — a second
    cosine cycle on top (7.63 %: no gain, kept for honesty).
  - `Recovery recipe — binary (full cache)` — **green, dashed** — the binary
    full-cache run: **10.70 %**, the quantizer's practical floor (the 10.70 is
    a sign-error/mirror floor, not an optimization failure).
  - `Aligned streaming SFT — ternary` — **yellow-green, solid** — the
    streaming-era SFT run (the aligned-targets recipe); its *offline* readout
    settles at ≈9.4 % while its *streaming* numbers are in figure 2.
- **Reference line:** the **black dotted horizontal line** at **2.44** is the
  fp16 pretrained 0.6B offline baseline (50-utt test-clean) — the floor a
  perfect quantization would approach.

### 2. `streaming_results.png` — the official streaming algorithm (Stage 4)

![Streaming results](docs/figures/streaming_results.png)

- **Left panel — the collapse-and-fix history.** One bar per recipe stage, all
  measured with the *official* prefix algorithm at **320 ms, 50-utt
  test-clean**, y on a **log scale** (8 → 2 000 %):
  - `ternary prefix (first attempt)` — **951.8** — the raw quantized model
    under the official algorithm: total collapse.
  - `+ DAgger` — **920.6** — on-policy DAgger on the student's own prefixes
    *poisons* the model (it trains its own loops back).
  - `+ distill v1 / v2 / v3` — **448 / 350 / 278** — streaming-state
    distillation, improving steadily but plateauing on the content/stop
    balance (v1 had a metadata-format conflict; v2 under-emitted).
  - `adapt5` — **175.0** — longer adaptation.
  - `+ LSP loop policy` — **144.6** — the inference-side loop-emission policy
    (truncate loops, keep one occurrence) halves the error and removes the
    visible degeneration.
  - `long2 state` — **33.6** — the offline state itself as the streaming base.
  - `binary + aligned` — **18.4** — the binary arm with aligned targets.
  - `ternary + aligned (prev)` — **12.9** — the previously promoted recipe
    (blue bar).
  - `ternary + LSP, longer cycle (BEST)` — **11.5** — **the current promoted
    state** (red bar): the same aligned recipe on a single 5 000-step cycle,
    selected at its *step-1 000* checkpoint (the trajectory peaks early:
    11.55 @1 000 → 12.82 @1 500 → 14.30 @~1 750), evaluated on full samples.
- **Right panel — the promoted matrix, current best vs previous.** Grouped
  bars, one group per chunk size (160/320/1 000 ms) plus the binary arm at
  320 ms. Each group has up to four bars: **pale blue = previous state,
  test-clean**, **pale orange = previous state, test-other**, **solid blue =
  current best, test-clean**, **solid orange = current best, test-other**;
  every bar is annotated with its WER, and a missing cell (the current best has
  no `other` eval at 160/1 000 ms, no binary counterpart) is simply not drawn.
  The four horizontal reference lines are the Stage-0 anchors and the gate:
  `fp16 1.7B clean 2.24` and `fp16 1.7B other 4.61` (dotted, the baseline) and
  `gate clean 3.74` / `gate other 6.11` (dashed, baseline + 1.5 — the Stage-4
  success criteria). The visible gap between the bars and the dashed lines *is*
  the open gate. Read the 320 ms group carefully: the current best (11.55)
  clearly beats the previous state (12.92), while at 160/1 000 ms the two are
  within ~0.5 of each other.
- **Why it is two panels:** the left answers "was the algorithm fixable?"
  (yes, 951.8 → 12.9), the right answers "where does the promoted recipe stand
  against the gate?" (close on clean at 1 000 ms, far on other).

### 3. `final_wer.png` — final offline WER per configuration

![Final WER by configuration](docs/figures/final_wer.png)

- **Axes:** x = configuration (chronological, with the recovery-era ones on the
  right); y = final WER % (50-utt test-clean, best checkpoint).
- **Bars:** **blue = ternary readout**, **green = binary readout** — both
  *real measurements of the same trained weights* at the two deployment
  precisions (**the recovery-era rows are read straight from each run's results
  JSON**, not hand-copied). A bar near 100 means the model cannot transcribe at
  that precision at all — the true "wrong-precision deployment" number
  (e.g. the ternary-trained run decoded at binary reads 98.4), not a missing
  checkpoint. Each bar is labelled with its value.
- **Reference line:** the black dotted **2.44** fp16 pretrained baseline.
- **Read it for:** the three-era progression left→right — naive ≈100 % → FFN
  ≈33 % → rollout ≈16 % → 8k ≈9–13 % → full cache **7.63 ternary / 10.70
  binary** (and the binary run's *ternary* readout 9.75 — precision-mixing is
  a real measurement, not a runner).

### 4. `quantizer_error.png` — what quantization costs in weights

![Quantizer error](docs/figures/quantizer_error.png)

- **Axes:** x = quantizer variant (`binary/ternary` × `tensor`-scale,
  `g64`, `g128` group scales); y = relative Frobenius error
  `‖Q(W) − W‖ / ‖W‖` on the real 196 decoder projections of Qwen3-ASR-0.6B.
- **Bars:** one per variant; **red/pink/purple = binary**, **blue/cyan/green =
  ternary**. Lower is better; the difference between tensor- and group-wise
  scaling is the headroom that the learnable-scale design exploits.
- **Reference line:** **0.60** — the Gaussian closed form for a 1-bit
  quantizer, confirming the implementation is neither better nor worse than the
  theory (a sanity check, not a target).

### 5. `fp16_drift.png` — shadow drift during QAT

![Shadow drift](docs/figures/fp16_drift.png)

- **Axes:** x = QAT step; y = WER % of the **fp16 readout (α = 0)** of the
  trained weights — i.e. what the *unquantized* model would do with the same
  weights, evaluated in-training.
- **Series:** one line per run (squares), historical runs plus the two
  full-cache runs (black/in green). The curve *rising* means the weights are
  drifting away from fp16-good territory into a quantization-friendly one —
  that is the intended effect of the frozen-anchor recovery recipe, not a bug.
- **Reference line:** the **2.44** pretrained fp16 baseline (black dotted).
  The full-cache runs rise to 21–27 %, which is why the quantized readout
  (figures 1–2) is the metric that matters.

### 6. `roadmap.png` — stage status overview

![Roadmap](docs/figures/roadmap.png)

- **Axes:** x = informal completion (0–125 %); y = the seven plan stages, most
  advanced at the bottom.
- **Bar colours:** **green** = completed (Stages 0–2), **orange** = in
  progress (Stage 3 paused on cluster capacity, Stage 4 gate open), **blue** =
  begun (Stage 6 write-up), **grey** = not started (Stage 5).
- **In-bar text:** the one-line state of each stage (e.g. Stage 4's
  "203.5 → 12.92 clean / 20.00 other, zero loops; gate (~3.7/6.1) open — gap is
  quantization damage").

---

## Plan — what happens next

1. **In flight (0.6B, cluster permitting).**
   - `qat_stream_aligned5b.pbs` — the resume of the longer cycle (to step 5 000,
     ending ~21:15); its step-2000/2500/3000 states are being scored on full
     samples (`eval_aligned5b.pbs`) because the 20-utt guardrail *improved*
     (8.94 @2 000, 9.18 @2 750) while the same schedule's early full samples
     fell (11.55 → 12.82 → 14.30) — the fourth guardrail-vs-metric divergence in
     this project, so only full samples decide.
   - `qat_stream_bin_long.pbs` — a **3 000-step** binary run starting from the
     new best ternary state (the 1 500-step budget was too short; the binary
     transition spread 18.4/27.5/36.4 across nominally identical runs), with six
     selectable checkpoints.
   - Cluster rules that gate all of this: submit only when a GPU genuinely has
     ≥20 GB free, ≥35 GB for training runs (`scripts/pick_gpu_wait.sh` waits
     inside the job); co-tenants grow mid-run and OOM'd four attempts on
     2026-10-15.
2. **Quantizer-side levers** for the quantization-damage gap — stochastic
   rounding and co-training (precision schedules / mixed ternary-binary) are the
   untried options, alongside scale.
3. **Stage 3 — the 1.7B.** The anchored fp32-AdamW retry ran clean (1 200 steps,
   **no permanent dip**) and gave the first streaming number: **29.24 clean**
   from an offline **24.05**. That is the finding: the streaming recipe
   transfers (gap +5.2 vs the 0.6B's +4.0), but the 1.7B's quantization damage
   is **+22.4** against its fp16 offline (1.63) versus the 0.6B's **+5.4** —
   because it never received the *complete* offline recovery recipe. **Next:**
   full-cache QAT + anneal cycles on the 1.7B (offline target sub-10) *before*
   repeating the aligned streaming SFT. 1.7B jobs need `mem=64gb` (host-RAM
   SIGKILL otherwise, as its own final eval showed).
4. **Stage 5 (edge)** — CPU/RTF work once the accuracy gates are judged.
5. **Stage 6 (write-up)** — keep the docs and figures regenerated from
   artifacts; `docs/stage4_results.md` holds the full streaming record.

Full report and history: [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md).
**Next agent: start with [`docs/HANDOFF.md`](docs/HANDOFF.md)** (state, recipe,
what's left, pitfalls) and [`docs/GPU_ACCESS.md`](docs/GPU_ACCESS.md) (cluster
access). Operating rules: [`rules.md`](rules.md); step-by-step work log:
[`docs/ACTIVITY_LOG.md`](docs/ACTIVITY_LOG.md).

---

## Development

Local dev/unit-test environment is Python **3.12** (pinned in `.python-version`),
created with [`uv`](https://docs.astral.sh/uv/):

```bash
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"   # Windows
# uv pip install --python .venv/bin/python -e ".[dev]"          # Linux/macOS
```

Run the suite and linter:

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
```

The Kaggle/Colab runtime (Linux + CUDA, includes `qwen-asr[vllm]`) is described
in `requirements-stage0.txt`. Qwen3-ASR streaming is **vLLM-only**, so streaming
baselines run on GPU hosts, not on the local dev machine.

## Layout

Contributing or extending the code? Start with
[`docs/CODE_GUIDE.md`](docs/CODE_GUIDE.md) (file-per-concern rules, commenting
style, the safe way to split a large file, and the telemetry/docs every run
must emit).

```
src/asr1bit/   # package (quant, bitlinear, replace, data, qwen, train, eval, deploy)
tests/         # unit tests
configs/       # run configurations
docs/          # planning, design, status reports, figures
scripts/       # run/launch/plot tooling (incl. PBS launchers)
rules.md       # operating rules for agents/developers
```

## Versioning

Tags use `MAJOR.MINOR.BUGS`: **MAJOR** = end-to-end milestone, **MINOR** = a
completed stage (gate passed), **BUGS** = fixes and non-stage checkpoints.
Stage work lands on the **current working branch** — `stage-1` today — and `main`
advances only when a stage gate passes (it still sits at the `stage-0` merge,
`0.2.0`). Branch names track *gates*, not the plan's stage numbers: the work of
plan-Stages 1–4 all lands on `stage-1` until the next gate passes. Checkpoints
are documented in [`CHANGELOG.md`](CHANGELOG.md).
