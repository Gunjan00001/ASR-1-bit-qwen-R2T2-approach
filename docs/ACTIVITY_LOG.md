# Activity log (append-only)

> New here — or preparing a viva/interview? Read
> [`docs/DEEPDIVE.md`](DEEPDIVE.md) first: it explains the project from the
> ground up and indexes every failed and successful attempt with its number.

Step-by-step record for **future agents and developers**. Every entry: what we
did, failure symptom, root cause, fix, evidence. Update after every meaningful
step (see `rules.md` R3). For protocol-level detail see `docs/LAB_NOTES.md`
(append-only, five fields) and `docs/stage1_results.md`.

---

## Live status (updated 2026-10-15 03:35 IST)

| Job | What | Progress | Result | ETA |
|---|---|---|---|---|
| `11383` | **0.6B — longer aligned cycle** (`qat_stream_aligned5r`) | **CUT by 6 h walltime at ~step 1 776** (13:01); guardrail track: 20.94 @250 → 11.76 @750 → 11.53 @1000 → …; checkpoints `step500/1000/1500.pt` + a **pre-resume snapshot** `last_aligned5r_end.pt` (the 1 750-step state) on disk | full-sample evals submitted (`11486`) | done |
| `11485` | **Resume of the longer cycle** (`qat_stream_aligned5b`: continues from the 1 750-step state to step 5 000 total, 8 h, `--keep-step-checkpoints 500`, ≥35 GB wait-picker) | queued/starting | pending | 8 h from start |
| `11486` | **Evaluator for the longer cycle** | **done — PROMOTED (tag `0.2.47`)** | winner = **step 1 000: 11.55 clean / 18.98 other** (CER 6.41 / 11.11, 0–1 truncations) — beats the previous promoted 12.92 / 20.00 on both splits; the trajectory peaks early (12.82 @1 500, 14.30 @~1 750) | — |
| `11512`/`11513` | **Chunk sweep for the new best** (160 ms / 1 000 ms, clean) | **done** | 160 ms **13.77**, 1 000 ms **10.70** — the new state wins at the gate metric (320 ms) but is marginally behind at the extremes (13.24/10.59); stated plainly in `stage4_results.md` | — |
| `11526` | **Resume-checkpoint evaluator** | **done** | step 2 000 / 2 500 / 3 000 all **11.76 clean** (CER 5.02/4.56/4.56; 0/3/0 truncations) — **no state beats the promoted 11.55** → nothing promoted. Fifth guardrail-vs-metric divergence (the guardrail said 8.94–9.88). The plateau is *stable*, which bodes well for long cycles. | — |
| `11514` | **1.7B final-state evaluator** | **done** | **29.24 clean / 33.75 other** — the 1.7B's first complete streaming row; the streaming recipe transfers, the offline base is its bottleneck | — |
| `11532` | **1.7B OFFLINE recovery cycle** (init attn5 20.55, anchor, lr 1.5e-6, `--quant-stochastic 1`) | **reached step 2 500** @01:11 (final eval + save running) | offline readout 24.15 (500) / 23.09 (1000) / 24.79 (1500) / **22.78 (2000)** — no permanent dip, but **not below the 20.55 init** ✗; still converging slowly | ends ~01:30 |
| `11491` | **Longer binary run** | **COMPLETED all 3 000 steps** (zero OOMs) — the first binary run to finish a full budget | `11542` | **DONE** — bin_long fully scored: best = **26.80 clean / 35.23 other** (step 2 500); trajectory 61.12/40.36/29.77/28.28/26.80 → **no promotion (binary best 18.43/25.45 stands)** | — |
| `11543` | **1.7B recovery — CONTROL arm** (`qat_17b_recover2.pbs`: byte-identical to recover1 **minus `--quant-stochastic`**) | running (picker waiting for ≥45 GB) | isolates whether stochastic rounding caused recover1's flat 23–25 readout | ~8 h |
| `11543` | **1.7B recovery — CONTROL arm** (no SR, lr 1.5e-6) | **completed 2 500 steps** | offline **20.76 / 20.55 / 19.17 / 19.92 / 19.17→19.92** — the flat-to-slow-gain shape; never below 19 | — |
| `11581` | **1.7B recovery — LR arm** (lr **5e-6**, no SR) | **completed 2 500 steps** | offline **19.92 → … → 16.74 → 16.53** — **4.0 below its 20.55 init** and the best 1.7B recovery seen; the schedule answer confirmed | streaming eval 11583 running |
| `11582` | **1.7B recovery — OPTIMIZER arm** (fp32 AdamW, lr 1.5e-6) | **completed 2 500 steps** (no OOMs at the 1.41 B fp32 footprint) | offline 19.49 (1500) / 19.28 (2000) / **21.40 (2500)** — a small edge over recover2 at 1.5e-6, but behind recover3's LR arm | — |
| `11667` | **1.7B recovery — PHASE 2 (multi-cycle)** | **COMPLETED 4 000 steps** | offline **16.53 → 11.76** (20.23 / 17.90 / 15.15 / 13.14 / 11.97 / **11.76**) — **the ~sub-12 target met**, still descending; fp16 readout improved throughout | — |
| `11754` | **STAGE B — aligned streaming SFT on the recovered base** (`aligned17b6`: init recover5/last **11.76**, aligned-n 2 400, mix 50/50, stop-stride 2, lr 5e-6, fp32 AdamW, 2 500 steps, keep-step 400, 12 h) | training, step ~500 @08:08 | **first guardrail @250 = 16.00** (CER 8.85) vs the same recipe on the shallower base's **26.59** — the deeper recovery *transfers* to the streaming head; full-sample evals decide | ~6 h |
| `11758` | **Cycle 3 — recovery phase 3** (`recover6`: init recover5/last 11.76, 4 000 steps, lr 5e-6, fp32 AdamW, keep-step 400) | launched 08:05 | pending — feeds the pipeline's next Stage-B (target offline ~9–10) | ~8 h |
| `11583` | **recover3's streaming eval** (`eval_streaming_17b.pbs`) | clean split **DONE**: **WER 285 with 354 loop truncations** (raw 629) — the recovered base (16.53 offline) *collapsed at streaming* ✓ the plain-offline recipe broke the streaming head | `other` split finishing | ~30 min |
| `11592` | **The two-stage test** (aligned recipe on the *recovered* base, lr 5e-6, 2 000 steps) | **completed** | offline preserved: **16.63 / 16.74** vs the recovered base's 16.53 ✓ (the base survived); guardrail trend 19.3–21.9 (a guardrail, not a metric) | full-sample streaming eval `11666` running |
*(the LR arm's rows are collapsed below into the 11581/11582/11543 summary rows above)*
| `11474` | **Binary early-stopping evaluator** | **done** | step 500 **56.14**, step 1000 **40.25**, step 1500 **36.44 clean / 43.18 other** — **no state recovered 18.43**; its curve was still descending at the cut | — |
| `11491` | **Longer binary budget** (`qat_stream_bin_long.pbs`: 3 000 steps, 8 h, from the new best ternary state) | **training** — step ~840, zero OOMs, guardrail descending 66.6 @250 → 47.8 @500 → 36.5 @750 | pending — full-sample evals at its end decide vs the binary best 18.43/25.45 | its 8 h end ~00:00 |
| `11514` | **1.7B final-state evaluator** (`eval_streaming_17b.pbs`, mem=64gb, clean → other) | running | pending — the 1.7B retry trained clean (1 200 steps, no dip: offline 27.97 → 24.05) and its checkpoint is being scored on full samples | ~2 h |
| `11384` | **Binary early-stopping re-run** (`qat_stream_bin_es`) | **completed** — 1 500 steps, zero OOMs, `step500/1000/1500.pt` on disk | `11474` (`eval_bin_es.pbs`) scoring now: **step 500 = 56.14 clean** (CER 38.55, 4 truncations — mid-transition as expected); step 1 000/1 500 + winner's other pending vs the binary best 18.43/25.45 | ~2 h |
| `11470` | **Stage-3 — 1.7B retry v4** (anchored fp32-AdamW) | **training COMPLETED 1 200/1 200 steps, zero GPU OOMs — no permanent dip** (offline 27.97 @500 → **24.05 @1 000**; guardrail 23.8 @750 → 25.4 @1 000). The job was then killed by a **host-RAM SIGKILL during its own final streaming eval** (no traceback) — the checkpoint (`last.pt`, 5.6 GB) survived | `11514` (`eval_streaming_17b.pbs`, mem=64gb, clean → other) scoring its final state | ~2 h |
| `11384` | **Binary early-stopping re-run** (`qat_stream_bin_es`: from the ternary aligned2, bit_width 1.0, `--keep-step-checkpoints 500`) | **started 07:55** — picked GPU 5 (≥35 GB free), loading → alignment | pending; then `eval_bin_es.pbs` scores step-500/1 000/1 500 to recover the best binary state (18.43/25.45 was lost) | ~3–4 h + evals |
| — | **0.6B OOM post-mortem (2026-10-15)** | four attempts died: three at the first training step, one at model *loading* (co-tenants took 20+ GB between pick and load) | fix: strict/self-waiting GPU picker + a 35 GB bar (≥2× the ~15 GB need); the 07:01 pick satisfied it | — |
| `11371` | **Binary early-stopping re-run** | **OOM at first step** (co-tenant growth after the picker sampled; the alignment was wasted) | resubmit `qat_stream_bin_es.pbs` when a GPU has ≥20 GB free, then `eval_bin_es.pbs` for its step-500/1000/1500 states | — |
| — | **Strict GPU picker** (`scripts/pick_gpu.sh`) — refuses (exit 3) when nothing has ≥20 GB free, instead of the least-used fallback that bought two OOMs | done (all resubmittable launchers wired) | — | — |
| — | **Binary arm** | final evals done | step-1 250 state **18.43 / 25.45** (weights overwritten by the resume ✗) → final step-1 500 state **27.54 / 26.93** (29–40 loop truncations: the resume broke the loop discipline) | re-run with early stopping if needed |
| — | 1.7B aligned | **paused** — all 8 GPUs 60–78 GB used | — | — |

**This round's lessons:** (1) snapshot before resuming — the binary's better
streaming state (18.43) is gone because the resume overwrote it; (2) a resume is
a *new experiment*: it improved the binary's offline (41.8 → 35.2) while the
streaming arm degraded 18.4 → 27.5, so schedule-based completions need an
early-stopping rule on the streaming eval; (3) figures were regenerated
(`MPLBACKEND=Agg`), the matrix cells in PROJECT_STATUS/HANDOFF now show both
binary states.

**The 0.6B aligned matrix (official algorithm, loop policy on, 50 utts):**

| checkpoint | 160 ms | 320 ms | 1 000 ms |
|---|---:|---:|---:|
| ternary aligned — clean | **13.24** (RTF 10.3) | **12.92** (RTF 3.3) | **10.59** (RTF 3.2) |
| ternary aligned — other | **20.80** | **20.00** | **16.59** |
| binary aligned — clean | — | **18.43** | — |
| binary aligned — other | — | **25.45** | — |
| fp16 pretrained (same path) | — | 2.33 | — |

Committed to `docs/PROJECT_STATUS.md` + `docs/stage1_results.md`; tag `0.2.43`.
[CORRECTION 2026-10-16] The note above ("the remaining gap is 0.6B capacity,
not the recipe") was wrong and is superseded by the measured decomposition in
docs/LAB_NOTES.md (2026-10-15): the fp16 **0.6B** streams at 2.54/6.27 — close to
the 1.7B's 2.24/4.61 — so size is not what limits the fp16 side; the gap is
**quantization damage**.

---

The `+1.5` gate (~3.7/6.1) is not met — the remaining gap is 0.6B capacity, not
the recipe (zero loops everywhere; binary only +5.5 behind ternary in streaming
vs +32 offline).

**MILESTONE (tag `0.2.42`): the aligned-targets recipe works and scales.**

| checkpoint (official algorithm, 320 ms, 50 utts, LSP policy on) | offline | clean | other | loops |
|---|---:|---:|---:|---:|
| aligned v1 | 14.62 | **12.61** | 22.50 | 0 |
| **scaled aligned** (7.5 base, 2 400 utts) | **9.1–10.7** | **12.92** (CER 5.00) | **20.00** | 0 |
| pretrained fp16 reference | 2.44 / **1.17 (1.7B)** | 2.33 | ~4.6 | — |
| previous best streaming (distill3 + loop policy) | 25.6 | 144.60 | — | 11 |

The 0.6B streaming has **saturated at ~12–13** (scaling improved offline and
`other`, not `clean`) — the remaining error is the model's own streaming content
ability, so the **1.7B** (pretrained offline 1.17 vs the 0.6B's 2.44) is the
lever. Stage 3 status: FFN warm start 22.03, anchor-free attention rollout
finished at **20.55** (2000 steps; the anchored variant awaits a quiet GPU).
The binary mirror (`11159`) is running.

**Root cause of the crash chain (fixed):** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
is incompatible with this cluster's driver 525 (`CUDA driver error: invalid
argument` on ordinary CUDA ops) — never set it. Also: `gradient_checkpointing_enable()`
breaks the frozen-only setup, and `bitsandbytes` (now installed, 0.50.2) is
required for `--optimizer adam8bit` to engage.

**1.7B attention rollout:** needs ~38–40 GB free (1.41 B trainable, fp32
student+teacher); the shared node is racy. `qat_17b_attn5.pbs` (`11146`) drops
the anchor to fit (watch its ternary WER — the 0.6B needed the anchor).
Banked Stage-3 result: **FFN warm start 22.03 ternary**.

**T4.1/T4.2/T4.3 done:** `data/align.py` (aligner adapter), `data/prefix.py`
(chunk-aligned targets; probe verified on real audio — e.g. chunk 1 → `'I'`,
chunk 2 → `'HAVE GREAT'`, 9/39 chunks empty for *"I HAVE GREAT THINGS…"*), and
`--aligned-targets 1` in the trainer with `--aligned-stop-stride 2`. The probe
also exposed the recipe's supervision quality: targets are exactly
"the words whose end time falls in this chunk" — the first recipe whose targets
are defined by the *audio heard*, not by the model or the teacher.

**1.7B attention rollout:** needs ~38–40 GB free (1.41 B trainable, fp32
student+teacher, fp32 Adam); the shared node's co-tenants make that racy. Retry
options: a quiet GPU (≥55 GB free), dropping the anchor
(`--frozen-teacher 0 --distill-weight 0`), or partial attention quantization.
Banked Stage-3 result: **FFN warm start 22.03 ternary**.

**Loop truncation at emission is the biggest single win:** cutting a trailing
repetition loop (one unit kept) at the LSP layer takes distill3 from **278.18 →
144.60** (confirmed through the real eval path, job `11067`), distill2
349.89 → 154.56, v1 448.31 → 179.34. Implemented as `truncate_trailing_loop` +
`LSPState(truncate_loops=True)` (`--lsp-truncate-loops 1`), 309 tests.

**WP2 binary hits a floor:** the binary full-cache run (from the 7.5 ternary,
8 000 steps) converged to **binary 10.70** — identical to the old 10.7 despite
the much better ternary base (readout 9.75 vs 22.35). Further binary gains need
the co-training/stochastic-precision stack, not more of the same recipe.

**Offline best (50-utt test-clean; fp16 pretrained 2.44):**

| Checkpoint | Offline ternary | Offline binary |
|---|---:|---:|
| `qat_ternary_long2` (8k data) | 9.2 | — |
| **`qat_ternary_full` (28 539 utts)** | **7.52 best / 7.63 final** | **10.70** |
| `qat_stream_distill3` | 25.6 | — |
| `qat_stream_content3` | 15.8 | — |
| `qat_stream_adapt5` | **14.51** | — |
| `qat_17b_ffn` (Stage 3) | 22.03 (FFN-only) | — |

**WP1 streaming results (official prefix algorithm, 50-utt test-clean, 320 ms,
32 tokens):**

| Recipe | Official-algorithm WER | RTF | Failure mode |
|---|---:|---:|---|
| Static prefix SFT (earlier session) | 203.5 | — | correct content, never stops |
| On-policy SFT, student prefixes (DAgger) | 920.55 | 4.49 | loops get **worse** (fits its own loops) |
| Distillation v1 (`language X<asr_text>` kept) | 448.31 | 1.76 | stops ✓, content garbled (format conflict) |
| Distillation v2 (metadata stripped) | 349.89 | 1.47 | loops gone, stops ✓, under-emits |
| Distillation v3 (three-way mix) | **278.18** | 2.44 | streaming policy **correct**; residual = offline content + trailing loops |
| **v3 + LSP loop policy** | **144.60** (confirmed) | 1.39 | remaining gap = ordinary content errors |
| Content phase (40 % share) + policy | 190.78 (raw 532.20) | 2.60 | offline 15.8 but stop policy eroded |
| adapt5 (three-way from content3) + policy | pending `11115` | — | aims to hold offline 14.5 *and* the policy |

**Streaming 320 ms (50 utts) — final picture:**

| Variant | Pretrained | Ternary | Binary |
|---|---:|---:|---:|
| Official prefix algorithm | **2.33** | collapse 1296 → 952 (short SFT) → **203.5** (scaled SFT) | — |
| Official + repetition_penalty=1.3 | — | 438.2 (worse) | — |
| No-prefix re-feed, 32 tokens | 19.28 | 33.58 | 33.26 |
| **No-prefix re-feed, 128 tokens (reported)** | **2.54** | **9.43** | **10.28** |

**Offline (50-utt test-clean; fp16 baseline 2.44):** ternary **9.2** (best 9.1);
binary **10.7**; ternary+streaming-SFT ~12.

**Conclusion:** the official prefix algorithm needs research-grade Stage 4 work
(on-policy streaming SFT / LSP emission); the chunked re-feed variant (9.43 vs
2.54) is the reported streaming metric.

**Session-cgroup rule:** never load model + checkpoint in the 32 GB interactive
session (OOM kills silently). Batch jobs only (rules.md R5).

**Resume rule:** `--resume` continues the same run (advances the LR schedule and
step counter); `--init-from` loads weights for a new phase (step counter
restarts at 0).

---

## 2026-10-10 — Session: Stage 1 on the JIIT A100 cluster

### 18:10 — Repository set up in the working folder

- **Did:** `git init` in `~/Desktop/ASR 1bit qwen`, added the GitHub remote,
  fetched, checked out `stage-1` (20 commits ahead of `main`; `main` at `0.2.0`,
  Stage 0 gate passed). `gpu.md` (cluster notes) kept untracked.
- **Failure:** none.
- **Evidence:** `git log` on `stage-1` head `a78334a`.

### 18:15 — Python environment for GPU + tests

- **Did:** installed `transformers 4.57.6`, `datasets 5.0.1`, `jiwer`, `peft`,
  `accelerate`, `qwen-asr 0.0.6`, `pytest`, `ruff` into `~/miniconda3`
  (torch `2.11.0+cu126` already present).
- **Failure:** `import pyarrow` → *"pyarrow requires NumPy 2.0 or newer, found
  1.26.4"* (datasets 5 requirement).
- **Cause:** env had numpy 1.26; `datasets>=5` needs numpy≥2.
- **Fix:** the `qwen-asr` install pulled `numpy 2.5.3`; suite re-run green.
- **Evidence:** `pip list`; `pytest -q` → 178 passed, 1 skipped.

### 18:35 — Quantizer audit (settled the "bug or expected?" question)

- **Did:** wrote `scripts/diag_quant_error.py`; measured relative Frobenius
  error `||Q(W)-W||/||W||` over all 196 decoder projections directly from
  `model.safetensors`.
- **Result:** binary per-tensor **0.648** (closed form for Gaussian weights
  ≈ 0.60); ternary 0.565; group-wise g64/128 binary 0.616/0.614, ternary
  0.519/0.522; bf16 round-trip ~0.
- **Conclusion:** quantizer math is **correct**; the previous session's
  single-projection collapse (WER 100) is inherent naive 1-bit behaviour, not a
  harness bug. Recovery must come from training, not a quantizer fix.
- **Evidence:** `artifacts/stage1/diag_quant_error.json`.

### 18:45 — Data cache: first attempt OOM-killed

- **Did:** built a 4 000-utterance `train.clean.100` cache on the GPU node with
  `scripts/build_qat_cache.py` (non-streaming `datasets` load).
- **Failure:** process killed (0-byte output, log stuck at 3 500 rows).
- **Cause:** the non-streaming load materialized the full split; the
  interactive session's **32 GB cgroup** (page cache included) hit the OOM
  killer (`dmesg`: `oom_memcg=.../10708.jiit-master`).
- **Fix:** added `streaming=True` support to `iter_librispeech` and built the
  cache **on the login node** (no cgroup limit). Result: 4 000 utts, 14.5 h
  audio, mean 13.0 s, 3.34 GB in 91 s.
- **Evidence:** `artifacts/stage1/build_qat_cache_login.log`,
  `data/qat_clean_train.100_n4000.pt`.

### 19:10 — "Hanging" processes: slow tokenizer, not a hang

- **Did:** debugging why smoke runs sat with empty logs for 10+ minutes.
- **Failure:** `py-spy` showed every process stuck in
  `nagisa/tagger.py` / `Qwen2Tokenizer` building `bpe_merges` line-by-line from
  `merges.txt` (line 76k of 151k after minutes).
- **Cause:** Qwen3-ASR checkpoints ship `vocab.json`/`merges.txt` but **no
  `tokenizer.json`**, so `transformers` builds the slow Python tokenizer on
  every process start; under cluster CPU/FS contention this takes minutes and
  looks like a hang.
- **Fix:** `scripts/install_fast_tokenizer.py` builds `tokenizer.json` once
  (`Qwen2TokenizerFast`); installed into the model dir. Full CPU model load
  dropped to **11.6 s**.
- **Evidence:** `scripts/install_fast_tokenizer.py`; loader check output.

### 19:30 — Node-local staging for speed

- **Did:** copied conda (20 GB), model (1.8 GB), and cache (3.2 GB) to
  `/tmp/asr1bit` on `jiit-gpu01`; all later runs use the local copy.
- **Failure:** shared BeeGFS reads stall under multi-job load (minute-long
  `D`-state waits).
- **Fix:** local `/tmp` staging; batch scripts fall back to shared paths if
  `/tmp/asr1bit` is absent.
- **Evidence:** `/tmp/asr1bit/{miniconda3,models,qat_clean_train.100_n4000.pt}`.

### 19:40 — Memorization test: loss controllability proven

- **Did:** `scripts/diag_generation.py` with `--train-n 1` (single utterance),
  600 steps, lr 1e-3, constant α: α=0 control, binary, ternary.
- **Result:** α=0 → loss **0.000** @300; ternary → **0.000** @300 with greedy
  text exactly matching the label; binary → 2.818 @600 (partial).
- **Conclusion:** harness, labels, STE, and learnable scales are correct;
  **ternary QAT is fully controllable**. The earlier "forward/objective bug"
  verdict is retracted — previous failures were budget/config (≤1 k steps,
  lr 1e-5, ≤200 utts, no learnable scales).
- **Evidence:** `artifacts/stage1/diag_generation_decoder_attn_bw1.{0,58}_steps600.json`.

### 20:40 — Learnable-scale pathology found and fixed

- **Did:** first real QAT run with `learnable_scale=True` (free parameter).
- **Failure:** one layer's scale went **negative** (−0.0044) and several
  collapsed ~10× (0.002–0.003 vs init 0.02); WER worsened while loss improved.
- **Cause:** unconstrained additive scale parameter; Adam updates are linear in
  the scale, so it can cross zero and shrink unchecked.
- **Fix:** **log-parameterization** — parameter stores `log(gamma)`,
  `gamma = exp(log_scale).clamp(1e-8, 1.0)`; updates are multiplicative and
  gamma is strictly positive. Gamma stats (`gamma_min/mean/max`) logged every
  5 steps in the runner and checked at checkpoints.
- **Evidence:** `src/asr1bit/bitlinear.py` (`gamma()`, `_init_weight_scale`),
  tests `TestLearnableScale`; checkpoint inspection at step 1000.

### 21:00 — Naive QAT v2 (6 000 steps) and self-distillation (4 000 steps)

- **Did:** `decoder_attn_ffn`, ternary, log-scaled learnable scales, lr 2e-4
  cosine, 5 % fp warmup, batch 4 (distill: 2), eval on fixed 50-utt test-clean.
- **Result:** naive periodic WER 127 → 121.6 → 96.5 → 96.9 → 103.5; final
  **fp16 readout 113.1**, binary 99.8, ternary 107.8. Self-distillation
  (teacher = same shadows at α=0) WER 114.6 → 125.0 → 111.3 → 98.1; final fp16
  108.3. Pretrained reference **2.44**.
- **Failure:** no recovery; the **fp16 readout itself collapsed** (2.44 → ~110)
  after 4–6 k steps at lr 2e-4 (the earlier T4 runs at lr 1e-5 kept it clean).
- **Cause:** per-tensor ternary (~52 % weight error) is too coarse for the
  attention pathway; the optimizer drifts the latent weights into a degenerate
  basin; the moving self-teacher drifts with the student so it cannot anchor.
- **Fix (direction):** keep attention fp (FFN-only), anchor with a *frozen*
  teacher, lower lr.
- **Evidence:** `artifacts/stage1/stage1_decoder_attn_ffn_*.json`,
  `qat_ternary_attnffn_ls2.log`, `qat_ternary_distill.log`.

### 22:30 — Failure mode confirmed: audio conditioning collapse

- **Did:** `scripts/inspect_checkpoint.py` on the v2 checkpoint, 5 utterances.
- **Result:** fp16 and ternary readouts both WER 112.2 and produce fluent
  generic text unrelated to the audio ("AND THE FIRST OF THE HUMAN BODY…").
- **Cause:** the decoder lost audio conditioning and fell back to the language
  prior; the frozen AuT encoder is fine — the ternary attention path is the
  damage site.
- **Evidence:** `artifacts/stage1/inspect_v2.json`.

### 22:40 — FFN-only run recovers (key positive result)

- **Did:** same protocol, policy `decoder_ffn` (attention stays fp; 84
  BitLinear).
- **Result:** ternary WER **118 → 84.3 → 42.2 → 25.7** at steps
  500/1000/1500/2000; training loss 4.1 → ~2; gamma healthy. Binary FFN run
  launched (97.0 @500).
- **Conclusion:** attention quantization was the blocker. FFN ternary (~2/3 of
  decoder params) is recoverable at this budget.
- **Evidence:** `artifacts/stage1/qat_ternary_ffn.log`,
  `artifacts/stage1/qat_binary_ffn.log`.

### 22:50 — Anchored distillation at w=0.5 is destructive (negative result)

- **Did:** frozen pretrained fp teacher + KL 0.5 (lr 5e-5), 1 000-step eval.
- **Result:** KL exploded to 11–15; fp16 WER 427, ternary 315; killed.
- **Cause:** the KL gradient from an unattainable frozen teacher dominates the
  CE term.
- **Lesson:** anchoring needs a much smaller weight/temperature or a
  trust-region formulation, not flat KL 0.5.
- **Evidence:** `artifacts/stage1/qat_ternary_anchored.log`.

### 23:00 — GitHub access verified and pushed

- **Did:** checked `~/.ssh` — `id_ed25519_github` already present with an
  `~/.ssh/config` entry; `ssh -T git@github.com` → *"Hi Gunjan00001!"*.
  Switched the remote from HTTPS to SSH and pushed.
- **Failure:** none (no new key needed).
- **Fix:** remote set to `git@github.com:Gunjan00001/ASR-1-bit-qwen-R2T2-approach.git`;
  `git push -u origin stage-1` + `git push origin --tags`.
- **Evidence:** remote refs updated `a78334a..375de28`; tags `0.2.19`–`0.2.33`.

### 23:05 — `rules.md` + this log created

- **Did:** codified GPU policy (never wait; switch GPUs), ETA reporting,
  activity logging, git/SSH, cluster etiquette, environment gotchas.
- **Evidence:** `rules.md`, `docs/ACTIVITY_LOG.md` (this file).

### 23:10 — GPU-switch rule applied to the slow ternary run (rules R1)

- **Did:** the FFN ternary job (`10737`) was running at **15 steps/min**
  (4 s/step) sharing a GPU with another agent's load; the binary job on a free
  GPU ran at ~117 steps/min. Verified the step-2000 checkpoint
  (`data/checkpoints/qat_ternary_ffn/last.pt`), wrote
  `scripts/qat_ternary_ffn_resume.pbs` with an improved GPU picker (least
  utilization among GPUs with ≥25 GiB free), then killed the slow job.
- **Failure:** the batch resume (`10767`) queued behind the per-user PBS CPU
  quota (16 ncpus total; the user's other-agent jobs + our binary job filled
  it). **CPU quota, not GPU, was the blocker.**
- **Fix:** launched the resume **in the persistent session** on the
  least-utilized GPU (GPU 5) instead of waiting; cancelled the queued batch
  copy to avoid duplicate work. This is the intended behaviour under `rules.md`
  R1 — never idle-wait.
- **Evidence:** `artifacts/stage1/qat_ternary_ffn_resume_session.log`.

### 23:15 — Binary FFN regression at step 1000

- **Did:** monitoring the FFN-only binary run (`10764`).
- **Failure:** binary WER 97.0 @500 → **387.3 @1000** (runaway generation).
- **Cause (hypothesis):** binary sign churn / latent-weight drift destabilizes
  training; consistent with the memorization result (binary loss floor 2.8 vs
  ternary 0.0) and with binary being the harder target in the literature.
- **Follow-up:** by step 1500 it recovered to **100.6** — the regression was a
  transient phase, not a terminal collapse. Final numbers will decide.
- **Evidence:** `artifacts/stage1/qat_binary_ffn.log`.

### 23:25 — Session OOM: checkpoint load killed; moved to batch queue

- **Did:** after the session resume launched on GPU 5, it printed
  `replaced=84 trainable=264241236` and then died.
- **Failure:** process became a zombie with an empty log; `dmesg` showed
  `oom-kill ... pid=3986997 ... oom_memcg=.../10708.jiit-master` (session
  32 GB cgroup).
- **Cause:** loading the 0.6 B fp32 model + 1.06 GB shadow checkpoint + 3.3 GB
  utterance cache + page cache exceeds the interactive session's 32 GB cgroup
  (page cache is charged to the cgroup).
- **Fix:** resubmitted the resume as a **batch job** with 4 ncpus / 48 GB
  (`10771`, `scripts/qat_ternary_ffn_resume.pbs`), which started immediately and
  picks the least-utilized GPU. Rule recorded: heavy loads never run in the
  interactive session (rules.md R5).
- **Evidence:** `dmesg` OOM record; job `10771`; `qat_ternary_ffn_resume.log`.

### 23:50 — Resume LR-schedule bug found and fixed

- **Failure:** the resumed run's step-2500 eval gave ternary WER **42.8** vs
  **25.7** at step 2000 (the checkpoint it resumed from) — the resume made the
  model *worse*.
- **Cause:** `scheduler.step()` restarts from 0 in the new process, so the
  cosine schedule re-applied a **fresh high LR (~1.9e-4)** to a converged
  checkpoint instead of the decaying tail (~1.4e-5). The high LR disturbed the
  adapted shadows.
- **Fix:** on `--resume`, set `scheduler.last_epoch = start_step` so the
  schedule continues at the right position. Also added `--init-from <ckpt>` to
  load shadows for a **new phase** without advancing the step counter (needed
  for Stage 2 rollouts).
- **Evidence:** `qat_ternary_ffn_resume.log` (lr column), commit
  `fix(T1.4): advance LR schedule on --resume; add --init-from for new phases`.

### 23:55 — Stage 2a designed and queued: attention rollout + small anchor

- **Plan:** init from the FFN-only ternary checkpoint (step 2000, WER 25.7),
  switch policy to `decoder_attn_ffn` (attention quantized too), add a **small
  KL anchor (w=0.1, T=2) to a frozen pretrained fp teacher**, low lr 5e-5,
  2 000 steps, batch 2, 5 % fp warmup, eval every 500 with fp16 readout.
- **Rationale:** the flat w=0.5 anchor exploded (KL 11–15, WER 315); the FFN
  warm start plus a small anchor should let attention adapt without losing
  audio conditioning.
- **Launcher:** `scripts/qat_attn_rollout.pbs` (job `10776`, queued behind the
  per-user ncpus quota; starts when the binary/ternary runs end).
- **Evidence:** `artifacts/stage1/qat_attn_rollout.log` (once running).

### 00:30 — FFN-only final tables + shadow-drift pattern

- **Did:** collected final ablations for both FFN-only runs.
- **Result:** ternary FFN **32.8** (best checkpoint 25.7 @2000); binary FFN
  **97.9** (no recovery); fp16 readouts **226.5 / 223.9** — the shadows drift
  far from the pretrained basin in every long run at lr 2e-4, even when the
  quantized readout recovers.
- **Lesson:** STE + lr 2e-4 + 3 000 steps ⇒ fp readout destroyed; the
  *quantized* function is what recovers. An anchored objective is needed to
  preserve both (Stage 2a tests this).
- **Stage 2a running:** at step ~400, CE 1.4–2.0 (much lower than the
  attn+ffn runs' ~6 — the FFN init gives a strong start), KL 5.8–6.6 (attention
  quantization error vs the frozen teacher), loss ≈ 0.9·CE + 0.1·KL.
- **Evidence:** `artifacts/stage1/stage1_decoder_ffn_bw1.{0,58}_*.json`.

### 01:00 — Stage 2a step-500 result + GPU switch (rules R1 again)

- **Result @500:** ternary **52.8**, fp16 **132.6**. Adding attention
  quantization to the FFN-adapted model (25.7) costs ~27 WER points; 500 rollout
  steps have not yet recovered it.
- **Failure:** the job landed on GPU 6 which became contended (100 % util);
  training crawled at ~5 steps/min and the eval took many minutes.
- **Fix:** launched `scripts/qat_attn_rollout_resume.pbs` (job `10780`) which
  picked GPU 7 (12 % util); killed the slow original (`10776`). The `--resume`
  LR fix was verified in the log: `lr schedule advanced to step 500`.
  Rate after the switch: **55 steps/min** (11× faster).
- **Evidence:** `artifacts/stage1/qat_attn_rollout.log` (step-500 eval),
  `artifacts/stage1/qat_attn_rollout_resume.log` (GPU 7, resumed).

### 01:30 — Stage 2a recovering: 52.8 → 38.0

- **Result @1000 (GPU 7 resume):** ternary **38.0** (from 52.8 @500), fp16
  107.6 (from 132.6); CE 1.34, KL 4.08, loss 1.61 at step 1121.
- **Interpretation:** attention quantization cost ~27 points on the FFN-adapted
  checkpoint; the rollout + small anchor is recovering it at roughly the
  FFN-only rate. If the trend holds, expect ~20–25 @1500 and ~15–20 @2000 —
  **full decoder ternary quantization approaching the FFN-only level (25.7)**.
- **Rate after GPU switch:** 97.5 steps/min (ETA ~25–30 min to finish).
- **Evidence:** `artifacts/stage1/qat_attn_rollout_resume.log`.

### 02:20 — Stage 2a final: full attention+FFN ternary at 22.6 WER (best yet)

- **Did:** finished the rollout (step 2000) and ran the final ablations.
- **Result (50-utt test-clean):** pretrained 2.44; **fp16 97.6; binary 99.5;
  ternary 22.6** (CER 13.9). Trajectory: 52.8 @500 → 38.0 @1000 → 23.1 @1500 →
  **22.6 @2000** (plateau as LR→0).
- **Why it matters:** this quantizes **all 196 decoder projections** (attention
  + FFN) at ternary — the configuration that collapsed to ~100 WER with naive
  QAT. The recipe (FFN warm start → attention rollout with a frozen-pretrained
  teacher anchor w=0.1, T=2, lr 5e-5) recovers it to 22.6, better than the
  FFN-only final (32.8) and close to the FFN-only best (25.7).
- **Next:** warm-restart extension (`10785`, cosine cycle 2, lr 2.5e-5 over
  5000 steps) to push below 20; then the same recipe for binary.
- **Evidence:** `artifacts/stage1/stage1_decoder_attn_ffn_bw1.58_gNone_lr5e-05_adamw_delay_ls1_n4000.json`.

### 02:45 — Warm-restart extension + second GPU switch (ext → ext2)

- **Result of ext (GPU 6):** ternary **21.5 @2500** (from 22.6 @2000) — the
  second cosine cycle continues the descent.
- **Failure:** GPU 6 became contended (94 % util, ~33 steps/min).
- **Fix:** launched `ext2` (job `10788`) resuming from the ext step-2500
  checkpoint on GPU 2 (picker chose it) → 80 steps/min; killed the slow ext.
  First ext2 submission had a wrong resume path (sed no-op) and was killed and
  resubmitted; the fix is in commit `fix(pbs): ext2 resumes from the ext
  step-2500 checkpoint`.
- **Result @3000:** ternary **19.4** — below 20, still descending.
- **Evidence:** `artifacts/stage1/qat_attn_rollout_ext.log`,
  `artifacts/stage1/qat_attn_rollout_ext2.log`.

### 03:20 — Stage 2b (binary rollout) + 8k cache launched

- **Did:** submitted `scripts/qat_binary_rollout.pbs` (job `10789`):
  `--bit-width 1.0`, `--init-from` the ternary rollout checkpoint
  (`qat_attn_rollout_ext2/last.pt`, ~step 3500, ternary 17.5), same frozen
  anchor (w=0.1) + lr 5e-5, 2 000 steps. Rationale: binary FFN-only stalled
  (~98 WER), so binary starts from **ternary-adapted shadows** (progressive
  precision) instead of a binary warm start.
- **Also:** started an 8k-utterance `train.clean.100` cache build on the login
  node (`data/qat_clean_train.100_n8000.pt`, streaming) for longer future runs.
- **Note:** `10789` started before ext2 finished, so it initialized from ext2's
  step-3500 checkpoint (not the final 5000) — acceptable, the ternary model at
  17.5 is a strong start.
- **Evidence:** `artifacts/stage1/qat_binary_rollout.log`,
  `artifacts/stage1/build_qat_cache_8k.log`.

### 04:10 — Stage 2a complete: 16.2 WER full attention+FFN ternary (headline)

- **Did:** finished ext2 (step 5000) and collected the final ablations.
- **Result (50-utt test-clean):** pretrained 2.44; **fp16 85.8; binary 99.9;
  ternary 16.2** (best 14.4 @4500; CER 8.4). Full trajectory:
  52.8 → 38.0 → 23.1 → 22.6 → 21.5 → 19.4 → 17.5 → 17.7 → **14.4** → 16.2.
- **Why it matters:** **all 196 decoder projections quantized to ternary** —
  the exact configuration that collapsed to ~100 WER with naive QAT — now
  reaches 14–16 WER. Recipe: FFN warm start → attention rollout with a
  frozen-pretrained-teacher anchor (w=0.1, T=2) at lr 5e-5 → warm-restart
  cosine cycle 2.
- **Binary (Stage 2b, progressive precision from the ternary checkpoint):**
  51.5 @500 → 38.1 @1000, still descending; far better than binary FFN-only
  (best 91.9).
- **8k cache ready:** `data/qat_clean_train.100_n8000.pt` (6.6 GB) + copy on
  `/tmp/asr1bit` for longer runs.
- **Evidence:**
  `artifacts/stage1/stage1_decoder_attn_ffn_bw1.58_gNone_lr2.5e-05_adamw_delay_ls1_n4000.json`.

### 05:00 — Stage 2b complete: binary 17.5 (progressive precision works)

- **Did:** finished the binary rollout (2 000 steps) and collected the final
  ablations.
- **Result (50-utt test-clean):** pretrained 2.44; fp16 99.8; **binary 17.5**
  (CER 9.0); ternary readout 30.5. Trajectory: 51.5 → 38.1 → 25.4 → **17.5**.
- **Why it matters:** binary (1-bit) weights across all 196 decoder projections
  reach 17.5 WER by **starting from the ternary rollout checkpoint** and
  switching `bit_width=1.0` (progressive precision) — versus ~98 for binary
  FFN-only and ~100 for naive binary.
- **Status:** all training jobs done. Final headline table in the live-status
  section above. Next: longer runs on the 8k cache, co-training/stochastic
  precision, then streaming (320 ms) evaluation.
- **Evidence:**
  `artifacts/stage1/stage1_decoder_attn_ffn_bw1.0_gNone_lr5e-05_adamw_delay_ls1_n4000.json`.

### 05:30 — Documentation pass: status report + figures

- **Did:** wrote `scripts/make_plots.py` (reproducible figures from logs +
  ablation JSONs) and generated five figures in `docs/figures/`:
  - `wer_trajectories.png` — WER vs step for all main runs (log scale);
  - `final_wer.png` — final WER per configuration (ternary/binary readouts);
  - `fp16_drift.png` — fp16 readout degradation during QAT (shadow drift);
  - `quantizer_error.png` — quantizer audit (binary/ternary, per-tensor/group);
  - `roadmap.png` — stage-by-stage status.
- **Also:** new consolidated report `docs/PROJECT_STATUS.md` (WER report,
  recipe, failures table, reproduce commands, future plan); README status
  updated with the headline results and the trajectory figure; figure pointers
  added to `docs/stage1_results.md`; summary data written to
  `artifacts/stage1/wer_summary.json`.
- **No training was run in this pass** (documentation only, per request).
- **Evidence:** `docs/PROJECT_STATUS.md`, `docs/figures/*.png`,
  `artifacts/stage1/wer_summary.json`.

### 06:00 — Long runs launched on the 8k cache (ternary + binary)

- **Did:** launched two batch jobs continuing the best checkpoints on the
  8k-utterance cache (`/tmp/asr1bit/qat_clean_train.100_n8000.pt`, 6.6 GB) with
  a fresh long cosine cycle and a halved anchor:
  - `10795` **ternary long** (GPU 6): init-from ext2 step 5000 (16.2 / 14.4),
    8 000 steps, lr 2.5e-5, anchor w=0.05, train-n 8 000;
  - `10798` **binary long** (GPU 2): init-from binary rollout step 2000 (17.5),
    6 000 steps, lr 2e-5, anchor w=0.05, train-n 8 000.
- **Failure:** both first submissions raced the GPU picker and landed on GPU 6
  together.
- **Fix:** killed the binary job and resubmitted; the picker (now seeing GPU 6
  occupied) chose GPU 2 (rules.md R1 — switch, never share/wait).
- **Rates:** ternary ~90 steps/min, binary ~85 steps/min.
- **ETA:** ternary ~2 h; binary ~1 h 30 m.
- **Evidence:** `artifacts/stage1/qat_ternary_long.log`,
  `artifacts/stage1/qat_binary_long.log`.

### 07:00 — Long-run check-in + gentle ternary hedge

- **Ternary long @2000:** 24.8 → 24.5 — the fresh lr 2.5e-5 cosine cycle
  perturbed the 16.2 checkpoint and has not recovered after 2000 steps.
  fp16 readout improved (56.7 → 46.4), i.e. much less shadow drift than the
  earlier runs (85–100) — the lower lr/anchor helps the fp path.
- **Binary long @2000:** 23.3 → **19.9** — recovering from the 17.5 start.
- **Action (hedge):** launched `10799` **gentle ternary** (GPU 7): same ext2
  checkpoint, lr 1e-5, anchor back to 0.1, 8k cache, 4 000 steps, eval every
  500 — a parallel path that preserves/improves 16.2 without the aggressive
  perturbation. All best checkpoints remain on disk, so nothing is at risk.
- **Evidence:** eval lines in `qat_ternary_long.log`, `qat_binary_long.log`,
  `qat_ternary_gentle.log`.

### 08:20 — Long runs deliver new bests: ternary 11.4 (best 10.6), binary 12.9

- **Did:** collected the final tables for both long runs.
- **Result (50-utt test-clean; pretrained 2.44):**
  - ternary long (8k data, 8 000 steps): **11.4 final / 10.6 best**, CER 5.2,
    fp16 50.7;
  - binary long (8k data, 6 000 steps): **12.9 final**, CER 6.7, fp16 37.2.
- **Why it worked:** the fresh long cosine cycle first perturbs the checkpoint
  (ternary 16.2 → 24.5 by step 2000) but then descends past the previous best as
  the LR decays (24.5 → 19.4 → 15.9 → 14.9 → 12.2 → 10.6). Lower lr + anchor
  also cut the fp16 drift (37–65 vs 85–230 before).
- **Decision:** killed the gentle hedge (`10799`, ~17 WER) — the aggressive path
  won. Launched `10811` **binary long2** (continuation from 12.9, lr 1e-5,
  4 000 steps) on GPU 2.
- **Docs:** figures regenerated (`make_plots.py` now includes both long runs);
  `docs/PROJECT_STATUS.md`, README headline table and this log updated.
- **Evidence:** `stage1_decoder_attn_ffn_bw1.58_gNone_lr2.5e-05_..._n8000.json`,
  `stage1_decoder_attn_ffn_bw1.0_gNone_lr2e-05_..._n8000.json`,
  `docs/figures/wer_trajectories.png`.

### 09:05 — Binary long2: 11.1 WER (binary overtakes ternary); next continuations launched

- **Did:** collected the binary long2 final table and launched the next pair of
  continuations.
- **Result (50-utt test-clean):** pretrained 2.44; fp16 36.2; **binary 11.1**
  (CER 4.6); ternary readout 23.6. Trajectory: 14.3 → 15.8 → 14.1 → 12.5 →
  13.1 → 12.6 → 12.8 → **11.1**. Binary (11.1) has overtaken ternary
  (11.4 final / 10.6 best).
- **Launched:**
  - `10813` **ternary long2**: from the 11.4/10.6 checkpoint, lr 1e-5, anchor
    0.05, 4 000 steps (GPU 6);
  - `10815` **binary long3**: from the 11.1 checkpoint, lr 5e-6, anchor 0.05,
    3 000 steps (GPU 2).
- **Failure:** both jobs raced onto GPU 6 again; the binary job was killed and
  resubmitted → GPU 2 (rules R1).
- **Note:** the 12 h GPU session ended; a new session (`10812`) is active and
  `gpu-send` works with it.
- **Evidence:** `stage1_decoder_attn_ffn_bw1.0_gNone_lr1e-05_..._n8000.json`,
  `qat_ternary_long2.log`, `qat_binary_long3.log`.

### 04:45 — Streaming 320 ms evaluation added; first continuation evals

- **Did:** wrote `scripts/eval_streaming.py` + `scripts/eval_streaming.pbs` —
  evaluates the **streaming 320 ms** primary metric with the portable
  `ChunkedStreamer` on the same fixed 50-utt sample, for both the fp16 readout
  (alpha=0) and the quantized readout (alpha=1), reporting WER + RTF. Running
  as job `10816` (GPU 5) on the binary long2 (11.1) and ternary long (11.4)
  checkpoints. This gives the exact `+X` comparison against the Stage 0 fp16
  streaming baselines (clean 2.24 / other 4.61).
- **First continuation evals:**
  - ternary long2 (lr 1e-5): **11.1 @500** — no transient this time (the
    gentler lr avoids the perturbation);
  - binary long3 (lr 5e-6): 12.2 @500 — small fresh-cycle dip from 11.1,
    expected to recover.
- **Evidence:** `artifacts/stage1/eval_streaming.log`,
  `artifacts/stage1/qat_ternary_long2.log`, `qat_binary_long3.log`.

### 05:20 — Binary long3 final 10.7; ternary long2 at 9.6 (best yet)

- **Result (50-utt test-clean; pretrained 2.44):**
  - binary long3 (from 11.1, lr 5e-6, 3 000 steps): **10.7 final**, CER 4.6,
    fp16 readout 31.7 (drift keeps shrinking: 99.8 → 37.2 → 36.2 → 31.7);
  - ternary long2 (in progress): 10.4 @2000 → 9.7 @2500 → **9.6 @3000**.
- **New bests:** ternary **9.6** (was 10.6), binary **10.7** (was 11.1).
- **Streaming eval:** first two attempts picked a noisy/busy GPU; added
  3-sample utilization averaging and an `EVAL_GPU` override; pinned to GPU 7
  (`10821`).
- **Evidence:** `stage1_decoder_attn_ffn_bw1.0_gNone_lr5e-06_..._n8000.json`,
  `qat_ternary_long2.log`.

### 05:50 — Ternary long2 final 9.2 (best 9.1); round-4 continuations launched

- **Result (50-utt test-clean; pretrained 2.44):** ternary long2 (lr 1e-5,
  4 000 steps) **9.2 final / 9.1 best**, CER 3.7, fp16 readout 37.7. The
  trajectory: 11.1 → 14.9 → 11.5 → 10.4 → 9.7 → 9.1 → 9.2.
- **Best numbers now:** ternary **9.2** (from 10.6), binary **10.7** (from
  11.1). Overall progression: naive ~100 → 32.8 (FFN-only) → 16.2 → 11.4 →
  **9.2** ternary; binary 17.5 → 12.9 → 11.1 → **10.7**.
- **Launched round 4:** `10826` ternary long3 (from 9.2, lr 5e-6, 3 000 steps,
  GPU 6) and `10827` binary long4 (from 10.7, lr 2.5e-6, 3 000 steps, GPU 5);
  staggered submission avoided the GPU race this time.
- **Docs/figures:** `make_plots.py` now includes long2/long3; figures and
  `docs/PROJECT_STATUS.md` + README updated with the new bests.
- **Evidence:** `stage1_decoder_attn_ffn_bw1.58_gNone_lr1e-05_..._n8000.json`,
  `docs/figures/*.png`.

### 06:40 — Streaming path diagnosed; recipe plateaued; quantized streaming eval launched

- **Streaming diagnostic (job `10833`, pretrained model, portable streamer,
  320 ms, 10 utts):** **WER 3.59, CER 1.02, RTF 1.16** with correct
  transcriptions → the portable path is sound.
- **Key finding:** the first streaming attempt (fp16 readout of a QAT model)
  scored **WER 1860 / RTF 6.0** — the drifted fp16 readout causes runaway
  generation in streaming. **The fp16 readout of QAT shadows is not a valid
  streaming baseline**; the meaningful metric is the trained precision
  (ternary/binary).
- **Recipe plateau:** round-4 continuations (ternary long3 lr 5e-6, binary
  long4 lr 2.5e-6) hovered at 9.3–10.3 / 10.7–11.8 — no improvement past
  **9.2 / 10.7**. Killed; further rounds are not worth it on 8k data.
- **Launched `10835`:** streaming 320 ms on the same 50 utts for (1) pretrained
  fp16 baseline, (2) ternary long2 quantized (9.2), (3) binary long3 quantized
  (10.7) — the primary-metric comparison.
- **Next levers after this:** more data (28.5k full `train-clean-100` /
  `train.360`), the 1.7B model, or the remaining Stage 2 stack (co-training /
  stochastic precision).
- **Evidence:** `artifacts/stage1/streaming_pretrained.json`,
  `artifacts/stage1/eval_streaming.log` (WER 1860 line),
  `docs/figures/wer_trajectories.png`.

### 08:05 — Streaming collapse root-caused; no-prefix re-feed workaround; full cache built

- **Did:** per-chunk streaming traces (`scripts/diag_stream_generation.py`)
  comparing the pretrained model and the ternary long2 checkpoint on the same
  utterance.
- **Pretrained trace:** 11 chunks, EOS emitted every chunk; first chunk
  (0.32 s) → `language None<asr_text>`; final text correct.
- **Ternary trace:** first chunk → **32 tokens of "assistant\nassistant…"
  without EOS** (runaway); later chunks OK; at the **first
  prefix-continuation chunk** (2.24 s) it collapses into a token repetition
  loop ("'T'T'T…"). Root cause: **QAT never trained partial-prefix
  continuation** (the streaming algorithm's rolled-back self-prefix), and the
  near-empty first chunk triggers runaway.
- **Warm-up skip alone did not fix it** (the prefix-continuation failure
  remains).
- **Workaround that works:** disable the prefix continuation
  (`unfixed_token_num=9999`, no-prefix re-feed) — the ternary model then emits
  the correct transcript ("WORDS WAS IT THEIR COLORS"). Added
  `--unfixed-token-num` to `eval_streaming.py` and relaunched the 50-utt
  evaluation (`10844`) for pretrained + ternary + binary with the same variant.
- **Proper fix:** Stage 4 streaming SFT (partial-prefix samples) — recorded in
  the future plan.
- **Full cache built:** `data/qat_clean_train.100_full.pt` (28 539 utts,
  100.6 h, 23 GB) after fixing a transient certifi failure.
- **Evidence:** `artifacts/stage1/diag_stream_pretrained.json`,
  `diag_stream_ternary.json`, `diag_stream_ternary_min05.json`,
  `diag_stream_ternary_noprefix.json`.

### 10:45 — No-prefix streaming measured; prefix-continuation SFT trained

- **Streaming no-prefix variant (job `10844`, 50 utts, 320 ms):**
  pretrained **19.28** (truncation-degraded vs 2.33 official), ternary long2
  **33.58** (CER 30.0, RTF 4.07) — no collapse, but a +14.3 gap; the variant
  also hurts the baseline, so it is only a stopgap.
- **Prefix-continuation SFT (job `10850`, `--prefix-sft-fracs
  0.0,0.5,0.8,0.95`):** 7 929 prefix samples from 2 000 utts; offline ternary
  **11.0** (was 9.2 → 1.8-point cost for streaming capability).
- **Decisive test running (`10859`):** prefix-trained checkpoint with the
  **official prefix algorithm** (unfixed_token_num=5), 50 utts.
- **Full cache ready:** 28 539 utts / 100.6 h / 23 GB
  (`data/qat_clean_train.100_full.pt`).
- **Evidence:** `artifacts/stage1/streaming_pretrained_50.json`,
  `streaming_ternary_long2_quantized.json`, `qat_prefix_sft.log`.

### 13:10 — Decisive streaming test: prefix-SFT insufficient; no-prefix 128-token run launched

- **Decisive test (`10859`):** prefix-continuation SFT (1 500 steps) + official
  prefix algorithm → **WER 952** (repetition loops: "HAVE HAVE HAVE…",
  "AT AT AT…"). The short prefix phase reduced the collapse (1296 → 952) but
  did **not** fix it. Proper streaming SFT at scale is required (Stage 4).
- **No-prefix re-feed (32 tokens, `10844`) complete:** pretrained 19.28,
  ternary **33.58**, binary **33.26** (same variant; no collapse).
- **Launched `10860`:** no-prefix re-feed with a 128-token budget (avoids
  truncation) for pretrained + ternary + binary — the most defensible
  chunked-re-feed streaming numbers available with the current models.
- **Docs:** `docs/PROJECT_STATUS.md` gained a full streaming section (variants,
  root causes, conclusions) and the future plan now puts streaming robustness
  on the critical path (Stage 4 brought forward, full 28.5 k cache ready).
- **Evidence:** `artifacts/stage1/streaming_ternary_prefix_quantized.json`,
  `eval_streaming_prefix.log`, `docs/PROJECT_STATUS.md`.

### 15:40 — 128-token streaming baseline valid; scaled streaming SFT launched

- **No-prefix re-feed with a 128-token budget (`10860`):** pretrained baseline
  **2.54 WER** (vs 19.28 with 32 tokens) — the truncation problem is fixed, so
  this variant is a valid streaming evaluation. Ternary/binary conditions
  running.
- **Scaled streaming SFT (`10865`):** full 28.5 k cache, `--prefix-sft-fracs
  0.0,0.5,0.8,0.95` + **near-empty-audio samples** (`--silence-samples-sec
  0.5`, new option, empty target → EOS) to attack both collapse modes; 5 000
  steps, lr 1e-5, anchor 0.05, init from the 9.2 offline ternary. ETA ~1 h 20 m,
  after which the official-algorithm streaming test is re-run.
- **Code:** `build_examples` gained `silence_sec`; runner gained
  `--silence-samples-sec`.
- **Note:** the persistent `gpu-send` session has become unresponsive after the
  environment restarts; batch jobs and log files are the reliable interface.
- **Evidence:** `artifacts/stage1/eval_streaming_noprefix128.log`,
  `qat_stream_sft.log`.

### 17:05 — Valid streaming number: ternary 9.43 (pretrained 2.54, +6.9)

- **Result (`10860`, no-prefix re-feed, 128-token budget, 50 utts, 320 ms):**
  pretrained **2.54** (CER 0.79, RTF 3.37); ternary quantized **9.43**
  (CER 3.86, RTF 9.72). The ternary streaming number essentially matches its
  offline quality (9.2) — **+6.9 absolute** vs the same-variant baseline.
- **Interpretation:** with a sufficient token budget, the chunked re-feed
  evaluation is valid and our model streams at 9.4 WER. The cost is RTF
  (~9.7) because every chunk regenerates the full transcript; restoring the
  official prefix algorithm (normal RTF) is what the scaled streaming SFT
  targets.
- **Streaming SFT (`10865`)** is at step ~4 300/5 000 with offline recovering
  (20.2 → 15.5 → 14.5 → 12.2 as the LR decays). Its official-algorithm test
  runs next.
- **Docs:** streaming tables in `docs/PROJECT_STATUS.md` and this log updated.
- **Evidence:** `artifacts/stage1/streaming_ternary_noprefix128.json`,
  `streaming_pretrained_noprefix128.json`.

### 18:05 — Streaming SFT finished (step 5000); decisive test queued

- **Result:** `qat_stream_sft/last.pt` saved at step 5000; offline ternary
  recovered to ~12 as the cosine decayed (20.2 → 15.5 → 14.5 → 12.2 → ~12).
  The 39 706-example mix (4 prefix fractions + silence) cost ~3 offline points
  versus the 9.2 baseline.
- **Note:** the 128-token streaming job (`10860`) hit its 4 h walltime after
  the ternary condition; the binary 128-token number was not collected.
- **Decisive test queued (`10875`):** official prefix algorithm on the
  scaled-SFT checkpoint. If the model now continues prefixes without
  degenerating, streaming returns to normal RTF at single-digit WER; if it
  still collapses, the LSP emission policy (Stage 4 / T4.4) is the next lever.
- **Evidence:** `data/checkpoints/qat_stream_sft/last.pt`,
  `qat_stream_sft.log`.

### 20:10 — Official-algorithm tests: scaled SFT 203.5; repetition controls fail

- **Scaled streaming SFT + official prefix algorithm (`10875`):** WER **203.5**
  (CER 865). Inspection shows the transcripts are **largely correct** followed
  by repetition loops ("ILLILL…", "HIS HIS…", "STANDSTAND…") that never stop —
  the model completes the text but does not terminate.
- **Repetition controls (`10980`):** `repetition_penalty=1.3` → WER **438.2**
  (worse; penalizes legitimate repeats). `no_repeat_ngram_size=4` → WER
  **264.5** (also worse; new loop patterns). **Both inference controls fail** —
  the collapse is a training-side problem.
- **Conclusion:** the official prefix-continuation algorithm needs
  research-grade Stage 4 work (on-policy streaming SFT / LSP emission), not a
  simple SFT add-on or inference trick. The **chunked re-feed variant
  (ternary 9.43 / binary 10.28 vs pretrained 2.54) is the reported streaming
  metric** until then.
- **Evidence:** `artifacts/stage1/streaming_ternary_sft_quantized.json`,
  `streaming_sft_reppen13.json`, `streaming_sft_ngram4.json`,
  `eval_streaming_reppen.log`.

### 22:55 — Session close-out: final numbers and handoff

**Offline (50-utt test-clean, seed 0; fp16 pretrained 2.44; naive QAT ≈ 100):**

| Configuration | WER |
|---|---|
| Ternary, all 196 decoder projections (8k, long2) | **9.2** (best 9.1) |
| Binary, all 196 decoder projections (8k, long3) | **10.7** |
| Ternary + short prefix-SFT | 11.0 |
| Ternary + scaled streaming SFT | 11.7 |

**Streaming 320 ms (50 utts):**

| Variant | Pretrained | Ternary | Binary |
|---|---:|---:|---:|
| Official prefix algorithm | 2.33 | collapse (1296 → 952 → **203.5** after scaled SFT) | — |
| Official + repetition_penalty 1.3 | — | 438.2 | — |
| Official + no_repeat_ngram 4 | — | 264.5 | — |
| **Chunked re-feed, no-prefix, 128 tokens (reported)** | **2.54** | **9.43** | **10.28** |

**Handoff:** see `docs/HANDOFF.md` (start-here guide for the next agent),
`docs/GPU_ACCESS.md` (cluster access), and the work packages below.

**Next work packages (in order):**
1. **On-policy streaming SFT**: prefix-continuation samples from the model's
   *own* partial generations, blended ≥50 % standard samples. Re-test the
   official algorithm.
2. **LSP emission policy** (Stage 4/T4.4): stable-prefix commit + stop handling.
3. **Offline scale-up** on the full 28.5 k cache (built, 100.6 h).
4. **`test-other`** evaluation for the clean/other pair.
5. **fp16 shadow-drift** trust-region fix.
6. **Stage 3 scale-up to 1.7B**, then the full evaluation matrix + writeup.

**Assets left on disk:** caches (4 k / 8 k / 28.5 k utterances), 22 QAT
checkpoints (`data/checkpoints/`), all artifacts under `artifacts/stage1/`,
figures in `docs/figures/`. Everything is committed and pushed (`stage-1`,
tags through `0.2.39`).

---

### 22:20 — WP1 implemented: on-policy streaming SFT + LSP emission (code + tests)

- **Did:** implemented the two WP1 work packages end-to-end; suite 277 passed,
  1 skipped, `ruff check .` clean.
  - **On-policy streaming SFT** (`src/asr1bit/train/sft.py`, was a stub):
    `collect_streaming_samples` replays the *official* algorithm (full audio
    re-feed per chunk + `unfixed_token_num` rollback) with the current model and
    records the exact `(prompt, audio heard, own text)` state — batched so all
    utterances' chunk-*k* decodes run as one `generate` call (identical math,
    much less wall time). `gold_coverage` aligns the model's own text to the
    reference by unit-cost edit distance, so the training target is the
    **unfinished remainder** (`""` when the own text already covers the
    transcript — the EOS-stop sample the previous SFT never trained).
    Warm-up chunks 0/1 (empty prefix) train EOS, killing the
    `assistant\nassistant…` runaway. `interleave_with_ratio` blends the pool
    with standard offline samples (target share `--op-mix-frac`, default 0.5).
  - **LSP emission policy** (`src/asr1bit/qwen/stream.py`, Stage 4 / T4.4):
    `LSPState` + `update_lsp` / `emit_stable` / `flush_lsp`. Text is committed
    only once it has survived a revision (longest common prefix of consecutive
    updates), always at a word boundary, with `margin_words` held back so the
    model's rollback window stays revisable; committed text is **append-only by
    construction**. `rollback_prefix` was factored out for reuse by the rollout.
  - **Runner:** `--op-rollout*` flags (rollout size, batch, stride, round cap,
    pool cap, mix fraction, refresh interval, warm-up EOS);
    `eval_streaming.py --lsp-margin-words` reports revisions and the committed
    fraction of the transcript at 50 % / 80 % of the audio.
- **New tests:** `tests/test_train_sft.py` (rollout replay incl. rollback chain,
  alignment, target construction, mixing), `tests/test_trainer_rollout.py`
  (trainer glue with a fake wrapper), extended `tests/test_stream.py`
  (LSP invariants) and `tests/test_backends.py` (batched decode fn, LSP trace).
- **Launched:**
  - `11020` **WP1 phase 1** (`scripts/qat_stream_opd.pbs`, GPU 1): init from the
    9.2 ternary; rollout 700 utts (320 ms, ≤16 rounds, batch 6), 2 400
    on-policy + 2 400 standard samples, 3 000 steps, lr 1e-5 cosine, anchor 0.05;
    checkpoint `data/checkpoints/qat_stream_opd/`.
  - `11022` **WP2 ternary full** (`scripts/qat_ternary_full.pbs`, pinned GPU 2):
    continuation from the 9.2 checkpoint on the full 28 539-utterance cache,
    10 000 steps, lr 1e-5 cosine, anchor 0.05;
    checkpoint `data/checkpoints/qat_ternary_full/`.
- **Failure:** the first WP2 submission (`11021`) raced `11020` onto GPU 1 —
  every other GPU was ≥72 GB used by other users at submit time. Killed and
  resubmitted with `-v QAT_GPU=2` (rules.md R1: switch, never share).
- **Held back:** `scripts/qat_binary_full.pbs` (WP2 binary) — no GPU with
  ≥35 GB free; submit when a GPU frees.
- **ETA:** WP1 ≈ 2 h (rollout ≈ 45–60 min + 3 000 steps ≈ 50 min + evals);
  WP2 ternary ≈ 3 h.
- **Next:** decisive official-algorithm eval on the phase-1 checkpoint
  (`scripts/eval_streaming_opd.pbs`, LSP margin 3), then phase 2
  (`scripts/qat_stream_opd2.pbs`, fresh rollout from the phase-1 checkpoint).
- **Evidence:** `src/asr1bit/train/sft.py`, `src/asr1bit/qwen/stream.py`,
  `scripts/run_stage1_qat.py`, `scripts/eval_streaming.py`,
  `artifacts/stage1/qat_stream_opd.log`, `artifacts/stage1/qat_ternary_full.log`.

---

### 03:45 — WP1 phase 1 negative (920.55 WER); root cause found; switched to teacher distillation

- **Did:** collected the first WP1 result set end-to-end.
  - `11030` phase-1 training finished its full 3 000-step cosine (2 h 04 m,
    `qat_seconds` 7 367): offline ternary **18.33** (19.70 @1000 → 18.75 @2000 →
    18.33 @3000), fp16 readout 62.2, binary readout 100.0;
  - `11036` decisive eval (official algorithm, 32 tokens, 320 ms, LSP margin 3):
    **WER 920.55**, CER 1052.73, RTF 4.49, wall 1 632 s; LSP: 64 revisions,
    committed@50 % 13.8 %, committed@80 % 25.9 %;
  - `11039` per-chunk traces (3 runs + pretrained reference).
- **Result:** the warm-up fix worked (chunks 0–2 emit `''` + EOS instead of
  `assistant\nassistant…`), but the model degenerates at the first content chunk
  (≥1.28 s) into `language'S'S'S…` and never stops; WER is *worse* than the old
  static-SFT checkpoint (203.5). Warm-up skip: no change.
- **Root cause:** DAgger poisoning — the student's own rollout prefixes already
  contained repetition loops (`WORDS WAS IT'S'T'T'T…`), so training
  "(looping prefix) → reference remainder" taught the loops as plausible
  continuations. Fixing the cold start is necessary but not sufficient.
- **Fix / next:** **streaming-state distillation** (`--op-distill`): the frozen
  pretrained teacher (verified clean streaming policy, 2.33 official-algorithm
  WER) drives the rollout; the student clones its continuation at each state;
  non-EOS generations are dropped. Implementation: `build_transformers_batch_generate_fn`
  (EOS flags), `StreamingSample.continuation/eos`, `build_distill_examples`,
  `--op-distill` in the runner, new tests (suite 288 passed, ruff clean).
  Launched `11040` (init 9.2, 350-utt teacher rollout, 50/50 mix, 4 000 steps,
  lr 1e-5; checkpoint `data/checkpoints/qat_stream_distill/`). Killed the
  phase-2 DAgger job `11038` (its rollout would see even more degenerate
  prefixes). ETA: rollout ≈1 h, then ≈2.5 h of training.
- **Evidence:** `artifacts/stage1/streaming_opd_quantized.json`,
  `eval_streaming_opd.log`, `diag_opd_*.json`,
  `stage1_decoder_attn_ffn_bw1.58_gNone_lr1e-05_adamw_delay_ls1_n2400.json`,
  `qat_stream_distill.log`.

---

### 04:15 — Distillation v1: stop behaviour learned (RTF 1.76) but content collapses (448.31) — format conflict fixed, v2 launched

- **Did:** ran streaming-state distillation v1 (`11040`): the frozen pretrained
  teacher drove the 350-utt rollout (3 min — short generations), 2 400 teacher
  states + 2 400 standard samples, 4 000 steps at lr 1e-5 from the 9.2 base
  (offline trajectory 33.58 @1000 → 35.81 @2000 → 26.59 @3000 → 32.52 @4000);
  then the decisive eval (`11041`, tag `_distill`).
- **Result:** official algorithm **WER 448.31** (vs 920.55 for DAgger, 203.5 for
  the earlier static SFT), CER 432.32, **RTF 1.76** (was 4.49) — the model now
  *stops* (short EOS-terminated generations) but the content collapses: empty
  hypotheses and garbled continuations (e.g. "…THAT LIPHER O", "").
  LSP: 331 revisions, committed@50 % 7.1 %.
- **Root cause (found by inspection of the targets):** the teacher's outputs are
  wrapped in `language English<asr_text>…`, the standard samples teach plain
  transcripts — the 50/50 mix taught contradictory output formats, and the
  student ended up in a confused middle (offline oscillated 27–35 all run).
- **Fix:** `build_distill_examples(strip_metadata=True)` removes the wrapper from
  the target and (relative to `prompt_raw`) from the recorded prompt; the
  rollout keeps the teacher's natural prompts. Launched **v2** (`11042`):
  metadata stripped, lr 7e-6, 5 000 steps (`scripts/qat_stream_distill2.pbs`,
  checkpoint `data/checkpoints/qat_stream_distill2/`).
- **ETA:** v2 ≈ 3 h (teacher rollout ~3 min + 5 000 steps).
- **Evidence:** `artifacts/stage1/streaming_opd_distill_quantized.json`,
  `eval_streaming_opd_distill.log`, `qat_stream_distill.log`,
  `qat_stream_distill2.log`.

---

### 07:05 — WP2 breaks the offline plateau (ternary 7.5 on the full cache); v3 anneal recovers offline quality

- **WP2 result (the offline scale-up worked):** the ternary continuation on the
  full 28 539-utterance `train-clean-100` cache (jobs `11022` + resume `11046`,
  10 000 steps total, lr 1e-5 cosine, anchor 0.05) reached **ternary 7.52 @8000,
  7.63 final** — from the 9.2 the 8k-data recipe had plateaued at. Trajectory:
  11.97 @4000 → 9.22 @6000 → 7.52 @8000 → 7.63 @10000; fp16 readout also
  improved (31.7 → 21.7 → 24.5). **New best offline ternary checkpoint:**
  `data/checkpoints/qat_ternary_full/last.pt`.
- **WP1 anneal of v3 (`11059`):** offline **15.25 final (14.83 best)** from v3's
  25.6, with the streaming share held at 25 % (800 distillation states + 2 400
  standard samples). The streaming eval on the annealed checkpoint (`11062`) is
  running.
- **Chained:** `11046` finished its 10 000-step schedule; the full-cache ternary
  run is complete. The binary full-cache job (`scripts/qat_binary_full.pbs`)
  should now be launched **from the ternary full-cache checkpoint** (progressive
  precision) instead of `qat_binary_long3` — update `--init-from` before
  submitting.
- **Next:** (1) read the anneal streaming eval; (2) **v4**: run the full WP1
  pipeline (three-way distillation + anneal) from the new 7.5 ternary base — the
  residual streaming failures were offline-content-driven, so the better base
  should carry through; (3) binary (progressive precision from the full-cache
  ternary) and the Stage-3 1.7B scaffold.
- **Evidence:** `artifacts/stage1/qat_ternary_full.log`,
  `qat_ternary_full2.log`,
  `stage1_decoder_attn_ffn_bw1.58_gNone_lr1e-05_adamw_delay_ls1_n28539.json`,
  `qat_stream_anneal.log`.

---

### 09:30 — Scheduled check: binary floor found, v4 killed, LSP loop policy implemented and **confirmed (144.60)**

- **Binary (WP2):** the full-cache binary run finished — **binary 10.70**
  (19.60 @2000 → 13.88 @4000 → 11.02 @6000 → 10.70 @8000), *identical* to the
  old binary long3 despite the much better ternary base (readout 9.75 vs 22.35):
  a quantizer floor, not a base-quality issue.
- **v4 killed** after 2 000 steps: live 10-utt streaming WER 134.9 → 704.8 and
  ~17 steps/min (the GPU contention returned).
- **Loop policy (the win):** diagnosed the anneal-3 loops (8/50 hypotheses
  5–20× the reference length), implemented `truncate_trailing_loop` +
  `LSPState(truncate_loops=True)` with tests, and confirmed the real-path eval
  `11067`: **raw WER 278.18 → policy WER 144.60** on distill3 (RTF 1.387).
  Offline re-scores of the other checkpoints: distill2 349.89 → 154.56,
  v1 448.31 → 179.34, anneal3 358.90 → 244.70. Best official-algorithm number
  so far (previous best recipe: static SFT 203.5).
- **Also fixed:** the eval's loop-truncation counter now aggregates across
  utterances (it read only the last state and always reported 0).
- **Next:** the remaining gap in the 144.6 is ordinary content error (offline
  25.6) — a content phase that keeps ≥40 % streaming pressure (the 25 % anneal
  lost the stop policy), with the live `STREAM` evals as the guardrail.
- **Evidence:** `artifacts/stage1/streaming_opd_lspdistill3_quantized.json`,
  `eval_streaming_opd_lspdistill3.log`, `qat_binary_full.log`,
  `qat_stream_distill4.log`.
