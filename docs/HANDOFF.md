# Handoff — ASR-1-bit-Qwen (for the next agent/developer)

**Date:** 2026-10-11 · **Branch:** `stage-1` · **Owner:** Gunjan00001
**Read this first**, then `docs/PROJECT_STATUS.md` (status + graphs),
`docs/ACTIVITY_LOG.md` (every step, failure, fix), `rules.md` (operating rules),
`docs/GPU_ACCESS.md` (cluster access).

---

## 0. Latest session (2026-10-12 night) — WP1 SOLVED-ish: aligned targets (T4.1–T4.3) take the official algorithm to ~14 WER

- **Recommended WP1 recipe (new):** forced-aligner chunk targets (T4.1–T4.3) +
  the LSP loop policy + the offline anchor, from the best-content streaming
  checkpoint:
  `scripts/qat_stream_aligned.pbs` → `--aligned-targets 1 --aligned-n 800
  --aligned-stop-stride 2 --op-mix-frac 0.5 --lr 3e-6` (init `qat_stream_adapt5`,
  which itself came from `qat_stream_distill3` ← the three-way mix ← the 9.2
  ternary). Eval with `scripts/eval_streaming_opd.pbs` (`TRUNC=1`).
- **The numbers (20-utt official-algorithm guardrail, raw = no loop policy):**
  step 750 → 20.7, 1 000 → 16.5, **1 250 → 13.4**, 1 500 → 14.4; offline held at
  **14.62** ternary. Every earlier recipe's *raw* number was 215–532 (and needed
  the loop policy to reach 144–190). Full 50-utt clean/other evals are running
  (`11143`, `11145`).
- **Why it works:** the target at each 320 ms chunk is exactly *the words whose
  end time falls in that chunk* (Qwen3-ForcedAligner-0.6B), i.e. "emit what you
  have heard, then stop" — the supervision that matches the official algorithm,
  instead of "the whole remainder" (DAgger) or "the teacher's continuation"
  (distillation), which made the model infer the target length and produced the
  under-emission / loop failure modes.
- **Streaming frontier (50-utt, official algorithm, LSP loop policy on):**
  distill3 144.60 (offline 25.6) · content3 190.78 (15.8) · adapt5 175.00 (14.5)
  · **aligned ~14 raw (offline 14.6)** — pending the 50-utt confirmation.
- **Stage 3:** the 1.7B FFN warm start is banked at **ternary 22.03**. The
  full-attention rollout (1.41 B trainable) needs ~38–40 GB free and is racy on
  the shared node; `qat_17b_attn5.pbs` (running `11146`) drops the anchor to fit
  (watch its ternary WER — the 0.6B needed the anchor).
- **Cluster lessons (binding):** NEVER set `PYTORCH_CUDA_ALLOC_CONF` (driver 525
  → `CUDA driver error: invalid argument`); `gradient_checkpointing_enable()`
  breaks this frozen-only setup; `bitsandbytes` is now installed (0.50.2) so
  `--optimizer adam8bit` really engages — verify it in the log.
- **Next:** confirm the 50-utt numbers, then scale the aligned recipe
  (aligned-n 2 400 → 8 000, longer schedule, the 7.5 offline base) and mirror it
  to binary via progressive precision; the 1.7B once its rollout fits.

---

## 1. Where the project stands (30-second version)

| Layer | Result | Reference |
|---|---|---|
| **Offline WER**, ternary, all 196 decoder projections | **9.2** (best 9.1) | fp16 pretrained 2.44; naive QAT ≈ 100 |
| **Offline WER**, binary, all 196 decoder projections | **10.7** | same |
| **Streaming 320 ms** (valid chunked re-feed variant) | ternary **9.43**, binary **10.28** | same-variant pretrained 2.54 |
| Streaming 320 ms, official prefix algorithm | **collapses** (203.5 best) | WP1 fix implemented, phase-1 result pending (see §0) |
| LSP emission (append-only) | policy implemented + tested | Stage 4 / T4.4; WER-neutral, reports commit latency |

- Stage 0 gate passed (`0.2.0`); Stage 1/2 machinery is implemented, tested
  (277 tests, ruff clean), documented, and pushed.
- The quantization pipeline works: ternary/binary weights across the whole
  decoder recover to single-digit/low-teens WER via the recipe in §2.
- **The open problem is streaming robustness of the official algorithm** — a
  training-side (Stage 4) issue, addressed by WP1 on-policy streaming SFT +
  LSP emission (see §3, §0).

## 2. The recipe that works (copy-paste ready)

Training stages (all `qsub`-able from `scripts/`; the full provenance is in
`docs/ACTIVITY_LOG.md`):

1. **FFN warm start** — quantize `gate/up/down` only, train from pretrained:
   `scripts/qat_ternary_ffn.pbs` (3 000 steps, lr 2e-4, 4k cache) → ~25.7 WER.
2. **Attention rollout + frozen-teacher anchor** — quantize attention too
   (196 layers), `L = 0.9·CE + 0.1·KL(student ‖ frozen pretrained teacher)`,
   lr 5e-5: `scripts/qat_attn_rollout.pbs` → 22.6; warm restart
   (`scripts/qat_attn_rollout_ext*.pbs`) → 14.4.
3. **Long runs on 8k data** (fresh cosine cycles, anchor 0.05):
   `scripts/qat_ternary_long.pbs` (lr 2.5e-5, 8 000 steps) → **11.4**;
   `scripts/qat_ternary_long2.pbs` (lr 1e-5, 4 000 steps) → **9.2**.
   Binary: `qat_binary_long.pbs` (lr 2e-5, 6 000) → 12.9 →
   `qat_binary_long2.pbs` (lr 1e-5) → 11.1 → `qat_binary_long3.pbs`
   (lr 5e-6) → **10.7**.
4. **Progressive precision for binary** — switch `bit_width=1.0` starting from
   the ternary checkpoint (never train binary from fp; it stalls at ~98).

Key flags: `--policy decoder_attn_ffn --bit-width 1.58|1.0 --learnable-scale 1
--frozen-teacher 1 --distill-weight 0.05..0.1 --distill-temp 2.0
--lr-schedule cosine --alpha-mode delay --alpha-warmup-frac 0.02..0.05
--lazy-collate 1 --train-cache <cache> --train-n <n>`.
`--resume` continues a run (advances LR + step); `--init-from` starts a new
phase from a checkpoint (step resets to 0). **Never `--resume` without the LR
advance — the bug is fixed but the distinction matters.**

Streaming evaluation (valid variant): `scripts/eval_streaming.py` with
`--unfixed-token-num 9999 --max-new-tokens 128` (chunked re-feed; every chunk
regenerates the full transcript; RTF ~10). The official prefix algorithm
(`--unfixed-token-num 5`) is broken for QAT models.

## 3. What's left (prioritized work packages)

**WP1 — Streaming robustness (critical path; Stage 4). — IMPLEMENTED, RESULT PENDING**
The official prefix algorithm collapsed because the model (a) runaway-generated
on near-empty warm-up chunks and (b) fell into repetition loops when continuing
its own rolled-back text. Scaled streaming SFT (5 000 steps, static gold prefixes
+ silence samples) improved 1296 → 203.5 but did not fix it; inference controls
fail (rep-penalty 438, ngram=4 264).
- **On-policy streaming SFT: done** — `--op-rollout` replays the official
  algorithm with the current model (batched), training on the reference
  remainder of its own prefixes; warm-up chunks train EOS; mixed ≥50 % standard
  samples (`docs/LAB_NOTES.md` 2026-10-11 has the full design argument).
  Launchers: `scripts/qat_stream_opd.pbs` (phase 1, running as `11020`),
  `scripts/qat_stream_opd2.pbs` (phase 2, fresh rollout).
- **LSP emission policy (Stage 4/T4.4): done** — `LSPState`, `update_lsp`,
  `emit_stable`, `flush_lsp` in `src/asr1bit/qwen/stream.py`; append-only by
  construction with a word-level margin; reported via
  `scripts/eval_streaming.py --lsp-margin-words N` (revision count + committed
  fraction at 50 %/80 % of the audio).
- **Pending:** the decisive official-algorithm eval (`scripts/eval_streaming_opd.pbs`,
  target single-digit WER); if phase 1 still loops, raise `--op-mix-frac` or run
  in-job refreshes (`--op-refresh-every`) before touching the offline recipe.

**WP2 — Offline scale-up.**
The 8k-data recipe plateaued at 9.2 / 10.7. The full **28 539-utterance**
`train-clean-100` cache is built (`data/qat_clean_train.100_full.pt`, 100.6 h,
23 GB). Continuations with fresh cosine cycles are the cheapest test:
`scripts/qat_ternary_full.pbs` (running as `11022`, 10 000 steps, lr 1e-5) and
`scripts/qat_binary_full.pbs` (held: no GPU ≥35 GB free) — re-run stages 1–3 on
the full cache only if these win.

**WP3 — `test-other`.**
All numbers are `test-clean` (50-utt fixed subsample, seed 0). Run the recipe's
evaluation on `test-other` for the clean/other pair the plan requires.

**WP4 — fp16 shadow-drift fix.**
The fp16 readout drifts (31–50 WER) during QAT; lower lr + anchor tamed it but
did not eliminate it. A trust-region/anchor formulation is the plan's
"co-train" story.

**WP5 — Stage 3: scale to 1.7B.**
Memory stack (LoRA/partial QAT), checkpoint/resume (ready), run + eval.

**WP6 — Full evaluation matrix + writeup** (160/320/1000 ms, clean/other,
packed size, CPU RTF).

## 4. Assets map

| Asset | Location |
|---|---|
| Best ternary checkpoint (9.2) | `data/checkpoints/qat_ternary_long2/last.pt` |
| Best binary checkpoint (10.7) | `data/checkpoints/qat_binary_long3/last.pt` |
| Streaming-SFT checkpoint | `data/checkpoints/qat_stream_sft/last.pt` |
| 22 QAT checkpoints (full history) | `data/checkpoints/` |
| Caches (4k / 8k / 28.5k utts) | `data/qat_clean_train.100_{n4000,n8000,full}.pt` |
| Run logs (one per job) | `artifacts/stage1/qat_*.log`, `eval_*.log` |
| Result JSONs (ablations, streaming) | `artifacts/stage1/*.json` |
| Figures (regenerate: `scripts/make_plots.py`) | `docs/figures/` |
| Launchers (all stages) | `scripts/*.pbs` |
| Diagnostics | `scripts/diag_*.py`, `scripts/eval_streaming.py`, `scripts/inspect_checkpoint.py` |

**Node-local staging** (`/tmp/asr1bit` on `jiit-gpu01`): conda copy, model
copy, 4k/8k caches. It can disappear on node reboot — launchers fall back to
shared paths automatically.

## 5. GPU access (quick reference)

Full guide: **`docs/GPU_ACCESS.md`**. The short version:
- **Batch jobs are the reliable path**: `qsub scripts/*.pbs` from the login
  node. Per-user quota: 16 ncpus, 5 jobs. Long runs ≤ 24 h.
- The **persistent `gpu-send` session is currently dead** (its job is no longer
  in the queue). Restart with `gpu-session reset && gpu-session start &&
  gpu-session wait` if you want interactive `gpu-send`; otherwise ignore it.
- Every launcher **picks the least-utilized GPU itself** (3-sample average) and
  supports `QAT_GPU=<i>` / `EVAL_GPU=<i>` overrides via `qsub -v`. Never wait
  for a busy GPU — switch (`rules.md` R1).
- PBS sets `CUDA_VISIBLE_DEVICES=""` — every script does
  `unset CUDA_VISIBLE_DEVICES`; keep that.

## 6. Pitfalls learned (read before running)

1. **Interactive-session cgroup is 32 GB including page cache** — loading a
   model + checkpoint in it gets OOM-killed silently (log just stops). Use
   batch jobs for anything heavier than a small probe.
2. **Qwen3-ASR ships no `tokenizer.json`** — without
   `scripts/install_fast_tokenizer.py` every process pays a multi-minute slow
   tokenizer rebuild that looks like a hang.
3. **GPU picker noise**: a single `nvidia-smi` utilization sample can pick a
   99 %-busy GPU. The launchers average 3 samples; if a job still lands badly,
   `qdel` and resubmit (or pin with `-v QAT_GPU=`).
4. **`--resume` must advance the LR schedule** (fixed in `run_stage1_qat.py`);
   use `--init-from` for new phases. A resumed run that gets a fresh high LR
   will regress (25.7 → 42.8 was observed).
5. **Fresh cosine cycles perturb first, then win**: judge a continuation at the
   LR tail, not the first 2 000 steps (16.2 → 24.5 → 10.6 was one cycle).
6. **Transient infra failures**: DNS/certifi errors on the login node
   (retry), and `gpu-send` may hang after environment restarts — use logs and
   batch jobs as the interface.
7. **The fp16 readout of a QAT model is not a valid streaming baseline** — its
   drift causes runaway generation; evaluate at the trained precision.

## 7. Reproduce the headline numbers

```bash
# offline 9.2 (ternary) — inspect_checkpoint.py on the best checkpoint
qsub scripts/inspect_checkpoint.pbs     # or:
python scripts/inspect_checkpoint.py --checkpoint data/checkpoints/qat_ternary_long2/last.pt \
  --policy decoder_attn_ffn --bit-width 1.58 --n 50

# streaming 9.43 (ternary, chunked re-feed, 128-token budget)
python scripts/eval_streaming.py --model /tmp/asr1bit/models/Qwen3-ASR-0.6B \
  --checkpoint data/checkpoints/qat_ternary_long2/last.pt \
  --policy decoder_attn_ffn --bit-width 1.58 --chunk-ms 320 --n 50 \
  --max-new-tokens 128 --unfixed-token-num 9999 --conditions quantized

# figures
PYTHONPATH=src python scripts/make_plots.py
```

## 8. Session history (where to read more)

- `docs/PROJECT_STATUS.md` — consolidated status + all graphs + future plan.
- `docs/ACTIVITY_LOG.md` — chronological: every launch, failure, cause, fix.
- `docs/LAB_NOTES.md` — research notes (append-only, five fields).
- `docs/stage1_results.md` — detailed Stage 1 tables.
- `rules.md` — binding operating rules (GPU switching, ETAs, logging, git).
- Git: `stage-1` branch, tags `0.2.19`–`0.2.39+`, remote
  `git@github.com:Gunjan00001/ASR-1-bit-qwen-R2T2-approach.git` (SSH works).
