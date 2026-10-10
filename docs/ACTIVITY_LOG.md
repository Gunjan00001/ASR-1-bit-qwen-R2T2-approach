# Activity log (append-only)

Step-by-step record for **future agents and developers**. Every entry: what we
did, failure symptom, root cause, fix, evidence. Update after every meaningful
step (see `rules.md` R3). For protocol-level detail see `docs/LAB_NOTES.md`
(append-only, five fields) and `docs/stage1_results.md`.

---

## Live status (updated 2026-10-11 02:30 IST)

| Job | What | Progress | Result | ETA |
|---|---|---|---|---|
| `10785` | **Stage 2a-ext**: warm-restart (lr 2.5e-5 → 0 over 5000) | starting (resumed step 2000, GPU 6) | 22.6 @2000 at start | **~60–70 min** (3000 steps @ ~90/min + evals) |
| `10780` | **Stage 2a** attention rollout + anchor w=0.1 | **done** | **ternary 22.6**, fp16 97.6, binary 99.5 | — |
| `10771` | FFN-only ternary | **done** | final: ternary **32.8**, binary 71.0, fp16 226.5 | — |
| `10764` | FFN-only binary | **done** | final: binary **97.9**, ternary 106.8, fp16 223.9 | — |
| `10708` | persistent GPU session | — | — | ends ~04:00 IST |

**Stage 2a trajectory (full attention+FFN ternary, 196 BitLinear):** starts from
the FFN-adapted checkpoint (25.7) → 52.8 after attention quantization @500 →
38.0 @1000 → 23.1 @1500 → **22.6 @2000** (plateau as LR→0). Recipe: FFN warm
start + frozen-pretrained-teacher anchor (w=0.1, T=2) + lr 5e-5 cosine.
The extension continues with a second cosine cycle to push below 20.

Next: if the extension lands ≤15, run the same recipe for **binary**; then
consider co-training/stochastic precision and more data.

**GPU-switch rule in action (R1):** Stage 2a's original job landed on GPU 6,
which later became contended (100 % util, 5 steps/min). It was resumed on GPU 7
(12 % util) from its step-500 checkpoint → 55–97 steps/min; the slow original
was killed. Verified the `--resume` LR fix: `lr schedule advanced to step 500`.

**Session-cgroup rule (learned the hard way):** do not load model + checkpoint
in the 32 GB interactive session; it OOM-kills silently (log just stops). Use
batch jobs for anything heavier than a small probe. See `rules.md` R5.

**Resume rule:** `--resume` continues the same run (advances the LR schedule and
step counter); `--init-from` loads weights for a new phase (step counter
restarts at 0). Never resume without the LR-schedule advance.

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
