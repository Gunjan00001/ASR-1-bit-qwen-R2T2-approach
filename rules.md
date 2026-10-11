# Rules — ASR-1-bit-Qwen (binding for agents and developers)

Owner: **Gunjan00001**. These rules are binding for AI agents and human
contributors working on this repository. Update this file when a new rule is
agreed; keep the numbering stable.

---

## R1 — GPU selection: never wait, always switch

- Before launching any GPU work, query the node and pick the **least-used GPU**
  (most free memory; tie-break: lowest utilization):

  ```bash
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | sort -t, -k2 -n | head -1
  ```

- If another agent's/user's job is using the current GPU, **use a different GPU
  immediately — do not wait for it to finish**. All launchers must set
  `CUDA_VISIBLE_DEVICES` dynamically (see `scripts/qat_*.pbs`).
- PBS does **not** pin GPUs; `ngpus=1` is accounting only. `nvidia-smi` lists all
  8 A100s for every job.

## R2 — Always report ETAs

- Every status update about a running GPU process must include an **estimate of
  time remaining**, derived from the observed step rate and the remaining work
  (training steps + eval overhead).
- When needed, sample the log twice ~2 min apart to measure the current rate
  (`grep -o '"step": [0-9]*' <log> | tail -1`).

## R3 — Keep the activity log

- Update **`docs/ACTIVITY_LOG.md`** after every meaningful step: launch, result,
  failure, fix, decision.
- Every entry records: what we did, failure symptom, root cause, what fixed it,
  evidence/artifact paths.
- **Failures are never deleted.** This log is for future agents/developers.

## R4 — Git and GitHub

- Remote is **SSH**: `git@github.com:Gunjan00001/ASR-1-bit-qwen-R2T2-approach.git`
  (key `~/.ssh/id_ed25519_github`, configured in `~/.ssh/config`).
- Work lands on the per-stage branch (`stage-1` currently); `main` advances only
  when a stage gate passes.
- Tag checkpoints `MAJOR.MINOR.BUGS` (no leading `v`); push branch + tags
  directly after each commit batch (`git push origin <branch> --tags`).
- Run `pytest -q` and `ruff check .` before committing; keep the suite green
  (203 passed, 1 skipped as of 2026-10-10).

## R5 — Cluster resource etiquette (JIIT PBS)

- Per-user limits on the `gpu` queue: **16 ncpus**, **5 running jobs**, walltime
  ≤ 24 h per job. Check `qstat -u $USER` before submitting.
- Long runs go to **batch jobs** (`qsub scripts/*.pbs`), not the interactive
  session. The session is for quick probes and process management.
- The interactive `gpu-session` cgroup is **32 GB including page cache** — do not
  run memory-heavy loads in it (OOM kills are silent in the log). Batch jobs
  request `mem=48–64gb` for training.
- Build data caches on the **login node** (no cgroup limit) with streaming reads;
  never materialize full HF splits inside a job cgroup.
- Always `unset CUDA_VISIBLE_DEVICES` in PBS scripts (PBS sets it empty, which
  hides all GPUs).

## R6 — Environment gotchas (verified on `jiit-gpu01`)

Full guide: **`docs/GPU_ACCESS.md`**.

- Driver 525 / CUDA 12.3 → only `+cu126` torch builds work.
- Stage conda/model/data on node-local `/tmp/asr1bit` for speed; the shared
  BeeGFS filesystem stalls under heavy multi-job load.
- Qwen3-ASR checkpoints ship **no `tokenizer.json`**; run
  `scripts/install_fast_tokenizer.py` into the model directory or every process
  pays a multi-minute slow-tokenizer rebuild (looks like a hang).
- `datasets>=5` requires `numpy>=2`.
- GPU work from the login node is impossible (no driver). **Batch jobs
  (`qsub scripts/*.pbs`) are the reliable path.** The persistent `gpu-send`
  session is currently **dead** (its job is no longer in the queue) — restart
  with `gpu-session reset && gpu-session start && gpu-session wait` if you want
  interactive probes, and never load models in it (32 GB cgroup, silent OOM).
- GPU selection is dynamic in every launcher; override with
  `qsub -v QAT_GPU=<i>` (training) or `-v EVAL_GPU=<i>` (evaluation).

## R7 — Project conventions

- Locked interfaces and design decisions live in `docs/PLANNING.md`; append
  decisions to its "Decisions log" section.
- Results and failures go to `docs/LAB_NOTES.md` (append-only, five fields) and
  `docs/stage*_results.md`.
- One test module per `src/asr1bit` unit; TDD for quantizers/training helpers.
- `0.3.0` stays untagged until the Stage 1 gate is genuinely met.
