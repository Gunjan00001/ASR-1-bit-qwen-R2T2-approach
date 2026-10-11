# GPU access guide — JIIT cluster (`jiit-gpu01`)

**Audience:** agents and developers on this project. Read `rules.md` R1/R5/R6
alongside this guide. Last verified: **2026-10-11**.

---

## 1. Hardware & scheduler

| Item | Value |
|---|---|
| GPU node | `jiit-gpu01` (login node: `jiit-master`) |
| GPUs | **8 × NVIDIA A100-SXM4-80GB** (sm_80, NVLink) |
| Scheduler | PBS (`qsub`/`qdel`/`qstat` from the login node) |
| GPU queue | `gpu` — **per-user limit 16 ncpus / 5 running jobs**, walltime ≤ 24 h, `mem` ≤ 64 GB per job |
| Driver | 525.147.05 (CUDA 12.3) → use **`+cu126` torch builds only** |
| Filesystem | home is a shared network FS; node-local `/tmp` is fast |

## 2. How to run work (in order of reliability)

### A. Batch jobs — **recommended, use this by default**

All project launchers are PBS scripts:

```bash
cd "/home/jiit_992401210050/Desktop/ASR 1bit qwen"
qsub scripts/qat_ternary_long2.pbs        # example training run
qstat -u $USER                            # check status
tail -f artifacts/stage1/<job>.log        # follow the run log
```

Every launcher:
- does `unset CUDA_VISIBLE_DEVICES` (PBS sets it **empty**, which hides GPUs);
- **picks the least-utilized GPU** (3-sample average) with enough free memory;
- supports overrides: `qsub -v QAT_GPU=3 scripts/...` (training) or
  `qsub -v EVAL_GPU=7 scripts/...` (evaluation);
- falls back from `/tmp/asr1bit` staging to shared paths automatically;
- logs to `artifacts/stage1/<name>.log` and PBS output to `~/<name>.out`.

**Template** (copy any `scripts/qat_*.pbs`): `#PBS -l select=1:ncpus=4:ngpus=1:mem=48gb`,
`unset CUDA_VISIBLE_DEVICES`, `cd` to the repo, export
`PYTHONPATH=src`, `TOKENIZERS_PARALLELISM=false`, `HF_HOME=/tmp/asr1bit/hf`,
run with `/tmp/asr1bit/miniconda3/bin/python` if present else `~/miniconda3/bin/python`.

### B. Interactive session helper (`gpu-send` / `gpu-run`) — currently DEAD

The persistent session (`~/.gpu-session`) held a long-lived Python interpreter on
the GPU node. Its job is **no longer in the queue** (checked 2026-10-11), so
`gpu-send` will hang. Restart it if you want interactive access:

```bash
gpu-session reset && gpu-session start && gpu-session wait
gpu-send <<'PY'
import torch; print(torch.cuda.get_device_name(0))
PY
```

**Do not load models/checkpoints in the session** — its cgroup is 32 GB
(page cache included) and OOM-kills silently. Use it only for quick probes
(`nvidia-smi`, file checks, tiny tensors).

### C. One-shot runner

`gpu-run 'python scripts/...'` wraps `qsub -I` with a PTY for non-TTY shells.
Batch jobs are preferable for anything long.

## 3. Choosing a GPU

PBS does **not** pin GPUs; every job sees all 8. Rules (`rules.md` R1):

- Always select dynamically: least utilization among GPUs with enough free
  memory (≥25 GB for inference, ≥35 GB for training with a frozen teacher).
- **Never wait** for a specific GPU; if the current one is busy, switch. Kill
  and resubmit if a job lands on a contended GPU (the picker's utilization
  sample is instantaneous — the launchers average 3 samples to reduce noise).
- If two of your own jobs race onto the same GPU, `qdel` one and resubmit
  (it will see the other's memory and pick a different GPU).

Quick manual check on the GPU node (via a batch probe or `gpu-send`):

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

## 4. Node-local staging (`/tmp/asr1bit`)

Created to avoid shared-FS stalls:

| Path | Contents |
|---|---|
| `/tmp/asr1bit/miniconda3` | conda copy (torch 2.11.0+cu126) |
| `/tmp/asr1bit/models/Qwen3-ASR-0.6B` | model copy **with `tokenizer.json`** |
| `/tmp/asr1bit/qat_clean_train.100_n8000.pt` | 8k-utterance cache |
| `/tmp/asr1bit/qat_clean_train.100_full.pt` | 28.5k cache (if copied) |

`/tmp` is node-local: it survives job boundaries but **not node reboots**. The
launchers fall back to the shared repo paths when staging is absent.

## 5. Known gotchas

| Symptom | Cause | Fix |
|---|---|---|
| `torch.cuda.is_available() == False` in a job | PBS sets `CUDA_VISIBLE_DEVICES=""` | `unset CUDA_VISIBLE_DEVICES` |
| Process killed with no error, log stops | 32 GB session cgroup (page cache) | run in a batch job |
| Multi-minute "hang" at startup | slow Python tokenizer rebuild | `scripts/install_fast_tokenizer.py` into the model dir |
| `Could not find a suitable TLS CA certificate bundle` | transient (pip/certifi replacement) | retry the command |
| Job lands on a 99 %-busy GPU | noisy instantaneous utilization sample | kill + resubmit, or `-v QAT_GPU=` |
| `qsub: "-lresource=" cannot be used with "select"` | `mem` passed outside `select` | use `select=1:ncpus=…:ngpus=1:mem=…` |
| Job queued with "per-user limit reached on resource ncpus" | 16-ncpu quota | wait for a job to end or reduce `ncpus` |
| `gpu-send` hangs | session dead after environment restart | restart the session or use batch jobs |

## 6. Environment

- Python: `/home/jiit_992401210050/miniconda3/bin/python` (3.13.12,
  torch 2.11.0+cu126, transformers 4.57.6, qwen-asr 0.0.6, datasets 5.0.1,
  numpy 2.5.3). Node-local copy at `/tmp/asr1bit/miniconda3`.
- `HF_HOME=/tmp/asr1bit/hf` in launchers; the shared HF cache
  (`~/.cache/huggingface`) also works but is slower.
- Always run from the repo root with `PYTHONPATH=src`.
