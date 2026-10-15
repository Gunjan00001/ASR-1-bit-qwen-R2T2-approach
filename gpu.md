# GPU Access on This Cluster

Everything here was discovered and **verified live** on this machine (login node
`jiit-master` → GPU node `jiit-gpu01`, PBS scheduler). Last updated 2026-10-10.

---

## 1. TL;DR — the shortest path

```bash
qsub -I -q gpu                 # from a real terminal only (needs a TTY)
unset CUDA_VISIBLE_DEVICES     # REQUIRED, otherwise all GPUs are hidden
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `True NVIDIA A100-SXM4-80GB`.

For automation / non-interactive shells, use the helpers in `~/bin`:

```bash
gpu-run 'python train.py'      # one-shot command on a GPU node
gpu-send <<'PY'                # run Python in the persistent GPU session
import torch; print(torch.cuda.get_device_name(0))
PY
```

---

## 2. GPU hardware configuration

| Item | Value |
| --- | --- |
| Node | `jiit-gpu01` |
| GPUs | **8 × NVIDIA A100-SXM4-80GB** |
| Compute capability | **sm_80** (Ampere), 108 SMs each |
| Memory per GPU | 80 GiB HBM2e (~85.0 GB reported) |
| Interconnect | **NVLink NV12**, fully connected (every GPU pair is `NV12`) |
| Driver | **525.147.05** |
| Max CUDA runtime | **12.3** (driver-reported) |
| VBIOS | 92.00.9E.00.02 |
| Power cap | 500 W per GPU |
| Persistence-M | On |
| ECC | Enabled (default) |
| MIG | **Disabled** |
| PCI bus IDs | `1B, 3D, 4E, 5F, 9D, BD, CD, DD` |
| NIC | `mlx5_0` (InfiniBand); NVLink is on-GPU, NIC is `SYS`/`PXB` to GPUs |

Per-GPU UUIDs:

```
0 GPU-017a78b9-6954-f51a-b8c8-4bf4be43856c
1 GPU-e7a819a5-4df2-221b-9db6-cfdd3da5b787
2 GPU-873bc349-7098-5209-a049-27e34ccdc02e
3 GPU-6530268d-ced9-923b-0cc7-0f7584e4075b
4 GPU-e3d8bcac-26ff-8705-9f01-7b3d9c05fd6b
5 GPU-01d359d4-45fd-9f18-d8a8-53c045a873fe
6 GPU-58a18fa2-9449-223b-68e1-cf6070b31cf9
7 GPU-b0782341-b053-6f08-ea39-5eccb71f8f4d
```

## 3. Node / system configuration

| Item | Value |
| --- | --- |
| CPU | 2 × Intel Xeon Platinum 8468, 48 cores/socket → 96 cores, 1 thread/core |
| RAM | ~2014 GB (~2 TB) on the node |
| Home filesystem | **network filesystem, shared between login and compute nodes** (see gotcha 3) |

## 4. PBS queue configuration

| Queue | Values |
| --- | --- |
| `gpu` | `resources_max.mem = 64gb`, `resources_max.ncpus = 16`, default `select = 1:ncpus=1:ngpus=1`, default `ngpus = 1`, default `mem = 32gb`, default `walltime = 24:00:00`, `max_run = 5` jobs per user |
| `workq` | CPU-only queue |

Notes:
- A job gets a **32 GB memory cgroup** by default (`-l mem=` up to `64gb`).
- You can request more CPUs/GPUs: `qsub -l select=1:ncpus=16:ngpus=8 -l mem=64gb`.

---

## 5. The three gotchas (this is what wastes everyone's time)

### Gotcha 1 — PBS hides the GPUs by setting `CUDA_VISIBLE_DEVICES` to an **empty string**
It is *set and empty*, not unset. An empty value means "no devices visible", so
`torch.cuda.is_available()` is `False` even though `nvidia-smi` lists 8 A100s.

```bash
unset CUDA_VISIBLE_DEVICES          # or: export CUDA_VISIBLE_DEVICES=0
```
`PBS_GPUFILE` is also empty, so PBS does **not** pin GPUs to your job.

### Gotcha 2 — the driver is CUDA 12.3, so CUDA-13 PyTorch builds fail
`~/miniconda3` shipped with `torch 2.11.0+cu130`, which errors:
> `CUDA initialization: The NVIDIA driver on your system is too old (found version 12030)`

Fix already applied: `~/miniconda3` now has **`torch 2.11.0+cu126`**
(+ `torchvision 0.26.0+cu126`, `torchaudio 2.11.0+cu126`). CUDA 12.x works with
the 525 driver via minor-version compatibility.

### Gotcha 3 — a FIFO cannot bridge the login node and GPU node
Home is a **network filesystem**. A named pipe (FIFO) created there has no shared
kernel object across nodes, so a writer on `jiit-master` blocks forever waiting
for a reader that lives on `jiit-gpu01`. Use a **file-based queue** instead
(`gpu-send` does exactly this).

### Bonus — all 8 GPUs are visible, and the node is busy
Because PBS doesn't pin `CUDA_VISIBLE_DEVICES`, any job can see all 8 A100s.
`nvidia-smi` frequently shows several at ~100% and 30–66 GB used by other users.
Pick an idle GPU before heavy work: `torch.cuda.set_device(<id>)`.

---

## 6. Ways to access the GPUs

### A. Interactive shell (normal workflow)
```bash
qsub -I -q gpu
unset CUDA_VISIBLE_DEVICES
nvidia-smi
exit
```
Requires a real terminal. `qsub -I` from a non-TTY errors with
`standard input and output must be a terminal for interactive job submission`.

### B. Batch job (recommended for training)
```bash
cat > job.pbs <<'EOF'
#!/bin/bash
#PBS -N myjob
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o /home/jiit_992401210050/Desktop/Humanoid/myjob.out

unset CUDA_VISIBLE_DEVICES
cd /home/jiit_992401210050/Desktop/Humanoid
python train.py
EOF
qsub job.pbs
qstat -u $USER
```

### C. PTY wrapper (for scripts/agents that have no TTY)
```bash
printf '%s\n' 'hostname' 'unset CUDA_VISIBLE_DEVICES' 'nvidia-smi -L' 'exit' \
| script -qec "qsub -I -q gpu -l select=1:ncpus=1:ngpus=1 -l walltime=00:05:00" /dev/null
```
`script` allocates a pseudo-TTY, which is what makes `qsub -I` accept the job.

### D. Jupyter notebook
```bash
qsub -I -q gpu
./notebook.sh          # ~/notebook.sh; opens Jupyter bound to the node
```
`~/notebook.sh` was recreated and contains `unset CUDA_VISIBLE_DEVICES` so kernels
see the GPU.

### E. Persistent GPU session (best for interactive development / agents)
A 12 h GPU job hosts a long-lived Python interpreter; code is sent via files.
State (imports, tensors in VRAM) persists across calls.
```bash
gpu-session start && gpu-session wait    # start/dequeue
gpu-send <<'PY'
import torch
x = torch.rand(8192, 8192, device="cuda")   # stays resident
PY
gpu-send <<'PY'
print(x.sum())                              # x still there
PY
gpu-session status | stop | reset
```

### F. One-shot runner
```bash
gpu-run 'python eval.py --ckpt best.pt'
gpu-run -t 08:00:00 -n 8 -g 1 -f train.sh
```

### G. Run the agent itself on a GPU node (fully native)
```bash
qsub -I -q gpu
module load opencode/2.0.18
unset CUDA_VISIBLE_DEVICES
opencode
```
Then the whole agent process has direct GPU access (no wrappers).

---

## 7. Helper tools installed for this

| Path | Purpose |
| --- | --- |
| `~/bin/gpu-run` | one-shot command on a GPU node (PTY + `qsub -I`) |
| `~/bin/gpu-session` | manage the persistent GPU job (`start/wait/status/stop/reset/logs`) |
| `~/bin/gpu-send` | run a Python block in the persistent GPU interpreter |
| `~/.gpu-session/` | session state: `server.py`, `start.pbs`, `out`, `ready`, `jobid` |
| `~/notebook.sh` | cluster-standard Jupyter launcher (+ GPU fix) |

---

## 8. Verified evidence

```
torch 2.11.0+cu126  | cuda build 12.6
cuda available: True
device count: 8
device 0: NVIDIA A100-SXM4-80GB
GPU matmul on persisted tensor: (8192, 8192) sum=1.3743e+11
```
Persistent session confirmed on `jiit-gpu01`; `nvidia-smi` lists 8 × A100-SXM4-80GB.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `qsub -I` → "must be a terminal" | no TTY | use batch, or wrap with `script -qec` |
| `torch.cuda.is_available()` False, but `nvidia-smi` works | empty `CUDA_VISIBLE_DEVICES` | `unset CUDA_VISIBLE_DEVICES` |
| "driver ... too old (found version 12030)" | CUDA-13 build on 525 driver | install `+cu126` (see Gotcha 2) |
| `gpu-send` blocks forever writing a FIFO | network-FS FIFO across nodes | use file queue (`gpu-send`) |
| job OOM-killed | 32 GB default cgroup | raise `-l mem=` (max `64gb`) |
| `Cgroup mem limit exceeded:` printed at job end | benign epilogue artifact observed on `jiit-gpu01` | ignore |
