# Kaggle setup — Stage 0 baseline & T0.0 spike

Qwen3-ASR streaming is **vLLM-only** and vLLM needs Linux + CUDA. The local dev
machine (Windows, AMD iGPU) cannot run it, so all Stage 0 streaming numbers are
produced here.

## Notebook settings

- **Accelerator:** GPU **T4 x2** (or T4).
  - Do **not** use P100. The vLLM nightly wheel is built for **cu129** and the
    Pascal P100 is `sm60`, below the `sm70` minimum. T4 is `sm75`.
- **Internet:** ON (needed for pip, Hugging Face, and the sample audio).
- Sessions cap at ~12 h and Kaggle gives ~30 GPU h/week, so the harness
  persists per-utterance results and can resume (see T0.3).

## Automated gate (recommended)

`scripts/run_kaggle_gate.sh` runs the whole Stage 0 gate with fail-fast guards.
After **Cell 1** (clone) and **Cell 2** (install), one command does the rest:

```bash
!bash scripts/run_kaggle_gate.sh --stage all
# or per stage:
!bash scripts/run_kaggle_gate.sh --stage setup
!bash scripts/run_kaggle_gate.sh --stage spike
!bash scripts/run_kaggle_gate.sh --stage probe
!bash scripts/run_kaggle_gate.sh --stage matrix
```

- **Preflight** asserts CUDA (`sm70+`; T4=`sm75`, P100 rejected) and records
  python/torch/vllm/qwen-asr versions to `artifacts/stage0/environment.txt`.
- **setup** installs nightly cu129 vLLM + extras + `pip install -e . --no-deps`;
  aborts on any failure.
- **spike** tees T0.0 to `artifacts/stage0/spike.log`. On non-zero it prints the
  fallback order (Colab Pro → Docker) and **stops — probe/matrix never run**.
- **probe** runs clean + other throughput probes and saves their JSON.
- **matrix** runs `run_stage0.py --backend vllm`, is re-runnable, and resumes
  from the per-utterance `.jsonl` files.
- At the end it writes `MANIFEST.txt` and commits the **small** artifacts
  (`reports.csv/json`, `throughput_*.json`, `spike.log`, `environment.txt`);
  the per-utterance `.jsonl` stay on Kaggle.

Useful flags: `--models <path>...` for cached weights (sets `HF_HUB_OFFLINE=1`;
`--no-offline` to override), `--dry-run` (prints the plan, passes `--dry-run` to
the matrix), `--push` (push the artifact commit; see auth below),
`--artifacts DIR`, `--budget-hours H`, `--probe-n N`.

### Artifacts are collected on exit (even on failure)

Collection runs from an `EXIT` trap (plus `INT`/`TERM`), so `spike.log`,
`environment.txt`, `MANIFEST.txt`, and any partial `reports.*` /
`throughput_*.json` are written **regardless of outcome** — a failed spike or a
killed session still leaves evidence. The exit code is preserved. The
per-utterance `.jsonl` are intentionally left untracked on Kaggle.

### Push auth (`GATE_PUSH=1`)

Store a **fine-grained GitHub token** (`contents: write` on this repo) as a
Kaggle Secret named `GITHUB_TOKEN`, then expose it before running:

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["GITHUB_TOKEN"] = UserSecretsClient().get_secret("GITHUB_TOKEN")
```

```bash
!GATE_PUSH=1 bash scripts/run_kaggle_gate.sh --stage all
```

`--push` commits the small artifacts locally with a fallback git identity, then
pushes `HEAD` to `GATE_BRANCH` (default `stage-0`) using the token. If the token
is missing, push is **skipped with a warning** and the run still succeeds; if the
push itself fails, it warns and leaves the commit local. The token is scrubbed
from any push output.

### Recoverability

Every stage logs `BEGIN stage=<name> artifacts=<dir>` at start. `--stage all`
runs preflight → setup → spike → probe → matrix and stops on the first failure.
`run_stage0.py` persists each utterance to `<run>.jsonl` and skips completed ids
on re-run, so a mid-matrix kill is resumed by simply re-running:

```bash
bash scripts/run_kaggle_gate.sh --stage matrix \
  --models /kaggle/input/qwen3-asr-weights/0.6B /kaggle/input/qwen3-asr-weights/1.7B
```

```bash
# cached weights + push artifacts for review:
!GATE_PUSH=1 bash scripts/run_kaggle_gate.sh --stage all \
  --models /kaggle/input/qwen3-asr-weights/0.6B /kaggle/input/qwen3-asr-weights/1.7B
# preview without touching the GPU:
!bash scripts/run_kaggle_gate.sh --stage all --dry-run
```

The manual cells below are the same steps, for reference / debugging.

## Cell 1 — get the code

```python
!git clone https://github.com/Gunjan00001/ASR-1-bit-qwen-R2T2-approach.git
%cd ASR-1-bit-qwen-R2T2-approach
!git checkout stage-0
```

## Cell 2 — install the vLLM backend (nightly, per Qwen docs)

```python
!pip install -q -U vllm --pre \
  --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
!pip install -q "vllm[audio]" qwen-asr jiwer soundfile soxr pandas pyyaml
!pip install -q -e . --no-deps
```

Kaggle images ship a preinstalled torch; if the nightly install leaves a broken
combination, `pip install -q --force-reinstall torch --index-url` a matching
cu129 build and re-run Cell 2. Record the resulting versions.

## Cell 2b — cache the models (do not re-download every session)

Weights are large (0.6B ~1.2 GB, 1.7B ~3.4 GB). Kaggle wipes `/kaggle/working`
between sessions, so download once into a directory and persist it as a Kaggle
Dataset (or attach an existing community dataset / Kaggle Model), then reuse it.

```python
# One-time: download to a persistent location outside /kaggle/working
import os
os.makedirs("/kaggle/working/qwen3-asr-weights", exist_ok=True)
!huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir /kaggle/working/qwen3-asr-weights/0.6B
!huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir /kaggle/working/qwen3-asr-weights/1.7B
# Then (Kaggle UI / API) create a Dataset from /kaggle/working/qwen3-asr-weights
# and attach it to this notebook at /kaggle/input/qwen3-asr-weights.
```

Then point the harness at the local directories (no hub access needed):

```python
!python scripts/run_stage0.py --backend vllm \
  --models /kaggle/input/qwen3-asr-weights/0.6B /kaggle/input/qwen3-asr-weights/1.7B
```

Set `HF_HOME=/kaggle/working/hf` for intra-session reuse of the LibriSpeech
parquet, and set `HF_HUB_OFFLINE=1` once the weights are attached.

## Cell 3 — T0.0 go/no-go spike

```python
!python scripts/spike_vllm_streaming.py --model Qwen/Qwen3-ASR-0.6B
```

Expected: `SPIKE: PASS`, with an environment block (torch / vllm / qwen-asr /
compute capability) and non-empty offline + streaming text at `chunk_size_sec`
2.0 and 0.32.

Copy the whole output block back — the environment lines are recorded with the
Stage 0 results, and the 0.32 run previews the low-latency chunk path.

## Cell 4 — throughput probe (before the matrix)

Measure whether full-split streaming fits a session and pick a subsample if not:

```python
!python scripts/throughput_probe.py --backend vllm --n 50 --config clean
!python scripts/throughput_probe.py --backend vllm --n 50 --config other
```

This prints mean wall time per utterance, extrapolated full-split hours, and the
coverage that fits `--budget-hours` (default 12). If coverage < 1, the script
recommends a labelled subsample size; record it for the report.

## Cell 5 — Stage 0 baseline matrix

After the spike passes, run the matrix (offline full clean/other; streaming
2.0 s on full test-clean; streaming 2.0 s + 320 ms on a fixed 500-utterance
test-other subsample):

```python
!python scripts/run_stage0.py --backend vllm
```

Preview the plan without loading models:

```python
!python scripts/run_stage0.py --backend vllm --dry-run
```

Outputs land in `outputs/stage0/`: a `<run>.jsonl` per-utterance file (reused on
re-run, so a killed session resumes) plus `reports.json` / `reports.csv` and a
printed summary. Copy the summary table and `environment` block back for the
gate.

If the vLLM backend is unusable but the transformers backend loads, the
portable path can be exercised with `--backend transformers` (streaming uses our
:class:`~asr1bit.qwen.stream.ChunkedStreamer`); the vLLM run remains the
reference for the published-number reproduction.

## If the spike fails

In order:

1. **Colab Pro** (A100 40 GB / L4 22 GB) — same install + spike commands.
2. **Official Docker image** `qwenllm/qwen3-asr` (see the Qwen3-ASR repo) on a
   CUDA host, mounting this repo and running the spike inside the container.
3. Only if both fail do we revisit whether the Stage 0 streaming gate is
   reachable, and raise it with the owner before proceeding.
