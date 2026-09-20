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

## Cell 3 — T0.0 go/no-go spike

```python
!python scripts/spike_vllm_streaming.py --model Qwen/Qwen3-ASR-0.6B
```

Expected: `SPIKE: PASS`, with an environment block (torch / vllm / qwen-asr /
compute capability) and non-empty offline + streaming text at `chunk_size_sec`
2.0 and 0.32.

Copy the whole output block back — the environment lines are recorded with the
Stage 0 results, and the 0.32 run previews the low-latency chunk path.

## Cell 4 — Stage 0 baseline matrix

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
