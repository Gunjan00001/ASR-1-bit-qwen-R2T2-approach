# Changelog / checkpoints

Versioning is **`MAJOR.MINOR.BUGS`**:

- **MAJOR** — an end-to-end milestone (e.g. `1.0.0` = 1.7B binary streaming
  pipeline works end-to-end).
- **MINOR** — a **completed stage** (its gate in `docs/PLANNING.md` has passed).
- **BUGS** — fixes and non-stage checkpoints.

Stage work lands on a per-stage branch; `main` only advances on a passed stage
gate. `stage-0` was merged to `main` at `0.2.0`.

## Checkpoint map

| Tag | Date | Branch | Checkpoint | Gate |
|---|---|---|---|---|
| `0.1.0` | 2026-09-20 | `main` | Repo scaffold + project plan | — |
| `0.1.1` | 2026-09-20 | `stage-0` | Stage 0 plan corrections (eval protocol, append-only scope, feasibility) | — |
| `0.1.2`–`0.1.17` | 2026-09-20 | `stage-0` | Stage 0 implementation + gate-runner hardening (17 checkpoints, summarized below) | — |
| `0.2.0` | 2026-09-20 | `stage-0` → `main` | **Stage 0 gate PASSED** — fp16 baselines reproduced; `X = +1.5` locked | **passed** |

## 0.2.0 — Stage 0 gate passed

T0.0 vLLM streaming spike **PASS**; published streaming WER reproduced within
tolerance on **Tesla T4, fp16** (torch 2.9.1+cu128, transformers 4.57.6,
vllm 0.14.0; Triton attention). Pinned protocol: greedy, `language=None`,
`max_new_tokens` 1024 offline / 32 streaming, `max_model_len=16384`,
`gpu_util=0.85`.

Results (`artifacts/stage0/gate_results.csv`; subsampled rows = fixed 500-utt
set, seed 0):

| Model | Mode | Chunk | Config | n | sub | WER | Published | Δ |
|---|---|---|---|---|---|---|---|---|
| Qwen3-ASR-1.7B | streaming | 2.0 s | clean | 2620 | full | **1.96** | 1.95 | **+0.01** |
| Qwen3-ASR-1.7B | streaming | 2.0 s | other | 500 | sub | **3.47** | 4.51 | −1.04 |
| Qwen3-ASR-1.7B | streaming | 320 ms | other | 500 | sub | **4.61** | — | — |
| Qwen3-ASR-1.7B | offline | — | clean | 500 | sub | **2.10** | 1.63 | +0.47 |
| Qwen3-ASR-1.7B | offline | — | other | 500 | sub | **3.47** | 3.38 | +0.09 |
| Qwen3-ASR-1.7B | streaming | 320 ms | clean | 500 | sub | **2.24** | — | — |

- **Primary metric** = streaming 320 ms, clean/other. fp16 baselines:
  clean **2.24**, other **4.61**.
- **`X = +1.5` absolute** locked (quantized model must stay within +1.5 WER of
  fp16 at the primary metric). Ternary fallback remains acceptable per the plan.
- Full-split offline and 0.6B confirmation are deferred (opportunistic).

## 0.1.2 – 0.1.17 — Stage 0 implementation & gate-runner hardening

Collapsed summary of 17 `stage-0` checkpoints (individual commits/tags retained).

**Implementation (T0.0–T0.4):**
- T0.0 vLLM go/no-go spike (`scripts/spike_vllm_streaming.py`) + Kaggle/Colab
  setup docs.
- T0.1 Python 3.12+ dev env, dependency split, `tests/conftest.py`.
- T0.2 LibriSpeech loader (`data/load.py`) + shared normalizer (`text.py`);
  hyphen-split normalizer (`librispeech_hyph`) is the default.
- T0.3 portable streaming engine (`qwen/stream.py`, official re-feed +
  prefix-rollback) and Qwen adapter (`qwen/backends.py`).
- T0.4 WER/CER, latency/RTF, matrix harness, Stage 0 runner, throughput probe.

**Gate-runner fixes found by running the gate (each a BUGS checkpoint):**
- `pip install -U pip` before the vLLM install; then install `qwen-asr[vllm]`
  from PyPI (nightly fallback without `--index-strategy`).
- vLLM `max_model_len=16384` / `gpu_memory_utilization=0.85` (KV cache did not
  fit the 65536 default on a T4).
- `VLLM_ATTENTION_BACKEND=TRITON_ATTN` (FlashInfer JIT build fails on sm75).
- `PYTHONPATH=$REPO_ROOT/src` so `asr1bit.*` imports without relying on the
  editable install.
- **`.gitignore` bug**: an unanchored `data/` had hidden `src/asr1bit/data/`
  from git (loader missing on the remote). Anchored to `/data/`.
- Offline-mode heuristic: only set `HF_HUB_OFFLINE=1` for local weight dirs.
- Matrix slicing: `--only-mode`, `--only-config`, `--only-chunk-ms`,
  `--offline-sample-n` (labelled subsamples so sessions fit).
- Load only the requested split parquet (the named builder materialized the
  ~104k-example `train.360` and failed); cast `Audio(decode=False)` to avoid
  `torchcodec`.
- Forward `max_new_tokens` to vLLM (offline 1024 / streaming 32 per mode;
  offline was truncated at 32).
- Skip artifact commit when `--artifacts` is outside the repo.
- EXIT/INT/TERM trap collects artifacts on any outcome; token-scoped `--push`
  warns and skips when `GITHUB_TOKEN` is absent.

## 0.2.1 – 0.2.10 — Stage 1 (BitLinear + QAT) work-in-progress

Tags `0.2.1`–`0.2.10` on branch `stage-1` (off `main` `0.2.0`). Stage 1 is **not**
complete; the gate is **not met** and `0.3.0` is intentionally untagged.

- T1.1 `quant.py` (binary/ternary + group-wise), T1.2 `bitlinear.py` (fp32
  shadow, STE, INT8 acts, progressive alpha), T1.3 `replace.py`
  (`apply_layer_policy`), T1.4 `train/qat.py` + `scripts/run_stage1_qat.py`.
- First QAT run **invalid** (post-QAT fp16 WER 256% — shadow corruption; see
  `docs/LAB_NOTES.md` and `docs/stage1_results.md`).
- Debug phase: STE correctness tests (overfit-one-batch at α=1), `grad_norms` /
  `check_finite_grads` diagnostics, and a stabilized runner (fp32 AdamW, α=1,
  no grad-ckpt/act-quant/autocast, QAT-only freeze). 171 tests pass.

## 0.1.1 — Stage 0 plan corrections

- Locked the evaluation protocol (greedy, `language=None`, `max_new_tokens`
  1024/32, record versions).
- Scoped the append-only invariant to Stage 4 / T4.4 (Qwen baseline is
  pseudo-streaming).
- Streaming feasibility policy (offline full; streaming full clean + fixed
  500-utt other subsample) and the T0.0 spike.
- Noted `configs/base.yaml` `unfixed_token_num: 1` is intentional.

## 0.1.0 — Repo scaffold + project plan

- Package layout (`src/asr1bit/{quant,bitlinear,replace,data,qwen,train,eval,deploy}`),
  `tests/`, `configs/`, `docs/PLANNING.md`, initial `README.md`, `pyproject.toml`.
