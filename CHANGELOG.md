# Changelog / checkpoints

Versioning is **`MAJOR.MINOR.BUGS`**:

- **MAJOR** — an end-to-end milestone (e.g. `1.0.0` = 1.7B binary streaming
  pipeline works end-to-end).
- **MINOR** — a **completed stage** (its gate in `docs/PLANNING.md` has passed).
- **BUGS** — fixes and non-stage checkpoints.

Tags mark checkpoints. `main` only advances on a passed stage gate; stage work is
committed on a per-stage branch and tagged there first.

## Checkpoint map

| Tag | Date | Branch | Checkpoint | Gate |
|---|---|---|---|---|
| `0.1.0` | 2026-09-20 | `main` | Repo scaffold + project plan | — |
| `0.1.1` | 2026-09-20 | `stage-0` | Stage 0 plan corrections (eval protocol, append-only scope, streaming feasibility) | — |
| `0.1.2` | 2026-09-20 | `stage-0` | Stage 0 implementation + local normalizer validation + automated Kaggle gate | **pending** |
| `0.1.3` | 2026-09-20 | `stage-0` | Gate runner fix: upgrade pip before vLLM nightly install | — |
| `0.1.4` | 2026-09-20 | `stage-0` | Gate runner fix: install `qwen-asr[vllm]` from PyPI; nightly fallback without `--index-strategy` | — |
| `0.1.5` | 2026-09-20 | `stage-0` | Cap vLLM `max_model_len` to fit the T4 KV cache | — |
| `0.1.6` | 2026-09-20 | `stage-0` | Force Triton attention backend (FlashInfer JIT fails on sm75) | — |
| `0.1.7` | 2026-09-20 | `stage-0` | Put `src` on `PYTHONPATH` so `asr1bit` imports without relying on the editable install | — |
| `0.1.8` | 2026-09-20 | `stage-0` | Fix `.gitignore` (`data/` → `/data/`); commit `src/asr1bit/data/` package | — |
| `0.1.9` | 2026-09-20 | `stage-0` | Slice the matrix (`--only-mode`) so gate sessions fit; probe uses `MODELS[0]` | — |
| `0.1.10` | 2026-09-20 | `stage-0` | Add `--only-config` matrix slice | — |
| `0.1.11` | 2026-09-20 | `stage-0` | Only set `HF_HUB_OFFLINE` for local weight dirs (not HF repo ids) | — |
| `0.1.12` | 2026-09-20 | `stage-0` | Skip artifact commit when `--artifacts` is outside the repo; record 1.7B result | — |
| `0.2.0` | — | `stage-0` → `main` | Stage 0 gate passed (published streaming WER reproduced, `X` locked) | **not yet** |

## Unreleased

### `0.2.0` — Stage 0 gate (planned)
- Run `scripts/run_kaggle_gate.sh --stage all` on Kaggle T4.
- Reproduce published streaming WER vs LibriSpeech `clean|other`
  (0.6B `2.54|6.27`, 1.7B `1.95|4.51`); lock `X`.
- Fill `docs/stage0_results.md`; merge `stage-0` → `main`; tag `0.2.0`.

## 0.1.12 — outside-repo artifacts + first gate number

- `commit_artifacts` now detects when `--artifacts` lies outside the git repo
  (e.g. `/kaggle/working` while the repo is at `/kaggle/temp/repo`) and skips the
  commit instead of failing at the end of an otherwise successful run.
- **Gate result:** Qwen3-ASR-1.7B streaming @ 2.0 s, full LibriSpeech test-clean
  (2620 utts): **WER 1.96** vs published **1.95** (+0.01), CER 0.61, RTF 0.199.
  Artifacts committed under `artifacts/stage0/`; see `docs/stage0_results.md`.

## 0.1.11 — offline-mode fix

- `run_kaggle_gate.sh` only sets `HF_HUB_OFFLINE=1` when every `--models` entry
  is an existing local directory. HF repo ids keep the hub online.
- Found by Kaggle run v8: vLLM raised `LocalEntryNotFoundError ... outgoing
  traffic has been disabled` because `--models Qwen/Qwen3-ASR-1.7B` was treated
  as a cached-weights signal.

## 0.1.10 — config slicing

- Added `--only-config {all,clean,other}` alongside `--only-mode` so a single
  run (e.g. 1.7B streaming clean) can be executed and downloaded per session.

## 0.1.9 — matrix slicing

- `run_stage0.py`/`run_kaggle_gate.sh` accept `--only-mode {all,offline,streaming}`
  so the 10-run matrix can be executed in slices that fit a session; each slice
  writes its own `reports.*` and per-utterance `.jsonl`.
- `stage_probe` now probes `MODELS[0]` instead of always loading 0.6B.
- Also fixed the `data` package to pass `ruff` (it had been skipped while ignored).
- Context: Kaggle run v7 (full matrix, 0.6B+1.7B) was killed with **no output**
  after ~34 min, so the matrix is now run in smaller, downloadable slices.

## 0.1.8 — data package was git-ignored

- `.gitignore` had an unanchored `data/`, which silently ignored
  `src/asr1bit/data/` — so `load.py` and the package were never committed and
  the Kaggle clone lacked them (`No module named 'asr1bit.data'`). Anchored the
  rule to `/data/` and committed the package.
- Found by Kaggle run v6 (spike PASS, probe import failure). Local tests passed
  because the files existed on disk but were untracked.

## 0.1.7 — import-path fix + first spike PASS

- `run_kaggle_gate.sh` prepends `$REPO_ROOT/src` to `PYTHONPATH` and logs an
  import check after the editable install, so `asr1bit.*` resolves even when the
  editable install does not expose subpackages.
- Found by Kaggle run v5: **T0.0 spike PASSED** (vLLM streaming works on T4),
  then the probe failed with `No module named 'asr1bit.data'`.
- First working T0.0 numbers (Qwen3-ASR-0.6B, 15 s sample, greedy):
  chunk 2.0 s RTF **0.083**; chunk 0.32 s RTF **0.313**.

## 0.1.6 — T4 attention backend fix

- `run_kaggle_gate.sh` exports `VLLM_ATTENTION_BACKEND=TRITON_ATTN` and
  `VLLM_USE_FLASHINFER_SAMPLER=0` by default. T4 (`sm75`) has no FlashAttention-2,
  and vLLM 0.14's FlashInfer JIT build fails on it
  (`ninja ... collect2: error: ld returned 1 exit status`), aborting engine init.
- Found by Kaggle run v4: KV cache now fit (7.21 GiB available, `max_model_len`
  16384), then FlashInfer `batch_prefill` JIT compile failed.

## 0.1.5 — vLLM KV-cache sizing fix

- `spike_vllm_streaming.py` and `backends.load_model` now pass
  `max_model_len=16384` and `gpu_memory_utilization=0.85` to vLLM. The model
  default (`max_seq_len=65536`) needs ~7 GiB of KV cache, exceeding the ~5 GiB
  available on a T4 at utilization 0.7.
- Found by Kaggle run v3: install and model resolution succeeded
  (torch 2.9.1+cu128, vllm 0.14.0, Tesla T4 sm75, fp16 fallback), then
  `ValueError: ... 7.0 GiB KV cache is needed ... available (5.02 GiB)`.

## 0.1.4 — Gate runner vLLM install fix

- `stage_setup` now installs `qwen-asr[vllm]` from PyPI (which pins a compatible
  vLLM) instead of requiring the nightly index with `--index-strategy`. If that
  fails it falls back to `pip install -U vllm --pre` with the cu129 nightly
  index, **without** `--index-strategy`.
- Reason: Kaggle's system pip ignores the `pip install -U pip` upgrade
  (`/usr/bin/python3 -m pip` still lacked `--index-strategy`) — see 0.1.3.

## 0.1.3 — Gate runner pip compatibility fix

- `scripts/run_kaggle_gate.sh` `stage_setup` now runs `pip install -U pip` before
  the vLLM nightly install. Kaggle's preinstalled pip predates
  `--index-strategy`, which the documented vLLM nightly command requires.
- Found by the first Kaggle run (version 1): preflight passed
  (`CUDA OK: Tesla T4 sm75`), setup stopped with
  `no such option: --index-strategy`, and the EXIT trap collected
  `environment.txt` + `MANIFEST.txt` — the fail-fast guard behaved as designed.

## 0.1.2 — Stage 0 implementation checkpoint (gate pending)

Code complete and locally validated; the GPU gate is pending.

- **T0.0** vLLM streaming go/no-go spike (`scripts/spike_vllm_streaming.py`) and
  Kaggle setup (`docs/kaggle_stage0.md`).
- **T0.1** Python 3.12 dev env, dependency split (`pyproject.toml`,
  `requirements-stage0.txt`, `.python-version`), `tests/conftest.py`.
- **T0.2** LibriSpeech loader (`src/asr1bit/data/load.py`) + shared normalizer
  (`src/asr1bit/text.py`): 16 kHz mono float32, manifests, deterministic fixed
  subsampling, soundfile byte decoding (no `torchcodec` dependency).
- **T0.3** portable streaming engine (`src/asr1bit/qwen/stream.py`, official
  re-feed + prefix-rollback semantics) and Qwen adapter
  (`src/asr1bit/qwen/backends.py`, official vLLM / portable transformers).
  No append-only assertion on the Qwen baseline (reserved for Stage 4).
- **T0.4** WER/CER (`eval/wer.py`), latency/RTF (`eval/latency.py`), matrix
  harness (`eval/harness.py`), Stage 0 runner (`scripts/run_stage0.py`) and
  throughput probe (`scripts/throughput_probe.py`).
- **Normalizer finding** (local CPU, transformers backend, offline):
  hyphen-splitting beats plain punctuation removal; `librispeech_hyph` is now the
  default. 1.7B within **+0.08** of published (1.71 vs 1.63, n=100); 0.6B 2.93
  vs 2.11 (n=100/300). See `docs/stage0_results.md`.
- **Gate automation** `scripts/run_kaggle_gate.sh`
  (`--stage setup|spike|probe|matrix|all`): preflight CUDA assert + version
  capture, vLLM nightly cu129 setup, spike fail-fast with fallback order,
  clean/other throughput probes, resumable matrix; EXIT-trap artifact collection
  (survives a failed spike or killed session); token-scoped `--push` that warns
  and skips when `GITHUB_TOKEN` is absent. Verified locally with shellcheck,
  arg/stage dispatch, `--dry-run` passthrough, and the `set -euo pipefail`
  spike rc/fallback path.
- 99 tests, `ruff` clean.

## 0.1.1 — Stage 0 plan corrections

- Locked the Stage 0 evaluation protocol (bf16, greedy, `language=None`,
  `max_new_tokens` 1024/32; record vLLM/torch/qwen-asr versions).
- Scoped the append-only invariant to Stage 4 / T4.4 (Qwen baseline is
  pseudo-streaming and may revise text).
- Added streaming feasibility policy (offline full splits; streaming full
  test-clean; fixed ~500-utterance test-other subsample shared across chunk
  sizes) and the T0.0 vLLM spike.
- Noted `configs/base.yaml` `unfixed_token_num: 1` is intentional.

## 0.1.0 — Repo scaffold + project plan

- Package layout (`src/asr1bit/{quant,bitlinear,replace,data,qwen,train,eval,deploy}`),
  `tests/`, `configs/`, `docs/PLANNING.md`, initial `README.md`, `pyproject.toml`.
