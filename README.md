# ASR-1-bit-Qwen

English streaming ASR with near-binary weight quantization (`{-1,+1}` target,
ternary fallback), built **only** from Qwen models and targeting R2T2-class
published streaming quality on CPU/edge.

- **Plan of record:** [`docs/PLANNING.md`](docs/PLANNING.md)
- **Base models (Apache-2.0):**
  - `Qwen/Qwen3-ASR-0.6B`
  - `Qwen/Qwen3-ASR-1.7B`
  - `Qwen/Qwen3-ForcedAligner-0.6B`
- **Scope:** English only, research use, no distribution.
- **No R2T2 weights/code are used** — R2T2 is referenced as a benchmark target only.

## Status

**Stage 0 gate PASSED (`0.2.0`)** — fp16 baselines reproduced on T4; primary
metric fp16 anchors locked (streaming 320 ms clean `2.24` / other `4.61`) and
`X = +1.5` absolute. See [`docs/stage0_results.md`](docs/stage0_results.md).
Next: Stage 1 (BitLinear + QAT on 0.6B).

## Development

Local dev/unit-test environment is Python **3.12** (pinned in `.python-version`),
created with [`uv`](https://docs.astral.sh/uv/):

```bash
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"   # Windows
# uv pip install --python .venv/bin/python -e ".[dev]"          # Linux/macOS
```

Run the suite and linter:

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
```

The Kaggle/Colab runtime (Linux + CUDA, includes `qwen-asr[vllm]`) is described
in `requirements-stage0.txt`. Qwen3-ASR streaming is **vLLM-only**, so streaming
baselines run on GPU hosts, not on the local dev machine.

## Layout

```
src/asr1bit/   # package (quant, bitlinear, replace, data, qwen, train, eval, deploy)
tests/         # unit tests
configs/       # run configurations
docs/          # planning and design docs
```

## Versioning

Tags use `MAJOR.MINOR.BUGS`: **MAJOR** = end-to-end milestone, **MINOR** = a
completed stage (gate passed), **BUGS** = fixes and non-stage checkpoints.
Stage work lands on a per-stage branch (e.g. `stage-0`) and `main` advances only
when a stage gate passes. Checkpoints are documented in
[`CHANGELOG.md`](CHANGELOG.md).
