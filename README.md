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

**Stage 1/2 (in progress, branch `stage-1`)** — BitLinear + QAT machinery is
implemented and tested (203 tests); loss controllability proven (ternary QAT
memorizes exactly). A working recovery recipe was found and scaled up on the
A100 cluster:

| Configuration (50-utt `test-clean`) | WER |
|---|---:|
| fp16 pretrained reference | **2.44** |
| Ternary, all 196 decoder projections (8k data, offline) | **9.2** (best 9.1) |
| Binary, all 196 decoder projections (8k data, offline) | **10.7** |
| Streaming 320 ms, chunked re-feed (ternary / binary) | **9.43 / 10.28** |
| Naive QAT (for comparison) | ~100 |

![WER trajectories](docs/figures/wer_trajectories.png)

Full report, graphs and the future plan: **[`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md)**.
**Next agent: start with [`docs/HANDOFF.md`](docs/HANDOFF.md)** (state, recipe,
what's left, pitfalls) and [`docs/GPU_ACCESS.md`](docs/GPU_ACCESS.md) (cluster
access). Operating rules: [`rules.md`](rules.md); step-by-step work log:
[`docs/ACTIVITY_LOG.md`](docs/ACTIVITY_LOG.md).

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
docs/          # planning, design, status reports, figures
scripts/       # run/launch/plot tooling (incl. PBS launchers)
rules.md       # operating rules for agents/developers
```

## Versioning

Tags use `MAJOR.MINOR.BUGS`: **MAJOR** = end-to-end milestone, **MINOR** = a
completed stage (gate passed), **BUGS** = fixes and non-stage checkpoints.
Stage work lands on a per-stage branch (e.g. `stage-0`) and `main` advances only
when a stage gate passes. Checkpoints are documented in
[`CHANGELOG.md`](CHANGELOG.md).
