"""Run the Stage 0 baseline matrix (CUDA host, e.g. Kaggle T4).

Matrix (see ``docs/PLANNING.md`` §9 and ``docs/kaggle_stage0.md``):
  - offline, full LibriSpeech clean + other
  - streaming @ 2.0 s, full test-clean
  - streaming @ 2.0 s and each ``--extra-chunk-ms``, fixed subsample of test-other

Per-utterance results are written next to each run so a session can resume.

Usage::

    python scripts/run_stage0.py --backend vllm --dry-run
    python scripts/run_stage0.py --backend vllm
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any

from asr1bit.eval.harness import QwenEngine, filter_runs, run_baseline, stage0_runs
from asr1bit.qwen.backends import environment_report

DEFAULT_MODELS = ["Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-1.7B"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--backend", default="vllm", choices=["vllm", "transformers"])
    parser.add_argument("--subsample-n", type=int, default=500)
    parser.add_argument("--subsample-seed", type=int, default=0)
    parser.add_argument("--baseline-chunk-ms", type=int, default=2000)
    parser.add_argument("--extra-chunk-ms", type=int, nargs="*", default=[320])
    parser.add_argument("--step-ms", type=int, default=500)
    parser.add_argument("--results-dir", default="outputs/stage0")
    parser.add_argument("--language", default=None, help="default None = official protocol")
    parser.add_argument(
        "--only-mode",
        choices=["all", "offline", "streaming"],
        default="all",
        help="Run only this mode (slice the matrix so sessions fit).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_id(run: dict[str, Any]) -> str:
    model = str(run["model_id"]).split("/")[-1]
    suffix = "_sub" if run["subsampled"] else ""
    return f"{model}_{run['mode']}_{run['config']}_{run['chunk_ms']}ms{suffix}"


def _free_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001, S110 - best effort
        pass


def _write_summary(out_dir: Path, reports: list) -> None:
    rows = [report.to_dict() for report in reports]
    (out_dir / "reports.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        with (out_dir / "reports.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow({k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in row.items()})


def main() -> int:
    args = parse_args()
    runs = stage0_runs(
        args.models,
        subsample_n=args.subsample_n,
        subsample_seed=args.subsample_seed,
        baseline_chunk_ms=args.baseline_chunk_ms,
        extra_chunk_ms=args.extra_chunk_ms,
    )
    if args.only_mode != "all":
        runs = filter_runs(runs, mode=args.only_mode)
        print(f"filtered to mode={args.only_mode}: {len(runs)} runs")

    print("=== environment ===")
    print(json.dumps(environment_report(), indent=2))

    if args.dry_run:
        print("=== plan ===")
        for run in runs:
            print(" ", run_id(run), run)
        return 0

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports: list = []

    for model_id in args.models:
        engine = QwenEngine(model_id, backend=args.backend, language=args.language)
        for run in [r for r in runs if r["model_id"] == model_id]:
            identifier = run_id(run)
            print(f"=== {identifier} ===", flush=True)
            report = run_baseline(
                run["model_id"],
                run["mode"],
                run["chunk_ms"],
                engine=engine,
                config=run["config"],
                split=run["split"],
                limit=run["limit"],
                sample_n=run["sample_n"],
                sample_seed=run["sample_seed"],
                results_path=out_dir / f"{identifier}.jsonl",
                subsampled=run["subsampled"],
                step_ms=args.step_ms,
                backend=args.backend,
            )
            reports.append(report)
            print(json.dumps(report.to_dict()), flush=True)
            _write_summary(out_dir, reports)
        del engine
        _free_cuda()

    print("=== summary ===")
    for report in reports:
        print(
            f"{report.model_id.split('/')[-1]:>22} {report.mode:>9} {report.config:>5} "
            f"{report.chunk_ms:>5}ms WER={report.wer:.2f} CER={report.cer:.2f} "
            f"RTF={report.rtf:.3f} sub={report.subsampled}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
