"""Measure streaming throughput and decide full-split feasibility.

Streams a small number of utterances at the given chunk size, extrapolates the
wall-clock cost to the full split, and reports the coverage that fits a session
budget. Run this before the Stage 0 matrix so the runtime is known.

Usage::

    python scripts/throughput_probe.py --backend vllm --n 50
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from asr1bit.data.load import iter_librispeech
from asr1bit.eval.latency import coverage_fraction, estimate_wall_seconds, rtf
from asr1bit.qwen.backends import environment_report

FULL_SPLIT_SIZES = {
    ("clean", "test"): 2620,
    ("other", "test"): 2939,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "transformers"])
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--config", default="clean")
    parser.add_argument("--split", default="test")
    parser.add_argument("--chunk-ms", type=int, default=2000)
    parser.add_argument("--step-ms", type=int, default=500)
    parser.add_argument("--budget-hours", type=float, default=12.0)
    parser.add_argument("--results", default="outputs/stage0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from asr1bit.qwen import backends

    print("=== environment ===")
    print(json.dumps(environment_report(), indent=2))

    mode = "official" if args.backend == "vllm" else "portable"
    model = backends.load_model(args.model, backend=args.backend)
    utterances = list(iter_librispeech(args.config, args.split, limit=args.n))
    print(f"probing {len(utterances)} utterances, chunk={args.chunk_ms}ms backend={args.backend}")

    wall_times: list[float] = []
    audio_secs: list[float] = []
    for index, utterance in enumerate(utterances):
        started = time.perf_counter()
        trace = backends.stream_transcribe(
            model,
            utterance.audio,
            mode=mode,
            chunk_size_sec=args.chunk_ms / 1000.0,
            step_sec=args.step_ms / 1000.0,
        )
        wall_times.append(time.perf_counter() - started)
        audio_secs.append(trace.audio_sec)
        print(f"  {index + 1}/{len(utterances)} wall={wall_times[-1]:.2f}s text={trace.text[:40]!r}")

    mean_wall = sum(wall_times) / len(wall_times)
    full_n = FULL_SPLIT_SIZES.get((args.config, args.split), len(utterances))
    full_wall = estimate_wall_seconds(mean_wall, full_n)
    coverage = coverage_fraction(full_wall, args.budget_hours)
    fits = coverage >= 1.0

    summary = {
        "model": args.model,
        "backend": args.backend,
        "chunk_ms": args.chunk_ms,
        "n_probed": len(utterances),
        "mean_wall_sec_per_utt": mean_wall,
        "mean_audio_sec_per_utt": sum(audio_secs) / len(audio_secs),
        "rtf": rtf(sum(wall_times), sum(audio_secs)),
        "full_n": full_n,
        "estimated_full_wall_hours": full_wall / 3600.0,
        "coverage_at_budget": coverage,
        "fits_budget": fits,
    }

    print("=== throughput ===")
    print(json.dumps(summary, indent=2))
    if not fits:
        affordable = int(len(utterances) * coverage * 0.8)
        print(
            f"WARNING: full {args.config}/{args.split} streaming will NOT fit "
            f"{args.budget_hours}h. Use a fixed subsample of ~{affordable} and label it."
        )

    out_dir = Path(args.results)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"throughput_{args.model.split('/')[-1]}_{args.backend}_{args.chunk_ms}ms.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
