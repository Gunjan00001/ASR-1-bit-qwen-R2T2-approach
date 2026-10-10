"""Build a local QAT training cache from LibriSpeech (Stage 1, T1.4).

``iter_librispeech`` streams rows from the Hub. For multi-epoch QAT we want a
fixed, fast, in-folder cache with duration filtering. This script materializes
``n`` utterances (deterministic subsample, seed 0) into a single ``.pt`` file
holding ids, transcripts, and 16 kHz float32 waveforms.

The cache is written under ``data/`` (gitignored) so runs are reproducible and
do not re-download shards.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from asr1bit.data.load import iter_librispeech


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="clean", help="LibriSpeech config (clean/other).")
    p.add_argument("--split", default="train.100", help="Split, e.g. train.100.")
    p.add_argument("--n", type=int, default=2000, help="Number of utterances to keep.")
    p.add_argument("--seed", type=int, default=0, help="Recorded for provenance; streaming takes first-n.")
    p.add_argument("--min-sec", type=float, default=0.5)
    p.add_argument("--max-sec", type=float, default=25.0)
    p.add_argument(
        "--stream", type=int, default=1,
        help="Read shards lazily (bounded memory; takes the first n valid rows).",
    )
    p.add_argument("--out", default=None, help="Output .pt path (default data/qat_<config>_<split>_n<n>.pt).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out) if args.out else Path("data") / f"qat_{args.config}_{args.split}_n{args.n}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    kept: list[dict] = []
    skipped = 0
    seen = 0

    def _iter():
        if args.stream:
            yield from iter_librispeech(args.config, args.split, streaming=True)
        else:
            pool = max(args.n * 2, args.n)
            yield from iter_librispeech(args.config, args.split, sample_n=pool, sample_seed=args.seed)

    for utt in _iter():
        seen += 1
        if utt.duration < args.min_sec or utt.duration > args.max_sec:
            skipped += 1
            continue
        kept.append(
            {
                "id": utt.id,
                "text": utt.text,
                "audio": torch.from_numpy(utt.audio.copy()),
            }
        )
        if len(kept) >= args.n:
            break
        if len(kept) % 500 == 0 and len(kept):
            print(f"  kept {len(kept)} ({seen} seen, {skipped} skipped)", flush=True)

    if len(kept) < args.n:
        print(f"warning: kept only {len(kept)}/{args.n} utterances", flush=True)
    durations = [len(item["audio"]) / 16000.0 for item in kept]
    meta = {
        "config": args.config,
        "split": args.split,
        "requested": args.n,
        "kept": len(kept),
        "seen": seen,
        "skipped": skipped,
        "seed": args.seed,
        "selection": "stream_first_n" if args.stream else "random_subsample",
        "min_sec": args.min_sec,
        "max_sec": args.max_sec,
        "total_audio_sec": sum(durations),
        "mean_sec": sum(durations) / max(1, len(durations)),
        "max_sec_seen": max(durations) if durations else 0.0,
    }
    torch.save({"meta": meta, "utterances": kept}, out)
    (out.with_suffix(".json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e9:.2f} GB) in {time.perf_counter() - started:.1f}s", flush=True)
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
