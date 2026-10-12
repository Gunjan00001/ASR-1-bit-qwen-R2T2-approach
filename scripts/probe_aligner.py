"""Verify the forced-aligner path + chunk-aligned targets (Stage 4, T4.1/T4.2).

Loads ``Qwen/Qwen3-ForcedAligner-0.6B``, aligns a few LibriSpeech test-clean
utterances, and prints the derived chunk-aligned samples (prefix / new words per
chunk) so the timing logic can be inspected on real audio before any training
run uses it.

Run on a CUDA host:  python scripts/probe_aligner.py --n 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from asr1bit.data.align import DEFAULT_ALIGNER_MODEL, align_words, alignment_covers
from asr1bit.data.load import iter_librispeech
from asr1bit.data.prefix import chunk_examples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_ALIGNER_MODEL)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--chunk-ms", type=int, default=320)
    p.add_argument("--lookahead-ms", type=int, default=0)
    p.add_argument("--config", default="clean", choices=["clean", "other"])
    p.add_argument("--out", default="artifacts/stage4/aligner_probe.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from qwen_asr import Qwen3ForcedAligner

    aligner = Qwen3ForcedAligner.from_pretrained(args.model)
    utterances = list(iter_librispeech(args.config, "test", sample_n=args.n, sample_seed=0))
    results = []
    for utt in utterances:
        words = align_words(utt.audio, utt.text, aligner=aligner)
        examples = chunk_examples(
            words,
            chunk_sec=args.chunk_ms / 1000.0,
            lookahead_sec=args.lookahead_ms / 1000.0,
            total_sec=utt.duration,
        )
        print(f"\n=== {utt.id} ({utt.duration:.2f}s) {utt.text[:70]}")
        print(f"words: {len(words)} | last end {words[-1].end:.2f}s" if words else "words: 0")
        print(f"coverage: {alignment_covers(words, utt.duration)}")
        for example in examples[:4]:
            print(
                f"  chunk {example.chunk_index} audio {example.audio_sec:.2f}s | "
                f"prefix {example.prefix_text[:40]!r} -> new {example.target_text[:50]!r}"
            )
        empty = sum(1 for e in examples if not e.target_text)
        print(f"  samples {len(examples)} | empty-target (stop) samples {empty}")
        results.append(
            {
                "id": utt.id,
                "duration": utt.duration,
                "reference": utt.text,
                "words": [{"text": w.text, "start": w.start, "end": w.end} for w in words],
                "examples": [
                    {
                        "chunk": e.chunk_index,
                        "audio_sec": e.audio_sec,
                        "prefix": e.prefix_text,
                        "target": e.target_text,
                    }
                    for e in examples
                ],
            }
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "chunk_ms": args.chunk_ms, "utterances": results}, indent=2), encoding="utf-8")
    print("\nwrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
