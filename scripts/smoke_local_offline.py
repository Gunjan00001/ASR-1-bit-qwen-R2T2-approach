"""Local CPU smoke: validate loader + normalizer + ChunkedStreamer plumbing.

Runs the portable transformers backend OFFLINE on a fixed LibriSpeech
test-clean subset and compares corpus WER to the published offline numbers under
two normalizer variants. Then streams a few utterances through
``ChunkedStreamer`` to exercise the plumbing (no WER target).

Run inside an isolated venv with ``qwen-asr`` installed, e.g.::

    python scripts/smoke_local_offline.py --model Qwen/Qwen3-ASR-0.6B --n 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from asr1bit.data.load import iter_librispeech
from asr1bit.eval.wer import corpus_cer, corpus_wer
from asr1bit.qwen.backends import environment_report
from asr1bit.text import DEFAULT_MODE
from asr1bit.text import MODES as _ALL_MODES

# Published offline LibriSpeech WER (%) from the Qwen3-ASR technical report.
PUBLISHED_OFFLINE = {
    "Qwen/Qwen3-ASR-0.6B": 2.11,
    "Qwen/Qwen3-ASR-1.7B": 1.63,
}
MODES = tuple(mode for mode in _ALL_MODES if mode != "identity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default="clean")
    parser.add_argument("--split", default="test")
    parser.add_argument("--streaming-n", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--results", default="outputs/smoke")
    parser.add_argument("--published", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch

    from asr1bit.qwen import backends

    published = args.published if args.published is not None else PUBLISHED_OFFLINE.get(args.model)

    print("=== environment ===")
    print(json.dumps(environment_report(), indent=2))
    print(f"model={args.model} published_offline_wer={published}")

    print("=== loading utterances ===")
    utterances = list(
        iter_librispeech(args.config, args.split, sample_n=args.n, sample_seed=args.seed)
    )
    print(f"loaded {len(utterances)} utterances (config={args.config} split={args.split} seed={args.seed})")

    print("=== loading model (transformers backend, offline) ===")
    model = backends.load_model(
        args.model, backend="transformers", dtype=torch.float32, device=args.device
    )

    print("=== offline transcribe ===")
    out_dir = Path(args.results)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"offline_{args.model.split('/')[-1]}.jsonl"

    references: list[str] = []
    hypotheses: list[str] = []
    with results_path.open("w", encoding="utf-8") as handle:
        for index, utterance in enumerate(utterances):
            hypothesis = backends.offline_transcribe(model, utterance.audio, language=None)
            references.append(utterance.text)
            hypotheses.append(hypothesis)
            handle.write(
                json.dumps(
                    {"id": utterance.id, "reference": utterance.text, "hypothesis": hypothesis},
                    ensure_ascii=False,
                )
                + "\n"
            )
            if (index + 1) % 10 == 0:
                print(f"  {index + 1}/{len(utterances)}", flush=True)

    print(f"=== WER by normalizer variant (default={DEFAULT_MODE}) ===")
    scores: dict[str, float] = {}
    for mode in MODES:
        value = corpus_wer(references, hypotheses, mode=mode)
        scores[mode] = value
        delta = "" if published is None else f" (published {published:.2f}, delta {value - published:+.2f})"
        print(f"  {mode:>18}: {value:.2f}{delta}")
    print(f"  {'cer(librispeech)':>18}: {corpus_cer(references, hypotheses):.2f}")

    best_mode = min(scores, key=scores.get)
    best = scores[best_mode]
    if published is not None:
        verdict = "OK" if best <= published + 1.0 else "INVESTIGATE"
        print(f"=== verdict: {verdict} (best={best_mode} {best:.2f}) ===")
    (out_dir / "offline_summary.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "n": len(utterances),
                "published": published,
                "wer": scores,
                "best_mode": best_mode,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=== streaming plumbing (ChunkedStreamer, no WER target) ===")
    for utterance in utterances[: args.streaming_n]:
        trace = backends.stream_transcribe_portable(
            model, utterance.audio, chunk_size_sec=2.0, step_sec=0.5
        )
        print(
            f"  id={utterance.id} chunks={len(trace.chunks)} text={trace.text!r} rtf={trace.rtf:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
