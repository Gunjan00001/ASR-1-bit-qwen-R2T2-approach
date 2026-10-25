"""Paired WER comparison with a bootstrap confidence interval.

Both evaluation JSONs must contain ``references`` and ``hypotheses`` for the
same split (the official evaluator writes them). WER is corpus-level over
normalized words; the confidence interval comes from resampling utterances
with replacement (paired design: the same utterance is scored for both
systems).

Usage::

    PYTHONPATH=src python scripts/compare_wer.py \
        --control artifacts/stage1/streaming_qat_stream_aligned17b12_step1600_clean.json \
        --candidate artifacts/stage1/streaming_sd_d20_step1600_clean.json \
        --out artifacts/stage1/compare_d20_step1600_clean.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from asr1bit.text import normalize_text


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    """Unit-cost word Levenshtein distance."""
    n, m = len(reference), len(hypothesis)
    previous = list(range(m + 1))
    for i in range(1, n + 1):
        current = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
        previous = current
    return previous[m]


def condition(payload: dict) -> dict:
    """The quantized condition (official evaluator format) or the payload itself."""
    return (payload.get("conditions") or {}).get("quantized") or payload


def per_utterance(payload: dict) -> tuple[list[int], list[int]]:
    payload = condition(payload)
    references = [normalize_text(text).split() for text in payload["references"]]
    hypotheses = [normalize_text(text).split() for text in payload["hypotheses"]]
    errors = [edit_distance(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    lengths = [len(ref) for ref in references]
    return errors, lengths


def wer(errors: list[int], lengths: list[int]) -> float:
    total = sum(lengths)
    return 100.0 * sum(errors) / total if total else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    control = condition(json.loads(Path(args.control).read_text()))
    candidate = condition(json.loads(Path(args.candidate).read_text()))
    if control.get("references") != candidate.get("references"):
        raise SystemExit("the two evaluations are not on the same references")
    n = len(control["references"])
    if n != len(candidate["hypotheses"]):
        raise SystemExit("hypothesis count mismatch")

    control_errors, lengths = per_utterance(control)
    candidate_errors, _ = per_utterance(candidate)
    control_wer = wer(control_errors, lengths)
    candidate_wer = wer(candidate_errors, lengths)
    deltas = [b - a for a, b in zip(candidate_errors, control_errors)]

    rng = random.Random(args.seed)
    estimates = []
    for _ in range(args.resamples):
        total_errors_control = total_errors_candidate = total_len = 0
        for _ in range(n):
            index = rng.randrange(n)
            total_errors_control += control_errors[index]
            total_errors_candidate += candidate_errors[index]
            total_len += lengths[index]
        if total_len:
            estimates.append(
                100.0 * (total_errors_candidate - total_errors_control) / total_len
            )
    estimates.sort()
    lo = estimates[int(0.025 * len(estimates))]
    hi = estimates[int(0.975 * len(estimates)) - 1]

    result = {
        "control": str(args.control),
        "candidate": str(args.candidate),
        "n_utterances": n,
        "control_wer": round(control_wer, 4),
        "candidate_wer": round(candidate_wer, 4),
        "delta_wer": round(candidate_wer - control_wer, 4),
        "bootstrap_95ci": [round(lo, 4), round(hi, 4)],
        "candidate_better_utterances": sum(1 for d in deltas if d < 0),
        "candidate_worse_utterances": sum(1 for d in deltas if d > 0),
        "ties": sum(1 for d in deltas if d == 0),
        "resamples": args.resamples,
        "seed": args.seed,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
