"""Task-aware layer sensitivity screening for the ternary 1.7B baseline.

Runs the first-order sensitivity analyzer on a small set of real streaming
chunk batches and writes a ranked JSON + CSV. This is the cheap screening
stage; it is not a WER measurement. The task-level validation is the
sensitivity-guided mixed-precision run.

Usage::

    PYTHONPATH=src python scripts/analyze_layer_sensitivity.py \
        --checkpoint data/checkpoints/qat_stream_aligned17b12/step1600.pt \
        --out artifacts/stage1/layer_sensitivity.json
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path

import torch

from asr1bit.analysis.sensitivity import SensitivityAnalyzer, aggregate_by_component
from asr1bit.data.align import DEFAULT_ALIGNER_MODEL
from asr1bit.data.load import load_utterance_cache
from asr1bit.qwen import backends
from asr1bit.replace import apply_layer_policy
from asr1bit.train.aligned import build_aligned_examples
from asr1bit.train.aligned_cache import WordsCache, utts_key
from asr1bit.train.checkpoints import load_checkpoint
from asr1bit.train.dataset import collate
from asr1bit.train.model_prep import patch_outer_forward
from asr1bit.train.samples import select_evenly

ROOT = Path(__file__).resolve().parents[1]
ALIGNED_CACHE = ROOT / "data" / "aligned_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--checkpoint", default="data/checkpoints/qat_stream_aligned17b12/step1600.pt")
    parser.add_argument("--policy", default="decoder_attn_ffn")
    parser.add_argument("--bit-width", type=float, default=1.58)
    parser.add_argument("--train-cache", default="data/qat_clean_train.100_n8000.pt")
    parser.add_argument("--aligned-n", type=int, default=6000)
    parser.add_argument("--calibration-utts", type=int, default=8)
    parser.add_argument("--calibration-examples", type=int, default=16)
    parser.add_argument("--chunk-ms", type=int, default=320)
    parser.add_argument("--out", default="artifacts/stage1/layer_sensitivity.json")
    parser.add_argument("--csv", default="docs/layer_sensitivity.csv")
    return parser.parse_args()


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return ""


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_words_cache(utts) -> tuple[WordsCache, int, int]:
    """Pick the largest aligned-words cache that covers the train cache."""
    for size in (8000, 6000, 2400, 1200, 300):
        if size > len(utts):
            continue
        key = utts_key(utts[:size], DEFAULT_ALIGNER_MODEL)
        if (ALIGNED_CACHE / f"{key}.json").exists():
            return WordsCache(key), size, key
    raise SystemExit(
        "no matching aligned-words cache; run the aligned builder once to populate "
        f"{ALIGNED_CACHE}"
    )


def main() -> int:
    args = parse_args()
    wrapper = backends.load_model(
        args.model, backend="transformers", dtype=torch.float32, device="cuda:0",
        max_new_tokens=128,
    )
    model = wrapper.model
    processor = wrapper.processor
    patch_outer_forward(model)
    apply_layer_policy(
        model, args.policy, bit_width=args.bit_width, learnable_scale=True
    )
    model.to("cuda:0")
    step = load_checkpoint(args.checkpoint, model)
    for name, parameter in model.named_parameters():
        if "self_attn" in name or ".mlp." in name:
            parameter.requires_grad_(True)
    model.eval()

    utterances = load_utterance_cache(args.train_cache)
    words_cache, cache_size, cache_key = _find_words_cache(utterances)
    n_cal = min(args.calibration_utts, cache_size)
    for index in range(n_cal):
        if words_cache.get(index) is None:
            raise SystemExit(f"aligned-words cache miss for calibration utterance {index}")
    examples = build_aligned_examples(
        utterances[:n_cal],
        processor,
        aligner=None,
        chunk_ms=args.chunk_ms,
        lookahead_ms=0,
        stop_stride=1,
        words_cache=words_cache,
    )
    examples = select_evenly(examples, args.calibration_examples)
    if not examples:
        raise SystemExit("no calibration examples were built")
    print(
        f"calibration: {len(examples)} examples from {n_cal} utterances "
        f"(cache {cache_key}, size {cache_size})",
        flush=True,
    )

    analyzer = SensitivityAnalyzer(model)
    total_loss = 0.0
    used = 0
    with torch.enable_grad():
        for index, example in enumerate(examples):
            batch = collate([example], processor)
            batch = {key: (value.cuda() if torch.is_tensor(value) else value) for key, value in batch.items()}
            batch.pop("state_mask", None)
            analyzer.reset()
            out = model(**batch)
            loss = out.loss if hasattr(out, "loss") else out[0]
            if loss is None or not torch.isfinite(loss):
                print(f"skip example {index}: non-finite loss", flush=True)
                continue
            analyzer.score(loss)
            total_loss += float(loss.detach())
            used += 1
            if (index + 1) % 4 == 0:
                print(f"  scored {index + 1}/{len(examples)}", flush=True)
    rows = analyzer.results()
    analyzer.close()
    if not rows:
        raise SystemExit("analyzer produced no rows")

    payload = {
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": step,
        "checkpoint_sha256": _sha256(Path(args.checkpoint)),
        "commit": _commit(),
        "policy": args.policy,
        "bit_width": args.bit_width,
        "chunk_ms": args.chunk_ms,
        "calibration_utterances": n_cal,
        "calibration_examples": used,
        "aligned_cache_key": cache_key,
        "mean_calibration_loss": total_loss / max(1, used),
        "method": "first-order task-loss impact of the ternary output error",
        "layers": [dataclasses.asdict(row) for row in rows],
        "components": aggregate_by_component(rows),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = ROOT / args.csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "name,component,params,local_relative_error,task_impact,"
            "task_impact_share,task_impact_per_param,batches\n"
        )
        for row in rows:
            handle.write(
                f"{row.name},{row.component},{row.params},{row.local_relative_error:.8g},"
                f"{row.task_impact:.8g},{row.task_impact_share:.8g},"
                f"{row.task_impact_per_param:.8g},{row.batches}\n"
            )
    print(f"wrote {out} and {csv_path}", flush=True)
    print("top 15 layers:", flush=True)
    for row in rows[:15]:
        print(
            f"  {row.task_impact_share:7.4f} impact | "
            f"{row.local_relative_error:8.4f} local | {row.component:6s} | {row.name}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
