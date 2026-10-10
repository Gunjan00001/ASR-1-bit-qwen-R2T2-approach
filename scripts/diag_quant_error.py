"""Stage 1 diagnostic: quantization error of the real Qwen3-ASR weights.

Reads ``model.safetensors`` directly (no model instantiation) and reports, for
every decoder projection, the relative Frobenius error
``||Q(W) - W|| / ||W||`` under several quantizer configurations:

- binary / ternary with per-tensor scaling,
- binary / ternary with group-wise scaling (group sizes 32/64/128/256),
- bf16 round-to-trip as a reference (the storage precision of the checkpoint).

The purpose is to separate "the quantizer math is broken" from "per-tensor
1-bit quantization of a 0.6B model is inherently coarse": for Gaussian weights
the per-tensor binary error has a closed form around 60%, and group-wise
scaling reduces it. Run on any host (CPU is fine).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file

from asr1bit.quant import dequantize_weights, quantize_binary, quantize_ternary

PROJ_PATTERN = re.compile(
    r"(?:^|\.)(self_attn\.(?P<attn>q|k|v|o)_proj|mlp\.(?P<mlp>gate|up|down)_proj)\.weight$"
)
GROUP_SIZES = (32, 64, 128, 256)
DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B"


def resolve_safetensors(model_id: str, local: str | None) -> Path:
    if local:
        path = Path(local)
        if path.is_dir():
            path = path / "model.safetensors"
        return path
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json", "tokenizer*"])
    )
    return root / "model.safetensors"


def is_decoder_projection(key: str) -> str | None:
    if "audio_tower" in key:
        return None
    match = PROJ_PATTERN.search(key)
    if not match:
        return None
    return match.group("attn") or match.group("mlp")


def rel_error(w: torch.Tensor, qw: torch.Tensor, scale: torch.Tensor, group_size: int | None) -> float:
    w_hat = dequantize_weights(qw, scale, group_size)
    return float((w_hat - w).norm() / w.norm())


def measure(w: torch.Tensor) -> dict[str, float]:
    w = w.float()
    row: dict[str, float] = {
        "rms": float(w.pow(2).mean().sqrt()),
        "mean_abs": float(w.abs().mean()),
        "max_over_rms": float(w.abs().max() / w.pow(2).mean().sqrt()),
    }
    qb, sb = quantize_binary(w)
    row["binary_tensor"] = rel_error(w, qb, sb, None)
    qt, st = quantize_ternary(w)
    row["ternary_tensor"] = rel_error(w, qt, st, None)
    for group in GROUP_SIZES:
        if w.shape[-1] % group:
            continue
        qb, sb = quantize_binary(w, group)
        row[f"binary_g{group}"] = rel_error(w, qb, sb, group)
        qt, st = quantize_ternary(w, group)
        row[f"ternary_g{group}"] = rel_error(w, qt, st, group)
    row["bf16_roundtrip"] = float((w.bfloat16().float() - w).norm() / w.norm())
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--safetensors", default=None, help="Local path (dir or file) to use.")
    p.add_argument("--out", default="artifacts/stage1")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    path = resolve_safetensors(args.model, args.safetensors)
    print(f"reading {path}", flush=True)
    tensors = load_file(str(path))

    layers: list[dict] = []
    for key, tensor in tensors.items():
        proj = is_decoder_projection(key)
        if proj is None:
            continue
        layers.append({"key": key, "proj": proj, "shape": list(tensor.shape), **measure(tensor)})

    metrics = [k for k in layers[0] if k not in ("key", "proj", "shape")]
    by_proj: dict[str, dict[str, float]] = {}
    for proj in ("q", "k", "v", "o", "gate", "up", "down"):
        subset = [layer for layer in layers if layer["proj"] == proj]
        by_proj[proj] = {
            metric: sum(layer[metric] for layer in subset) / len(subset) for metric in metrics
        }
    overall = {metric: sum(layer[metric] for layer in layers) / len(layers) for metric in metrics}

    header = f"{'proj':6s} {'n':>3s} " + " ".join(f"{m:>15s}" for m in metrics)
    print(header, flush=True)
    for proj, row in by_proj.items():
        print(
            f"{proj:6s} {sum(1 for layer in layers if layer['proj'] == proj):3d} "
            + " ".join(f"{row[m]:15.4f}" for m in metrics),
            flush=True,
        )
    print(
        f"{'ALL':6s} {len(layers):3d} " + " ".join(f"{overall[m]:15.4f}" for m in metrics),
        flush=True,
    )

    worst = sorted(layers, key=lambda layer: -layer["binary_tensor"])[:8]
    print("\nworst per-tensor binary error layers:", flush=True)
    for layer in worst:
        print(f"  {layer['key']}: binary {layer['binary_tensor']:.4f}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "safetensors": str(path),
        "n_layers": len(layers),
        "metrics": metrics,
        "by_projection": by_proj,
        "overall": overall,
        "layers": layers,
    }
    out_path = out_dir / "diag_quant_error.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", out_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
