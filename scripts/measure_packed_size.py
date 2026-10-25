"""Measure the serialized size of a QAT checkpoint's packed representation.

The deployed model is not the fp32 shadow checkpoint: the decoder projections
are ternary ``{-1,0,+1}`` (2-bit slots) with a small scale, and everything else
stays at the base model's bf16. This script reads the trained shadows and the
base model's measured safetensors byte count, actually packs the ternary codes,
and reports the resulting model bytes, compression ratio and effective
bits/parameter.

It measures a *representation*, not a runtime: no packed inference kernel
exists in this repository, so latency/throughput are not claimed here.

Usage::

    python scripts/measure_packed_size.py \
        --checkpoint data/checkpoints/qat_stream_aligned17b12/step1600.pt \
        --base-bytes 4698521512 \
        --out artifacts/stage1/packed_size_aligned17b12_step1600.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

PROJ = re.compile(r"(?:^|\.)(self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)\.weight$")


def is_decoder_projection(key: str) -> bool:
    return "audio_tower" not in key and bool(PROJ.search(key))


def pack_ternary_2bit(weight: torch.Tensor) -> int:
    """Pack a ternary matrix into 2-bit slots; return the byte count.

    ``q in {-1,0,+1}`` is stored as ``q+1 in {0,1,2}``; four values per byte.
    The function materialises the bytes so the size is an actual serialization,
    not a formula.
    """
    flat = weight.reshape(-1)
    scale = flat.abs().mean().clamp(min=1e-12)
    q = torch.clamp(torch.round(flat / scale), -1, 1).to(torch.int64) + 1
    if q.numel() % 4:
        q = torch.cat([q, torch.zeros(4 - q.numel() % 4, dtype=torch.int64)])
    packed = (
        q[0::4]
        | (q[1::4] << 2)
        | (q[2::4] << 4)
        | (q[3::4] << 6)
    ).to(torch.uint8)
    return int(packed.numel())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-bytes", type=int, required=True,
                        help="Measured byte size of the bf16 base model safetensors.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    shadows = payload.get("shadows", {})
    if not shadows:
        raise SystemExit(f"no shadows in {args.checkpoint}")

    quantized_params = 0
    quantized_bf16_bytes = 0
    packed_bytes = 0
    scale_bytes = 0
    layers = 0
    for key, tensor in shadows.items():
        if not key.endswith(".weight") or tensor.ndim != 2 or not is_decoder_projection(key):
            continue
        layers += 1
        quantized_params += tensor.numel()
        quantized_bf16_bytes += tensor.numel() * 2
        packed_bytes += pack_ternary_2bit(tensor)
        scale_bytes += 2  # one fp16 scale per tensor

    compressed = args.base_bytes - quantized_bf16_bytes + packed_bytes + scale_bytes
    total_params = args.base_bytes // 2
    effective_bits = compressed * 8 / total_params
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": payload.get("step"),
        "base_bytes": args.base_bytes,
        "base_params_bf16_equivalent": total_params,
        "quantized_layers": layers,
        "quantized_params": quantized_params,
        "quantized_fraction": quantized_params / total_params,
        "quantization": "ternary {-1,0,+1}, 2-bit slots, per-tensor fp16 scale",
        "packed_weight_bytes": packed_bytes,
        "scale_bytes": scale_bytes,
        "compressed_bytes": compressed,
        "compressed_mib": compressed / 2**20,
        "compression_ratio_vs_bf16": args.base_bytes / compressed,
        "effective_bits_per_param": effective_bits,
        "non_quantized_bytes_bf16": args.base_bytes - quantized_bf16_bytes,
        "note": "Serialized representation only; no packed runtime exists, so "
                "latency/throughput are NOT measured.",
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
