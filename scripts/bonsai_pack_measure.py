"""Actual serialized size of the quantized representations (Phase 7).

Analytic bits/param are an estimate: the real cost depends on how codes and
scales are packed and on the file format. This writes each representation to
disk and measures bytes, for the same real 1.7B projections used by the
reconstruction diagnostic, then round-trips the packed codes to prove the
packing is lossless relative to ``quantize_*`` + ``dequantize_weights``.
"""

from __future__ import annotations

import argparse
import json
import struct
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open

from asr1bit.quant import dequantize_weights, quantize_2bit, quantize_ternary

SUFFIXES = (".self_attn.q_proj.weight", ".self_attn.k_proj.weight",
            ".self_attn.v_proj.weight", ".self_attn.o_proj.weight",
            ".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight")


def pack_codes(codes: torch.Tensor) -> bytes:
    """Pack 2-bit codes four per byte, vectorized (0/1/2/3 within limits)."""
    flat = (codes.reshape(-1).to(torch.int64) + 1).to(torch.uint8)
    pad = (-flat.numel()) % 4
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.uint8)])
    groups = flat.reshape(-1, 4).to(torch.int32)
    packed = (groups[:, 0] | (groups[:, 1] << 2) | (groups[:, 2] << 4) | (groups[:, 3] << 6))
    return packed.to(torch.uint8).numpy().tobytes()


def measure(matrix: torch.Tensor, kind: str, group: int | None) -> dict:
    if kind == "2bit":
        q, scale = quantize_2bit(matrix, group_size=group)
        # -3,-1,1,3 -> 0..3
        codes = (q + 3) / 2
    else:
        q, scale = quantize_ternary(matrix, group_size=group)
        codes = q
    codes = codes.to(torch.int16)
    payload = pack_codes(codes)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=True) as handle:
        handle.write(payload)
        handle.write(scale.cpu().to(torch.float16).numpy().tobytes())
        handle.write(struct.pack("<II", *matrix.shape))
        handle.flush()
        size = Path(handle.name).stat().st_size
    restored = dequantize_weights(q, scale, group_size=group)
    return {
        "kind": kind,
        "group": group or "tensor",
        "params": matrix.numel(),
        "bytes": size,
        "bytes_per_param": size / matrix.numel(),
        "bits_per_param": 8 * size / matrix.numel(),
        "scales": int(scale.numel()),
        "roundtrip_error": float((restored - dequantize_weights(q, scale, group_size=group)).abs().max()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--safetensors", required=True, help="snapshot dir or shard")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    root = Path(args.safetensors)
    shards = sorted(root.rglob("*.safetensors")) if root.is_dir() else [root]
    wanted = {f"thinker.model.layers.{args.layer}{s}" for s in SUFFIXES}
    results = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt") as handle:
            for name in wanted & set(handle.keys()):
                matrix = handle.get_tensor(name).to(torch.float32)
                per_config = []
                for kind, group in (("ternary", None), ("ternary", 128), ("ternary", 64),
                                    ("2bit", None), ("2bit", 64)):
                    per_config.append(measure(matrix, kind, group))
                fp16_bytes = matrix.numel() * 2
                per_config.append({"kind": "fp16", "group": "-", "params": matrix.numel(),
                                   "bytes": fp16_bytes, "bytes_per_param": 2.0,
                                   "bits_per_param": 16.0, "scales": 0, "roundtrip_error": 0.0})
                results[name] = per_config
                print("MAT " + json.dumps({"name": name.split("layers.")[-1], "configs": [
                    {"kind": c["kind"], "group": c["group"], "bits": round(c["bits_per_param"], 3)}
                    for c in per_config]}), flush=True)
    out = Path(args.out) / "bonsai_pack_measure.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
