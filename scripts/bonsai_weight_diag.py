"""Real-weight diagnostics for the Bonsai investigation (1.7B, CPU).

Measures, per decoder projection and layer, the ternary reconstruction error of:
  A  original weights, per-tensor scale (baseline)
  B  original weights, group sizes 128/64/32
  C  Hadamard-rotated weights (block 1024/512/256, seeded signs), per-tensor
  D  rotated weights + group scales

No training, no GPU: reads ``model.safetensors`` directly. Output is one JSON
per run plus ``PROJ`` lines so a log reader can follow along.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

from asr1bit.quant import dequantize_weights, quantize_ternary
from asr1bit.rotate import rotate_weight

PROJECTIONS = {
    "q": ".self_attn.q_proj.weight",
    "k": ".self_attn.k_proj.weight",
    "v": ".self_attn.v_proj.weight",
    "o": ".self_attn.o_proj.weight",
    "gate": ".mlp.gate_proj.weight",
    "up": ".mlp.up_proj.weight",
    "down": ".mlp.down_proj.weight",
}
GROUPS = (None, 128, 64, 32)
BLOCKS = (1024, 512, 256)
SEED = 0


def rel_and_mse(matrix: torch.Tensor, group_size: int | None) -> tuple[float, float]:
    q, scale = quantize_ternary(matrix, group_size=group_size)
    restored = dequantize_weights(q, scale, group_size=group_size)
    return (
        ((restored - matrix).norm() / matrix.norm()).item(),
        ((restored - matrix) ** 2).mean().item(),
    )


def resolve_shards(model_dir: str | None) -> list[Path]:
    candidates: list[str] = []
    if model_dir:
        p = Path(model_dir)
        candidates += [str(p / "model.safetensors"), str(p)]
    candidates += [
        "/tmp/asr1bit/models/Qwen3-ASR-1.7B/model.safetensors",
        os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/*/model.safetensors"),
    ]
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(f"{hf_home}/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/*/model.safetensors")
    for pattern in candidates:
        hits = [Path(h) for h in sorted(glob.glob(pattern)) if Path(h).is_file()]
        if hits:
            if hits[0].is_dir():
                hits = sorted(hits[0].glob("*.safetensors"))
            elif len(hits) == 1 and hits[0].suffix == ".safetensors":
                pass
            return hits
    raise FileNotFoundError(
        "no Qwen3-ASR-1.7B safetensors found; pass --model-dir or --safetensors"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--safetensors", default=None)
    p.add_argument("--layers", default="0,7,14,21,27", help="layer indices to sample")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if args.safetensors:
        path = Path(args.safetensors)
        shards = sorted(path.rglob("*.safetensors")) if path.is_dir() else [path]
    else:
        shards = resolve_shards(args.model_dir)
    print(f"shards: {[str(s) for s in shards]}", flush=True)

    # Pass 1: key -> (shard, layer, projection). No tensors are loaded here.
    located: dict[str, tuple[Path, int, str]] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt") as handle:
            for name in handle.keys():  # noqa: SIM118 - safetensors handle, not a dict
                if ".model.layers." not in name or "audio_tower" in name:
                    continue
                for key, suffix in PROJECTIONS.items():
                    if name.endswith(suffix):
                        layer = int(name.split(".model.layers.")[1].split(".")[0])
                        located[name] = (shard, layer, key)
                        break
    if not located:
        raise RuntimeError("no decoder projections matched; check the key naming")
    n_layers = max(layer for _, layer, _ in located.values()) + 1
    layers = [int(x) for x in args.layers.split(",") if int(x) < n_layers]
    print(f"decoder layers: {n_layers}; sampling {layers}", flush=True)

    rows = []
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        wanted = {n: (layer, key) for n, (sh, layer, key) in located.items()
                  if sh == shard and layer in layers}
        if not wanted:
            continue
        with safe_open(str(shard), framework="pt") as handle:
            for name, (layer, key) in wanted.items():
                w = handle.get_tensor(name).to(torch.float32)
                row = {"layer": layer, "projection": key, "shape": list(w.shape),
                       "rms": w.pow(2).mean().sqrt().item(),
                       "max_over_rms": (w.abs().max() / w.pow(2).mean().sqrt()).item()}
                for group in GROUPS:
                    rel, mse = rel_and_mse(w, group)
                    row[f"rel_g{group or 'tensor'}"] = rel
                    row[f"mse_g{group or 'tensor'}"] = mse
                in_features = w.shape[1]
                for block in BLOCKS:
                    if in_features % block:
                        continue
                    rotated, _ = rotate_weight(w, block, SEED)
                    row[f"rel_rot{block}"] = rel_and_mse(rotated, None)[0]
                    row[f"rel_rot{block}_g64"] = rel_and_mse(rotated, 64)[0]
                rows.append(row)
                print("PROJ " + json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                                            for k, v in row.items()}), flush=True)
                del w

    payload = {"shards": [str(s) for s in shards], "layers": layers, "seed": SEED,
               "groups": [g or "tensor" for g in GROUPS], "blocks": list(BLOCKS),
               "rows": rows}
    out = out_dir / "bonsai_weight_diag.json"
    out.write_text(json.dumps(payload, indent=2))
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
