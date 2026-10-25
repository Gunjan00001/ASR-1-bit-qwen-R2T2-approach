"""Collect official streaming evaluations into one machine-readable CSV.

Every row comes from an ``artifacts/stage1/streaming_*.json`` written by
``scripts/eval_streaming.py``. Files whose name contains ``WRONGPOLICY`` are
kept on disk but excluded from the authoritative table (they measured a
policy/checkpoint mismatch, not the arm).

Usage:
    PYTHONPATH=src python scripts/collect_results.py \
        --out docs/final_experiment_results.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001 - provenance is best effort
        return ""


BRANCH = _git("branch", "--show-current")
COMMIT = _git("rev-parse", "--short", "HEAD")
ARTIFACTS = ROOT / "artifacts" / "stage1"

# Known 1.7B Stage-B arms: the policy/width actually used for training. The
# evaluator JSON does not record these, so they come from the committed
# launchers (and the b17/b20/b21 lesson: an evaluation with the wrong policy or
# width is invalid).
ARM_CONFIG = {
    "aligned17b5": ("decoder_attn_ffn", 1.58, "per-tensor", "5e-6"),
    "aligned17b6": ("decoder_attn_ffn", 1.58, "per-tensor", "5e-6"),
    "aligned17b7": ("decoder_attn_ffn", 1.58, "per-tensor", "5e-6"),
    "aligned17b8": ("decoder_attn_ffn", 1.58, "per-tensor", "5e-6"),
    "aligned17b9": ("decoder_attn_ffn", 1.58, "per-tensor", "5e-6"),
    "aligned17b10": ("decoder_attn_ffn", 1.58, "per-tensor", "5e-6"),
    "aligned17b11": ("decoder_attn_ffn", 1.58, "per-tensor", "2.5e-6"),
    "aligned17b12": ("decoder_attn_ffn", 1.58, "per-tensor", "2.5e-6"),
    "aligned17b13": ("decoder_attn_ffn", 1.58, "per-tensor", "2.5e-6"),
    "aligned17b14": ("decoder_attn_ffn", 1.58, "per-tensor", "2.5e-6"),
    "aligned17b15": ("decoder_attn_ffn", 1.58, "per-tensor", "2.5e-6"),
    "aligned17b16": ("decoder_attn_ffn", 1.58, "per-tensor", "1.25e-6"),
    "aligned17b17": ("decoder_ffn", 1.58, "per-tensor", "2.5e-6"),
    "aligned17b20": ("decoder_attn_ffn", 2.0, "per-tensor", "2.5e-6"),
    "aligned17b21": ("decoder_attn_ffn", 1.58, "g64", "2.5e-6"),
    "b12": ("decoder_attn_ffn", 1.58, "per-tensor", "2.5e-6"),
    "b12_step1600_n100": ("decoder_attn_ffn", 1.58, "per-tensor", "2.5e-6"),
    "b05": ("decoder_attn_ffn", 1.58, "per-tensor", "2.5e-6"),
    "c01": ("decoder_attn_ffn", 1.58, "per-tensor", "2.5e-6"),
    "d20": ("decoder_attn_ffn", 1.58, "per-tensor", "2.5e-6"),
    "e08": ("decoder_attn_ffn", 1.58, "per-tensor", "2.5e-6"),
    "f20": ("decoder_attn_ffn", 1.58, "per-tensor", "2.5e-6"),
    "g_noLN": ("decoder_attn_ffn", 1.58, "per-tensor", "2.5e-6"),
}

PATTERNS = (
    re.compile(r"streaming_qat_stream_(?P<arm>aligned17b\d+)_(?P<state>step\d+|last)_?\(?(?P<split>clean|other)\)?\.json$"),
    re.compile(r"streaming_sd_(?P<arm>b05|c01|d20|e08|f20|g_noLN)_(?P<state>step\d+)(?:_n(?P<n>\d+))?_(?P<split>clean|other)\.json$"),
    re.compile(r"streaming_(?P<arm>b12)_(?P<state>step\d+)_n(?P<n>\d+)_(?P<split>clean|other)\.json$"),
    re.compile(r"streaming_(?P<arm>[a-z0-9]+(?:_[a-z0-9]+)*)_?(?P<state>c?\d+)?_?(?P<split>clean|other)?\.json$"),
)


def _state_step(state: str | None) -> str:
    if not state:
        return ""
    match = re.search(r"(\d+)", state)
    return match.group(1) if match else state


def parse(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    quant = (payload.get("conditions") or {}).get("quantized")
    if not quant:
        return None
    name = path.name
    if "WRONGPOLICY" in name:
        return None
    match = None
    for pattern in PATTERNS:
        match = pattern.search(name)
        if match and match.groupdict().get("arm"):
            break
    arm = match.group("arm") if match else name
    state = match.groupdict().get("state") if match else ""
    split = payload.get("config") or (match.groupdict().get("split") if match else "") or ""
    lsp = quant.get("lsp") or {}
    policy, bit_width, group, lr = ARM_CONFIG.get(arm, ("", "", "", ""))
    checkpoint = payload.get("checkpoint") or ""
    return {
        "arm": arm,
        "branch": BRANCH,
        "commit": COMMIT,
        "config": f"{policy}/{bit_width}/{group}/lr{lr}" if policy else "historical",
        "seed": "0 (explicit for new arms; implicit deterministic for historical)",
        "state": Path(checkpoint).name if checkpoint else (state or ""),
        "checkpoint_step": _state_step(Path(checkpoint).stem if checkpoint else state),
        "split": split,
        "chunk_ms": payload.get("chunk_ms"),
        "policy": policy,
        "bit_width": bit_width,
        "group": group,
        "lr": lr,
        "wer": round(float(quant["wer"]), 4),
        "cer": round(float(quant["cer"]), 4),
        "loops": lsp.get("loop_truncations"),
        "revisions": lsp.get("revisions"),
        "wer_raw": round(float(lsp["wer_raw"]), 4) if lsp.get("wer_raw") is not None else None,
        "n": payload.get("n"),
        "checkpoint": checkpoint,
        "artifact": str(path.relative_to(ROOT)),
        "evaluated_at": dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        "status": "COMPLETED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/final_experiment_results.csv")
    parser.add_argument("--artifacts", default=str(ARTIFACTS))
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.artifacts).glob("streaming_*.json")):
        row = parse(path)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda row: (row["arm"], row["checkpoint_step"], row["split"], row["chunk_ms"] or 0))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else [
        "arm", "state", "checkpoint_step", "split", "chunk_ms", "policy",
        "bit_width", "group", "lr", "wer", "cer", "loops", "revisions",
        "wer_raw", "n", "checkpoint", "artifact", "status",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} ({len(rows)} official evaluations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
