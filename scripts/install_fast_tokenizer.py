"""Install a fast `tokenizer.json` into a Qwen3-ASR checkpoint (perf fix).

The released Qwen3-ASR checkpoints ship only ``vocab.json`` / ``merges.txt``.
Without ``tokenizer.json``, every process that loads the processor builds the
slow Python ``Qwen2Tokenizer`` line-by-line from ``merges.txt`` — on a contended
network filesystem that can take many minutes. Adding ``tokenizer.json`` makes
``AutoProcessor`` resolve ``Qwen2TokenizerFast`` (Rust) instead.

Usage::

    python scripts/install_fast_tokenizer.py --model Qwen/Qwen3-ASR-0.6B
    python scripts/install_fast_tokenizer.py --model /path/to/checkpoint
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from transformers import Qwen2TokenizerFast


def resolve_dir(model: str) -> Path:
    """Return a local directory for ``model`` (hub id or path)."""
    path = Path(model)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, allow_patterns=["*.json", "*.txt", "tokenizer*"]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    p.add_argument(
        "--out-dir", default=None,
        help="Directory to install into (defaults to the model directory itself).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = resolve_dir(args.model)
    vocab = root / "vocab.json"
    merges = root / "merges.txt"
    if not vocab.exists() or not merges.exists():
        raise FileNotFoundError(f"vocab.json/merges.txt not found in {root}")

    out_dir = Path(args.out_dir) if args.out_dir else root
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Qwen2TokenizerFast(vocab_file=str(vocab), merges_file=str(merges))
    with tempfile.TemporaryDirectory() as tmp:
        tokenizer.save_pretrained(tmp)
        source = Path(tmp) / "tokenizer.json"
        target = out_dir / "tokenizer.json"
        shutil.copy2(source, target)
    print(f"installed {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
