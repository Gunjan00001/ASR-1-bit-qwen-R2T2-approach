"""Cache for the forced aligner's word timestamps (the slow part of T4.1).

Why this exists: building aligned targets runs Qwen3-ForcedAligner over every
training utterance (~0.8 s/utt, 20-40 min per Stage-B run), and every retry or
re-run pays it again. The *chunking* of words into per-chunk targets is cheap;
only the aligner output is expensive. So this module caches **the aligner's
words**, keyed by the exact utterance set and aligner model, and the aligned
builder reuses them transparently.

Key design points:
- the key covers the *utterance identities* (their text and duration) and the
  aligner model name, so a different split, order or model can never hit a
  stale entry;
- the payload is plain JSON (word, start, end per entry), which keeps it
  small and human-inspectable;
- a corrupt or missing cache is treated as a miss, never as an error.
"""

from __future__ import annotations

import json
from pathlib import Path

CACHE_DIR = Path("data/aligned_cache")


def utts_key(utterances, aligner_model: str) -> str:
    """Stable key for the (utterance set, aligner) pair.

    Uses text + duration rather than audio hashes: they are cheap, and two
    utterances agreeing on both are interchangeable for alignment purposes.
    """
    import hashlib

    digest = hashlib.sha1()
    digest.update(aligner_model.encode("utf-8"))
    for utt in utterances:
        digest.update(b"\x00")
        digest.update(utt.text.encode("utf-8"))
        digest.update(f"{utt.duration:.3f}".encode())
    return digest.hexdigest()[:16]


class WordsCache:
    """Align words per utterance, persisted as JSON so retries are cheap."""

    def __init__(self, key: str, directory: Path | None = None):
        self.dir = Path(directory or CACHE_DIR)
        self.path = self.dir / f"{key}.json"
        self._entries: dict[str, list[list[float]]] | None = None

    def _load(self) -> dict[str, list[list[float]]]:
        if self._entries is not None:
            return self._entries
        try:
            self._entries = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError, ValueError):
            self._entries = {}
        return self._entries

    def get(self, index: int):
        """Cached words, shaped like the aligner's output (``.word/.start/.end``)."""
        from types import SimpleNamespace

        words = self._load().get(str(index))
        if words is None:
            return None
        return [SimpleNamespace(text=str(w), start=float(s), end=float(e)) for w, s, e in words]

    def put(self, index: int, words) -> None:
        """Accept either the aligner's word objects or plain ``(word, start, end)`` tuples."""
        rows = []
        for item in words:
            if hasattr(item, "text"):
                rows.append([item.text, float(item.start), float(item.end)])
            else:
                w, s, e = item
                rows.append([w, float(s), float(e)])
        self._load()[str(index)] = rows

    def save(self) -> None:
        if self._entries is None:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._entries))
