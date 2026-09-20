"""Shared text normalization for data loading and WER scoring (Stage 0).

Qwen3-ASR does not release its evaluation normalization code; its published
LibriSpeech numbers follow the common ASR convention of lowercasing and
stripping punctuation. ``normalize_text`` is the single source of truth used by
both the data loader and the WER harness so references and hypotheses are
normalized identically.
"""

from __future__ import annotations

import unicodedata

MODES = ("librispeech", "librispeech_hyph", "identity")
# Stage 0 smoke on 0.6B offline (100/300 test-clean utts) scored hyph 2.93 vs
# plain 3.31 against the published 2.11, so hyphen-splitting is the default.
DEFAULT_MODE = "librispeech_hyph"


def to_lowercase(text: str) -> str:
    """Lowercase ``text``."""
    return text.lower()


def remove_punctuation(text: str) -> str:
    """Delete every Unicode punctuation character (categories ``P*``)."""
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))


def split_hyphens(text: str) -> str:
    """Replace hyphens with spaces so ``well-known`` becomes two tokens."""
    return text.replace("-", " ")


def collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace to single spaces and strip the ends."""
    return " ".join(text.split())


def normalize_text(text: str | None, mode: str = DEFAULT_MODE) -> str:
    """Normalize ``text`` according to ``mode``.

    Args:
        text: Raw transcript. ``None`` is treated as the empty string.
        mode: ``"librispeech"`` (default) lowercases, removes punctuation, and
            collapses whitespace; ``"identity"`` returns the input unchanged.

    Returns:
        The normalized string.

    Raises:
        ValueError: If ``mode`` is not one of :data:`MODES`.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown normalization mode {mode!r}; expected one of {MODES}")
    if text is None:
        return ""
    if mode == "identity":
        return text
    text = to_lowercase(text)
    if mode == "librispeech_hyph":
        text = split_hyphens(text)
    return collapse_whitespace(remove_punctuation(text))
