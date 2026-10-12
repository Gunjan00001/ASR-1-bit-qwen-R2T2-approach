"""Word-level timestamps from Qwen3-ForcedAligner-0.6B (Stage 4, T4.1).

The official streaming algorithm's training targets need to know *which words
have been heard* at each chunk; a forced aligner supplies that timing. This
module adapts the official aligner to a small, backend-free dataclass so the
target construction (:mod:`asr1bit.data.prefix`) stays pure and testable.

The heavy ``qwen_asr`` import is deferred: ``align_words`` is the only entry
point that needs it, and it needs CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from asr1bit.data.load import SAMPLE_RATE

DEFAULT_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"


@dataclass(frozen=True)
class AlignedWord:
    """One aligned word/token: text plus its start/end time in seconds."""

    text: str
    start: float
    end: float


def align_words(
    audio: np.ndarray,
    text: str,
    *,
    language: str = "English",
    aligner: Any | None = None,
    model_id: str = DEFAULT_ALIGNER_MODEL,
    sample_rate: int = SAMPLE_RATE,
) -> list[AlignedWord]:
    """Align ``text`` to ``audio`` and return per-word timestamps.

    Args:
        audio: Mono waveform (16 kHz float32).
        text: The transcript to align (raw or normalized; punctuation is
            ignored by the aligner's tokenizer).
        language: Forced-aligner language name, e.g. ``"English"``.
        aligner: An already-loaded ``Qwen3ForcedAligner`` (avoids reloading the
            0.6B aligner for every call). When omitted one is loaded lazily from
            ``model_id``.
        model_id: Hub id used when ``aligner`` is None.

    Returns:
        Words in time order. Empty when the aligner returns no items.
    """
    if aligner is None:  # lazy heavy import (needs qwen_asr + CUDA)
        from qwen_asr import Qwen3ForcedAligner

        aligner = Qwen3ForcedAligner.from_pretrained(model_id)
    array = np.asarray(audio, dtype=np.float32).reshape(-1)
    results = aligner.align(audio=(array, sample_rate), text=text, language=language)
    if not results:
        return []
    items = getattr(results[0], "items", None) or []
    return [
        AlignedWord(text=str(item.text), start=float(item.start_time), end=float(item.end_time))
        for item in items
    ]


def alignment_covers(words: list[AlignedWord], duration_sec: float, *, tolerance: float = 0.5) -> bool:
    """Sanity check: the last word ends within ``tolerance`` of the audio end."""
    if not words:
        return False
    return abs(duration_sec - max(word.end for word in words)) <= tolerance
