"""Stable-prefix + chunk/lookahead targets (Stage 4, T4.2).

Converts word timestamps (:mod:`asr1bit.data.align`) into chunk-aligned SFT
samples: at chunk ``k`` the model has heard the words that end before
``(k + 1) * chunk_sec`` (plus an optional lookahead), so the training sample is

- ``audio_sec = (k + 1) * chunk_sec`` (the audio the streamer has re-fed),
- ``prefix_text`` = the words heard by chunk ``k - 1`` (the *stable prefix* the
  emission policy may have committed), and
- ``target_text`` = the words heard *in* chunk ``k`` — the only text the model
  should emit at this chunk before stopping.

This is the supervision the earlier streaming recipes lacked: DAgger/distillation
taught "the whole remainder" or "the teacher's continuation", while the official
prefix algorithm needs "exactly what has been heard so far, then stop"
(docs/LAB_NOTES.md 2026-10-12). The construction is append-only by design: the
sample sequence's ``prefix_text + target_text`` only ever grows.

Invariant (locked in ``docs/PLANNING.md``): an emitted stable prefix is always a
true prefix of the full transcript.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from asr1bit.data.align import AlignedWord


@dataclass(frozen=True)
class ChunkExample:
    """One chunk-aligned training sample (text pieces only; audio is sliced by the caller)."""

    chunk_index: int
    audio_sec: float
    prefix_text: str
    target_text: str

    @property
    def heard_text(self) -> str:
        """The full transcript heard so far (prefix + this chunk's new words)."""
        return (self.prefix_text + " " + self.target_text).strip()


def chunk_examples(
    words: Sequence[AlignedWord],
    *,
    chunk_sec: float,
    lookahead_sec: float = 0.0,
    total_sec: float | None = None,
) -> list[ChunkExample]:
    """Build chunk-aligned samples from word timestamps.

    Args:
        words: Aligned words in time order (already sorted or sorted here).
        chunk_sec: Model chunk length in seconds (> 0).
        lookahead_sec: Extra audio the target may include (0 = strictly causal).
        total_sec: Full audio duration. Defaults to the last word's end; pass the
            real duration so the final (tail) chunk is represented faithfully.

    Returns:
        One :class:`ChunkExample` per chunk, from k = 0 to the chunk that
        contains the end of the audio. Chunks with no new words get an empty
        target — those are the samples that teach the model to stop.
    """
    if chunk_sec <= 0:
        raise ValueError(f"chunk_sec must be > 0, got {chunk_sec}")
    if lookahead_sec < 0:
        raise ValueError(f"lookahead_sec must be >= 0, got {lookahead_sec}")
    ordered = sorted(words, key=lambda word: (word.start, word.end))
    last_end = max((word.end for word in ordered), default=0.0)
    duration = float(total_sec) if total_sec is not None else last_end
    if duration <= 0 and not ordered:
        return []
    duration = max(duration, last_end)

    examples: list[ChunkExample] = []
    previous: list[AlignedWord] = []
    n_chunks = max(1, math.ceil(duration / chunk_sec))
    for index in range(n_chunks):
        audio_sec = min((index + 1) * chunk_sec, duration)
        heard = [word for word in ordered if word.end <= audio_sec + lookahead_sec]
        new = heard[len(previous) :]
        examples.append(
            ChunkExample(
                chunk_index=index,
                audio_sec=audio_sec,
                prefix_text=" ".join(word.text for word in previous),
                target_text=" ".join(word.text for word in new),
            )
        )
        previous = heard
    return examples
