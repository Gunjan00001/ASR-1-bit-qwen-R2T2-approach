"""Chunked streaming decode replicating the official Qwen3-ASR algorithm.

Stage 0 (T0.3): a backend-agnostic engine that mirrors the official
``streaming_transcribe`` semantics — each time a full chunk accumulates, *all*
audio seen so far is re-fed to the model and the prompt is rebuilt with a
rollback prefix.

Qwen3-ASR is pseudo-streaming: it may revise text between chunks, so this module
makes **no append-only guarantee**. The Longest-Stable-Prefix (LSP) emission
policy and its append-only invariant are Stage 4 / T4.4.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

SAMPLE_RATE = 16000
ASR_TEXT_TAG = "<asr_text>"
LANG_PREFIX = "language "
REPLACEMENT_CHAR = "\ufffd"


def chunk_size_samples(chunk_ms: float, sample_rate: int = SAMPLE_RATE) -> int:
    """Convert a chunk length in milliseconds to a sample count."""
    return max(1, round(chunk_ms / 1000.0 * sample_rate))


class TokenCodec(Protocol):
    """Minimal tokenizer interface needed for prefix rollback."""

    def encode(self, text: str) -> Sequence[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


class WhitespaceTokenCodec:
    """Word-level codec used for tests and tokenizer-free runs."""

    def encode(self, text: str) -> list[str]:
        return text.split()

    def decode(self, ids: Sequence[str]) -> str:
        return " ".join(ids)


def parse_stream_output(raw: str | None, user_language: str | None = None) -> tuple[str, str]:
    """Split raw model output into ``(language, text)``.

    Mirrors the official ``parse_asr_output`` for the tag form
    ``language English<asr_text>...``; when ``user_language`` is set the raw
    string is treated as text-only (the model was prompted to omit metadata).
    """
    if raw is None:
        return "", ""
    text = str(raw).strip()
    if not text:
        return "", ""

    if user_language:
        return user_language, text

    if ASR_TEXT_TAG not in text:
        return "", text

    meta, remainder = text.split(ASR_TEXT_TAG, 1)
    if "language none" in meta.lower():
        return "", remainder.strip()

    language = ""
    for line in meta.splitlines():
        line = line.strip()
        if line.lower().startswith(LANG_PREFIX):
            value = line[len(LANG_PREFIX) :].strip()
            if value:
                language = value[:1].upper() + value[1:].lower()
            break
    return language, remainder.strip()


@dataclass
class ASRStreamingState:
    """Mutable per-stream state (one utterance).

    Mirrors the fields of the official ``ASRStreamingState``.
    """

    unfixed_chunk_num: int
    unfixed_token_num: int
    chunk_size_sec: float
    chunk_size_samples: int
    chunk_id: int = 0
    buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    audio_accum: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    prompt_raw: str = ""
    context: str = ""
    force_language: str | None = None
    language: str = ""
    text: str = ""
    _raw_decoded: str = ""


DecodeFn = Callable[[str, np.ndarray], str]


class ChunkedStreamer:
    """Backend-agnostic chunked streamer.

    Args:
        decode_fn: ``(prompt, accumulated_audio) -> generated_text``. The
            generated text is the *continuation* after ``prompt``.
        codec: Tokenizer used for prefix rollback.
        chunk_size_sec: Model chunk length. Audio is decoded once this much has
            accumulated.
        unfixed_chunk_num: For the first N chunks the prefix is empty.
        unfixed_token_num: Number of trailing tokens rolled back from the
            previous decode before it is reused as a prefix.
        sample_rate: Sample rate of incoming PCM.
    """

    def __init__(
        self,
        decode_fn: DecodeFn,
        codec: TokenCodec,
        *,
        chunk_size_sec: float = 2.0,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
        sample_rate: int = SAMPLE_RATE,
    ):
        if chunk_size_sec <= 0:
            raise ValueError(f"chunk_size_sec must be > 0, got {chunk_size_sec}")
        self.decode_fn = decode_fn
        self.codec = codec
        self.chunk_size_sec = float(chunk_size_sec)
        self.chunk_size_samples = max(1, round(self.chunk_size_sec * sample_rate))
        self.unfixed_chunk_num = int(unfixed_chunk_num)
        self.unfixed_token_num = int(unfixed_token_num)
        self.sample_rate = int(sample_rate)

    def init_state(
        self,
        prompt_raw: str = "",
        context: str = "",
        force_language: str | None = None,
    ) -> ASRStreamingState:
        """Create a fresh state for one stream."""
        return ASRStreamingState(
            unfixed_chunk_num=self.unfixed_chunk_num,
            unfixed_token_num=self.unfixed_token_num,
            chunk_size_sec=self.chunk_size_sec,
            chunk_size_samples=self.chunk_size_samples,
            prompt_raw=prompt_raw,
            context=context or "",
            force_language=force_language,
        )

    @staticmethod
    def _as_float32(pcm: np.ndarray) -> np.ndarray:
        audio = np.asarray(pcm)
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        if audio.dtype == np.int16:
            return (audio.astype(np.float32) / 32768.0).astype(np.float32)
        return audio.astype(np.float32, copy=False)

    def _rollback_prefix(self, text: str) -> str:
        if not text:
            return ""
        ids = list(self.codec.encode(text))
        k = self.unfixed_token_num
        while True:
            end = max(0, len(ids) - k)
            prefix = self.codec.decode(ids[:end]) if end > 0 else ""
            if REPLACEMENT_CHAR not in prefix:
                return prefix
            if end == 0:
                return ""
            k += 1

    def _decode_chunk(self, chunk: np.ndarray, state: ASRStreamingState) -> None:
        if state.audio_accum.shape[0] == 0:
            state.audio_accum = chunk
        else:
            state.audio_accum = np.concatenate([state.audio_accum, chunk])

        prefix = (
            ""
            if state.chunk_id < state.unfixed_chunk_num
            else self._rollback_prefix(state._raw_decoded)
        )
        prompt = state.prompt_raw + prefix
        generated = self.decode_fn(prompt, state.audio_accum)
        state._raw_decoded = (prefix + generated) if prefix is not None else generated
        state.language, state.text = parse_stream_output(
            state._raw_decoded, user_language=state.force_language
        )
        state.chunk_id += 1

    def push(self, pcm16k: np.ndarray, state: ASRStreamingState) -> ASRStreamingState:
        """Append audio and decode every full chunk that becomes available."""
        audio = self._as_float32(pcm16k)
        if audio.shape[0] > 0:
            if state.buffer.shape[0] == 0:
                state.buffer = audio
            else:
                state.buffer = np.concatenate([state.buffer, audio])

        while state.buffer.shape[0] >= state.chunk_size_samples:
            chunk = state.buffer[: state.chunk_size_samples]
            state.buffer = state.buffer[state.chunk_size_samples :]
            self._decode_chunk(chunk, state)
        return state

    def finish(self, state: ASRStreamingState) -> ASRStreamingState:
        """Flush any remaining buffered audio (no padding)."""
        if state.buffer is None or state.buffer.shape[0] == 0:
            return state
        tail = state.buffer
        state.buffer = np.zeros(0, dtype=np.float32)
        self._decode_chunk(tail, state)
        return state


class LSPState:
    """Streaming state for Longest-Stable-Prefix emission (Stage 4, T4.4)."""


def emit_stable(state):
    """Return ``(committed, pending)`` where committed text never changes.

    Stage 4 / T4.4.
    """
    raise NotImplementedError("Stage 4, T4.4")
