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

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

SAMPLE_RATE = 16000
ASR_TEXT_TAG = "<asr_text>"
LANG_PREFIX = "language "
REPLACEMENT_CHAR = "\ufffd"

WORD_RE = re.compile(r"\S+")


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


def rollback_prefix(codec: TokenCodec, text: str, unfixed_token_num: int) -> str:
    """Drop the last ``unfixed_token_num`` tokens of ``text`` (official rollback).

    Mirrors the official ``streaming_transcribe`` prefix policy: encode the
    accumulated decode, drop the trailing ``unfixed_token_num`` tokens and
    decode the rest. The rollback is extended token-by-token while the decoded
    prefix would end inside a byte-BPE token (replacement character), so the
    returned prefix is always decodable text.
    """
    if not text:
        return ""
    ids = list(codec.encode(text))
    k = unfixed_token_num
    while True:
        end = max(0, len(ids) - k)
        prefix = codec.decode(ids[:end]) if end > 0 else ""
        if REPLACEMENT_CHAR not in prefix:
            return prefix
        if end == 0:
            return ""
        k += 1


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
        return rollback_prefix(self.codec, text, self.unfixed_token_num)

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


def common_prefix_len(a: str, b: str) -> int:
    """Length of the longest common prefix of two strings."""
    limit = min(len(a), len(b))
    index = 0
    while index < limit and a[index] == b[index]:
        index += 1
    return index


def word_safe_cut(text: str, limit: int, margin_words: int = 0) -> int:
    """Largest word-boundary cut ``<= limit`` leaving ``margin_words`` words out.

    The returned cut never splits a word, and it absorbs the whitespace that
    follows the last included word so ``pending`` cannot glue onto ``committed``.
    """
    limit = max(0, min(int(limit), len(text)))
    spans = [(m.start(), m.end()) for m in WORD_RE.finditer(text) if m.end() <= limit]
    keep = max(0, len(spans) - max(0, int(margin_words)))
    if keep == 0:
        return 0
    cut = spans[keep - 1][1]
    while cut < len(text) and text[cut].isspace():
        cut += 1
    return cut


@dataclass
class LSPState:
    """Longest-Stable-Prefix emission state (Stage 4, T4.4).

    Emission policy: text is **committed** only once it has survived a revision
    — i.e. it lies inside the longest common prefix of two consecutive updates —
    and only up to a word boundary. The last ``margin_words`` words of the
    surviving text stay **pending** so they can still be revised (the streaming
    algorithm's rollback window is a few words wide; a margin of that size keeps
    normal revisions inside ``pending``).

    ``committed`` is append-only by construction: once text is committed it is
    never changed, even if the model later revises it. A revision that reaches
    into committed text is appended after the divergence point (or held if the
    text diverges from its very beginning), so the committed transcript stays a
    stable, never-revised stream.
    """

    margin_words: int = 0
    committed: str = ""
    pending: str = ""
    last_full: str = ""
    updates: int = 0
    revisions: int = 0
    commit_lengths: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.margin_words < 0:
            raise ValueError(f"margin_words must be >= 0, got {self.margin_words}")

    @property
    def text(self) -> str:
        """Current emitted text (``committed`` + ``pending``)."""
        return self.committed + self.pending

    def update(self, full_text: str) -> None:
        """Ingest the latest full transcript produced by the streamer."""
        full_text = str(full_text or "")
        previous = self.committed + self.pending
        if full_text != previous:
            self.updates += 1
            if not full_text.startswith(previous):
                self.revisions += 1

        keep = len(self.committed)
        if full_text.startswith(self.committed):
            stable = min(common_prefix_len(previous, full_text), len(full_text))
            cut = max(keep, word_safe_cut(full_text, stable, self.margin_words))
            cut = min(cut, len(full_text))
            if cut > keep:
                self.commit_lengths.append(cut)
            self.committed = full_text[:cut]
            self.pending = full_text[cut:]
        else:
            # The model revised already-committed text: the append-only policy
            # keeps committed as-is. Re-anchor after the divergence point so the
            # stream keeps moving; if the text diverges from the very start,
            # hold the pending tail until the model re-converges.
            divergence = common_prefix_len(self.committed, full_text)
            if divergence > 0:
                self.pending = full_text[divergence:]
        self.last_full = full_text

    def flush(self) -> None:
        """Commit the pending tail — call when the stream ends (no revisions left)."""
        if self.pending:
            self.committed = self.text
            self.pending = ""
            self.commit_lengths.append(len(self.committed))


def update_lsp(state: LSPState, full_text: str) -> None:
    """Functional form of :meth:`LSPState.update`."""
    state.update(full_text)


def emit_stable(state: LSPState) -> tuple[str, str]:
    """Return ``(committed, pending)`` where committed text never changes.

    Stage 4 / T4.4.
    """
    return state.committed, state.pending


def flush_lsp(state: LSPState) -> str:
    """Flush the stream tail and return the final append-only transcript."""
    state.flush()
    return state.committed
