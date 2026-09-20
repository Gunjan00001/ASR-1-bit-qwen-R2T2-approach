"""Tests for the chunked streaming engine (Stage 0, T0.3).

This replicates the official Qwen3-ASR streaming algorithm (per-chunk re-feed of
all accumulated audio + prefix rollback). Qwen3-ASR is pseudo-streaming and may
revise text, so we assert only that rollback matches the official algorithm —
the append-only invariant belongs to Stage 4 / T4.4.
"""

import numpy as np

from asr1bit.qwen.stream import (
    ChunkedStreamer,
    WhitespaceTokenCodec,
    chunk_size_samples,
    parse_stream_output,
)

SR = 16000


class _StubDecoder:
    """Records (prompt, audio_len) calls and replays scripted outputs."""

    def __init__(self, outputs=None):
        self.calls = []
        self._outputs = list(outputs or [])
        self._index = 0

    def __call__(self, prompt, audio):
        self.calls.append((prompt, int(np.asarray(audio).shape[0])))
        if self._index < len(self._outputs):
            output = self._outputs[self._index]
            self._index += 1
            return output
        return f"segment {len(self.calls)}"


def _streamer(decoder, **kwargs):
    kwargs.setdefault("chunk_size_sec", 2.0)
    return ChunkedStreamer(decoder, WhitespaceTokenCodec(), **kwargs)


class TestChunkSizing:
    def test_samples_for_ms(self):
        assert chunk_size_samples(320) == 5120
        assert chunk_size_samples(2000) == 32000

    def test_state_chunk_size_matches_seconds(self):
        streamer = _streamer(_StubDecoder(), chunk_size_sec=0.32)
        state = streamer.init_state()
        assert state.chunk_size_samples == 5120


class TestBuffering:
    def test_no_decode_before_full_chunk(self):
        decoder = _StubDecoder()
        streamer = _streamer(decoder)
        state = streamer.init_state(prompt_raw="P")
        for _ in range(3):
            streamer.push(np.zeros(SR // 2, dtype=np.float32), state)
        assert decoder.calls == []
        assert state.buffer.shape[0] == SR + SR // 2
        assert state.chunk_id == 0

    def test_one_decode_per_full_chunk(self):
        decoder = _StubDecoder()
        streamer = _streamer(decoder)
        state = streamer.init_state(prompt_raw="P")
        for _ in range(4):
            streamer.push(np.zeros(SR, dtype=np.float32), state)
        assert len(decoder.calls) == 2
        assert state.chunk_id == 2

    def test_refeeds_all_accumulated_audio(self):
        decoder = _StubDecoder()
        streamer = _streamer(decoder)
        state = streamer.init_state(prompt_raw="P")
        for _ in range(4):
            streamer.push(np.zeros(SR, dtype=np.float32), state)
        assert decoder.calls[0][1] == 2 * SR
        assert decoder.calls[1][1] == 4 * SR
        assert state.audio_accum.shape[0] == 4 * SR
        assert state.audio_accum.dtype == np.float32

    def test_accepts_int16_input(self):
        decoder = _StubDecoder()
        streamer = _streamer(decoder)
        state = streamer.init_state(prompt_raw="P")
        streamer.push(np.zeros(2 * SR, dtype=np.int16), state)
        assert len(decoder.calls) == 1
        assert state.audio_accum.dtype == np.float32


class TestPrefixRollback:
    def test_prefix_empty_for_first_unfixed_chunks(self):
        decoder = _StubDecoder(["a b c", "a b c d", "a b c d e"])
        streamer = _streamer(decoder, unfixed_chunk_num=2, unfixed_token_num=2)
        state = streamer.init_state(prompt_raw="P", force_language="English")
        for _ in range(6):
            streamer.push(np.zeros(SR, dtype=np.float32), state)
        assert decoder.calls[0][0] == "P"
        assert decoder.calls[1][0] == "P"
        assert decoder.calls[2][0] == "Pa b"

    def test_rollback_matches_official_algorithm(self):
        decoder = _StubDecoder(["one two three four five"])
        streamer = _streamer(decoder, unfixed_chunk_num=0, unfixed_token_num=2)
        state = streamer.init_state(prompt_raw="Q", force_language="English")
        streamer.push(np.zeros(2 * SR, dtype=np.float32), state)
        streamer.push(np.zeros(2 * SR, dtype=np.float32), state)
        assert decoder.calls[1][0] == "Qone two three"


class TestParsing:
    def test_parses_language_and_text(self):
        decoder = _StubDecoder(["language English<asr_text>hello world"])
        streamer = _streamer(decoder)
        state = streamer.init_state()
        streamer.push(np.zeros(2 * SR, dtype=np.float32), state)
        assert state.language == "English"
        assert state.text == "hello world"

    def test_forced_language_treats_raw_as_text(self):
        decoder = _StubDecoder(["hello there"])
        streamer = _streamer(decoder)
        state = streamer.init_state(force_language="English")
        streamer.push(np.zeros(2 * SR, dtype=np.float32), state)
        assert state.language == "English"
        assert state.text == "hello there"

    def test_parse_stream_output_no_tag(self):
        assert parse_stream_output("plain text") == ("", "plain text")

    def test_parse_stream_output_empty(self):
        assert parse_stream_output("") == ("", "")


class TestFinish:
    def test_finish_flushes_tail(self):
        decoder = _StubDecoder()
        streamer = _streamer(decoder)
        state = streamer.init_state(prompt_raw="P")
        streamer.push(np.zeros(3 * SR, dtype=np.float32), state)
        assert len(decoder.calls) == 1
        streamer.finish(state)
        assert len(decoder.calls) == 2
        assert decoder.calls[1][1] == 3 * SR
        assert state.buffer.shape[0] == 0

    def test_finish_without_tail_is_noop(self):
        decoder = _StubDecoder()
        streamer = _streamer(decoder)
        state = streamer.init_state(prompt_raw="P")
        streamer.push(np.zeros(2 * SR, dtype=np.float32), state)
        streamer.finish(state)
        assert len(decoder.calls) == 1


class TestTextRevision:
    def test_text_is_latest_decode_not_accumulated(self):
        # Qwen re-decodes from scratch each chunk, so text is replaced, not
        # appended. (Append-only is a Stage 4 invariant, not a T0.3 one.)
        decoder = _StubDecoder(
            [
                "language English<asr_text>alpha",
                "language English<asr_text>beta",
            ]
        )
        streamer = _streamer(decoder, unfixed_chunk_num=5)
        state = streamer.init_state(prompt_raw="P")
        streamer.push(np.zeros(2 * SR, dtype=np.float32), state)
        assert state.text == "alpha"
        streamer.push(np.zeros(2 * SR, dtype=np.float32), state)
        assert state.text == "beta"
