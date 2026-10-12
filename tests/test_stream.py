"""Tests for the chunked streaming engine (Stage 0, T0.3).

This replicates the official Qwen3-ASR streaming algorithm (per-chunk re-feed of
all accumulated audio + prefix rollback). Qwen3-ASR is pseudo-streaming and may
revise text, so we assert only that rollback matches the official algorithm —
the append-only invariant belongs to Stage 4 / T4.4.
"""

import numpy as np

from asr1bit.qwen.stream import (
    ChunkedStreamer,
    LSPState,
    WhitespaceTokenCodec,
    chunk_size_samples,
    emit_stable,
    flush_lsp,
    parse_stream_output,
    rollback_prefix,
    truncate_trailing_loop,
    update_lsp,
    word_safe_cut,
)

SR = 16000


class _ByteCodec:
    """Byte-level tokenizer: an incomplete UTF-8 sequence decodes to U+FFFD."""

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")


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


class TestRollbackPrefixFunction:
    def test_drops_last_tokens(self):
        codec = WhitespaceTokenCodec()
        assert rollback_prefix(codec, "one two three four", 2) == "one two"
        assert rollback_prefix(codec, "one two", 2) == ""
        assert rollback_prefix(codec, "", 2) == ""

    def test_large_rollback_disables_prefix(self):
        codec = WhitespaceTokenCodec()
        assert rollback_prefix(codec, "a b c", 9999) == ""

    def test_extends_rollback_to_avoid_partial_multibyte_token(self):
        codec = _ByteCodec()
        # "aé" -> [97, 195, 169]; rolling back one token would leave a bare
        # 0xC3 byte, so the guard extends the rollback to the last whole token.
        assert rollback_prefix(codec, "aé", 1) == "a"
        # A lone multi-byte character rolls back to nothing.
        assert rollback_prefix(codec, "é", 1) == ""


class TestWordSafeCut:
    def test_never_splits_a_word(self):
        assert word_safe_cut("hello world", 8) == 6  # "hello " + ws

    def test_respects_limit(self):
        assert word_safe_cut("hello world", 3) == 0
        assert word_safe_cut("hello world", 0) == 0

    def test_margin_words_held_back(self):
        assert word_safe_cut("alpha beta gamma", 16, margin_words=1) == 11
        assert word_safe_cut("alpha beta gamma", 16, margin_words=3) == 0

    def test_empty_text(self):
        assert word_safe_cut("", 10) == 0


class TestLSPState:
    def test_first_update_commits_nothing(self):
        state = LSPState()
        update_lsp(state, "hello world")
        assert state.committed == ""
        assert state.pending == "hello world"
        assert emit_stable(state) == ("", "hello world")

    def test_text_that_survives_an_update_is_committed(self):
        state = LSPState()
        update_lsp(state, "hello wor")
        update_lsp(state, "hello world")
        assert state.committed == "hello "
        assert state.pending == "world"

    def test_committed_is_append_only_under_revisions(self):
        state = LSPState(margin_words=1)
        texts = [
            "one two three",
            "one two thre",
            "one two three four",
            "one two three four five",
        ]
        committed = ""
        for text in texts:
            update_lsp(state, text)
            assert state.committed.startswith(committed)
            committed = state.committed
        assert state.revisions == 1  # only "one two thre" retracts text
        assert state.committed == "one two three "
        assert state.pending == "four five"

    def test_committed_always_ends_on_a_word_boundary(self):
        state = LSPState()
        for text in ["alpha", "alpha be", "alpha beta", "alpha beta gamma delta"]:
            update_lsp(state, text)
            assert state.committed == "" or state.committed.endswith(" ")
        assert state.committed == "alpha beta "

    def test_margin_words_are_held_back(self):
        state = LSPState(margin_words=1)
        update_lsp(state, "alpha beta gamma")
        update_lsp(state, "alpha beta gamma")
        assert state.committed == "alpha beta "
        assert state.pending == "gamma"

    def test_flush_commits_pending(self):
        state = LSPState(margin_words=1)
        update_lsp(state, "alpha beta gamma")
        update_lsp(state, "alpha beta gamma")
        assert flush_lsp(state) == "alpha beta gamma"
        assert state.pending == ""
        assert emit_stable(state) == ("alpha beta gamma", "")

    def test_word_boundary_guard_holds_partial_word(self):
        # A cut inside a word commits nothing past the previous boundary.
        assert word_safe_cut("alpha beta", 7) == 6
        state = LSPState()
        update_lsp(state, "alpha beta")
        update_lsp(state, "alpha beta")
        assert state.committed == "alpha beta"

    def test_revision_into_committed_text_is_held(self):
        state = LSPState(margin_words=1)
        update_lsp(state, "aaa bbb")
        update_lsp(state, "aaa bbb")
        assert state.committed == "aaa "
        update_lsp(state, "xxx yyy")  # diverges at the very start -> held
        assert state.committed == "aaa "
        assert state.text == "aaa bbb"
        update_lsp(state, "aaa bbb ccc")  # re-converges
        assert state.committed == "aaa "
        assert state.pending == "bbb ccc"

    def test_divergence_after_prefix_appends_new_tail(self):
        state = LSPState(margin_words=1)
        update_lsp(state, "the cat sat")
        update_lsp(state, "the cat sat")
        assert state.committed == "the cat "
        update_lsp(state, "the dog ran")  # revision past the committed prefix
        assert state.committed == "the cat "
        assert state.pending == "dog ran"

    def test_rejects_negative_margin(self):
        import pytest

        with pytest.raises(ValueError):
            LSPState(margin_words=-1)


class TestTruncateTrailingLoop:
    def test_cuts_subword_loops(self):
        # One occurrence of the looped unit is kept (never delete everything).
        cleaned, looped = truncate_trailing_loop("WORDS WAS IT THEIR ILLILLILLILLILL")
        assert looped
        assert cleaned == "WORDS WAS IT THEIR ILL"

    def test_cuts_word_loops_keeping_the_first_occurrence(self):
        cleaned, looped = truncate_trailing_loop("HE NODS HIS CONSENT HIS HIS HIS HIS")
        assert looped
        assert cleaned == "HE NODS HIS CONSENT HIS"

    def test_cuts_repeated_phrases(self):
        cleaned, looped = truncate_trailing_loop("THE CAT SAT ON THE CAT SAT ON THE CAT SAT ON")
        assert looped
        assert cleaned.startswith("THE CAT SAT ON")

    def test_clean_text_is_untouched(self):
        assert truncate_trailing_loop("WORDS WAS IT THEIR COLOURS") == (
            "WORDS WAS IT THEIR COLOURS",
            False,
        )

    def test_few_repeats_are_tolerated(self):
        # Two repeats can be legitimate ("no no"); the policy needs min_repeats.
        assert truncate_trailing_loop("OH NO NO") == ("OH NO NO", False)

    def test_empty_text(self):
        assert truncate_trailing_loop("") == ("", False)


class TestLSPLoopPolicy:
    def test_looping_tail_is_never_committed(self):
        state = LSPState(truncate_loops=True)
        for text in [
            "HE NODS HIS",
            "HE NODS HIS CONSENT",
            "HE NODS HIS CONSENT ILLILLILL",
            "HE NODS HIS CONSENT ILLILLILLILLILLILL",
        ]:
            update_lsp(state, text)
        final = flush_lsp(state)
        assert final == "HE NODS HIS CONSENT ILL"  # transcript + one kept unit
        assert "ILLILL" not in final
        assert state.loop_truncations >= 1

    def test_loop_truncations_are_counted(self):
        state = LSPState(truncate_loops=True)
        update_lsp(state, "A B C D D D D")
        assert state.loop_truncations == 1
        assert emit_stable(state)[0] + emit_stable(state)[1] == "A B C D"

    def test_disabled_by_default(self):
        state = LSPState()
        update_lsp(state, "A B C D D D D")
        assert state.loop_truncations == 0
        assert state.text == "A B C D D D D"


class TestLSPWithStreamer:
    def test_chunk_task_stream_is_append_only(self):
        decoder = _StubDecoder(
            [
                "language English<asr_text>he nods",
                "language English<asr_text>he nods his",
                "language English<asr_text>he nods his consent",
                "language English<asr_text>he nods his consent",
            ]
        )
        streamer = _streamer(decoder, unfixed_chunk_num=5)
        state = streamer.init_state(prompt_raw="P")
        lsp = LSPState()
        committed = ""
        for _ in range(4):
            streamer.push(np.zeros(2 * SR, dtype=np.float32), state)
            update_lsp(lsp, state.text)
            assert lsp.committed.startswith(committed)
            committed = lsp.committed
        assert flush_lsp(lsp) == "he nods his consent"
