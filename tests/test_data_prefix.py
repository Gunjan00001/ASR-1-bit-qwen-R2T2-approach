"""Tests for the forced-alignment adapter and stable-prefix targets (Stage 4)."""

import numpy as np
import pytest

from asr1bit.data.align import AlignedWord, align_words, alignment_covers
from asr1bit.data.prefix import chunk_examples


def _word(text, start, end):
    return AlignedWord(text=text, start=start, end=end)


class TestChunkExamples:
    def test_words_split_across_chunks(self):
        words = [_word("ONE", 0.1, 0.4), _word("TWO", 0.5, 0.9), _word("THREE", 1.1, 1.5)]
        examples = chunk_examples(words, chunk_sec=0.5, total_sec=2.0)
        # chunk 0 hears ONE (ends 0.4 <= 0.5); chunk 1 hears TWO; chunk 2 hears
        # THREE; chunk 3 hears nothing (its empty target is a stop sample).
        assert [e.target_text for e in examples] == ["ONE", "TWO", "THREE", ""]
        assert [e.prefix_text for e in examples] == ["", "ONE", "ONE TWO", "ONE TWO THREE"]
        assert examples[0].heard_text == "ONE"
        assert examples[3].heard_text == "ONE TWO THREE"

    def test_audio_sec_grows_and_tail_uses_total(self):
        words = [_word("A", 0.1, 0.2), _word("B", 1.0, 1.1)]
        examples = chunk_examples(words, chunk_sec=0.5, total_sec=1.4)
        assert [e.audio_sec for e in examples] == [0.5, 1.0, 1.4]

    def test_prefix_is_append_only(self):
        words = [_word(f"W{i}", i * 0.3, i * 0.3 + 0.2) for i in range(6)]
        examples = chunk_examples(words, chunk_sec=0.7, total_sec=2.0)
        heard = ""
        for example in examples:
            assert example.heard_text.startswith(heard)
            assert heard.startswith(example.prefix_text)
            heard = example.heard_text

    def test_chunks_with_no_new_words_teach_the_stop(self):
        words = [_word("ONLY", 0.0, 0.1)]
        examples = chunk_examples(words, chunk_sec=0.5, total_sec=1.6)
        assert examples[0].target_text == "ONLY"
        assert all(e.target_text == "" for e in examples[1:])
        assert all(e.prefix_text == "ONLY" for e in examples[1:])

    def test_lookahead_pulls_words_forward(self):
        words = [_word("LATE", 0.6, 0.7)]
        without = chunk_examples(words, chunk_sec=0.5, total_sec=1.0)
        with_lookahead = chunk_examples(words, chunk_sec=0.5, lookahead_sec=0.25, total_sec=1.0)
        assert without[0].target_text == ""
        assert with_lookahead[0].target_text == "LATE"

    def test_words_ending_exactly_on_the_boundary_are_heard(self):
        words = [_word("EDGE", 0.0, 0.5)]
        examples = chunk_examples(words, chunk_sec=0.5, total_sec=1.0)
        assert examples[0].target_text == "EDGE"

    def test_unsorted_input_is_sorted(self):
        words = [_word("B", 0.6, 0.9), _word("A", 0.1, 0.3)]
        examples = chunk_examples(words, chunk_sec=0.5, total_sec=1.0)
        assert examples[1].heard_text == "A B"

    def test_empty_words(self):
        assert chunk_examples([], chunk_sec=0.5) == []

    def test_total_sec_extends_past_the_last_word(self):
        words = [_word("A", 0.0, 0.2)]
        examples = chunk_examples(words, chunk_sec=0.5, total_sec=2.0)
        assert len(examples) == 4
        assert examples[-1].audio_sec == 2.0

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            chunk_examples([], chunk_sec=0)
        with pytest.raises(ValueError):
            chunk_examples([_word("A", 0.0, 0.1)], chunk_sec=0.5, lookahead_sec=-1)


class TestAlignWords:
    def test_uses_the_provided_aligner_and_maps_items(self):
        class _Item:
            def __init__(self, text, start, end):
                self.text, self.start_time, self.end_time = text, start, end

        class _Result:
            def __init__(self):
                self.items = [_Item("HELLO", 0.0, 0.4), _Item("WORLD", 0.5, 0.9)]

        class _Aligner:
            def __init__(self):
                self.calls = []

            def align(self, audio, text, language):
                self.calls.append({"audio": audio, "text": text, "language": language})
                return [_Result()]

        aligner = _Aligner()
        audio = np.zeros(16000, dtype=np.float32)
        words = align_words(audio, "HELLO WORLD", aligner=aligner)
        assert [w.text for w in words] == ["HELLO", "WORLD"]
        assert words[0].start == 0.0 and words[1].end == 0.9
        call = aligner.calls[0]
        assert call["text"] == "HELLO WORLD"
        assert call["language"] == "English"
        assert call["audio"][0].shape[0] == 16000

    def test_empty_result(self):
        class _Aligner:
            def align(self, audio, text, language):
                return []

        assert align_words(np.zeros(10, dtype=np.float32), "X", aligner=_Aligner()) == []

    def test_alignment_covers(self):
        words = [_word("A", 0.0, 0.9), _word("B", 1.0, 2.5)]
        assert alignment_covers(words, 2.6)
        assert not alignment_covers(words, 4.0)
        assert not alignment_covers([], 1.0)
