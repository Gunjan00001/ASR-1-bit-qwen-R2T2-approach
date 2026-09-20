"""Tests for WER/CER scoring (Stage 0, T0.4)."""

import pytest

from asr1bit.eval.wer import corpus_cer, corpus_wer, wer


class TestWer:
    def test_identical_is_zero(self):
        assert wer("hello world", "hello world") == 0.0

    def test_substitution(self):
        # 1 substitution over 2 reference words
        assert wer("hello world", "hello there") == 50.0

    def test_insertion(self):
        assert wer("hello", "hello world") == 100.0

    def test_deletion(self):
        assert wer("hello world", "hello") == 50.0

    def test_empty_hypothesis(self):
        assert wer("a b c", "") == 100.0

    def test_empty_both_is_zero(self):
        assert wer("", "") == 0.0

    def test_normalizes_by_default(self):
        assert wer("Hello, world!", "hello world") == 0.0

    def test_raw_mode_keeps_punctuation(self):
        assert wer("Hello", "hello", normalize=False) != 0.0

    def test_cer_via_wer_on_chars(self):
        assert wer("abc", "abd", unit="char") == pytest.approx(100.0 / 3.0)


class TestCorpus:
    def test_aggregates_over_words(self):
        refs = ["hello world", "good morning"]
        hyps = ["hello world", "good evening"]
        assert corpus_wer(refs, hyps) == 25.0

    def test_skips_normalization_when_disabled(self):
        assert corpus_wer(["Hello"], ["hello"], normalize=False) != 0.0

    def test_corpus_cer(self):
        assert corpus_cer(["abc"], ["abd"]) == pytest.approx(100.0 / 3.0)
