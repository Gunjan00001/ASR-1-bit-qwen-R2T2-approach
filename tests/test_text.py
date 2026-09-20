"""Tests for the shared text normalizer (Stage 0, T0.2).

Qwen3-ASR does not publish its evaluation normalizer. The published LibriSpeech
numbers follow the common convention (lowercase, strip punctuation) used by
Whisper and the Open ASR Leaderboard, so that is our default mode. The Stage 0
gate validates it against the published numbers.
"""

import pytest

from asr1bit.text import (
    MODES,
    collapse_whitespace,
    normalize_text,
    remove_punctuation,
    to_lowercase,
)


class TestBuildingBlocks:
    def test_to_lowercase(self):
        assert to_lowercase("Hello WORLD") == "hello world"

    def test_remove_punctuation_deletes_chars(self):
        assert remove_punctuation("Hello, world!") == "Hello world"

    def test_remove_punctuation_removes_hyphen_without_space(self):
        assert remove_punctuation("well-known") == "wellknown"

    def test_collapse_whitespace(self):
        assert collapse_whitespace("a   b\n\tc") == "a b c"


class TestNormalize:
    def test_lowercases_and_strips_punctuation(self):
        assert normalize_text("Hello, world!") == "hello world"

    def test_removes_apostrophe(self):
        assert normalize_text("Don't stop") == "dont stop"

    def test_removes_unicode_punctuation(self):
        assert normalize_text("你好，世界。") == "你好世界"

    def test_collapses_and_strips_whitespace(self):
        assert normalize_text("  a   b  ") == "a b"

    def test_preserves_digits(self):
        assert normalize_text("Chapter 12") == "chapter 12"

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_none_becomes_empty(self):
        assert normalize_text(None) == ""

    def test_idempotent(self):
        text = "  Well-known, isn't it?! "
        once = normalize_text(text)
        assert normalize_text(once) == once

    def test_identity_mode_is_noop(self):
        text = "  Hello, World! "
        assert normalize_text(text, mode="identity") == text

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            normalize_text("hi", mode="does-not-exist")

    def test_default_mode_is_librispeech(self):
        assert "librispeech" in MODES
