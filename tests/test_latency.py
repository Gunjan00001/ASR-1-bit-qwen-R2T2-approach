"""Tests for latency / RTF metrics (Stage 0, T0.4)."""

import pytest

from asr1bit.eval.latency import chunk_latency_stats, rtf


class TestRtf:
    def test_basic(self):
        assert rtf(total_decode_sec=2.0, audio_sec=4.0) == 0.5

    def test_zero_audio_is_infinite(self):
        assert rtf(total_decode_sec=1.0, audio_sec=0.0) == float("inf")


class TestChunkLatencyStats:
    def test_basic_stats(self):
        stats = chunk_latency_stats([0.1, 0.2, 0.3], chunk_size_sec=2.0)
        assert stats["n_chunks"] == 3
        assert stats["mean_decode_sec"] == pytest.approx(0.2)
        assert stats["median_decode_sec"] == pytest.approx(0.2)
        assert stats["max_decode_sec"] == pytest.approx(0.3)
        assert stats["total_decode_sec"] == pytest.approx(0.6)

    def test_retrospective_latency_is_chunk_plus_decode(self):
        stats = chunk_latency_stats([0.1, 0.3], chunk_size_sec=2.0)
        assert stats["retrospective_latency_sec"] == pytest.approx(2.2)

    def test_empty(self):
        stats = chunk_latency_stats([], chunk_size_sec=2.0)
        assert stats["n_chunks"] == 0
        assert stats["total_decode_sec"] == 0.0
        assert stats["retrospective_latency_sec"] == 0.0
