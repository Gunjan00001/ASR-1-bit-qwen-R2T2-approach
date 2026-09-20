"""Tests for latency / RTF metrics (Stage 0, T0.4)."""

import pytest

from asr1bit.eval.latency import (
    chunk_latency_stats,
    coverage_fraction,
    estimate_wall_seconds,
    recommend_subsample,
    rtf,
)


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


class TestThroughput:
    def test_estimate_wall_seconds(self):
        assert estimate_wall_seconds(mean_utt_wall_sec=2.0, n_utterances=100) == 200.0

    def test_coverage_fraction(self):
        assert coverage_fraction(total_sec=6 * 3600, budget_hours=12) == 0.5

    def test_coverage_capped_at_one(self):
        assert coverage_fraction(total_sec=20 * 3600, budget_hours=12) == 1.0

    def test_coverage_zero_budget(self):
        assert coverage_fraction(total_sec=10.0, budget_hours=0) == 0.0

    def test_recommend_subsample_scales_with_full_size(self):
        assert recommend_subsample(full_n=2620, coverage=0.807, safety=0.8) == int(2620 * 0.807 * 0.8)

    def test_recommend_subsample_full_when_fits(self):
        assert recommend_subsample(full_n=2620, coverage=1.0) == 2620
