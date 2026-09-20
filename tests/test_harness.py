"""Tests for the baseline harness (Stage 0, T0.3-T0.4)."""

import json

import numpy as np
import pytest

from asr1bit.data.load import Utterance
from asr1bit.eval.harness import (
    WERReport,
    filter_runs,
    run_baseline,
    stage0_runs,
    subsample_offline_runs,
)
from asr1bit.qwen.backends import StreamTrace

SR = 16000


class _FakeEngine:
    def __init__(self, text="hello world", decode_sec=0.5):
        self.text = text
        self.decode_sec = decode_sec
        self.offline_calls = 0
        self.stream_calls = 0
        self.stream_kwargs = None

    def versions(self):
        return {"torch": "test-version"}

    def offline(self, audio):
        self.offline_calls += 1
        return self.text

    def stream(self, audio, chunk_size_sec, step_sec):
        self.stream_calls += 1
        self.stream_kwargs = {"chunk_size_sec": chunk_size_sec, "step_sec": step_sec}
        return StreamTrace(
            text=self.text,
            chunks=[self.text],
            total_decode_sec=self.decode_sec,
            audio_sec=len(audio) / float(SR),
        )


def _utterances(n=2):
    return [
        Utterance(
            id=str(i),
            audio=np.zeros(SR, dtype=np.float32),
            text="Hello, World!",
            reference="hello world",
            duration=1.0,
        )
        for i in range(n)
    ]


class TestOffline:
    def test_report_fields(self):
        engine = _FakeEngine()
        report = run_baseline("m", "offline", 2000, utterances=_utterances(), engine=engine)
        assert isinstance(report, WERReport)
        assert report.mode == "offline"
        assert report.chunk_ms == 2000
        assert report.n_utterances == 2
        assert report.wer == 0.0
        assert report.versions == {"torch": "test-version"}
        assert report.subsampled is False
        assert engine.offline_calls == 2


class TestStreaming:
    def test_wer_and_rtf(self):
        engine = _FakeEngine()
        report = run_baseline("m", "streaming", 2000, utterances=_utterances(2), engine=engine)
        assert report.wer == 0.0
        assert report.rtf == pytest.approx(0.5)
        assert report.mean_latency_sec == pytest.approx(0.5)

    def test_chunk_ms_converted_to_seconds(self):
        engine = _FakeEngine()
        run_baseline("m", "streaming", 320, utterances=_utterances(1), engine=engine)
        assert engine.stream_kwargs["chunk_size_sec"] == pytest.approx(0.32)

    def test_subsampled_flagged(self):
        engine = _FakeEngine()
        report = run_baseline(
            "m", "streaming", 2000, utterances=_utterances(1), engine=engine, subsampled=True
        )
        assert report.subsampled is True


class TestResultsPersistence:
    def test_writes_per_utterance_results(self, tmp_path):
        path = tmp_path / "results.jsonl"
        run_baseline(
            "m", "offline", 2000, utterances=_utterances(2), engine=_FakeEngine(), results_path=path
        )
        rows = [json.loads(line) for line in path.read_text().strip().splitlines()]
        assert len(rows) == 2
        assert {row["id"] for row in rows} == {"0", "1"}
        assert rows[0]["hypothesis"] == "hello world"

    def test_resume_skips_completed(self, tmp_path):
        path = tmp_path / "results.jsonl"
        run_baseline(
            "m", "offline", 2000, utterances=_utterances(2), engine=_FakeEngine(), results_path=path
        )
        second = _FakeEngine()
        report = run_baseline(
            "m", "offline", 2000, utterances=_utterances(2), engine=second, results_path=path
        )
        assert second.offline_calls == 0
        assert report.n_utterances == 2


class TestErrors:
    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            run_baseline("m", "nope", 2000, utterances=_utterances(1), engine=_FakeEngine())


class TestStage0Plan:
    def test_plan_size_and_shape(self):
        # per model: offline clean/other (full), streaming clean/other @2.0s,
        # plus 320ms clean+other subsampled = 6
        runs = stage0_runs(["a", "b"])
        assert len(runs) == 12
        keys = {"model_id", "mode", "config", "split", "chunk_ms", "subsampled", "limit"}
        assert all(keys <= set(run) for run in runs)

    def test_streaming_clean_is_full(self):
        runs = stage0_runs(["a"])
        clean = next(r for r in runs if r["mode"] == "streaming" and r["config"] == "clean")
        assert clean["subsampled"] is False
        assert clean["limit"] is None

    def test_streaming_other_is_subsampled(self):
        runs = stage0_runs(["a"])
        other = next(
            r
            for r in runs
            if r["mode"] == "streaming" and r["config"] == "other" and r["chunk_ms"] == 2000
        )
        assert other["subsampled"] is True
        assert other["limit"] == 500

    def test_extra_chunk_size_present_for_other(self):
        runs = stage0_runs(["a"])
        assert any(r["chunk_ms"] == 320 and r["config"] == "other" for r in runs)

    def test_extra_chunk_size_present_for_clean_subsampled(self):
        # primary metric is streaming 320ms clean, on the fixed labelled subsample
        runs = stage0_runs(["a"])
        clean320 = next(
            r for r in runs if r["chunk_ms"] == 320 and r["config"] == "clean" and r["mode"] == "streaming"
        )
        assert clean320["subsampled"] is True
        assert clean320["limit"] == 500


class TestFilterRuns:
    def test_none_returns_all(self):
        runs = stage0_runs(["a"])
        assert filter_runs(runs) == runs

    def test_filter_by_mode(self):
        runs = stage0_runs(["a", "b"])
        offline = filter_runs(runs, mode="offline")
        assert offline
        assert all(r["mode"] == "offline" for r in offline)

    def test_filter_by_config(self):
        runs = stage0_runs(["a"])
        other = filter_runs(runs, config="other")
        assert other
        assert all(r["config"] == "other" for r in other)

    def test_filter_by_mode_and_config(self):
        runs = stage0_runs(["a"])
        result = filter_runs(runs, mode="streaming", config="other")
        assert result
        assert all(r["mode"] == "streaming" and r["config"] == "other" for r in result)

    def test_filter_by_chunk_ms(self):
        runs = stage0_runs(["a"])
        result = filter_runs(runs, chunk_ms=320)
        assert result
        assert all(r["chunk_ms"] == 320 for r in result)


class TestSubsampleOffline:
    def test_offline_runs_become_subsampled(self):
        out = subsample_offline_runs(stage0_runs(["a"]), 500, seed=0)
        for run in out:
            if run["mode"] == "offline":
                assert run["limit"] == 500
                assert run["sample_n"] == 500
                assert run["subsampled"] is True

    def test_streaming_runs_untouched(self):
        out = subsample_offline_runs(stage0_runs(["a"]), 500)
        streaming = next(r for r in out if r["mode"] == "streaming" and r["config"] == "clean")
        assert streaming["subsampled"] is False
        assert streaming["limit"] is None

    def test_zero_is_noop_and_input_not_mutated(self):
        runs = stage0_runs(["a"])
        assert subsample_offline_runs(runs, 0) == runs
        subsample_offline_runs(runs, 500)
        assert runs[0]["limit"] is None
