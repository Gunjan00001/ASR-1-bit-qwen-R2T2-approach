"""Tests for the baseline harness (Stage 0, T0.3-T0.4)."""

import json

import numpy as np
import pytest

from asr1bit.data.load import Utterance
from asr1bit.eval.harness import WERReport, run_baseline, stage0_runs
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
        runs = stage0_runs(["a", "b"])
        assert len(runs) == 10
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

    def test_extra_chunk_size_present(self):
        runs = stage0_runs(["a"])
        assert any(r["chunk_ms"] == 320 and r["config"] == "other" for r in runs)
