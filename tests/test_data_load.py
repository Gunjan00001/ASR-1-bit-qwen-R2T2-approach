"""Tests for the LibriSpeech data loader (Stage 0, T0.2)."""

import io
import json

import numpy as np
import pytest
import soundfile as sf

from asr1bit.data.load import (
    Utterance,
    _librispeech_parquet_pattern,
    iter_librispeech,
    iter_utterances,
    load_audio,
    load_manifest,
    select_subsample,
    write_manifest,
)


class TestLoadAudio:
    def test_reads_mono_float32(self, write_wav, sine_wave, sample_rate):
        path = write_wav("a.wav", sine_wave(0.5))
        audio = load_audio(path)
        assert audio.dtype == np.float32
        assert audio.ndim == 1
        assert audio.shape[0] == sample_rate // 2

    def test_resamples_to_16k(self, write_wav, sine_wave):
        path = write_wav("a8k.wav", sine_wave(1.0, sr=8000), sr=8000)
        audio = load_audio(path, target_sr=16000)
        assert audio.dtype == np.float32
        assert abs(audio.shape[0] - 16000) <= 2

    def test_downmixes_stereo(self, tmp_path):
        sr = 16000
        left = np.full(sr, 0.5, dtype=np.float32)
        right = np.full(sr, -0.25, dtype=np.float32)
        path = tmp_path / "stereo.wav"
        sf.write(str(path), np.stack([left, right], axis=1), sr)
        audio = load_audio(path)
        assert audio.ndim == 1
        assert abs(float(audio.mean()) - 0.125) < 1e-3

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_audio(tmp_path / "nope.wav")


class TestManifest:
    def test_round_trip(self, tmp_path):
        records = [
            {"id": "u1", "audio": "a.wav", "text": "Hello, World!"},
            {"id": "u2", "audio": "b.wav", "text": "Second one."},
        ]
        path = tmp_path / "m.jsonl"
        write_manifest(records, path)
        loaded = load_manifest(path)
        assert loaded == records

    def test_writes_one_json_object_per_line(self, tmp_path):
        records = [{"id": "u1", "audio": "a.wav", "text": "hi"}]
        path = tmp_path / "m.jsonl"
        write_manifest(records, path)
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["id"] == "u1"

    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_manifest(tmp_path / "nope.jsonl")


class TestIterUtterances:
    def test_decodes_audio_and_keeps_raw_text(self, tmp_path, write_wav, sine_wave):
        path = write_wav("u1.wav", sine_wave(0.25))
        records = [{"id": "u1", "audio": str(path), "text": "Hello, World!"}]
        utterances = list(iter_utterances(records))
        assert len(utterances) == 1
        utt = utterances[0]
        assert isinstance(utt, Utterance)
        assert utt.id == "u1"
        assert utt.text == "Hello, World!"  # raw transcript
        assert utt.reference == "hello world"  # shared normalizer
        assert utt.audio.dtype == np.float32
        assert abs(utt.duration - 0.25) < 1e-3


class _StubDataset:
    def __init__(self, records):
        self._records = records

    def __iter__(self):
        return iter(self._records)

    def __len__(self):
        return len(self._records)

    def select(self, indices):
        return _StubDataset([self._records[i] for i in indices])


def _hf_record(uid, text, sr=16000, seconds=1.0):
    audio = np.zeros(int(sr * seconds), dtype=np.float32)
    return {"id": uid, "text": text, "audio": {"array": audio, "sampling_rate": sr}}


def _hf_record_bytes(uid, text, sr=16000, seconds=0.5):
    audio = np.zeros(int(sr * seconds), dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV")
    return {"id": uid, "text": text, "audio": {"bytes": buffer.getvalue(), "path": None}}


class TestLibrispeechPattern:
    def test_pattern_targets_only_requested_split(self):
        pattern = _librispeech_parquet_pattern("clean", "test")
        assert pattern == "hf://datasets/openslr/librispeech_asr/clean/test/*.parquet"

    def test_pattern_for_other(self):
        assert _librispeech_parquet_pattern("other", "test").endswith(
            "openslr/librispeech_asr/other/test/*.parquet"
        )


class TestIterLibrispeech:
    def test_yields_normalized_records(self):
        dataset = _StubDataset([_hf_record("1", "Hello, World!")])
        utterances = list(iter_librispeech("clean", "test", dataset=dataset))
        assert len(utterances) == 1
        assert utterances[0].id == "1"
        assert utterances[0].text == "Hello, World!"
        assert utterances[0].reference == "hello world"
        assert utterances[0].audio.dtype == np.float32

    def test_resamples_non_16k_audio(self):
        dataset = _StubDataset([_hf_record("1", "hi", sr=8000, seconds=1.0)])
        utterances = list(iter_librispeech("clean", "test", dataset=dataset))
        assert abs(utterances[0].audio.shape[0] - 16000) <= 2

    def test_limit_restricts_count(self):
        dataset = _StubDataset([_hf_record(str(i), "hi") for i in range(10)])
        utterances = list(iter_librispeech("clean", "test", dataset=dataset, limit=3))
        assert len(utterances) == 3

    def test_indices_selects_rows(self):
        dataset = _StubDataset([_hf_record(str(i), "hi") for i in range(10)])
        utterances = list(iter_librispeech("clean", "test", dataset=dataset, indices=[0, 3, 5]))
        assert [u.id for u in utterances] == ["0", "3", "5"]

    def test_decodes_encoded_audio_bytes(self):
        dataset = _StubDataset([_hf_record_bytes("1", "Hello, World!")])
        utterances = list(iter_librispeech("clean", "test", dataset=dataset))
        assert utterances[0].audio.dtype == np.float32
        assert abs(utterances[0].duration - 0.5) < 1e-3
        assert utterances[0].reference == "hello world"

    def test_sample_n_is_deterministic(self):
        def fresh():
            return _StubDataset([_hf_record(str(i), "hi") for i in range(100)])

        first = list(iter_librispeech("clean", "test", dataset=fresh(), sample_n=5, sample_seed=1))
        second = list(iter_librispeech("clean", "test", dataset=fresh(), sample_n=5, sample_seed=1))
        assert [u.id for u in first] == [u.id for u in second]
        assert len(first) == 5


class TestSelectSubsample:
    def test_returns_all_when_n_exceeds_length(self):
        items = list(range(5))
        assert select_subsample(items, 10, seed=0) == items

    def test_deterministic_for_seed(self):
        items = list(range(100))
        assert select_subsample(items, 10, seed=7) == select_subsample(items, 10, seed=7)

    def test_size_and_membership(self):
        items = list(range(100))
        sample = select_subsample(items, 10, seed=1)
        assert len(sample) == 10
        assert all(x in items for x in sample)

    def test_sorted_for_stable_output(self):
        items = list(range(100))
        sample = select_subsample(items, 10, seed=2)
        assert sample == sorted(sample)
