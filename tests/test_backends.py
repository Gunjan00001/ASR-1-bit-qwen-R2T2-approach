"""Tests for the Qwen3-ASR backend adapter (Stage 0, T0.3).

The actual models need a GPU and `qwen-asr`; these tests use fakes that mimic
the official wrapper API so the adapter logic (trace extraction, prompt
building, timing) is verified without a GPU.
"""

import types

import numpy as np
import pytest

from asr1bit.qwen.backends import (
    SAMPLE_RATE,
    StreamTrace,
    TokenizerCodec,
    build_transformers_decode_fn,
    environment_report,
    stream_transcribe,
    stream_transcribe_official,
)


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.calls = []

    def __call__(self, text, audio, return_tensors=None, padding=None):
        self.calls.append({"text": text, "audio": audio})
        return _FakeInputs(7)

    def batch_decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return [" continued"]


class _FakeTokenizer:
    def encode(self, text):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)


class _FakeInputs:
    def __init__(self, n):
        self._n = n

    def to(self, *args, **kwargs):
        return self

    def __getitem__(self, key):
        return self

    def keys(self):
        return []

    @property
    def shape(self):
        return (1, self._n)


class _FakeInner:
    device = "cpu"
    dtype = "float32"

    def __init__(self):
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return types.SimpleNamespace(sequences=np.zeros((1, 9), dtype=int))


class _FakeTransformersModel:
    def __init__(self):
        self.processor = _FakeProcessor()
        self.model = _FakeInner()
        self.max_new_tokens = 32


class _FakeOfficialModel:
    """Mimics the official wrapper: buffers audio and decodes per full chunk."""

    def __init__(self, step_samples):
        self._step = step_samples
        self.calls = 0
        self.finished = 0
        self.init_kwargs = None
        self._chunk_samples = 2 * SAMPLE_RATE
        self._buffer = np.zeros(0, dtype=np.float32)

    def init_streaming_state(self, **kwargs):
        self.init_kwargs = kwargs
        self._chunk_samples = round(kwargs.get("chunk_size_sec", 2.0) * SAMPLE_RATE)
        self._buffer = np.zeros(0, dtype=np.float32)
        return types.SimpleNamespace(text="", language="")

    def streaming_transcribe(self, pcm, state):
        pcm = np.asarray(pcm)
        assert pcm.dtype == np.float32
        self._buffer = np.concatenate([self._buffer, pcm])
        while self._buffer.shape[0] >= self._chunk_samples:
            self._buffer = self._buffer[self._chunk_samples :]
            self.calls += 1
            state.text = f"chunk {self.calls}"
            state.language = "English"
        return state

    def finish_streaming_transcribe(self, state):
        self.finished += 1
        state.text = state.text + " done"
        return state


class TestEnvironmentReport:
    def test_reports_expected_keys(self):
        report = environment_report()
        assert "python" in report
        assert "torch" in report
        assert "vllm" in report
        assert "qwen_asr" in report
        assert "device" in report

    def test_reports_torch_locally(self):
        report = environment_report()
        assert report["torch"] is not None
        assert "cuda_available" in report


class TestTokenizerCodec:
    def test_delegates_to_tokenizer(self):
        codec = TokenizerCodec(_FakeTokenizer())
        assert codec.encode("a b c") == ["a", "b", "c"]
        assert codec.decode(["a", "b"]) == "a b"


class TestBuildTransformersDecodeFn:
    def test_returns_continuation_and_passes_prompt_audio(self):
        model = _FakeTransformersModel()
        decode_fn = build_transformers_decode_fn(model, max_new_tokens=5)
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        out = decode_fn("PROMPT", audio)
        assert out == " continued"
        call = model.processor.calls[0]
        assert call["text"] == ["PROMPT"]
        assert np.asarray(call["audio"][0]).shape[0] == SAMPLE_RATE
        assert model.model.generate_kwargs["max_new_tokens"] == 5


class TestStreamTrace:
    def test_rtf(self):
        trace = StreamTrace(text="x", total_decode_sec=2.0, audio_sec=4.0)
        assert trace.rtf == 0.5

    def test_rtf_infinite_for_zero_audio(self):
        trace = StreamTrace(text="", total_decode_sec=1.0, audio_sec=0.0)
        assert trace.rtf == float("inf")


class TestOfficialStreaming:
    def test_extracts_trace_and_flushes(self):
        model = _FakeOfficialModel(SAMPLE_RATE)
        audio = np.zeros(4 * SAMPLE_RATE, dtype=np.float32)
        trace = stream_transcribe_official(model, audio, chunk_size_sec=2.0, step_sec=1.0)
        assert model.calls == 2
        assert model.finished == 1
        assert trace.text == "chunk 2 done"
        assert trace.chunks == ["chunk 1", "chunk 2", "chunk 2 done"]
        assert trace.audio_sec == pytest.approx(4.0)
        assert model.init_kwargs["chunk_size_sec"] == 2.0


class TestDispatch:
    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            stream_transcribe(object(), np.zeros(1, dtype=np.float32), mode="nope")

    def test_official_mode_dispatches(self):
        model = _FakeOfficialModel(SAMPLE_RATE)
        audio = np.zeros(2 * SAMPLE_RATE, dtype=np.float32)
        trace = stream_transcribe(model, audio, mode="official", chunk_size_sec=2.0)
        assert trace.text == "chunk 1 done"
