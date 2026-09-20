"""Shared pytest fixtures for the asr1bit test suite."""

import wave

import numpy as np
import pytest

SAMPLE_RATE = 16000


@pytest.fixture
def sample_rate():
    return SAMPLE_RATE


@pytest.fixture
def write_wav(tmp_path):
    """Return a factory that writes a mono 16-bit PCM wav and returns its path."""

    def _write(name, audio, sr=SAMPLE_RATE):
        path = tmp_path / name
        pcm = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype("<i2")
        with wave.open(str(path), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(sr)
            fh.writeframes(pcm16.tobytes())
        return path

    return _write


@pytest.fixture
def sine_wave():
    """Return a factory producing a mono float32 sine wave."""

    def _sine(duration_sec, freq=440.0, sr=SAMPLE_RATE, amplitude=0.5):
        t = np.arange(round(duration_sec * sr), dtype=np.float32) / sr
        return (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)

    return _sine
