"""Qwen3-ASR backend adapter (Stage 0, T0.3).

Thin glue over the official ``qwen-asr`` toolkit. Heavy imports (torch, qwen_asr)
are deferred so the module can be imported and unit-tested without a GPU.

Two streaming paths are supported:

- ``official``: the vLLM backend's own ``streaming_transcribe`` (reference).
- ``portable``: :class:`~asr1bit.qwen.stream.ChunkedStreamer` driven by the
  transformers backend. This is the path that will later accept BitLinear
  models, which vLLM cannot serve.
"""

from __future__ import annotations

import importlib
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from asr1bit.qwen.stream import SAMPLE_RATE, ChunkedStreamer

__all__ = [
    "SAMPLE_RATE",
    "StreamTrace",
    "TokenizerCodec",
    "build_transformers_decode_fn",
    "environment_report",
    "load_model",
    "offline_transcribe",
    "stream_transcribe",
    "stream_transcribe_official",
    "stream_transcribe_portable",
]


@dataclass
class StreamTrace:
    """Text and timing captured from one streaming run."""

    text: str
    chunks: list[str] = field(default_factory=list)
    total_decode_sec: float = 0.0
    audio_sec: float = 0.0
    language: str = ""

    @property
    def rtf(self) -> float:
        """Real-time factor (decode seconds per audio second)."""
        if self.audio_sec <= 0:
            return float("inf")
        return self.total_decode_sec / self.audio_sec


class TokenizerCodec:
    """Adapt a Hugging Face tokenizer to the :class:`TokenCodec` protocol."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def encode(self, text: str):
        return self.tokenizer.encode(text)

    def decode(self, ids) -> str:
        return self.tokenizer.decode(ids)


def environment_report() -> dict[str, Any]:
    """Collect version/device facts to record alongside every result table."""
    report: dict[str, Any] = {"python": platform.python_version()}
    for name in ("torch", "transformers", "vllm", "qwen_asr"):
        try:
            module = importlib.import_module(name)
            report[name] = getattr(module, "__version__", "unknown")
        except Exception:  # noqa: BLE001 - report absence, don't crash
            report[name] = None
    try:
        import torch

        report["cuda_available"] = torch.cuda.is_available()
        report["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            report["device"] = torch.cuda.get_device_name(0)
        else:
            report["device"] = "cpu"
    except Exception:  # noqa: BLE001
        report.setdefault("device", None)
    return report


def load_model(
    model_id: str,
    backend: str = "transformers",
    *,
    dtype: Any = None,
    device: str | None = None,
    max_new_tokens: int = 1024,
    gpu_memory_utilization: float = 0.7,
):
    """Load a Qwen3-ASR model with the requested backend."""
    from qwen_asr import Qwen3ASRModel

    if backend == "vllm":
        return Qwen3ASRModel.LLM(
            model=model_id,
            gpu_memory_utilization=gpu_memory_utilization,
            max_new_tokens=32,
        )

    kwargs: dict[str, Any] = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if device is not None:
        kwargs["device_map"] = device
    return Qwen3ASRModel.from_pretrained(model_id, max_new_tokens=max_new_tokens, **kwargs)


def offline_transcribe(
    model: Any,
    audio: np.ndarray,
    language: str | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> str:
    """Run the official offline ``transcribe`` and return the text."""
    results = model.transcribe(audio=(np.asarray(audio, dtype=np.float32), sample_rate), language=language)
    return results[0].text


def build_transformers_decode_fn(model: Any, max_new_tokens: int | None = None) -> Callable[[str, np.ndarray], str]:
    """Build a ``decode_fn(prompt, audio) -> continuation`` for the transformers backend."""
    inner = model.model
    processor = model.processor
    limit = max_new_tokens if max_new_tokens is not None else getattr(model, "max_new_tokens", 512)

    def decode_fn(prompt: str, audio: np.ndarray) -> str:
        inputs = processor(
            text=[prompt],
            audio=[np.asarray(audio, dtype=np.float32)],
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(inner.device).to(inner.dtype)
        generated = inner.generate(**inputs, max_new_tokens=limit)
        prompt_len = inputs["input_ids"].shape[1]
        decoded = processor.batch_decode(
            generated.sequences[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0]

    return decode_fn


def _snap_text(state: Any, chunks: list[str]) -> tuple[str, str]:
    text = getattr(state, "text", "") or ""
    language = getattr(state, "language", "") or ""
    if text and (not chunks or chunks[-1] != text):
        chunks.append(text)
    return text, language


def stream_transcribe_official(
    model: Any,
    audio: np.ndarray,
    *,
    chunk_size_sec: float = 2.0,
    step_sec: float = 0.5,
    unfixed_chunk_num: int = 2,
    unfixed_token_num: int = 5,
    language: str | None = None,
) -> StreamTrace:
    """Stream with the official vLLM backend and capture a trace."""
    audio = np.asarray(audio, dtype=np.float32)
    state = model.init_streaming_state(
        unfixed_chunk_num=unfixed_chunk_num,
        unfixed_token_num=unfixed_token_num,
        chunk_size_sec=chunk_size_sec,
        language=language,
    )
    step = max(1, round(step_sec * SAMPLE_RATE))
    chunks: list[str] = []
    total = 0.0
    for position in range(0, audio.shape[0], step):
        started = time.perf_counter()
        model.streaming_transcribe(audio[position : position + step], state)
        total += time.perf_counter() - started
        _snap_text(state, chunks)
    started = time.perf_counter()
    model.finish_streaming_transcribe(state)
    total += time.perf_counter() - started
    text, language = _snap_text(state, chunks)
    return StreamTrace(
        text=text,
        chunks=chunks,
        total_decode_sec=total,
        audio_sec=audio.shape[0] / float(SAMPLE_RATE),
        language=language,
    )


def stream_transcribe_portable(
    model: Any,
    audio: np.ndarray,
    *,
    chunk_size_sec: float = 2.0,
    step_sec: float = 0.5,
    unfixed_chunk_num: int = 2,
    unfixed_token_num: int = 5,
    language: str | None = None,
) -> StreamTrace:
    """Stream with :class:`ChunkedStreamer` over the transformers backend."""
    audio = np.asarray(audio, dtype=np.float32)
    base_decode = build_transformers_decode_fn(model)
    durations: list[float] = []

    def timed_decode(prompt: str, accumulated: np.ndarray) -> str:
        started = time.perf_counter()
        result = base_decode(prompt, accumulated)
        durations.append(time.perf_counter() - started)
        return result

    streamer = ChunkedStreamer(
        timed_decode,
        TokenizerCodec(model.processor.tokenizer),
        chunk_size_sec=chunk_size_sec,
        unfixed_chunk_num=unfixed_chunk_num,
        unfixed_token_num=unfixed_token_num,
    )
    prompt_raw = model._build_text_prompt(context="", force_language=language)
    state = streamer.init_state(prompt_raw=prompt_raw, force_language=language)
    step = max(1, round(step_sec * SAMPLE_RATE))
    for position in range(0, audio.shape[0], step):
        streamer.push(audio[position : position + step], state)
    streamer.finish(state)
    return StreamTrace(
        text=state.text,
        chunks=[state.text] if durations else [],
        total_decode_sec=float(sum(durations)),
        audio_sec=audio.shape[0] / float(SAMPLE_RATE),
        language=state.language,
    )


def stream_transcribe(
    model: Any,
    audio: np.ndarray,
    mode: str = "official",
    **kwargs: Any,
) -> StreamTrace:
    """Dispatch to the official (vLLM) or portable (transformers) streamer."""
    if mode == "official":
        return stream_transcribe_official(model, audio, **kwargs)
    if mode in ("portable", "transformers"):
        return stream_transcribe_portable(model, audio, **kwargs)
    raise ValueError(f"Unknown streaming mode {mode!r}; expected 'official' or 'portable'")
