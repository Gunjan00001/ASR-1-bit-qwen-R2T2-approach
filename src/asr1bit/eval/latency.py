"""Retrospective chunk-wise latency and real-time factor (Stage 0, T0.4).

Official Qwen3-ASR streaming emits no timestamps, so latency is derived from
decode wall-times per chunk. Retrospective chunk latency is approximated as
``chunk_size_sec + mean(decode_sec)``: the text for a chunk is only available
after the chunk's audio is complete, plus the decode cost.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean, median


def rtf(total_decode_sec: float, audio_sec: float) -> float:
    """Real-time factor: decode seconds per audio second."""
    if audio_sec <= 0:
        return float("inf")
    return total_decode_sec / audio_sec


def chunk_latency_stats(decode_secs: Sequence[float], chunk_size_sec: float) -> dict[str, float]:
    """Summarize per-chunk decode times.

    Returns a dict with ``n_chunks``, ``mean_decode_sec``,
    ``median_decode_sec``, ``max_decode_sec``, ``total_decode_sec`` and
    ``retrospective_latency_sec``.
    """
    n = len(decode_secs)
    if n == 0:
        return {
            "n_chunks": 0,
            "mean_decode_sec": 0.0,
            "median_decode_sec": 0.0,
            "max_decode_sec": 0.0,
            "total_decode_sec": 0.0,
            "retrospective_latency_sec": 0.0,
        }
    return {
        "n_chunks": n,
        "mean_decode_sec": float(mean(decode_secs)),
        "median_decode_sec": float(median(decode_secs)),
        "max_decode_sec": float(max(decode_secs)),
        "total_decode_sec": float(sum(decode_secs)),
        "retrospective_latency_sec": float(chunk_size_sec + mean(decode_secs)),
    }
