"""Config-matrix evaluation harness -> result tables (Stage 0 + 6).

Locked interface::

    run_baseline(model_id, mode, chunk_ms) -> WERReport

The heavy model engine is injectable (``engine=``) so the harness can be
unit-tested without a GPU. Per-utterance results are persisted to
``results_path`` and reused on re-run, which keeps Kaggle sessions resumable.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from asr1bit.data.load import Utterance, iter_librispeech
from asr1bit.eval.latency import chunk_latency_stats, rtf
from asr1bit.eval.wer import corpus_cer, corpus_wer
from asr1bit.qwen import backends

MODES = ("offline", "streaming")


@dataclass
class WERReport:
    """Container for WER, latency, RTF, and provenance."""

    model_id: str
    mode: str
    chunk_ms: int
    n_utterances: int
    wer: float
    cer: float
    rtf: float = 0.0
    mean_latency_sec: float = 0.0
    retrospective_latency_sec: float = 0.0
    subsampled: bool = False
    dataset: str = "openslr/librispeech_asr"
    config: str = "clean"
    split: str = "test"
    versions: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QwenEngine:
    """Adapter exposing the uniform offline/stream interface over qwen-asr."""

    def __init__(
        self,
        model_id: str,
        *,
        backend: str = "transformers",
        language: str | None = None,
        **load_kwargs: Any,
    ):
        self.model_id = model_id
        self.backend = backend
        self.language = language
        self.model = backends.load_model(model_id, backend=backend, **load_kwargs)
        self.stream_mode = "official" if backend == "vllm" else "portable"

    def versions(self) -> dict[str, Any]:
        return backends.environment_report()

    def offline(self, audio) -> str:
        return backends.offline_transcribe(self.model, audio, language=self.language)

    def stream(self, audio, chunk_size_sec: float, step_sec: float):
        return backends.stream_transcribe(
            self.model,
            audio,
            mode=self.stream_mode,
            chunk_size_sec=chunk_size_sec,
            step_sec=step_sec,
            language=self.language,
        )


def _load_completed(results_path: str | Path | None) -> dict[str, dict]:
    if results_path is None:
        return {}
    path = Path(results_path)
    if not path.exists():
        return {}
    completed: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            completed[str(row["id"])] = row
    return completed


def _append_row(results_path: str | Path | None, row: dict) -> None:
    if results_path is None:
        return
    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_baseline(
    model_id: str,
    mode: str,
    chunk_ms: int,
    *,
    utterances: Sequence[Utterance] | None = None,
    engine: Any | None = None,
    config: str = "clean",
    split: str = "test",
    limit: int | None = None,
    sample_n: int | None = None,
    sample_seed: int = 0,
    results_path: str | Path | None = None,
    subsampled: bool = False,
    step_ms: int = 500,
    language: str | None = None,
    backend: str = "transformers",
    notes: str = "",
    **load_kwargs: Any,
) -> WERReport:
    """Run an offline or streaming config and return a :class:`WERReport`.

    Args:
        model_id: Qwen3-ASR repo id (e.g. ``Qwen/Qwen3-ASR-0.6B``).
        mode: ``"offline"`` or ``"streaming"``.
        chunk_ms: Model chunk length in milliseconds.
        utterances: Optional pre-loaded utterances (skips dataset loading).
        engine: Optional engine overriding the default :class:`QwenEngine`.
        config: LibriSpeech config (``"clean"`` / ``"other"``).
        split: Dataset split.
        limit: Optional cap on utterances.
        sample_n: Optional deterministic subsample size (fixed via seed).
        sample_seed: Seed for ``sample_n`` selection.
        results_path: JSONL path for per-utterance persistence/resume.
        subsampled: Whether the evaluated set is a subsample (labelled in report).
        step_ms: Caller feed granularity for streaming.
        language: Optional forced language.
        backend: ``"transformers"`` or ``"vllm"`` for the default engine.
        notes: Free-form note attached to the report.
        **load_kwargs: Forwarded to the model loader.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")

    if utterances is None:
        utterances = list(
            iter_librispeech(config, split, limit=limit, sample_n=sample_n, sample_seed=sample_seed)
        )
    if engine is None:
        engine = QwenEngine(model_id, backend=backend, language=language, **load_kwargs)

    completed = _load_completed(results_path)
    references: list[str] = []
    hypotheses: list[str] = []
    decode_secs: list[float] = []
    audio_secs: list[float] = []

    for utt in utterances:
        previous = completed.get(utt.id)
        if previous is not None:
            hypothesis = previous["hypothesis"]
            decode_sec = float(previous.get("decode_sec", 0.0))
            audio_sec = float(previous.get("audio_sec", utt.duration))
        elif mode == "offline":
            hypothesis = engine.offline(utt.audio)
            decode_sec = 0.0
            audio_sec = utt.duration
        else:
            trace = engine.stream(
                utt.audio,
                chunk_size_sec=chunk_ms / 1000.0,
                step_sec=step_ms / 1000.0,
            )
            hypothesis = trace.text
            decode_sec = trace.total_decode_sec
            audio_sec = trace.audio_sec

        if previous is None:
            _append_row(
                results_path,
                {
                    "id": utt.id,
                    "text": utt.text,
                    "reference": utt.reference,
                    "hypothesis": hypothesis,
                    "decode_sec": decode_sec,
                    "audio_sec": audio_sec,
                },
            )

        references.append(utt.reference)
        hypotheses.append(hypothesis)
        decode_secs.append(decode_sec)
        audio_secs.append(audio_sec)

    wer_value = corpus_wer(references, hypotheses)
    cer_value = corpus_cer(references, hypotheses)

    if mode == "streaming" and audio_secs:
        rtf_value = rtf(sum(decode_secs), sum(audio_secs))
        latency = chunk_latency_stats(decode_secs, chunk_ms / 1000.0)["retrospective_latency_sec"]
        mean_latency = float(mean(decode_secs))
    else:
        rtf_value = 0.0
        latency = 0.0
        mean_latency = float(mean(decode_secs)) if decode_secs else 0.0

    return WERReport(
        model_id=model_id,
        mode=mode,
        chunk_ms=int(chunk_ms),
        n_utterances=len(utterances),
        wer=wer_value,
        cer=cer_value,
        rtf=rtf_value,
        mean_latency_sec=mean_latency,
        retrospective_latency_sec=latency,
        subsampled=subsampled,
        config=config,
        split=split,
        versions=engine.versions(),
        notes=notes,
    )


def max_new_tokens_for_mode(mode: str) -> int:
    """Official Stage 0 generation budget: 32 streaming, 1024 offline."""
    return 32 if mode == "streaming" else 1024


def subsample_offline_runs(
    runs: Sequence[dict[str, Any]],
    n: int,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Return copies of ``runs`` with offline runs capped to an ``n`` subsample.

    Offline full splits can exceed a Kaggle session; this labels the affected
    runs as subsampled. ``n <= 0`` returns the runs unchanged.
    """
    if n <= 0:
        return list(runs)
    out: list[dict[str, Any]] = []
    for run in runs:
        run = dict(run)
        if run["mode"] == "offline":
            run["limit"] = n
            run["sample_n"] = n
            run["sample_seed"] = seed
            run["subsampled"] = True
        out.append(run)
    return out


def filter_runs(
    runs: Sequence[dict[str, Any]],
    mode: str | None = None,
    config: str | None = None,
    chunk_ms: int | None = None,
) -> list[dict[str, Any]]:
    """Filter Stage 0 runs by ``mode``, ``config`` and/or ``chunk_ms`` (None = keep all)."""
    selected = list(runs)
    if mode is not None:
        selected = [run for run in selected if run["mode"] == mode]
    if config is not None:
        selected = [run for run in selected if run["config"] == config]
    if chunk_ms is not None:
        selected = [run for run in selected if run["chunk_ms"] == chunk_ms]
    return selected


def stage0_runs(
    models: Sequence[str],
    *,
    subsample_n: int = 500,
    subsample_seed: int = 0,
    baseline_chunk_ms: int = 2000,
    extra_chunk_ms: Sequence[int] = (320,),
) -> list[dict[str, Any]]:
    """Build the locked Stage 0 matrix.

    Offline runs use full clean + other; streaming runs full test-clean at the
    official 2.0 s chunk, plus a fixed ``subsample_n`` subsample of test-other at
    2.0 s and each extra chunk size. Subsample selection uses a fixed seed so
    every chunk size evaluates the identical utterances.
    """

    def run(mode: str, config: str, chunk_ms: int, subsampled: bool) -> dict[str, Any]:
        return {
            "model_id": None,  # filled per model below
            "mode": mode,
            "config": config,
            "split": "test",
            "chunk_ms": chunk_ms,
            "subsampled": subsampled,
            "limit": subsample_n if subsampled else None,
            "sample_n": subsample_n if subsampled else None,
            "sample_seed": subsample_seed,
        }

    runs: list[dict[str, Any]] = []
    for model_id in models:
        runs.append({**run("offline", "clean", baseline_chunk_ms, False), "model_id": model_id})
        runs.append({**run("offline", "other", baseline_chunk_ms, False), "model_id": model_id})
        runs.append({**run("streaming", "clean", baseline_chunk_ms, False), "model_id": model_id})
        runs.append({**run("streaming", "other", baseline_chunk_ms, True), "model_id": model_id})
        for chunk_ms in extra_chunk_ms:
            # Primary metric covers streaming 320 ms on clean and other, so add
            # the extra chunk sizes for both configs (labelled subsamples).
            runs.append({**run("streaming", "clean", chunk_ms, True), "model_id": model_id})
            runs.append({**run("streaming", "other", chunk_ms, True), "model_id": model_id})
    return runs
