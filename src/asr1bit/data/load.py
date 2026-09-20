"""Dataset loading (Stage 0, T0.2).

Produce 16 kHz mono float32 audio plus a raw transcript and its normalized
reference from LibriSpeech manifests or the Hugging Face
``openslr/librispeech_asr`` dataset. Normalization is delegated to
:mod:`asr1bit.text` so references and hypotheses are scored consistently.
"""

from __future__ import annotations

import io
import json
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import soxr

from asr1bit.text import normalize_text

SAMPLE_RATE = 16000
LIBRISPEECH_DATASET = "openslr/librispeech_asr"


@dataclass(frozen=True)
class Utterance:
    """One audio/reference pair.

    Attributes:
        id: Stable utterance id.
        audio: Mono float32 waveform at ``target_sr``.
        text: Raw transcript as provided by the dataset.
        reference: Normalized transcript used for WER scoring.
        duration: Duration in seconds.
    """

    id: str
    audio: np.ndarray
    text: str
    reference: str
    duration: float


def _to_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=-1)
    if audio.ndim != 1:
        raise ValueError(f"Expected 1-D or 2-D audio, got shape {audio.shape}")
    return audio.astype(np.float32, copy=False)


def load_audio(path: str | Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Read an audio file as mono float32, resampling to ``target_sr``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    audio = _to_mono(audio)
    if int(sr) != target_sr:
        audio = soxr.resample(audio, int(sr), int(target_sr)).astype(np.float32)
    return audio


def write_manifest(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Write ``records`` as JSON Lines to ``path``, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load JSON Lines records (``id``, ``audio``, ``text``) from ``path``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def iter_utterances(
    records: Sequence[dict[str, Any]],
    target_sr: int = SAMPLE_RATE,
    limit: int | None = None,
) -> Iterator[Utterance]:
    """Yield :class:`Utterance` objects for manifest ``records``."""
    for index, record in enumerate(records):
        if limit is not None and index >= limit:
            break
        audio = load_audio(record["audio"], target_sr=target_sr)
        text = record.get("text", "")
        yield Utterance(
            id=str(record.get("id", index)),
            audio=audio,
            text=text,
            reference=normalize_text(text),
            duration=audio.shape[0] / float(target_sr),
        )


def _hf_audio(row: dict[str, Any]) -> tuple[np.ndarray, int]:
    """Extract mono float32 audio + sample rate from a HF ``audio`` field.

    Handles both decoded fields (``array``/``sampling_rate``) and undecoded
    fields (``bytes``/``path``), so we never depend on ``torchcodec``.
    """
    audio = row["audio"]
    if audio.get("array") is not None:
        return _to_mono(audio["array"]), int(audio["sampling_rate"])
    if audio.get("bytes") is not None:
        array, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
        return _to_mono(array), int(sr)
    if audio.get("path") is not None:
        array, sr = sf.read(audio["path"], dtype="float32", always_2d=False)
        return _to_mono(array), int(sr)
    raise ValueError(f"Unsupported audio field: {sorted(audio.keys())}")


def _select_rows(dataset: Any, indices: Sequence[int]) -> Any:
    if hasattr(dataset, "select"):
        return dataset.select(list(indices))
    return [dataset[i] for i in indices]


def iter_librispeech(
    config: str,
    split: str,
    *,
    dataset: Iterable[dict[str, Any]] | None = None,
    target_sr: int = SAMPLE_RATE,
    limit: int | None = None,
    indices: Sequence[int] | None = None,
    sample_n: int | None = None,
    sample_seed: int = 0,
) -> Iterator[Utterance]:
    """Yield :class:`Utterance` objects from ``openslr/librispeech_asr``.

    Args:
        config: Dataset config, e.g. ``"clean"`` or ``"other"``.
        split: Split name, e.g. ``"test"``.
        dataset: Optional pre-loaded dataset (used for tests / offline runs).
            When omitted, the dataset is streamed from the Hugging Face Hub.
        target_sr: Output sample rate.
        limit: Optional cap on the number of utterances.
        indices: Optional explicit row indices to evaluate.
        sample_n: Optional deterministic subsample size. Uses ``sample_seed``
            so the same fixed subsample is reused across chunk sizes.
        sample_seed: Seed for ``sample_n`` selection.
    """
    if dataset is None:
        from datasets import Audio, load_dataset

        dataset = load_dataset(LIBRISPEECH_DATASET, config, split=split)
        # Decode audio ourselves (soundfile) instead of relying on the
        # `torchcodec` backend that newer `datasets` versions require.
        if hasattr(dataset, "cast_column"):
            dataset = dataset.cast_column("audio", Audio(decode=False))

    if indices is not None:
        dataset = _select_rows(dataset, indices)
    elif sample_n is not None:
        total = dataset.num_rows if hasattr(dataset, "num_rows") else len(dataset)
        dataset = _select_rows(dataset, select_subsample(list(range(total)), sample_n, seed=sample_seed))

    for index, row in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        audio, sr = _hf_audio(row)
        if sr != target_sr:
            audio = soxr.resample(audio, sr, target_sr).astype(np.float32)
        text = row.get("text", "")
        yield Utterance(
            id=str(row.get("id", index)),
            audio=audio,
            text=text,
            reference=normalize_text(text),
            duration=audio.shape[0] / float(target_sr),
        )


def select_subsample(items: Sequence[Any], n: int, seed: int = 0) -> list[Any]:
    """Return ``n`` items chosen deterministically, preserving input order."""
    if n >= len(items):
        return list(items)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(items)), n))
    return [items[i] for i in indices]
