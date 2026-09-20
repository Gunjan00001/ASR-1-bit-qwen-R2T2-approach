"""Word/character error rate computation (Stage 0, T0.4).

Scoring runs on normalized text by default (see :mod:`asr1bit.text`); pass
``normalize=False`` for raw comparison. Rates are returned in percent.
"""

from __future__ import annotations

from collections.abc import Sequence

import jiwer

from asr1bit.text import DEFAULT_MODE, normalize_text

Unit = str  # "word" | "char"


def _prepare(texts: Sequence[str], normalize: bool, mode: str) -> list[str]:
    if not normalize:
        return [t if t is not None else "" for t in texts]
    return [normalize_text(t, mode=mode) for t in texts]


def _metric(references: Sequence[str], hypotheses: Sequence[str], unit: Unit) -> float:
    if unit == "char":
        return float(jiwer.cer(list(references), list(hypotheses)))
    return float(jiwer.wer(list(references), list(hypotheses)))


def wer(
    reference: str,
    hypothesis: str,
    *,
    normalize: bool = True,
    mode: str = DEFAULT_MODE,
    unit: Unit = "word",
) -> float:
    """Error rate (percent) for a single pair. ``unit='char'`` gives CER."""
    ref, hyp = _prepare([reference, hypothesis], normalize, mode)
    return 100.0 * _metric([ref], [hyp], unit)


def corpus_wer(
    references: Sequence[str],
    hypotheses: Sequence[str],
    *,
    normalize: bool = True,
    mode: str = DEFAULT_MODE,
) -> float:
    """Corpus-level WER (percent) aggregated over total edit distance."""
    refs = _prepare(references, normalize, mode)
    hyps = _prepare(hypotheses, normalize, mode)
    if not refs:
        return 0.0
    return 100.0 * _metric(refs, hyps, "word")


def corpus_cer(
    references: Sequence[str],
    hypotheses: Sequence[str],
    *,
    normalize: bool = True,
    mode: str = DEFAULT_MODE,
) -> float:
    """Corpus-level CER (percent)."""
    refs = _prepare(references, normalize, mode)
    hyps = _prepare(hypotheses, normalize, mode)
    if not refs:
        return 0.0
    return 100.0 * _metric(refs, hyps, "char")
