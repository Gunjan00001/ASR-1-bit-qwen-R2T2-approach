"""Degenerate-output detection and cleanup (the DAgger guard).

Why this exists: training on the student's *own* outputs is standard practice
(DAgger) - but when the student has already collapsed, those outputs are loops,
and training on them teaches the loops back. We measured exactly that: a DAgger
round pushed streaming WER to 920. These helpers identify degenerate text
(loops, empty content, silent failures) so such examples can be dropped instead
of learned.
"""
from __future__ import annotations

from asr1bit.qwen.stream import ASR_TEXT_TAG, LANG_PREFIX


def is_degenerate(
    text: str, *, max_period: int = 16, min_repeats: int = 4
) -> bool:
    """Heuristic repetition-loop detector for streamed text.

    True when the tail of ``text`` is a small unit repeated ``min_repeats`` or
    more times (``'S'S'S'S…``, ``HIS HIS HIS HIS``). Used by
    :func:`build_onpolicy_examples` to drop training states whose own text is
    already a loop — fitting those teaches the loop back (see
    ``docs/LAB_NOTES.md`` 2026-10-12).
    """
    if not text:
        return False
    tail = text.rstrip()
    if len(tail) < 2:
        return False
    for period in range(1, max_period + 1):
        unit = tail[-period:]
        if not unit.strip():
            continue
        repeats = 0
        index = len(tail)
        while index >= period and tail[index - period : index] == unit:
            repeats += 1
            index -= period
        if repeats >= min_repeats:
            return True
    return False


def clean_stream_text(text: str) -> str:
    """Drop a leading ``language X<asr_text>`` block from raw model text.

    The pretrained model emits the structured ``language English<asr_text>...``
    form; the QAT student is trained on plain transcripts everywhere else (the
    standard samples), so distillation targets must use the same plain format or
    the two halves of the mix teach contradictory output formats.
    """
    if ASR_TEXT_TAG not in text:
        return text
    tag_index = text.index(ASR_TEXT_TAG)
    language_index = text.rfind(LANG_PREFIX.strip(), 0, tag_index)
    if language_index >= 0:
        line_start = text.rfind("\n", 0, language_index)
        cut = line_start + 1 if line_start >= 0 else 0
    else:
        cut = 0
    return text[:cut] + text[tag_index + len(ASR_TEXT_TAG) :]
