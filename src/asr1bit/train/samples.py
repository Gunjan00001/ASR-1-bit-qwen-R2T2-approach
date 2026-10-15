"""Streaming sample primitives: the record, its prompt/continuation text, mixing.

A streaming training example answers one question: *given what the model had
committed before this chunk, what should it say for this chunk?* This module
holds the data record for that question plus the small pure helpers used to
build and mix examples. It has no model dependency, which makes it the easiest
place in the codebase to read the recipe from.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class StreamingSample:
    """One streaming state recorded during a rollout.

    ``continuation`` is the text the *rollout model* generated at this state and
    ``eos`` whether that generation stopped on the EOS token (False = it hit the
    token cap). For on-policy SFT the targets are built from the reference
    (:func:`build_onpolicy_examples`); for streaming-state distillation the
    continuation itself is the target (:func:`build_distill_examples`).
    """

    audio: np.ndarray
    prompt: str
    own_text: str
    chunk_id: int
    audio_sec: float
    final: bool = False
    continuation: str = ""
    eos: bool = False


def split_result(result) -> tuple[str, bool]:
    """Normalize a rollout decode result: ``str`` or ``(text, eos)``."""
    if isinstance(result, tuple):
        text, eos = result
        return str(text or ""), bool(eos)
    return str(result or ""), False


def prompt_tail(sample: StreamingSample) -> str:
    """The text handed to the model before its own generation (base prompt + prefix)."""
    if sample.own_text and sample.prompt.endswith(sample.own_text):
        return sample.prompt[: len(sample.prompt) - len(sample.own_text)]
    return sample.prompt


def continuation(prompt_text: str, remainder: Sequence[str]) -> str:
    """Text the model should generate after ``prompt_text`` (``""`` = EOS only).

    ``prompt_text`` is the full prompt the model saw (base template + its own
    rolled-back text). The streamer's rollback can leave the prompt inside a
    word; in that case the target completes the partial word before continuing
    with the remainder. A single leading space is inserted when the prompt ends
    on a word (mirroring how the model continues its own text); a prompt that is
    empty or ends with whitespace needs no separator.
    """
    remainder = list(remainder)
    if not remainder:
        return ""
    if prompt_text and not prompt_text[-1].isspace():
        fragment = prompt_text.split()[-1] if prompt_text.split() else ""
        first = remainder[0]
        if (
            fragment
            and len(fragment) < len(first)
            and first[: len(fragment)].lower() == fragment.lower()
        ):
            head = first[len(fragment) :]
            rest = " ".join(remainder[1:])
            return head + (" " + rest if rest else "")
        return " " + " ".join(remainder)
    return " ".join(remainder)


def interleave_with_ratio(
    standard: Sequence[dict],
    onpolicy: Sequence[dict],
    op_frac: float,
) -> list[dict]:
    """Interleave two example lists so ~``op_frac`` of the result is on-policy.

    ``op_frac`` is the *target* share of on-policy samples: no on-policy sample
    is ever dropped (they are the expensive ones), so an oversized pool raises
    the share instead. Size the pools as
    ``len(onpolicy) == op_frac / (1 - op_frac) * len(standard)`` to land exactly
    on the target and consume both lists. The interleave is deterministic (no
    shuffling) so runs are reproducible.
    """
    standard = list(standard)
    onpolicy = list(onpolicy)
    if not onpolicy or op_frac <= 0:
        return standard
    if not standard or op_frac >= 1:
        return onpolicy
    out: list[dict] = []
    op_frac = float(op_frac)
    si = oi = 0
    for k in range(len(standard) + len(onpolicy)):
        want_onpolicy = oi < len(onpolicy) and (
            si >= len(standard) or (oi + 1) <= op_frac * (k + 1)
        )
        if want_onpolicy:
            out.append(onpolicy[oi])
            oi += 1
        else:
            out.append(standard[si])
            si += 1
    return out


def select_evenly(items: Sequence, n: int) -> list:
    """Deterministic order-preserving subsample of ``items`` down to ``n``."""
    items = list(items)
    if n <= 0:
        return []
    if len(items) <= n:
        return items
    step = len(items) / float(n)
    return [items[min(len(items) - 1, int(index * step))] for index in range(n)]
