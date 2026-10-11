"""Streaming stable-prefix supervised fine-tuning (Stage 4, T4.3).

On-policy streaming SFT: the official algorithm re-feeds a rolled-back copy of
the model's *own* text as the next prompt (``qwen/stream.py``), so training
samples must be built from the model's own partial generations — not from gold
transcripts — or the model degenerates into repetition loops when it has to
continue itself (docs/PROJECT_STATUS.md, streaming section).

This module provides the pieces of that data pipeline:

- :func:`collect_streaming_samples` replays the official chunked algorithm
  (full audio re-feed + prefix rollback) with the *current* model, batched over
  utterances, and records the exact ``(prompt, audio heard, own text)`` state at
  selected chunks.
- :func:`gold_coverage` aligns the model's own text against the reference so the
  training target is the *unfinished remainder* of the transcript.
- :func:`build_onpolicy_examples` turns the rollout into ``(audio, prefix_text,
  target)`` examples for the SFT collator.
- :func:`interleave_with_ratio` blends them with standard samples (the mix must
  keep a large share of offline samples to protect offline quality).

The rollout is the streaming algorithm itself — only reorganized so that all
utterances' chunk-``k`` decodes run as one batch instead of one stream at a
time (same math, less wall time).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from asr1bit.qwen.stream import (
    SAMPLE_RATE,
    TokenCodec,
    parse_stream_output,
    rollback_prefix,
)
from asr1bit.text import normalize_text

BatchDecodeFn = Callable[[Sequence[str], Sequence[np.ndarray]], Sequence[str]]


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


def _split_result(result) -> tuple[str, bool]:
    """Normalize a rollout decode result: ``str`` or ``(text, eos)``."""
    if isinstance(result, tuple):
        text, eos = result
        return str(text or ""), bool(eos)
    return str(result or ""), False


def collect_streaming_samples(
    decode_batch: BatchDecodeFn,
    codec: TokenCodec,
    prompt_raw: str,
    audios: Sequence[np.ndarray],
    *,
    chunk_size_sec: float = 0.32,
    unfixed_chunk_num: int = 2,
    unfixed_token_num: int = 5,
    sample_rate: int = SAMPLE_RATE,
    keep: Callable[[int, bool], bool] | None = None,
    batch_size: int = 4,
    max_rounds: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[list[StreamingSample]]:
    """Replay the official streaming algorithm and record on-policy states.

    Args:
        decode_batch: ``(prompts, audios) -> continuations``. Each continuation
            is the text generated *after* its prompt; audios are the accumulated
            chunks (what the official algorithm re-feeds).
        codec: Tokenizer used for the official prefix rollback.
        prompt_raw: Chat-template prompt without any prefix (the official
            ``_build_text_prompt(context="", force_language=None)``).
        audios: One waveform per utterance (16 kHz float32).
        keep: ``(chunk_id, final) -> bool`` selection of states to record.
            Defaults to keeping every state (memory-heavy; callers normally
            subsample).
        batch_size: Utterances decoded per model call.
        max_rounds: Optional cap on the number of chunk rounds.
        progress: ``(completed_utterances, total)`` callback per round.

    Returns:
        Per-utterance lists of :class:`StreamingSample` (input order preserved).
    """
    chunk_samples = max(1, round(chunk_size_sec * sample_rate))
    keep = keep or (lambda chunk_id, final: True)

    states = []
    for index, audio in enumerate(audios):
        array = np.asarray(audio, dtype=np.float32).reshape(-1)
        states.append(
            {"index": index, "audio": array, "position": 0, "chunk_id": 0, "raw": ""}
        )
    results: list[list[StreamingSample]] = [[] for _ in states]

    round_id = 0
    while True:
        active = [state for state in states if state["position"] < state["audio"].shape[0]]
        if not active or (max_rounds is not None and round_id >= max_rounds):
            break

        records: list[dict] = []
        for state in active:
            start = state["position"]
            end = min(start + chunk_samples, state["audio"].shape[0])
            prefix = (
                ""
                if state["chunk_id"] < unfixed_chunk_num
                else rollback_prefix(codec, state["raw"], unfixed_token_num)
            )
            records.append(
                {
                    "state": state,
                    "prefix": prefix,
                    "prompt": prompt_raw + prefix,
                    "feed": state["audio"][start:end],
                    "end": end,
                    "final": end >= state["audio"].shape[0],
                }
            )

        for batch_start in range(0, len(records), batch_size):
            sub_records = records[batch_start : batch_start + batch_size]
            results_batch = decode_batch(
                [record["prompt"] for record in sub_records],
                [record["feed"] for record in sub_records],
            )
            for record, result in zip(sub_records, results_batch):
                state = record["state"]
                prefix = record["prefix"]
                continuation, eos = _split_result(result)
                if keep(state["chunk_id"], record["final"]):
                    results[state["index"]].append(
                        StreamingSample(
                            audio=state["audio"][: record["end"]].copy(),
                            prompt=record["prompt"],
                            own_text=parse_stream_output(prefix)[1],
                            chunk_id=state["chunk_id"],
                            audio_sec=record["end"] / float(sample_rate),
                            final=record["final"],
                            continuation=continuation,
                            eos=eos,
                        )
                    )
                state["raw"] = prefix + (continuation or "")
                state["position"] = record["end"]
                state["chunk_id"] += 1

        round_id += 1
        if progress is not None:
            done = sum(1 for state in states if state["position"] >= state["audio"].shape[0])
            progress(done, len(states))
    return results


def _alignment_key(word: str) -> str:
    """Case/punctuation-insensitive key for one word (1:1 with the raw word)."""
    return normalize_text(word, mode="librispeech")


def gold_coverage(
    own_words: Sequence[str],
    gold_words: Sequence[str],
    *,
    key: Callable[[str], str] | None = None,
) -> int:
    """Number of leading reference words the model's own text covers.

    Unit-cost edit-distance alignment between the model's own words and the
    reference; the returned coverage is the last reference index aligned to a
    matching word (+1), so repetitions and insertions in the own text do not
    inflate it. Returns 0 when nothing matches.
    """
    own = list(own_words)
    gold = list(gold_words)
    n, m = len(own), len(gold)
    if n == 0 or m == 0:
        return 0
    if key is not None:
        own = [key(word) for word in own]
        gold = [key(word) for word in gold]

    def matches(a: str, b: str) -> bool:
        return a == b and (key is None or a != "")

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        row, previous = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            cost = 0 if matches(own[i - 1], gold[j - 1]) else 1
            row[j] = min(previous[j] + 1, row[j - 1] + 1, previous[j - 1] + cost)

    i, j = n, m
    covered = 0
    while i > 0 and j > 0:
        if matches(own[i - 1], gold[j - 1]) and dp[i][j] == dp[i - 1][j - 1]:
            covered = max(covered, j)
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
        elif dp[i][j] == dp[i][j - 1] + 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    return covered


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


def build_onpolicy_examples(
    samples: Sequence[Sequence[StreamingSample]],
    references: Sequence[str],
    *,
    warmup_chunk_num: int = 2,
    warmup_target_eos: bool = True,
    drop_degenerate: bool = True,
    key: Callable[[str], str] | None = None,
) -> list[dict]:
    """Turn collected rollout states into SFT examples.

    Each example's target is the reference remainder the model has not produced
    yet (``""`` when the own text already covers the transcript, which trains
    the EOS stop). Warm-up chunks (``chunk_id < warmup_chunk_num``) get an empty
    target so the model learns to stay silent — instead of runaway-generating —
    on the near-empty first chunks.

    ``drop_degenerate`` skips states whose own text is already a repetition loop
    (:func:`is_degenerate`): without it, DAgger training fits the loop patterns
    back into the model (measured: official-algorithm WER 920.55 vs 203.5 for
    the unfiltered phase-1 run, ``docs/LAB_NOTES.md`` 2026-10-12).

    Args:
        samples: Output of :func:`collect_streaming_samples`.
        references: Reference transcript per utterance (raw ``utt.text``).
        warmup_chunk_num: First N chunks are warm-up (empty prefix by design).
        warmup_target_eos: Whether warm-up chunks train an immediate EOS.
        drop_degenerate: Drop non-warm-up states with looping own text.
        key: Optional alignment key function (default: case/punctuation-
            insensitive words).
    """
    key = key or _alignment_key
    examples: list[dict] = []
    for utt_samples, reference in zip(samples, references):
        gold_words = list(reference.split())
        for sample in utt_samples:
            if warmup_target_eos and sample.chunk_id < warmup_chunk_num:
                target = ""
            else:
                if drop_degenerate and is_degenerate(sample.own_text):
                    continue
                covered = gold_coverage(sample.own_text.split(), gold_words, key=key)
                remainder = gold_words[covered:]
                if sample.own_text:
                    target = continuation(sample.prompt, remainder)
                else:
                    # Fresh start (no own text yet): the reference follows the
                    # base prompt directly, exactly like the standard samples.
                    target = " ".join(remainder)
            examples.append(
                {"audio": sample.audio, "prefix_text": sample.prompt, "target": target}
            )
    return examples


def build_distill_examples(
    samples: Sequence[Sequence[StreamingSample]],
    *,
    require_eos: bool = True,
) -> list[dict]:
    """Streaming-state distillation examples from a *teacher* rollout.

    Behavior cloning at the exact states the streaming algorithm produces at
    inference: the target is the rollout model's own continuation for that
    prompt/audio (teacher = the frozen pretrained model, whose streaming policy
    is clean and EOS-terminated). ``require_eos`` drops generations that hit the
    token cap, so every kept sample teaches "…continuation… then stop" (the
    collator appends the EOS token itself).

    Unlike :func:`build_onpolicy_examples` there is no reference alignment: the
    teacher's continuation is the target verbatim (its leading space is part of
    the tokenization).
    """
    examples: list[dict] = []
    for utt_samples in samples:
        for sample in utt_samples:
            if require_eos and not sample.eos:
                continue
            examples.append(
                {
                    "audio": sample.audio,
                    "prefix_text": sample.prompt,
                    "target": sample.continuation,
                }
            )
    return examples


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


@dataclass
class OnPolicyPool:
    """Container for a rollout's on-policy examples plus bookkeeping."""

    examples: list[dict] = field(default_factory=list)
    utterances: int = 0
    chunks: int = 0
    kept: int = 0

    def __len__(self) -> int:
        return len(self.examples)
