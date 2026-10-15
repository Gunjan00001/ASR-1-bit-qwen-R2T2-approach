"""Aligned-target example builder — the recipe that fixed streaming.

The idea (and the reason this project's streaming number moved from 144.6 to
~12): do not ask the model to *infer* how much text belongs to how much audio.
Use Qwen3-ForcedAligner-0.6B at data-preparation time to find when each word is
spoken, then build, for every chunk boundary, a training example whose target is
exactly the words that end inside that chunk — and, when no word ends there, a
target that says "nothing new yet" (the stop samples).

This module holds only the glue that assembles those examples into training
rows; the alignment itself lives in ``asr1bit/data/align.py`` and the chunk
arithmetic in ``asr1bit/data/prefix.py``.
"""
from __future__ import annotations


def build_aligned_examples(
    utterances,
    processor,
    aligner,
    *,
    prompt: str = "",
    chunk_ms: int = 320,
    lookahead_ms: int = 0,
    stop_stride: int = 2,
    progress=None,
) -> list[dict]:
    """Chunk-aligned SFT samples from forced-aligner word timestamps (T4.3).

    For every utterance: align the reference (T4.1) → chunk targets (T4.2) →
    ``(audio_slice, template + prefix, target)`` examples. The target is exactly
    the words the model should have emitted at that chunk ("what I have heard,
    then stop"), which is the behaviour the official prefix algorithm needs.

    Roughly half of all chunks have *no* new words (an empty target = a stop
    sample); ``stop_stride`` keeps only every N-th of those so the mix is not
    dominated by silence teaching.
    """
    from asr1bit.data.align import align_words
    from asr1bit.data.prefix import chunk_examples

    examples: list[dict] = []
    empty_seen = 0
    for index, utt in enumerate(utterances):
        words = align_words(utt.audio, utt.text, aligner=aligner)
        if not words:
            continue
        empty_seen = 0
        for example in chunk_examples(
            words,
            chunk_sec=chunk_ms / 1000.0,
            lookahead_sec=lookahead_ms / 1000.0,
            total_sec=utt.duration,
        ):
            if not example.target_text:
                empty_seen += 1
                if stop_stride > 1 and (empty_seen - 1) % stop_stride != 0:
                    continue
            samples = max(1, round(example.audio_sec * 16000))
            audio = utt.audio[:samples]
            prefix = processor.apply_chat_template(
                [
                    {"role": "system", "content": prompt or ""},
                    {"role": "user", "content": [{"type": "audio", "audio": None}]},
                ],
                add_generation_prompt=True,
                tokenize=False,
            )
            if isinstance(prefix, list):
                prefix = prefix[0]
            examples.append(
                {
                    "audio": audio,
                    "prefix_text": prefix + example.prefix_text,
                    "target": example.target_text,
                }
            )
        if progress is not None and (index + 1) % 50 == 0:
            progress(index + 1, len(utterances))
    return examples
