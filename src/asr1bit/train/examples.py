"""Turn rollouts into training examples: on-policy and distilled variants.

The two builders differ only in *what the target is*:

* ``build_onpolicy_examples`` - target = the words the reference transcript
  says belong next (after 'gold' alignment of what the model already said).
* ``build_distill_examples`` - target = the teacher's continuation for the same
  state (the teacher is a frozen fp16 copy of the model). This transfers the
  teacher's *behaviour*, which is what we could not get from labels alone; it
  plateaued at WER 278 before aligned targets finally solved the problem.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from asr1bit.train.filters import clean_stream_text, is_degenerate
from asr1bit.train.gold import alignment_key, gold_coverage
from asr1bit.train.samples import StreamingSample, continuation


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
    key = key or alignment_key
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
    prompt_raw: str | None = None,
    require_eos: bool = True,
    strip_metadata: bool = True,
) -> list[dict]:
    """Streaming-state distillation examples from a *teacher* rollout.

    Behavior cloning at the exact states the streaming algorithm produces at
    inference: the target is the rollout model's own continuation for that
    prompt/audio (teacher = the frozen pretrained model, whose streaming policy
    is clean and EOS-terminated). ``require_eos`` drops generations that hit the
    token cap, so every kept sample teaches "…continuation… then stop" (the
    collator appends the EOS token itself).

    ``strip_metadata`` removes the teacher's ``language X<asr_text>`` wrapper
    from both the recorded prompt (its prefix part, relative to ``prompt_raw``)
    and the target, keeping the distillation format identical to the standard
    samples (plain transcript text). The rollout itself still uses the teacher's
    natural, metadata-bearing prompts — only the *training records* are cleaned.
    """
    examples: list[dict] = []
    for utt_samples in samples:
        for sample in utt_samples:
            if require_eos and not sample.eos:
                continue
            prompt = sample.prompt
            target = sample.continuation
            if strip_metadata:
                target = clean_stream_text(target)
                if prompt_raw is not None and prompt.startswith(prompt_raw):
                    prompt = prompt_raw + clean_stream_text(prompt[len(prompt_raw) :])
            examples.append(
                {"audio": sample.audio, "prefix_text": prompt, "target": target}
            )
    return examples


@dataclass
class OnPolicyPool:
    """Container for a rollout's on-policy examples plus bookkeeping."""

    examples: list[dict] = field(default_factory=list)
    utterances: int = 0
    chunks: int = 0
    kept: int = 0

    def __len__(self) -> int:
        return len(self.examples)
