"""Training-data assembly: examples, collation, utterance loading.

Why: the standard (non-streaming) half of every training batch is built here.
Streaming-specific example construction lives in ``aligned.py`` (forced-aligner
targets) and ``onpolicy.py`` (rollouts + distillation); this module is the
"ordinary transcription" side plus the batching glue that turns examples into
model inputs.
"""
from __future__ import annotations

import argparse

from asr1bit.data.load import iter_librispeech, load_utterance_cache
from asr1bit.train.qat import make_sft_labels


def build_examples(
    utterances, processor, prompt: str = "", prefix_fracs=None, silence_sec: float | None = None
) -> list[dict]:
    """Build (audio, prefix_text, target) triples.

    Mirrors Qwen's official ``finetuning/qwen3_asr_sft.py``: the prefix template
    is rendered with a *dummy* audio placeholder (``audio=None``); the real audio
    is passed separately to the processor. Using the real audio here changes the
    rendered prefix and misaligns the label mask.

    ``prefix_fracs`` (e.g. ``(0.0, 0.5, 0.8, 0.95)``) additionally creates
    **prefix-continuation** samples for streaming robustness: the chat template
    prefix is followed by the first ``frac`` of the transcript words and the
    target is the remaining words — mimicking the streaming algorithm's
    rolled-back self-prefix.

    ``silence_sec`` additionally creates a **near-empty-audio** sample per
    utterance (audio truncated to ``silence_sec``, empty target) so the model
    learns to emit EOS instead of hallucinating on the streaming warm-up chunks.
    """
    examples = []
    for utt in utterances:
        messages = [
            {"role": "system", "content": prompt or ""},
            {"role": "user", "content": [{"type": "audio", "audio": None}]},
        ]
        prefix = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if isinstance(prefix, list):
            prefix = prefix[0]
        if prefix_fracs:
            words = utt.text.split()
            for frac in prefix_fracs:
                k = round(len(words) * frac)
                partial = " ".join(words[:k])
                target = " ".join(words[k:])
                if not target:
                    continue
                examples.append(
                    {"audio": utt.audio, "prefix_text": prefix + partial, "target": target}
                )
        else:
            examples.append({"audio": utt.audio, "prefix_text": prefix, "target": utt.text})
        if silence_sec:
            n = max(1, int(silence_sec * 16000))
            examples.append(
                {"audio": utt.audio[:n], "prefix_text": prefix, "target": ""}
            )
    return examples


def collate(examples: list[dict], processor) -> dict:
    audios = [e["audio"] for e in examples]
    prefixes = [e["prefix_text"] for e in examples]
    eos = processor.tokenizer.eos_token or ""
    full = [p + e["target"] + eos for p, e in zip(prefixes, examples)]
    full_inputs = processor(text=full, audio=audios, return_tensors="pt", padding=True, truncation=False)
    prefix_inputs = processor(text=prefixes, audio=audios, return_tensors="pt", padding=True, truncation=False)
    prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
    full_inputs["labels"] = make_sft_labels(
        full_inputs["input_ids"], prefix_lens, processor.tokenizer.pad_token_id
    )
    return full_inputs


def load_train_utterances(args: argparse.Namespace) -> list:
    """Training utterances from a local cache when given, else Hub validation sample."""
    if args.train_cache:
        utterances = load_utterance_cache(args.train_cache)
        if args.train_n and args.train_n < len(utterances):
            return utterances[: args.train_n]
        return utterances
    return list(iter_librispeech("clean", "validation", sample_n=args.train_n, sample_seed=0))
