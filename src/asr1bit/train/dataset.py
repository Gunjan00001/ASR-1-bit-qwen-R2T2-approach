"""Training-data assembly: examples, collation, utterance loading.

Why: the standard (non-streaming) half of every training batch is built here.
Streaming-specific example construction lives in ``aligned.py`` (forced-aligner
targets) and ``onpolicy.py`` (rollouts + distillation); this module is the
"ordinary transcription" side plus the batching glue that turns examples into
model inputs.
"""
from __future__ import annotations

import argparse

import torch

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
                    {
                        "audio": utt.audio,
                        "prefix_text": prefix + partial,
                        "target": target,
                        "streaming": False,
                    }
                )
        else:
            examples.append(
                {
                    "audio": utt.audio,
                    "prefix_text": prefix,
                    "target": utt.text,
                    "streaming": False,
                }
            )
        if silence_sec:
            n = max(1, int(silence_sec * 16000))
            examples.append(
                {
                    "audio": utt.audio[:n],
                    "prefix_text": prefix,
                    "target": "",
                    "streaming": False,
                }
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
    # Mark which rows are streaming chunk states. The state-distillation loss
    # (train/state_distill.py) uses this to supervise only samples whose audio
    # is truncated at the chunk boundary; ordinary full-utterance samples must
    # not contribute, or the loss would see future audio.
    if any("streaming" in example for example in examples):
        full_inputs["state_mask"] = torch.tensor(
            [1.0 if example.get("streaming", False) else 0.0 for example in examples],
            dtype=torch.float32,
        )
    return full_inputs


def load_train_utterances(args: argparse.Namespace, limit: int | None = None) -> list:
    """Training utterances from a local cache when given, else Hub validation sample.

    ``limit`` overrides ``--train-n``. The aligned-targets path passes its own
    ``--aligned-n`` so that a large aligned pool is not silently capped by the
    *training* sample count (2026-10-20: a 6 000-aligned run aligned 2 400).
    """
    cap = args.train_n if limit is None else limit
    if args.train_cache:
        utterances = load_utterance_cache(args.train_cache)
        if cap and cap < len(utterances):
            return utterances[:cap]
        return utterances
    return list(iter_librispeech("clean", "validation", sample_n=cap, sample_seed=0))
