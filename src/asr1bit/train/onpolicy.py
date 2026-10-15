"""On-policy example builders: rollouts, distillation, and their filtering.

Why these exist: the model's *streaming* behaviour is what we care about, so
training should visit the states the model actually reaches at inference time.
Two hard-won lessons live in this file:

* DAgger on the student's own (collapsed) prefixes made everything worse (WER
  920) — the degenerate-loop filter below is the fix.
* Distilled targets must carry the same *format* as the ordinary targets, or the
  student's content comes out garbled (that bug cost a full run).
"""
from __future__ import annotations

import argparse
import time

import torch

from asr1bit.qwen import backends
from asr1bit.train.examples import build_distill_examples, build_onpolicy_examples
from asr1bit.train.inloop_eval import set_precision
from asr1bit.train.rollout import collect_streaming_samples
from asr1bit.train.samples import select_evenly


def build_op_keep(args: argparse.Namespace):
    """State-selection policy for the on-policy rollout.

    Always keep the warm-up chunks (the runaway-generation fix) and the last
    chunk; keep mid-stream chunks on a fixed stride to bound memory.
    """

    def keep(chunk_id: int, final: bool) -> bool:
        if chunk_id < 2 or final:
            return True
        if args.op_max_rounds and chunk_id >= args.op_max_rounds - 1:
            return True
        return args.op_chunk_stride > 0 and chunk_id % args.op_chunk_stride == 0

    return keep


def rollout_onpolicy_examples(
    args: argparse.Namespace, wrapper, utterances, *, teacher_wrapper=None
) -> list[dict]:
    """Build streaming training examples by replaying the official algorithm.

    Two modes:
    - **on-policy SFT** (default): the *current* model generates the prefixes and
      the target is the reference remainder (own-prefix distribution);
    - **streaming-state distillation** (``--op-distill``): the *frozen pretrained
      teacher* generates them and the target is the teacher's own continuation —
      behavior cloning of a clean streaming policy at exactly the inference
      states (the student's own states can be degenerate loops, which DAgger
      would reinforce).
    """
    if args.op_distill:
        if teacher_wrapper is None:
            raise ValueError("--op-distill requires --frozen-teacher 1 (teacher rollout)")
        decode_batch = backends.build_transformers_batch_generate_fn(
            teacher_wrapper, max_new_tokens=args.op_max_new_tokens
        )
    else:
        decode_batch = backends.build_transformers_batch_decode_fn(
            wrapper, max_new_tokens=args.op_max_new_tokens
        )
    codec = backends.TokenizerCodec(wrapper.processor.tokenizer)
    prompt_raw = wrapper._build_text_prompt(context="", force_language=None)
    audios = [utt.audio for utt in utterances]
    started = time.perf_counter()

    def progress(done: int, total: int) -> None:
        if done == total or (done > 0 and done % 50 == 0):
            elapsed = time.perf_counter() - started
            eta = elapsed / done * (total - done)
            print(
                f"rollout {done}/{total} utts ({elapsed:.0f}s elapsed, ~{eta:.0f}s left)",
                flush=True,
            )

    samples = collect_streaming_samples(
        decode_batch,
        codec,
        prompt_raw,
        audios,
        chunk_size_sec=args.op_chunk_ms / 1000.0,
        unfixed_chunk_num=2,
        unfixed_token_num=5,
        keep=build_op_keep(args),
        batch_size=args.op_batch_size,
        max_rounds=args.op_max_rounds or None,
        progress=progress,
    )
    if args.op_distill:
        examples = build_distill_examples(samples, prompt_raw=prompt_raw)
    else:
        examples = build_onpolicy_examples(
            samples,
            [utt.text for utt in utterances],
            warmup_target_eos=bool(args.op_warmup_eos),
            drop_degenerate=bool(args.op_filter_degenerate),
        )
    total_states = sum(len(utt_samples) for utt_samples in samples)
    print(
        f"rollout states {total_states} -> {len(examples)} examples after filtering",
        flush=True,
    )
    return select_evenly(examples, args.op_max_samples)


def collect_op_examples(
    args: argparse.Namespace,
    wrapper,
    model,
    teacher_wrapper,
    op_utterances,
    *,
    restore_alpha: float | None = None,
) -> list[dict]:
    """Roll out and return streaming examples, restoring the training state."""
    if args.op_distill:
        with torch.no_grad():
            return rollout_onpolicy_examples(
                args, wrapper, op_utterances, teacher_wrapper=teacher_wrapper
            )
    set_precision(model, args.bit_width, 1.0)
    model.eval()
    try:
        with torch.no_grad():
            return rollout_onpolicy_examples(args, wrapper, op_utterances)
    finally:
        model.train()
        if restore_alpha is not None:
            set_precision(model, args.bit_width, restore_alpha)
