"""In-training evaluation helpers (offline precisions + the streaming guardrail).

Why: training runs are long and GPUs are contended, so we cannot afford a
full evaluation every few hundred steps — but flying blind is worse. These
helpers run *small, cheap* evaluations inside the loop. They are deliberately
labelled guardrails: the streaming one uses 20 utterances, and it has diverged
from the full 50-utterance metric four times in this project, so it is never
used to promote a checkpoint (see docs/DEEPDIVE.md, "the promotion rule").
"""
from __future__ import annotations

import torch

from asr1bit.bitlinear import BitLinear
from asr1bit.eval.wer import corpus_cer, corpus_wer
from asr1bit.qwen import backends


def evaluate(wrapper, utterances, limit: int) -> dict:
    refs, hyps = [], []
    for utt in utterances[:limit]:
        hyps.append(backends.offline_transcribe(wrapper, utt.audio, language=None))
        refs.append(utt.text)
    return {"n": len(refs), "wer": corpus_wer(refs, hyps), "cer": corpus_cer(refs, hyps)}


def set_precision(model, bit_width: float, alpha: float) -> None:
    for module in model.modules():
        if isinstance(module, BitLinear):
            module.bit_width = bit_width
            module.alpha = alpha


def evaluate_precisions(
    wrapper, model, utterances, n, *, bit_width: float = 1.0, both: bool = True,
    include_fp16: bool = False,
) -> dict:
    """WER at alpha=1 for binary and/or ternary (used for periodic/zero-shot).

    With ``both=False`` only the training precision (``bit_width``) is scored,
    halving eval time on long runs. ``include_fp16`` adds the alpha=0 readout,
    which detects shadow drift during QAT.
    """
    model.eval()
    results: dict = {}
    with torch.no_grad():
        if include_fp16:
            set_precision(model, 1.0, 0.0)
            results["fp16"] = evaluate(wrapper, utterances, n)
        if both or bit_width == 1.0:
            set_precision(model, 1.0, 1.0)
            results["binary"] = evaluate(wrapper, utterances, n)
        if both or bit_width != 1.0:
            set_precision(model, 1.58, 1.0)
            results["ternary"] = evaluate(wrapper, utterances, n)
    model.train()
    return results


def evaluate_streaming(
    wrapper, model, utterances, n: int, *, chunk_ms: int = 320, max_new_tokens: int = 32,
    unfixed_token_num: int = 5,
) -> dict:
    """Official-algorithm streaming WER on a small eval slice (training feedback).

    The periodic eval is offline-only, which made the streaming iterations blind
    (docs/ACTIVITY_LOG.md 2026-10-12). This runs the portable streamer on the
    first ``n`` eval utterances, exactly like ``scripts/eval_streaming.py``.
    """
    refs, hyps = [], []
    previous = getattr(wrapper, "max_new_tokens", None)
    try:
        wrapper.max_new_tokens = max_new_tokens
        for utt in utterances[:n]:
            trace = backends.stream_transcribe_portable(
                wrapper,
                utt.audio,
                chunk_size_sec=chunk_ms / 1000.0,
                step_sec=chunk_ms / 1000.0,
                unfixed_token_num=unfixed_token_num,
            )
            refs.append(utt.text)
            hyps.append(trace.text)
    finally:
        if previous is not None:
            wrapper.max_new_tokens = previous
    return {"n": len(refs), "wer": corpus_wer(refs, hyps), "cer": corpus_cer(refs, hyps)}
