"""Streaming (320 ms) evaluation of a quantized checkpoint — Stage 1 primary metric.

Loads a QAT checkpoint, runs the portable ``ChunkedStreamer`` at a given chunk
size on a fixed LibriSpeech sample, and reports WER + RTF for the fp16 readout
(alpha=0) and the quantized readout (alpha=1). The fp16 streaming numbers are
measured on the *same* utterances so the ``+X`` comparison is exact.

Note: Qwen3-ASR streaming is vLLM-only for the official backend, and vLLM cannot
serve BitLinear models; the portable transformers streamer re-feeds accumulated
audio per chunk (O(n^2) re-feed), so this evaluation is deliberately subsampled
(per the Stage 0 feasibility policy).

Run on a CUDA host:  python scripts/eval_streaming.py --checkpoint <ckpt> ...
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from run_stage1_qat import load_checkpoint, patch_outer_forward

from asr1bit.data.load import iter_librispeech
from asr1bit.eval.wer import corpus_cer, corpus_wer
from asr1bit.qwen import backends
from asr1bit.replace import apply_layer_policy
from asr1bit.train.qat import set_qat_alpha


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--policy", default="decoder_attn_ffn")
    p.add_argument("--bit-width", type=float, default=1.58)
    p.add_argument("--chunk-ms", type=int, default=320)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    wrapper = backends.load_model(
        args.model, backend="transformers", dtype=torch.float32, device="cuda:0",
        max_new_tokens=args.max_new_tokens,
    )
    model = wrapper.model
    patch_outer_forward(model)
    apply_layer_policy(model, args.policy, bit_width=args.bit_width, learnable_scale=True)
    model.to("cuda:0")
    step = load_checkpoint(args.checkpoint, model)
    print(f"restored checkpoint step {step}", flush=True)

    utterances = list(iter_librispeech("clean", "test", sample_n=args.n, sample_seed=0))
    model.eval()
    results = {
        "checkpoint": str(args.checkpoint),
        "step": step,
        "chunk_ms": args.chunk_ms,
        "n": len(utterances),
        "conditions": {},
    }
    with torch.no_grad():
        for alpha, name in ((0.0, "fp16"), (1.0, "quantized")):
            set_qat_alpha(model, alpha)
            refs, hyps = [], []
            total_decode = 0.0
            total_audio = 0.0
            started = time.perf_counter()
            for utt in utterances:
                trace = backends.stream_transcribe_portable(
                    wrapper, utt.audio,
                    chunk_size_sec=args.chunk_ms / 1000.0,
                    step_sec=args.chunk_ms / 1000.0,
                )
                refs.append(utt.text)
                hyps.append(trace.text)
                total_decode += trace.total_decode_sec
                total_audio += trace.audio_sec
            wer = corpus_wer(refs, hyps)
            cer = corpus_cer(refs, hyps)
            wall = time.perf_counter() - started
            results["conditions"][name] = {
                "alpha": alpha,
                "wer": wer,
                "cer": cer,
                "rtf_decode": total_decode / total_audio if total_audio else None,
                "wall_sec": wall,
                "hypotheses": hyps,
                "references": refs,
            }
            print(
                f"--- {name} (alpha={alpha}) chunk {args.chunk_ms} ms: "
                f"WER {wer:.2f} | CER {cer:.2f} | RTF {total_decode / total_audio:.3f} | "
                f"wall {wall:.0f}s",
                flush=True,
            )
            for ref, hyp in list(zip(refs, hyps))[:3]:
                print(f"  ref: {ref[:70]}", flush=True)
                print(f"  hyp: {hyp[:70]}", flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("wrote", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
