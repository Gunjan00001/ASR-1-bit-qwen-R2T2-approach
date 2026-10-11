"""Diagnose the streaming collapse of QAT models (per-chunk trace).

Runs the portable ``ChunkedStreamer`` on one utterance and logs, per chunk:
audio duration, generated token count, whether EOS was emitted, and the text.
Runs the same utterance for the pretrained model and (optionally) a QAT
checkpoint so the divergence point is visible.

Run on a CUDA host:
    python scripts/diag_stream_generation.py --utterance 0
    python scripts/diag_stream_generation.py --checkpoint <ckpt> --bit-width 1.58
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from run_stage1_qat import load_checkpoint, patch_outer_forward

from asr1bit.data.load import iter_librispeech
from asr1bit.qwen import backends
from asr1bit.qwen.stream import ChunkedStreamer, parse_stream_output
from asr1bit.replace import apply_layer_policy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--policy", default="decoder_attn_ffn")
    p.add_argument("--bit-width", type=float, default=1.58)
    p.add_argument("--utterance", type=int, default=0)
    p.add_argument("--chunk-ms", type=int, default=320)
    p.add_argument("--min-decode-sec", type=float, default=0.0,
                   help="Skip decoding until the accumulated audio reaches this duration "
                        "(warm-up skip for near-empty first chunks).")
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
    processor = wrapper.processor
    patch_outer_forward(model)
    if args.checkpoint:
        apply_layer_policy(model, args.policy, bit_width=args.bit_width, learnable_scale=True)
        model.to("cuda:0")
        step = load_checkpoint(args.checkpoint, model)
        print(f"restored checkpoint step {step}", flush=True)
    else:
        print("no checkpoint: pretrained model", flush=True)

    utterances = list(iter_librispeech("clean", "test", sample_n=args.utterance + 1, sample_seed=0))
    utt = utterances[args.utterance]
    print(f"utterance {utt.id} ({utt.duration:.2f}s): {utt.text[:80]}", flush=True)

    eos_id = processor.tokenizer.eos_token_id
    traces: list[dict] = []
    chunk_index = {"i": 0}

    def decode_fn(prompt: str, audio: np.ndarray) -> str:
        if audio.shape[0] / 16000.0 < args.min_decode_sec:
            traces.append({
                "chunk": chunk_index["i"],
                "audio_sec": round(float(audio.shape[0]) / 16000.0, 2),
                "prompt_chars": len(prompt),
                "generated_tokens": 0,
                "eos": False,
                "text": "<skipped: warm-up>",
            })
            chunk_index["i"] += 1
            return ""
        inputs = processor(
            text=[prompt], audio=[np.asarray(audio, dtype=np.float32)],
            return_tensors="pt", padding=True,
        )
        inputs = inputs.to(model.device).to(model.dtype)
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        prompt_len = inputs["input_ids"].shape[1]
        ids = generated.sequences[0, prompt_len:].tolist()
        text = processor.batch_decode(
            generated.sequences[:, prompt_len:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        traces.append({
            "chunk": chunk_index["i"],
            "audio_sec": round(float(audio.shape[0]) / 16000.0, 2),
            "prompt_chars": len(prompt),
            "generated_tokens": len(ids),
            "eos": eos_id in ids,
            "text": text[:120],
        })
        chunk_index["i"] += 1
        return text

    streamer = ChunkedStreamer(
        decode_fn,
        backends.TokenizerCodec(processor.tokenizer),
        chunk_size_sec=args.chunk_ms / 1000.0,
        unfixed_chunk_num=2,
        unfixed_token_num=5,
    )
    prompt_raw = wrapper._build_text_prompt(context="", force_language=None)
    state = streamer.init_state(prompt_raw=prompt_raw)
    step_samples = max(1, round(args.chunk_ms / 1000.0 * 16000))
    model.eval()
    with torch.no_grad():
        for position in range(0, utt.audio.shape[0], step_samples):
            streamer.push(utt.audio[position : position + step_samples], state)
        streamer.finish(state)

    _language, text = parse_stream_output(state._raw_decoded, user_language=state.force_language)
    result = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "utterance_id": utt.id,
        "reference": utt.text,
        "final_text": text[:400],
        "chunks": traces,
    }
    print(f"final text ({len(text)} chars): {text[:200]}", flush=True)
    n_no_eos = sum(1 for t in traces if not t["eos"])
    print(f"chunks: {len(traces)} | without EOS: {n_no_eos}", flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("wrote", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
