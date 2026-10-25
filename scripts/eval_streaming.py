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
from asr1bit.qwen.stream import LSPState
from asr1bit.replace import apply_layer_policy
from asr1bit.train.qat import set_qat_alpha


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    p.add_argument("--checkpoint", default=None,
                   help="QAT checkpoint to restore; omit to evaluate the pretrained model.")
    p.add_argument("--policy", default="decoder_attn_ffn")
    p.add_argument("--bit-width", type=float, default=1.58)
    p.add_argument("--group-size", type=int, default=0,
                   help="0 = per-tensor; must match the checkpoint's training layout.")
    p.add_argument("--chunk-ms", type=int, default=320)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--config", default="clean", choices=["clean", "other"],
                   help="LibriSpeech config: clean (default) or other (WP3 clean/other pair).")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--conditions", default="fp16,quantized",
                   help="Comma-separated subset of {fp16,quantized}.")
    p.add_argument("--unfixed-token-num", type=int, default=5,
                   help="Tokens rolled back for the prefix continuation. Large values (e.g. "
                        "9999) disable the prefix — required for QAT models, which were never "
                        "trained to continue partial text (see docs/ACTIVITY_LOG.md).")
    p.add_argument("--repetition-penalty", type=float, default=None,
                   help="generate() repetition penalty (breaks streaming repetition loops).")
    p.add_argument("--no-repeat-ngram", type=int, default=None,
                   help="generate() no_repeat_ngram_size (breaks streaming repetition loops).")
    p.add_argument("--lsp-margin-words", type=int, default=None,
                   help="Enable Longest-Stable-Prefix emission holding back this many words "
                        "(Stage 4 / T4.4). The final text is unchanged; the report adds the "
                        "append-only commit latency and revision count.")
    p.add_argument("--lsp-truncate-loops", type=int, default=0,
                   help="1 = LSP stop policy: cut a trailing repetition loop at emission "
                        "(measured: corpus WER 278 -> 144 on the 2026-10-12 checkpoints). "
                        "Implies the LSP state; pass --lsp-margin-words for a margin too.")
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
    apply_layer_policy(model, args.policy, bit_width=args.bit_width,
                       group_size=args.group_size or None, learnable_scale=True)
    model.to("cuda:0")
    if args.checkpoint:
        step = load_checkpoint(args.checkpoint, model)
        print(f"restored checkpoint step {step}", flush=True)
    else:
        step = None
        print("no checkpoint: evaluating the pretrained model", flush=True)

    utterances = list(
        iter_librispeech(args.config, "test", sample_n=args.n, sample_seed=0)
    )
    model.eval()
    results = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "step": step,
        "chunk_ms": args.chunk_ms,
        "config": args.config,
        "n": len(utterances),
        "unfixed_token_num": args.unfixed_token_num,
        # Provenance: the evaluator must record the architecture it built, or a
        # policy/width mismatch is invisible after the fact (the b17/b21
        # invalid-evaluation lesson).
        "policy": args.policy,
        "bit_width": args.bit_width,
        "group_size": args.group_size,
        "learnable_scale": True,
        "dataset": "openslr/librispeech_asr",
        "split": "test",
        "sample_seed": 0,
        "conditions": {},
    }
    wanted = {name.strip() for name in args.conditions.split(",") if name.strip()}
    generate_kwargs: dict = {}
    if args.repetition_penalty:
        generate_kwargs["repetition_penalty"] = args.repetition_penalty
    if args.no_repeat_ngram:
        generate_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram
    results["generate_kwargs"] = generate_kwargs
    with torch.no_grad():
        for alpha, name in ((0.0, "fp16"), (1.0, "quantized")):
            if name not in wanted:
                continue
            set_qat_alpha(model, alpha)
            refs, hyps = [], []
            raw_hyps: list[str] = []
            total_decode = 0.0
            total_audio = 0.0
            started = time.perf_counter()
            lsp_revisions = 0
            lsp_truncations = 0
            lsp_progress: dict[float, list[float]] = {}
            lsp_enabled = args.lsp_margin_words is not None or bool(args.lsp_truncate_loops)
            for utt in utterances:
                lsp_state = (
                    LSPState(
                        margin_words=args.lsp_margin_words or 0,
                        truncate_loops=bool(args.lsp_truncate_loops),
                    )
                    if lsp_enabled
                    else None
                )
                trace = backends.stream_transcribe_portable(
                    wrapper, utt.audio,
                    chunk_size_sec=args.chunk_ms / 1000.0,
                    step_sec=args.chunk_ms / 1000.0,
                    unfixed_token_num=args.unfixed_token_num,
                    generate_kwargs=generate_kwargs or None,
                    lsp=lsp_state,
                )
                refs.append(utt.text)
                hyps.append(trace.text)
                total_decode += trace.total_decode_sec
                total_audio += trace.audio_sec
                if lsp_state is not None:
                    lsp_revisions += trace.revisions
                    lsp_truncations += lsp_state.loop_truncations
                    raw_hyps.append(trace.chunks[-1] if trace.chunks else trace.text)
                    for fraction in (0.5, 0.8):
                        cutoff = fraction * trace.audio_sec
                        committed = 0
                        for audio_sec, committed_len in trace.commits:
                            if audio_sec <= cutoff:
                                committed = committed_len
                        denominator = max(1, len(trace.text))
                        lsp_progress.setdefault(fraction, []).append(committed / denominator)
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
            if lsp_enabled:
                results["conditions"][name]["lsp"] = {
                    "margin_words": args.lsp_margin_words,
                    "truncate_loops": bool(args.lsp_truncate_loops),
                    "revisions": lsp_revisions,
                    "loop_truncations": lsp_truncations,
                    "wer_raw": corpus_wer(refs, raw_hyps) if raw_hyps else None,
                    "committed_frac": {
                        str(fraction): sum(values) / len(values)
                        for fraction, values in sorted(lsp_progress.items())
                    },
                }
            print(
                f"--- {name} (alpha={alpha}) chunk {args.chunk_ms} ms: "
                f"WER {wer:.2f} | CER {cer:.2f} | RTF {total_decode / total_audio:.3f} | "
                f"wall {wall:.0f}s",
                flush=True,
            )
            if lsp_enabled:
                lsp_report = results["conditions"][name]["lsp"]
                raw = lsp_report["wer_raw"]
                print(
                    f"    LSP(margin={args.lsp_margin_words}, truncate_loops={lsp_report['truncate_loops']}): "
                    f"revisions {lsp_report['revisions']} | loop truncations {lsp_report['loop_truncations']} | "
                    + (f"raw WER {raw:.2f} -> policy {wer:.2f} | " if raw is not None else "")
                    + " | ".join(
                        f"committed@{fraction}: {value * 100:.1f}%"
                        for fraction, value in lsp_report["committed_frac"].items()
                    ),
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
