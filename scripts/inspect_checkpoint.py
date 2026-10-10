"""Inspect generations from a saved QAT checkpoint (fp16 vs quantized).

Loads the model, applies the same layer policy, restores the shadow checkpoint,
and prints hypotheses at alpha=0 (full precision) and alpha=1 (quantized) for a
fixed test sample, plus WER for each. Useful for understanding *how* a QAT run
fails (empty output, repetition, wrong words) beyond the WER number.

Run on a CUDA host with the same policy the checkpoint was trained with.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from run_stage1_qat import load_checkpoint, patch_outer_forward

from asr1bit.data.load import iter_librispeech
from asr1bit.eval.wer import corpus_wer
from asr1bit.qwen import backends
from asr1bit.replace import apply_layer_policy
from asr1bit.train.qat import set_qat_alpha


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--policy", default="decoder_attn_ffn")
    p.add_argument("--bit-width", type=float, default=1.58)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=128)
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
    results = {"checkpoint": str(args.checkpoint), "step": step, "conditions": {}}
    with torch.no_grad():
        for alpha, name in ((0.0, "fp16"), (1.0, "quantized")):
            set_qat_alpha(model, alpha)
            refs, hyps = [], []
            for utt in utterances:
                text = backends.offline_transcribe(wrapper, utt.audio, language=None)
                refs.append(utt.text)
                hyps.append(text)
            results["conditions"][name] = {
                "wer": corpus_wer(refs, hyps),
                "references": refs,
                "hypotheses": hyps,
            }
            print(f"--- {name} (alpha={alpha}) WER {results['conditions'][name]['wer']:.2f} ---", flush=True)
            for ref, hyp in zip(refs, hyps):
                print(f"  ref: {ref[:80]}", flush=True)
                print(f"  hyp: {hyp[:80]}", flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("wrote", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
