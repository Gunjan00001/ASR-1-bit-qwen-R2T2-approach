"""Layer-sensitivity screening: flip individual projections to their fp16 shadow.

Why: a BitLinear trained under QAT keeps both a quantized forward and an fp16
shadow, so the *marginal* streaming cost of quantizing one component can be
measured on the trained model without retraining anything. Each mask sets the
selected components' ``alpha`` to 0 (fp16 shadow) and leaves the rest ternary.

This is a screening instrument: 20 utterances, the official prefix algorithm
with the LSP loop policy, one JSON line per mask. Confirmed findings must be
re-checked with the full 50-utterance protocol.

Usage (on a GPU node, after the model is loaded once):

    python scripts/screen_alpha_masks.py --checkpoint <state>.pt \
        --masks none,attn,ffn,q,k,v,o,ffn_in,ffn_out,all --n 20 --out out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from run_stage1_qat import load_checkpoint, patch_outer_forward

from asr1bit.bitlinear import BitLinear
from asr1bit.data.load import iter_librispeech
from asr1bit.eval.wer import corpus_cer, corpus_wer
from asr1bit.qwen import backends
from asr1bit.qwen.stream import LSPState
from asr1bit.replace import COMPONENTS, apply_layer_policy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--policy", default="decoder_attn_ffn")
    p.add_argument("--bit-width", type=float, default=1.58)
    p.add_argument("--chunk-ms", type=int, default=320)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--config", default="clean", choices=["clean", "other"])
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--masks", default="none,attn,ffn,q,k,v,o,ffn_in,ffn_out,all",
                   help="'none' = all ternary, 'all' = everything on its fp16 shadow.")
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
    print(f"restored checkpoint step {load_checkpoint(args.checkpoint, model)}", flush=True)

    utterances = list(iter_librispeech(args.config, "test", sample_n=args.n, sample_seed=0))
    model.eval()

    def apply_mask(mask: str) -> int:
        """Set alpha=1 (ternary) everywhere, then 0 (fp16 shadow) for the mask."""
        keys = set(COMPONENTS) if mask == "all" else ({mask} if mask != "none" else set())
        patterns = [COMPONENTS[k] for k in keys]
        touched = 0
        for name, module in model.named_modules():
            if not isinstance(module, BitLinear):
                continue
            module.alpha = 0.0 if any(p.search(name) for p in patterns) else 1.0
            touched += int(module.alpha == 0.0)
        return touched

    records = []
    with torch.no_grad():
        for mask in (m.strip() for m in args.masks.split(",") if m.strip()):
            flipped = apply_mask(mask)
            refs, hyps, truncations = [], [], 0
            for utt in utterances:
                state = LSPState(margin_words=0, truncate_loops=True)
                trace = backends.stream_transcribe_portable(
                    wrapper, utt.audio,
                    chunk_size_sec=args.chunk_ms / 1000.0,
                    step_sec=args.chunk_ms / 1000.0,
                    unfixed_token_num=5,
                    lsp=state,
                )
                refs.append(utt.text)
                hyps.append(trace.text)
                truncations += state.loop_truncations
            record = {
                "mask": mask,
                "flipped_modules": flipped,
                "wer": corpus_wer(refs, hyps),
                "cer": corpus_cer(refs, hyps),
                "loop_truncations": truncations,
                "hyp_words": sum(len(h.split()) for h in hyps),
                "ref_words": sum(len(r.split()) for r in refs),
            }
            records.append(record)
            print("SCREEN " + json.dumps(record), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"checkpoint": args.checkpoint, "policy": args.policy,
             "chunk_ms": args.chunk_ms, "n": len(utterances), "records": records}, indent=2))
        print("wrote", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
