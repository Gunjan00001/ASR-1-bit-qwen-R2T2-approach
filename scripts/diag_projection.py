"""Stage 1 diagnostic: which attention projection is sensitive to binarization?

Applies BitLinear to all decoder q/k/v/o, then sets alpha=1 for exactly one
projection at a time (q, k, v, o), all, or none, and reports teacher-forced
token accuracy (on a training utterance) plus greedy WER on a fixed eval set.

This runs only after label masking is verified (fp16 teacher-forced token
accuracy ~1.0). Run in a fresh subprocess on a CUDA host.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from run_stage1_qat import build_examples, collate, patch_outer_forward

from asr1bit.bitlinear import BitLinear
from asr1bit.data.load import iter_librispeech
from asr1bit.eval.wer import corpus_wer
from asr1bit.qwen import backends
from asr1bit.replace import apply_layer_policy
from asr1bit.train.qat import set_activation_quant

PROJECTIONS = ["none", "q", "k", "v", "o", "all"]


def set_alpha_by_projection(model, proj: str) -> None:
    for name, module in model.named_modules():
        if not isinstance(module, BitLinear):
            continue
        if proj == "all":
            module.alpha = 1.0
        elif proj == "none":
            module.alpha = 0.0
        else:
            module.alpha = 1.0 if name.endswith(f".self_attn.{proj}_proj") else 0.0


def teacher_forced(model, batch, processor) -> dict:
    model.eval()
    with torch.no_grad():
        out = model(**batch)
    logits = out.logits
    labels = batch["labels"]
    preds = logits[:, :-1].argmax(dim=-1)
    target = labels[:, 1:]
    mask = target != -100
    accuracy = float((preds[mask] == target[mask]).float().mean()) if mask.any() else 0.0
    return {"loss": float(out.loss), "token_accuracy": accuracy,
            "n_label_tokens": int(mask.sum()),
            "teacher_forced_text": processor.tokenizer.decode(preds[mask].tolist(), skip_special_tokens=True)}


def greedy_wer(wrapper, utterances, n) -> float:
    refs, hyps = [], []
    for utt in utterances[:n]:
        hyps.append(backends.offline_transcribe(wrapper, utt.audio, language=None))
        refs.append(utt.text)
    return corpus_wer(refs, hyps)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    p.add_argument("--policy", default="decoder_attn")
    p.add_argument("--eval-n", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--out", default="artifacts/stage1")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    wrapper = backends.load_model(args.model, backend="transformers", dtype=torch.float32,
                                  device="cuda:0", max_new_tokens=args.max_new_tokens)
    model = wrapper.model
    processor = wrapper.processor
    patch_outer_forward(model)

    train_utts = list(iter_librispeech("clean", "validation", sample_n=1, sample_seed=0))
    eval_utts = list(iter_librispeech("clean", "test", sample_n=args.eval_n, sample_seed=0))
    utterance = train_utts[0]
    batch = {k: (v.cuda() if torch.is_tensor(v) else v)
             for k, v in collate(build_examples([utterance], processor), processor).items()}
    print("label:", utterance.text, "| unmasked tokens:", int((batch["labels"] != -100).sum()), flush=True)

    apply_layer_policy(model, args.policy, bit_width=1.0, group_size=None)
    model.to("cuda:0")
    set_activation_quant(model, False)

    results = {"utterance_id": utterance.id, "label": utterance.text, "eval_n": len(eval_utts),
               "conditions": []}
    for proj in PROJECTIONS:
        set_alpha_by_projection(model, proj)
        tf = teacher_forced(model, batch, processor)
        wer = greedy_wer(wrapper, eval_utts, args.eval_n)
        row = {"projection": proj, "wer": wer, **tf}
        results["conditions"].append(row)
        print(json.dumps({k: row[k] for k in ("projection", "loss", "token_accuracy", "wer")}), flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"diag_projection_{args.policy}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
