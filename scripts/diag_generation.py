"""Stage 1 diagnostic: is the failure teacher-forced or the greedy decode path?

For a fixed utterance, at alpha=0 (control) and alpha=1 (binary), report:
  - teacher-forced loss and token accuracy (argmax over label positions),
  - the teacher-forced predicted text,
  - the greedy generated text,
and assert the loss forward actually used Q(W) (effective weight != shadow).

Optionally train a few steps first (constant alpha=1) and repeat, to test the
"low training loss but ~100% WER" case.

Run in a fresh subprocess on a CUDA host.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from run_stage1_qat import build_examples, collate, patch_outer_forward, set_precision

from asr1bit.bitlinear import BitLinear
from asr1bit.data.load import iter_librispeech
from asr1bit.qwen import backends
from asr1bit.replace import apply_layer_policy
from asr1bit.train.qat import grad_norms, set_activation_quant, set_qat_alpha


class QuantInstrument:
    """Wrap BitLinear._quantized_weight to record effective vs shadow weights."""

    def __init__(self):
        self.records: list[tuple[float, float, float]] = []

    def __enter__(self):
        self._orig = BitLinear._quantized_weight

        def wrapped(layer):
            effective = self._orig(layer)
            if len(self.records) < 8:
                self.records.append((
                    float(layer.weight.detach().norm()),
                    float(effective.detach().norm()),
                    float((effective.detach() - layer.weight.detach()).norm()),
                ))
            return effective

        BitLinear._quantized_weight = wrapped
        return self

    def __exit__(self, *exc):
        BitLinear._quantized_weight = self._orig
        return False


def teacher_forced(model, batch, processor) -> dict:
    with torch.no_grad():
        out = model(**batch)
    loss = float(out.loss) if hasattr(out, "loss") and out.loss is not None else None
    logits = out.logits
    labels = batch["labels"]
    preds = logits[:, :-1].argmax(dim=-1)
    target = labels[:, 1:]
    mask = target != -100
    accuracy = float((preds[mask] == target[mask]).float().mean()) if mask.any() else 0.0
    pred_ids = preds[mask].tolist()
    pred_text = processor.tokenizer.decode(pred_ids, skip_special_tokens=True)
    ref_ids = target[mask].tolist()
    ref_text = processor.tokenizer.decode(ref_ids, skip_special_tokens=True)
    return {"loss": loss, "token_accuracy": accuracy,
            "teacher_forced_text": pred_text, "label_text": ref_text}


def greedy_text(wrapper, utterance) -> str:
    return backends.offline_transcribe(wrapper, utterance.audio, language=None)


def run_condition(model, wrapper, batch, utterance, processor, bit_width, alpha, label) -> dict:
    set_precision(model, bit_width, alpha)
    with QuantInstrument() as inst:
        tf = teacher_forced(model, batch, processor)
    effective_changed = any(delta > 0 for _, _, delta in inst.records)
    result = {
        "condition": label,
        "bit_width": bit_width,
        "alpha": alpha,
        "quantized_effective_weight_changed": effective_changed,
        "weight_norms": inst.records[:3],
        **tf,
        "greedy_text": greedy_text(wrapper, utterance),
    }
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    p.add_argument("--policy", default="decoder_attn")
    p.add_argument("--train-steps", type=int, default=0)
    p.add_argument("--train-n", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-5)
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

    train_utts = list(iter_librispeech("clean", "validation", sample_n=max(1, args.train_n), sample_seed=0))
    utterance = train_utts[0]
    examples = build_examples([utterance], processor)
    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in collate(examples, processor).items()}
    print("utterance:", utterance.id, "| label:", utterance.text, flush=True)

    results: dict = {"utterance_id": utterance.id, "label": utterance.text, "conditions": []}

    # Control: full precision (alpha=0), before binarization.
    results["conditions"].append(run_condition(model, wrapper, batch, utterance, processor, 1.0, 0.0, "fp16_control"))

    apply_layer_policy(model, args.policy, bit_width=1.0, group_size=None)
    model.to("cuda:0")
    for param in model.parameters():
        param.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, BitLinear):
            module.weight.requires_grad_(True)
            if module.bias is not None:
                module.bias.requires_grad_(True)
    set_activation_quant(model, False)

    results["conditions"].append(run_condition(model, wrapper, batch, utterance, processor, 1.0, 1.0, "binary_naive"))

    if args.train_steps > 0:
        train_examples = build_examples(train_utts, processor)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        set_qat_alpha(model, 1.0)
        print(f"training {args.train_steps} steps (constant alpha=1)...", flush=True)
        for step in range(args.train_steps):
            b = {k: (v.cuda() if torch.is_tensor(v) else v)
                 for k, v in collate([train_examples[step % len(train_examples)]], processor).items()}
            optimizer.zero_grad()
            out = model(**b)
            loss = out.loss if hasattr(out, "loss") else out[0]
            loss.backward()
            norms = grad_norms(model)
            optimizer.step()
            if step % 100 == 0 or step == args.train_steps - 1:
                print(f"  step {step} loss={float(loss.detach()):.3f} max_grad={max(norms.values()):.2f}", flush=True)
        results["conditions"].append(run_condition(model, wrapper, batch, utterance, processor, 1.0, 1.0, "binary_trained"))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"diag_generation_{args.policy}_steps{args.train_steps}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)
    print("wrote", path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
