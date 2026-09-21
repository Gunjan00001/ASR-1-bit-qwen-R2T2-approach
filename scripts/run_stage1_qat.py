"""Stage 1 (T1.4/T1.5): progressive QAT on Qwen3-ASR-0.6B + offline WER ablations.

Stabilized debug configuration (see docs/LAB_NOTES.md):
- only BitLinear shadows are trainable (frozen QAT setup, not a fallback),
- fp32 master weights + fp32 AdamW by default (8-bit Adam behind a flag),
- no autocast, gradient checkpointing OFF, activation quant OFF by default,
- constant alpha=1.0 by default (alpha ramp behind a flag),
- per-layer gradient norms logged and finiteness asserted before every step.

Enable the risky features one at a time via flags to find which one breaks
training. Streaming/vLLM eval is deferred.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from asr1bit.bitlinear import BitLinear
from asr1bit.data.load import iter_librispeech
from asr1bit.eval.wer import corpus_cer, corpus_wer
from asr1bit.qwen import backends
from asr1bit.replace import apply_layer_policy
from asr1bit.train.qat import (
    build_optimizer,
    check_finite_grads,
    grad_norms,
    progressive_alpha,
    quantization_delay_alpha,
    set_activation_quant,
    set_qat_alpha,
)


def patch_outer_forward(model) -> None:
    """Route the outer forward to ``model.thinker`` (official Qwen3-ASR API)."""
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return
    if not hasattr(model, "thinker"):
        raise RuntimeError("model has no .thinker; cannot patch forward")

    def forward(self, input_ids=None, attention_mask=None, input_features=None,
                feature_attention_mask=None, labels=None, **kwargs):
        return self.thinker.forward(
            input_ids=input_ids, attention_mask=attention_mask,
            input_features=input_features, feature_attention_mask=feature_attention_mask,
            labels=labels, **kwargs)

    cls.forward = forward
    cls._forward_patched = True


def build_examples(utterances, processor, prompt: str = "") -> list[dict]:
    examples = []
    for utt in utterances:
        messages = [
            {"role": "system", "content": prompt or ""},
            {"role": "user", "content": [{"type": "audio", "audio": utt.audio}]},
        ]
        prefix = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if isinstance(prefix, list):
            prefix = prefix[0]
        examples.append({"audio": utt.audio, "prefix_text": prefix, "target": utt.text})
    return examples


def collate(examples: list[dict], processor) -> dict:
    audios = [e["audio"] for e in examples]
    prefixes = [e["prefix_text"] for e in examples]
    eos = processor.tokenizer.eos_token or ""
    full = [p + e["target"] + eos for p, e in zip(prefixes, examples)]
    full_inputs = processor(text=full, audio=audios, return_tensors="pt", padding=True, truncation=False)
    prefix_inputs = processor(text=prefixes, audio=audios, return_tensors="pt", padding=True, truncation=False)
    prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
    labels = full_inputs["input_ids"].clone()
    for i, plen in enumerate(prefix_lens):
        labels[i, :plen] = -100
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = -100
    full_inputs["labels"] = labels
    return full_inputs


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


def evaluate_precisions(wrapper, model, utterances, n) -> dict:
    """WER at alpha=1 for binary and ternary (used for periodic/zero-shot)."""
    model.eval()
    with torch.no_grad():
        set_precision(model, 1.0, 1.0)
        binary = evaluate(wrapper, utterances, n)
        set_precision(model, 1.58, 1.0)
        ternary = evaluate(wrapper, utterances, n)
    model.train()
    return {"binary": binary, "ternary": ternary}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    p.add_argument("--policy", default="decoder_attn")
    p.add_argument("--bit-width", type=float, default=1.0)
    p.add_argument("--group-size", type=int, default=0, help="0 = per-tensor")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--train-n", type=int, default=64)
    p.add_argument("--eval-n", type=int, default=100, help="Fixed eval set; same for floor and every checkpoint.")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--optimizer", choices=["adamw", "adam8bit"], default="adamw")
    p.add_argument("--grad-checkpointing", type=int, default=0)
    p.add_argument("--activation-quant", type=int, default=0)
    p.add_argument("--autocast", type=int, default=0)
    p.add_argument(
        "--alpha-mode", choices=["constant", "delay", "ramp"], default="delay",
        help="delay = fp warmup then constant alpha=1 (safe); ramp = blended (unsafe, see LAB_NOTES).",
    )
    p.add_argument("--alpha-warmup-frac", type=float, default=0.3, help="Delay fraction before alpha=1.")
    p.add_argument("--log-grad-norms", type=int, default=1)
    p.add_argument("--zero-shot", type=int, default=0, help="Evaluate alpha=1 without training")
    p.add_argument("--eval-every", type=int, default=0, help="WER eval interval during training (0=off)")
    p.add_argument("--out", default="artifacts/stage1")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    group_size = args.group_size or None
    config_record = vars(args) | {"group_size": group_size}
    print("=== environment ===")
    print(json.dumps(backends.environment_report(), indent=2), flush=True)
    print("=== config ===")
    print(json.dumps(config_record, indent=2), flush=True)

    wrapper = backends.load_model(
        args.model, backend="transformers", dtype=torch.float32, device="cuda:0",
        max_new_tokens=args.max_new_tokens,
    )
    model = wrapper.model
    processor = wrapper.processor
    patch_outer_forward(model)

    train_utts = list(iter_librispeech("clean", "validation", sample_n=args.train_n, sample_seed=0))
    eval_utts = list(iter_librispeech("clean", "test", sample_n=args.eval_n, sample_seed=0))
    train_examples = build_examples(train_utts, processor)
    print(f"train={len(train_examples)} eval={len(eval_utts)}", flush=True)

    results: dict = {"config": config_record, "eval_n": len(eval_utts)}

    print("=== baseline (pretrained) ===", flush=True)
    results["pretrained"] = evaluate(wrapper, eval_utts, args.eval_n)
    print(json.dumps(results["pretrained"]), flush=True)

    apply_layer_policy(model, args.policy, bit_width=args.bit_width, group_size=group_size)
    model.to("cuda:0")
    results["bitlinear_replaced"] = getattr(model, "bitlinear_replaced", 0)

    # QAT-only: freeze everything except BitLinear shadows.
    for param in model.parameters():
        param.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, BitLinear):
            module.weight.requires_grad_(True)
            if module.bias is not None:
                module.bias.requires_grad_(True)
    set_activation_quant(model, bool(args.activation_quant))
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results["trainable_params"] = trainable
    print(f"replaced={results['bitlinear_replaced']} trainable={trainable}", flush=True)

    if args.zero_shot:
        print("=== zero-shot naive (no training, alpha=1) ===", flush=True)
        set_precision(model, 1.0, 0.0)
        results["fp16"] = evaluate(wrapper, eval_utts, args.eval_n)
        prec = evaluate_precisions(wrapper, model, eval_utts, args.eval_n)
        results["binary"] = prec["binary"]
        results["ternary"] = prec["ternary"]
        print(json.dumps({k: results[k] for k in ("pretrained", "fp16", "binary", "ternary")}, indent=2), flush=True)
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"zeroshot_{args.policy}_g{group_size}"
        (out_dir / f"stage1_{tag}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("wrote", out_dir / f"stage1_{tag}.json", flush=True)
        return 0

    optimizer = build_optimizer(model, lr=args.lr, use_8bit=(args.optimizer == "adam8bit"))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / max(1, args.warmup_steps))
    )
    scaler = torch.amp.GradScaler("cuda") if args.autocast else None

    batches = [collate(train_examples[i : i + args.batch_size], processor)
               for i in range(0, len(train_examples), args.batch_size)]
    alpha_warmup = max(1, int(args.steps * args.alpha_warmup_frac))
    history = []
    print("=== QAT ===", flush=True)
    started = time.perf_counter()
    for step in range(args.steps):
        if args.alpha_mode == "ramp":
            alpha = progressive_alpha(step, alpha_warmup)
        elif args.alpha_mode == "delay":
            alpha = quantization_delay_alpha(step, alpha_warmup)
        else:
            alpha = 1.0
        set_qat_alpha(model, alpha)
        batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batches[step % len(batches)].items()}
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(**batch)
                loss = out.loss if hasattr(out, "loss") else out[0]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            check_finite_grads(model)
            norms = grad_norms(model)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(**batch)
            loss = out.loss if hasattr(out, "loss") else out[0]
            loss.backward()
            check_finite_grads(model)
            norms = grad_norms(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        max_norm = max(norms.values()) if norms else 0.0
        if step == 0 and args.log_grad_norms:
            top = sorted(norms.items(), key=lambda kv: -kv[1])[:5]
            print("top grad norms:", [(n.split(".")[-2] + "." + n.split(".")[-1], round(v, 4)) for n, v in top], flush=True)
        if step % 5 == 0 or step == args.steps - 1:
            record = {"step": step + 1, "alpha": alpha, "loss": float(loss.detach()),
                      "lr": scheduler.get_last_lr()[0], "max_grad_norm": max_norm}
            history.append(record)
            print(json.dumps(record), flush=True)
        if args.eval_every and (step + 1) % args.eval_every == 0:
            prec = evaluate_precisions(wrapper, model, eval_utts, args.eval_n)
            ev = {"step": step + 1, "alpha": alpha,
                  "binary_wer": prec["binary"]["wer"], "ternary_wer": prec["ternary"]["wer"]}
            history.append(ev)
            print("EVAL " + json.dumps(ev), flush=True)
            set_precision(model, args.bit_width, alpha)  # restore training precision
    results["history"] = history
    results["qat_seconds"] = time.perf_counter() - started

    print("=== ablations (offline WER) ===", flush=True)
    disable = getattr(model, "gradient_checkpointing_disable", None)
    if callable(disable):
        disable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    model.eval()
    with torch.no_grad():
        set_precision(model, 1.0, 0.0)
        results["fp16"] = evaluate(wrapper, eval_utts, args.eval_n)
        set_precision(model, 1.0, 1.0)
        results["binary"] = evaluate(wrapper, eval_utts, args.eval_n)
        set_precision(model, 1.58, 1.0)
        results["ternary"] = evaluate(wrapper, eval_utts, args.eval_n)
    print(json.dumps({k: results[k] for k in ("pretrained", "fp16", "binary", "ternary")}, indent=2), flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"{args.policy}_bw{args.bit_width}_g{group_size}_lr{args.lr}_"
        f"{args.optimizer}_{args.alpha_mode}_n{args.train_n}"
    )
    (out_dir / f"stage1_{tag}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", out_dir / f"stage1_{tag}.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
