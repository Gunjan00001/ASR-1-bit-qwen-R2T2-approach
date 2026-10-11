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
import math
import sys
import time
from pathlib import Path

import torch

from asr1bit.bitlinear import BitLinear
from asr1bit.data.load import iter_librispeech, load_utterance_cache
from asr1bit.eval.wer import corpus_cer, corpus_wer
from asr1bit.qwen import backends
from asr1bit.replace import apply_layer_policy
from asr1bit.train.qat import (
    build_optimizer,
    check_finite_grads,
    distillation_loss,
    grad_norms,
    make_sft_labels,
    progressive_alpha,
    quantization_delay_alpha,
    scale_stats,
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


def build_examples(utterances, processor, prompt: str = "", prefix_fracs=None) -> list[dict]:
    """Build (audio, prefix_text, target) triples.

    Mirrors Qwen's official ``finetuning/qwen3_asr_sft.py``: the prefix template
    is rendered with a *dummy* audio placeholder (``audio=None``); the real audio
    is passed separately to the processor. Using the real audio here changes the
    rendered prefix and misaligns the label mask.

    ``prefix_fracs`` (e.g. ``(0.0, 0.5, 0.8, 0.95)``) additionally creates
    **prefix-continuation** samples for streaming robustness: the chat template
    prefix is followed by the first ``frac`` of the transcript words and the
    target is the remaining words — mimicking the streaming algorithm's
    rolled-back self-prefix.
    """
    examples = []
    for utt in utterances:
        messages = [
            {"role": "system", "content": prompt or ""},
            {"role": "user", "content": [{"type": "audio", "audio": None}]},
        ]
        prefix = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if isinstance(prefix, list):
            prefix = prefix[0]
        if prefix_fracs:
            words = utt.text.split()
            for frac in prefix_fracs:
                k = round(len(words) * frac)
                partial = " ".join(words[:k])
                target = " ".join(words[k:])
                if not target:
                    continue
                examples.append(
                    {"audio": utt.audio, "prefix_text": prefix + partial, "target": target}
                )
        else:
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
    full_inputs["labels"] = make_sft_labels(
        full_inputs["input_ids"], prefix_lens, processor.tokenizer.pad_token_id
    )
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


def shadow_state(model) -> dict:
    """Trainable tensor snapshot (BitLinear shadows and learnable scales)."""
    return {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}


def save_checkpoint(path: Path, model, step: int, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "config": vars(args), "shadows": shadow_state(model)}, path)
    print(f"saved checkpoint {path}", flush=True)


def load_checkpoint(path: str | Path, model) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    shadows = payload.get("shadows", {})
    restored = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in shadows and shadows[name].shape == param.shape:
                param.copy_(shadows[name])
                restored += 1
    print(f"resumed {restored} tensors from {path} (step {payload.get('step')})", flush=True)
    return int(payload.get("step", 0))


def build_lr_lambda(args: argparse.Namespace):
    """Warmup then constant or cosine decay to zero over ``--steps``."""
    warmup = max(1, args.warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        if args.lr_schedule == "cosine":
            progress = (step - warmup) / max(1, args.steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return 1.0

    return lr_lambda


def load_train_utterances(args: argparse.Namespace) -> list:
    """Training utterances from a local cache when given, else Hub validation sample."""
    if args.train_cache:
        utterances = load_utterance_cache(args.train_cache)
        if args.train_n and args.train_n < len(utterances):
            return utterances[: args.train_n]
        return utterances
    return list(iter_librispeech("clean", "validation", sample_n=args.train_n, sample_seed=0))


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
    p.add_argument("--distill-weight", type=float, default=0.0,
                   help="Weight of self-distillation KL (teacher = same shadows at alpha=0).")
    p.add_argument("--distill-temp", type=float, default=2.0, help="Distillation temperature.")
    p.add_argument("--frozen-teacher", type=int, default=0,
                   help="1 = distillation teacher is a frozen fp copy of the pretrained model.")
    p.add_argument("--log-grad-norms", type=int, default=1)
    p.add_argument("--zero-shot", type=int, default=0, help="Evaluate alpha=1 without training")
    p.add_argument("--eval-every", type=int, default=0, help="WER eval interval during training (0=off)")
    p.add_argument("--eval-both", type=int, default=1, help="1 = score binary and ternary at evals; 0 = training precision only.")
    p.add_argument("--eval-fp16", type=int, default=0, help="1 = also score the alpha=0 readout at periodic evals (shadow drift).")
    p.add_argument("--learnable-scale", type=int, default=0,
                   help="BitNet b1.58-style learnable per-tensor/group dequantization gain.")
    p.add_argument("--lr-schedule", choices=["const", "cosine"], default="const")
    p.add_argument(
        "--lazy-collate", type=int, default=0,
        help="1 = tokenize/featurize batches on the fly (no upfront precompute of all batches).",
    )
    p.add_argument(
        "--train-cache", default=None,
        help="Utterance .pt cache from scripts/build_qat_cache.py (fast multi-epoch QAT).",
    )
    p.add_argument("--save-dir", default=None, help="Directory for shadow checkpoints.")
    p.add_argument("--save-every", type=int, default=0, help="Checkpoint interval in steps (0=only at end).")
    p.add_argument("--resume", default=None, help="Checkpoint path to resume shadows from (same run; advances step counter).")
    p.add_argument(
        "--init-from", default=None,
        help="Checkpoint path to initialize shadows from for a NEW phase (step counter starts at 0).",
    )
    p.add_argument(
        "--prefix-sft-fracs", default=None,
        help="Comma-separated transcript fractions for prefix-continuation samples "
             "(streaming robustness), e.g. '0.0,0.5,0.8,0.95'.",
    )
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

    teacher = None
    if args.frozen_teacher:
        if not args.distill_weight:
            raise ValueError("--frozen-teacher requires --distill-weight > 0")
        teacher_wrapper = backends.load_model(
            args.model, backend="transformers", dtype=torch.float32, device="cuda:0",
            max_new_tokens=8,
        )
        teacher = teacher_wrapper.model
        patch_outer_forward(teacher)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
        print("frozen teacher loaded", flush=True)

    train_utts = load_train_utterances(args)
    eval_utts = list(iter_librispeech("clean", "test", sample_n=args.eval_n, sample_seed=0))
    prefix_fracs = None
    if args.prefix_sft_fracs:
        prefix_fracs = tuple(float(x) for x in args.prefix_sft_fracs.split(",") if x.strip())
    train_examples = build_examples(train_utts, processor, prefix_fracs=prefix_fracs)
    print(f"train={len(train_examples)} eval={len(eval_utts)} prefix_fracs={prefix_fracs}", flush=True)

    results: dict = {"config": config_record, "eval_n": len(eval_utts), "frozen_teacher": teacher is not None}

    print("=== baseline (pretrained) ===", flush=True)
    results["pretrained"] = evaluate(wrapper, eval_utts, args.eval_n)
    print(json.dumps(results["pretrained"]), flush=True)

    apply_layer_policy(
        model,
        args.policy,
        bit_width=args.bit_width,
        group_size=group_size,
        learnable_scale=bool(args.learnable_scale),
    )
    model.to("cuda:0")
    results["bitlinear_replaced"] = getattr(model, "bitlinear_replaced", 0)

    # QAT-only: freeze everything except BitLinear shadows and learnable scales.
    for param in model.parameters():
        param.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, BitLinear):
            for param in (module.weight, module.bias, module.weight_scale):
                if param is not None:
                    param.requires_grad_(True)
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
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=build_lr_lambda(args))
    scaler = torch.amp.GradScaler("cuda") if args.autocast else None

    batches = None
    if not args.lazy_collate:
        batches = [collate(train_examples[i : i + args.batch_size], processor)
                   for i in range(0, len(train_examples), args.batch_size)]
    alpha_warmup = max(1, int(args.steps * args.alpha_warmup_frac))
    history = []
    start_step = 0
    if args.init_from:
        loaded = load_checkpoint(args.init_from, model)
        print(f"initialized shadows from {args.init_from} (checkpoint step {loaded}); new phase starts at 0", flush=True)
    if args.resume:
        start_step = load_checkpoint(args.resume, model)
        # Advance the LR schedule to the resumed position; otherwise the cosine
        # restarts and applies a fresh high LR to a converged checkpoint.
        if start_step > 0:
            scheduler.last_epoch = start_step
            print(f"lr schedule advanced to step {start_step}", flush=True)
    print("=== QAT ===", flush=True)
    started = time.perf_counter()
    for step in range(start_step, args.steps):
        if args.alpha_mode == "ramp":
            alpha = progressive_alpha(step, alpha_warmup)
        elif args.alpha_mode == "delay":
            alpha = quantization_delay_alpha(step, alpha_warmup)
        else:
            alpha = 1.0
        set_qat_alpha(model, alpha)
        if batches is None:
            base = ((step - start_step) * args.batch_size) % len(train_examples)
            chunk = train_examples[base : base + args.batch_size]
            if len(chunk) < args.batch_size:  # wrap around the end of the epoch
                chunk = chunk + train_examples[: args.batch_size - len(chunk)]
            batch = collate(chunk, processor)
        else:
            batch = batches[step % len(batches)]
        batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
        optimizer.zero_grad()
        ce_value = None
        kl_value = None
        if args.distill_weight > 0:
            if teacher is not None:
                # Anchored distillation: frozen fp copy of the pretrained model.
                with torch.no_grad():
                    teacher_out = teacher(**batch)
            else:
                # Self-distillation: the teacher is the same shadows at alpha=0.
                set_qat_alpha(model, 0.0)
                with torch.no_grad():
                    teacher_out = model(**batch)
                set_qat_alpha(model, alpha)
            out = model(**batch)
            ce = out.loss if hasattr(out, "loss") else out[0]
            kl = distillation_loss(
                out.logits, teacher_out.logits, batch["labels"], temperature=args.distill_temp
            )
            loss = (1.0 - args.distill_weight) * ce + args.distill_weight * kl
            ce_value = float(ce.detach())
            kl_value = float(kl.detach())
            loss.backward()
            check_finite_grads(model)
            norms = grad_norms(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        elif scaler is not None:
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
            if ce_value is not None:
                record["ce"] = ce_value
                record["kl"] = kl_value
            record.update(scale_stats(model))
            history.append(record)
            print(json.dumps(record), flush=True)
        if args.eval_every and (step + 1) % args.eval_every == 0:
            prec = evaluate_precisions(
                wrapper, model, eval_utts, args.eval_n,
                bit_width=args.bit_width, both=bool(args.eval_both),
                include_fp16=bool(args.eval_fp16),
            )
            ev = {"step": step + 1, "alpha": alpha,
                  "fp16_wer": prec["fp16"]["wer"] if "fp16" in prec else None,
                  "binary_wer": prec["binary"]["wer"] if "binary" in prec else None,
                  "ternary_wer": prec["ternary"]["wer"] if "ternary" in prec else None}
            history.append(ev)
            print("EVAL " + json.dumps(ev), flush=True)
            set_precision(model, args.bit_width, alpha)  # restore training precision
        if args.save_dir and args.save_every and (step + 1) % args.save_every == 0:
            save_checkpoint(Path(args.save_dir) / "last.pt", model, step + 1, args)
    results["history"] = history
    results["qat_seconds"] = time.perf_counter() - started
    if args.save_dir:
        save_checkpoint(Path(args.save_dir) / "last.pt", model, args.steps, args)

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
        f"{args.optimizer}_{args.alpha_mode}_ls{args.learnable_scale}_n{args.train_n}"
    )
    (out_dir / f"stage1_{tag}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", out_dir / f"stage1_{tag}.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
