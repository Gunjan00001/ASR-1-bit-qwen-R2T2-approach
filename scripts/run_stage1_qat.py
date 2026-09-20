"""Stage 1 (T1.4/T1.5): progressive QAT on Qwen3-ASR-0.6B + offline WER ablations.

Loads the transformers backend, binarizes/ternarizes the decoder projections via
``apply_layer_policy``, runs a short progressive-alpha QAT (fp32 shadow + STE,
optional 8-bit Adam + LoRA + gradient checkpointing), then evaluates offline WER
on a fixed labelled subset for fp16 / binary / ternary.

Streaming/vLLM eval is deferred; this uses the Stage 0 harness (offline,
portable transformers backend). Run in a fresh subprocess (vLLM spawn-safety is
not needed here, but isolation keeps the notebook kernel clean).
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
    QATConfig,
    build_optimizer,
    enable_gradient_checkpointing,
    lora_targets,
    progressive_alpha,
    set_qat_alpha,
    trainable_parameter_count,
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
    return {
        "n": len(refs),
        "wer": corpus_wer(refs, hyps),
        "cer": corpus_cer(refs, hyps),
    }


def set_precision(model, bit_width: float, alpha: float) -> None:
    for module in model.modules():
        if isinstance(module, BitLinear):
            module.bit_width = bit_width
            module.alpha = alpha


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    p.add_argument("--policy", default="decoder_attn")
    p.add_argument("--bit-width", type=float, default=1.0)
    p.add_argument("--group-size", type=int, default=0, help="0 = per-tensor")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--train-n", type=int, default=64)
    p.add_argument("--eval-n", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=256, help="Cap generation (bounds degenerate outputs).")
    p.add_argument("--use-lora", type=int, default=1)
    p.add_argument("--use-8bit", type=int, default=1)
    p.add_argument("--out", default="artifacts/stage1")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    group_size = args.group_size or None
    print("=== environment ===")
    print(json.dumps(backends.environment_report(), indent=2), flush=True)

    wrapper = backends.load_model(
        args.model, backend="transformers", dtype=torch.float32, device="cuda:0",
        max_new_tokens=args.max_new_tokens,
    )
    model = wrapper.model
    processor = wrapper.processor
    patch_outer_forward(model)

    print("=== data ===", flush=True)
    train_utts = list(iter_librispeech("clean", "validation", sample_n=args.train_n, sample_seed=0))
    eval_utts = list(iter_librispeech("clean", "test", sample_n=args.eval_n, sample_seed=0))
    train_examples = build_examples(train_utts, processor)
    print(f"train={len(train_examples)} eval={len(eval_utts)}", flush=True)

    results: dict = {"model": args.model, "policy": args.policy, "bit_width": args.bit_width,
                     "group_size": group_size, "steps": args.steps, "eval_n": len(eval_utts)}

    print("=== baseline (pretrained, fp32 shadow, alpha=0) ===", flush=True)
    results["pretrained"] = evaluate(wrapper, eval_utts, args.eval_n)
    print(json.dumps(results["pretrained"]), flush=True)

    apply_layer_policy(model, args.policy, bit_width=args.bit_width, group_size=group_size)
    model.to("cuda:0")
    replaced = getattr(model, "bitlinear_replaced", 0)
    results["bitlinear_replaced"] = replaced
    print(f"replaced {replaced} linears with BitLinear", flush=True)

    used_lora = False
    if args.use_lora:
        try:
            from asr1bit.train.qat import apply_lora

            model = apply_lora(model, target_modules=lora_targets(model))
            model.to("cuda:0")
            wrapper.model = model
            used_lora = True
            patch_outer_forward(model)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: LoRA unavailable ({exc}); training all params", flush=True)
    results["used_lora"] = used_lora
    enable_gradient_checkpointing(model)
    results["trainable_params"] = trainable_parameter_count(model)

    optimizer = build_optimizer(model, lr=args.lr, use_8bit=bool(args.use_8bit))
    config = QATConfig(steps=args.steps, lr=args.lr, use_8bit=bool(args.use_8bit))

    history = []
    batches = [
        collate(train_examples[i : i + args.batch_size], processor)
        for i in range(0, len(train_examples), args.batch_size)
    ]
    warmup = max(1, int(args.steps * config.alpha_warmup_frac))
    print("=== QAT ===", flush=True)
    started = time.perf_counter()
    for step in range(args.steps):
        alpha = progressive_alpha(step, warmup)
        set_qat_alpha(model, alpha)
        batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batches[step % len(batches)].items()}
        optimizer.zero_grad()
        out = model(**batch)
        loss = out.loss if hasattr(out, "loss") else out[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        if (step + 1) % 5 == 0 or step == 0:
            record = {"step": step + 1, "alpha": alpha, "loss": float(loss.detach())}
            history.append(record)
            print(json.dumps(record), flush=True)
    results["history"] = history
    results["qat_seconds"] = time.perf_counter() - started

    print("=== ablations (offline WER) ===", flush=True)
    # Restore fast inference: grad checkpointing off + KV cache on.
    disable = getattr(model, "gradient_checkpointing_disable", None)
    if callable(disable):
        disable()
    config_obj = getattr(model, "config", None)
    if config_obj is not None and hasattr(config_obj, "use_cache"):
        config_obj.use_cache = True
    model.eval()
    with torch.no_grad():
        set_precision(model, 1.0, 0.0)
        results["fp16"] = evaluate(wrapper, eval_utts, args.eval_n)
        set_precision(model, 1.0, 1.0)
        results["binary"] = evaluate(wrapper, eval_utts, args.eval_n)
        set_precision(model, 1.58, 1.0)
        results["ternary"] = evaluate(wrapper, eval_utts, args.eval_n)
    print(json.dumps({k: results[k] for k in ("fp16", "binary", "ternary")}, indent=2), flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.policy}_bw{args.bit_width}_g{group_size}"
    (out_dir / f"stage1_{tag}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", out_dir / f"stage1_{tag}.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
