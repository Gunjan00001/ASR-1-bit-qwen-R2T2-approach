"""Trainer CLI for quantized (ternary / near-binary) Qwen3-ASR fine-tuning.

This file is intentionally the *orchestrator only*: argument parsing, the
phase order, and the training loop. The pieces it drives live in focused
modules so each one can be read (and tested) on its own::

    src/asr1bit/train/checkpoints.py   save/load/select checkpoints
    src/asr1bit/train/dataset.py       ordinary transcription examples + batching
    src/asr1bit/train/aligned.py       forced-aligner streaming targets (the fix)
    src/asr1bit/train/onpolicy.py      rollouts, distillation, degenerate filtering
    src/asr1bit/train/inloop_eval.py   cheap in-training evaluations (guardrails)
    src/asr1bit/train/schedules.py     warmup + cosine LR
    src/asr1bit/train/qat.py           the loss, labels and distillation terms
    src/asr1bit/train/samples.py       the streaming-sample record + mixing
    src/asr1bit/train/gold.py          aligning the transcript against what was said
    src/asr1bit/train/filters.py       degenerate-output detection (the DAgger guard)
    src/asr1bit/train/rollout.py       batched replay of the official streaming algorithm
    src/asr1bit/train/examples.py      on-policy + distilled example builders

Run phases (each marked in ``main`` with a comment):

    1. read config + dump the environment (reproducibility)
    2. load the model, optionally a frozen fp16 teacher (the KL "anchor")
    3. evaluate the untouched model  -> the ruler for everything after
    4. replace chosen projections with quantized layers (the policy)
    5. build the batch: ordinary samples + (optional) streaming samples
    6. train: clip, step, log, save, and run the cheap in-loop checks
    7. final save + end-of-run evaluation, all recorded into one JSON

Stabilized debug configuration (see docs/LAB_NOTES.md):
- only BitLinear shadows are trainable (frozen QAT setup, not a fallback),
- fp32 master weights + fp32 AdamW by default (8-bit Adam behind a flag),
- no autocast, gradient checkpointing OFF, activation quant OFF by default,
- constant alpha=1.0 by default (alpha ramp behind a flag),
- per-layer gradient norms logged and finiteness asserted before every step.

Enable the risky features one at a time via flags to find which one breaks
training. Streaming/vLLM eval is deferred to scripts/eval_streaming.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

from asr1bit.data.load import iter_librispeech
from asr1bit.qwen import backends
from asr1bit.replace import apply_layer_policy, apply_mixed_precision
from asr1bit.train.aligned import build_aligned_examples
from asr1bit.train.checkpoints import (
    load_checkpoint,
    save_checkpoint,
    step_checkpoint_path,
)
from asr1bit.train.cotrain import cotrain_alpha
from asr1bit.train.dataset import build_examples, collate, load_train_utterances
from asr1bit.train.inloop_eval import (
    evaluate,
    evaluate_precisions,
    evaluate_streaming,
    set_precision,
)
from asr1bit.train.model_prep import patch_outer_forward
from asr1bit.train.onpolicy import (
    collect_op_examples,
)
from asr1bit.train.qat import (
    build_optimizer,
    check_finite_grads,
    distillation_loss,
    freeze_non_bitlinear,
    grad_norms,
    progressive_alpha,
    quantization_delay_alpha,
    scale_stats,
    set_activation_quant,
    set_qat_alpha,
)
from asr1bit.train.qec import QECController
from asr1bit.train.samples import interleave_with_ratio, select_evenly
from asr1bit.train.schedules import build_lr_lambda


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    p.add_argument("--policy", default="decoder_attn")
    p.add_argument(
        "--fp16-layers-file", default=None,
        help="JSON list (or one-per-line text) of decoder projection names kept "
             "at full precision while every other decoder projection is "
             "quantized. The mixed model is the only representation change.",
    )
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
    p.add_argument(
        "--qec-weight", type=float, default=0.0,
        help="Weight of the output-aware quantization-error compensation term "
             "(0 = disabled; training-only, adds no parameters and does not "
             "change the deployed ternary model).",
    )
    p.add_argument("--qec-stride", type=int, default=1,
                   help="Apply QEC to every N-th BitLinear (1 = all layers).")
    p.add_argument("--qec-eps", type=float, default=1e-6,
                   help="Denominator floor for the relative output error.")
    p.add_argument(
        "--cotrain-fp-every", type=int, default=0,
        help="Every N-th step after the quantization delay runs at alpha=0 "
             "(full precision) while the rest run at the scheduled ternary "
             "alpha. 0 = disabled; the deployed model remains ternary.",
    )
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
    p.add_argument("--quant-stochastic", type=int, default=0,
                   help="1 = unbiased stochastic rounding of the quantized weights during training "
                        "(the 'stochastic precision' lever); evals always round deterministically. "
                        "CAVEAT: the first controlled A/B (1.7B recovery, 2026-10-16) measured it "
                        "costing ~3.4 WER at step 500 (24.15 vs 20.76 without) - do not enable "
                        "without a measured reason (docs/LAB_NOTES.md).")
    p.add_argument("--keep-step-checkpoints", type=int, default=0,
                   help="Also write step-numbered checkpoints every N steps (0=off) so the best "
                        "state stays selectable (a last.pt-only run loses earlier states).")
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
    p.add_argument(
        "--silence-samples-sec", type=float, default=None,
        help="Also add a near-empty-audio sample per utterance (truncated to this many "
             "seconds, empty target) so the model learns to emit EOS on warm-up chunks.",
    )
    p.add_argument(
        "--op-rollout", type=int, default=0,
        help="1 = on-policy streaming SFT: replay the official streaming algorithm with "
             "the current model, train on continuations of its own prefixes, and mix "
             "them with the standard samples (Stage 4).",
    )
    p.add_argument("--op-rollout-n", type=int, default=600, help="Utterances for the on-policy rollout.")
    p.add_argument("--op-distill", type=int, default=0,
                   help="1 = streaming-state distillation: the frozen pretrained teacher drives the "
                        "rollout and the student clones its continuations (requires --frozen-teacher 1).")
    p.add_argument("--op-batch-size", type=int, default=6, help="Utterances decoded per rollout call.")
    p.add_argument("--op-max-new-tokens", type=int, default=32,
                   help="Per-chunk generation cap during the rollout (match the streaming eval).")
    p.add_argument("--op-chunk-ms", type=int, default=320, help="Chunk length of the on-policy rollout.")
    p.add_argument("--op-chunk-stride", type=int, default=3, help="Keep every Nth mid-stream chunk state.")
    p.add_argument("--op-max-rounds", type=int, default=16,
                   help="Cap rollout rounds per utterance (bounds the O(n^2) re-feed; 0 = no cap).")
    p.add_argument("--op-max-samples", type=int, default=2400,
                   help="Cap on the on-policy example pool. NOTE: this also caps the "
                        "*aligned* example pool (the aligned builder shares the pool), so "
                        "raising --aligned-n alone cannot exceed this default - set both "
                        "(2026-10-18: an intended 6000-aligned run pooled exactly 2400 "
                        "until this flag was raised too).")
    p.add_argument("--op-mix-frac", type=float, default=0.5,
                   help="Target share of on-policy samples in the training mix.")
    p.add_argument("--op-refresh-every", type=int, default=0,
                   help="Re-run the rollout every N steps with the current model (0 = once).")
    p.add_argument("--op-warmup-eos", type=int, default=1,
                   help="1 = warm-up chunks (no prefix) train an immediate EOS.")
    p.add_argument("--op-filter-degenerate", type=int, default=1,
                   help="1 = drop on-policy states whose own text is already a repetition loop "
                        "(guards against DAgger poisoning; distillation ignores this flag).")
    p.add_argument("--stream-eval-every", type=int, default=0,
                   help="Run the official-algorithm streaming eval on a small eval slice every N "
                        "steps (0 = off). The offline eval cannot see streaming regressions.")
    p.add_argument("--stream-eval-n", type=int, default=10,
                   help="Utterances in the in-training streaming eval (kept small: it is slow).")
    p.add_argument("--aligned-targets", type=int, default=0,
                   help="1 = build streaming SFT samples from forced-aligner word timestamps "
                        "(T4.3): each chunk's target is exactly the words whose end time falls "
                        "inside it (+ lookahead). Mixed with the standard samples via --op-mix-frac.")
    p.add_argument("--aligned-n", type=int, default=800,
                   help="Train utterances to align for --aligned-targets (aligning is cheap).")
    p.add_argument("--aligned-lookahead-ms", type=int, default=0,
                   help="Extra audio the aligned target may include (0 = strictly causal).")
    p.add_argument("--aligned-stop-stride", type=int, default=2,
                   help="Keep every N-th empty-target (stop) aligned sample (half of all chunks "
                        "have no new words; stride 1 keeps them all).")
    p.add_argument("--out", default="artifacts/stage1")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.op_rollout and not args.lazy_collate:
        print("--op-rollout enables --lazy-collate (the mix changes across refreshes)", flush=True)
        args.lazy_collate = 1
    group_size = args.group_size or None
    fp16_layers = None
    if args.fp16_layers_file:
        path = Path(args.fp16_layers_file)
        if path.suffix == ".json":
            fp16_layers = json.loads(path.read_text(encoding="utf-8"))
        else:
            fp16_layers = [
                line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        if not fp16_layers:
            raise ValueError(f"empty fp16 layer list: {path}")
        print(f"mixed precision: {len(fp16_layers)} fp16 layers from {path}", flush=True)
# Freeze the exact configuration into the results: every number a run
# produces must be traceable to the flags that produced it.
    config_record = vars(args) | {"group_size": group_size, "fp16_layers": fp16_layers}
    print("=== environment ===")
    print(json.dumps(backends.environment_report(), indent=2), flush=True)
    print("=== config ===")
    print(json.dumps(config_record, indent=2), flush=True)

# Load the ASR model through our portable backend (transformers rather than
# vLLM: the cluster's driver predates what vLLM needs).
    wrapper = backends.load_model(
        args.model, backend="transformers", dtype=torch.float32, device="cuda:0",
        max_new_tokens=args.max_new_tokens,
    )
    model = wrapper.model
    processor = wrapper.processor
    patch_outer_forward(model)

    teacher = None
    teacher_wrapper = None
# Optional frozen fp16 teacher (the 'anchor'). Its KL term is what stops a
# quantized model from drifting into a bad basin and never coming back - the
# exact failure the 1.7B retry hit before the anchor was added.
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

# Training data: a small cache while developing, the full 28.5k-utterance
# cache for production runs. Cached audio means the GPU never waits on I/O.
    train_utts = load_train_utterances(args)
    eval_utts = list(iter_librispeech("clean", "test", sample_n=args.eval_n, sample_seed=0))
    prefix_fracs = None
    if args.prefix_sft_fracs:
        prefix_fracs = tuple(float(x) for x in args.prefix_sft_fracs.split(",") if x.strip())
    train_examples = build_examples(
        train_utts, processor, prefix_fracs=prefix_fracs,
        silence_sec=args.silence_samples_sec,
    )
    print(f"train={len(train_examples)} eval={len(eval_utts)} prefix_fracs={prefix_fracs} "
          f"silence_sec={args.silence_samples_sec}", flush=True)

    results: dict = {"config": config_record, "eval_n": len(eval_utts), "frozen_teacher": teacher is not None}

    print("=== baseline (pretrained) ===", flush=True)
# Measure the model BEFORE any modification, on the same fixed utterances
# used later. This is the ruler for every comparison that follows.
    results["pretrained"] = evaluate(wrapper, eval_utts, args.eval_n)
    print(json.dumps(results["pretrained"]), flush=True)

    if fp16_layers:
        apply_mixed_precision(
            model,
            set(fp16_layers),
            bit_width=args.bit_width,
            group_size=group_size,
            learnable_scale=bool(args.learnable_scale),
            stochastic=bool(args.quant_stochastic),
        )
    else:
        apply_layer_policy(
            model,
            args.policy,
            bit_width=args.bit_width,
            group_size=group_size,
            learnable_scale=bool(args.learnable_scale),
            stochastic=bool(args.quant_stochastic),
        )
# Replace the chosen projections with quantized layers. The policy decides
# *which* matrices (attention only / +FFN / all), the bit width decides how
# aggressive the quantization is.
    model.to("cuda:0")
    results["bitlinear_replaced"] = getattr(model, "bitlinear_replaced", 0)
    results["fp16_layers"] = fp16_layers

    # QAT-only: freeze everything except BitLinear shadows and learnable scales.
    trainable = freeze_non_bitlinear(model)
    set_activation_quant(model, bool(args.activation_quant))
    qec = None
    if args.qec_weight > 0:
        qec = QECController(model, stride=args.qec_stride, eps=args.qec_eps)
        print(
            f"QEC enabled: weight={args.qec_weight}, stride={args.qec_stride}, "
            f"eps={args.qec_eps}",
            flush=True,
        )
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
# Count what we are about to train. Frozen modules must not require grad, or
# the optimizer silently allocates buffers for them and the run OOMs.
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

# Optimizer + LR schedule (warmup then cosine). adam8bit trades a little
# precision for ~4x less optimizer memory; fp32 AdamW is the conservative
# choice for the 1.7B, where a bad update can be permanent.
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
# Two ways to continue: init_from starts a fresh phase with a new schedule
# (the recipe's 'warm start'); resume also restores optimizer + scheduler
# state and continues the same schedule.
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

    standard_examples = train_examples
    op_utterances: list = []
    op_examples: list[dict] = []
# Assemble the training batch. The aligned-targets path builds streaming
# examples from forced-aligner timestamps and interleaves them with ordinary
# transcription samples so content quality does not regress.
    if args.aligned_targets:
        # The aligned pool has its own size: load a separate corpus slice with
        # `--aligned-n` as the cap. Using `train_utts` here silently capped a
        # 6 000-aligned run at `--train-n` (2 400) twice (2026-10-18, 2026-10-20).
        aligned_utts = load_train_utterances(args, limit=args.aligned_n)
        strip = len(aligned_utts)
        print(f"=== forced-aligner targets: aligning {strip} train utterances (T4.1) ===", flush=True)
        from qwen_asr import Qwen3ForcedAligner

        from asr1bit.data.align import DEFAULT_ALIGNER_MODEL
        from asr1bit.train.aligned_cache import WordsCache, utts_key

        # Cache the aligner's words (the expensive part) so retries and re-runs
        # with the same utterance set never pay it twice. Keyed by the exact
        # utterance set + aligner model, so a stale hit is impossible.
        words_cache = WordsCache(utts_key(aligned_utts, DEFAULT_ALIGNER_MODEL))
        aligner = Qwen3ForcedAligner.from_pretrained(DEFAULT_ALIGNER_MODEL)
        align_started = time.perf_counter()

        def align_progress(done: int, total: int) -> None:
            print(f"aligned {done}/{total} utts ({time.perf_counter() - align_started:.0f}s)", flush=True)

        aligned = build_aligned_examples(
            aligned_utts,
            processor,
            aligner,
            chunk_ms=args.op_chunk_ms,
            lookahead_ms=args.aligned_lookahead_ms,
            stop_stride=args.aligned_stop_stride,
            progress=align_progress,
            words_cache=words_cache,
        )
        words_cache.save()
        del aligner
        torch.cuda.empty_cache()
        op_utterances = aligned_utts
        op_examples = select_evenly(aligned, args.op_max_samples)
        train_examples = interleave_with_ratio(standard_examples, op_examples, args.op_mix_frac)
        print(
            f"aligned targets: {len(aligned)} examples -> {len(op_examples)} pooled | "
            f"mix {len(train_examples)} ({len(standard_examples)} standard + {len(op_examples)} aligned)",
            flush=True,
        )
    elif args.op_rollout:
        op_utterances = train_utts[: args.op_rollout_n]
        mode = (
            "streaming-state distillation (teacher rollout)"
            if args.op_distill
            else "on-policy rollout"
        )
        print(
            f"=== {mode} ({len(op_utterances)} utts, {args.op_chunk_ms} ms chunks) ===",
            flush=True,
        )
        op_examples = collect_op_examples(args, wrapper, model, teacher_wrapper, op_utterances)
        train_examples = interleave_with_ratio(standard_examples, op_examples, args.op_mix_frac)
        print(
            f"{mode}: {len(op_examples)} examples | mix {len(train_examples)} "
            f"({len(standard_examples)} standard + {len(op_examples)} streaming)",
            flush=True,
        )

    print("=== QAT ===", flush=True)
    started = time.perf_counter()
# --- the training loop -------------------------------------------------
# Each iteration: build the batch (possibly with a fresh on-policy rollout),
# forward/backward, step the scheduler, and - at the requested cadence -
# run the cheap in-loop evaluations and write checkpoints.
    for step in range(start_step, args.steps):
        if args.alpha_mode == "ramp":
            alpha = progressive_alpha(step, alpha_warmup)
        elif args.alpha_mode == "delay":
            alpha = quantization_delay_alpha(step, alpha_warmup)
        else:
            alpha = 1.0
        if args.cotrain_fp_every > 0:
            alpha = cotrain_alpha(
                step, alpha, fp_every=args.cotrain_fp_every, start_step=alpha_warmup
            )
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
        qec_value = None
        if qec is not None:
            qec.reset()
        if args.distill_weight > 0 or qec is not None:
            teacher_out = None
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
            loss = ce
            if args.distill_weight > 0:
                kl = distillation_loss(
                    out.logits, teacher_out.logits, batch["labels"],
                    temperature=args.distill_temp,
                )
                loss = (1.0 - args.distill_weight) * ce + args.distill_weight * kl
                kl_value = float(kl.detach())
            qec_loss = qec.loss() if qec is not None else None
            if qec_loss is not None:
                loss = loss + args.qec_weight * qec_loss
                qec_value = float(qec_loss.detach())
            ce_value = float(ce.detach())
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
            if qec_value is not None:
                record["qec"] = qec_value
                record["qec_weight"] = args.qec_weight
            if args.cotrain_fp_every > 0:
                record["cotrain_fp"] = alpha == 0.0
            record.update(scale_stats(model))
            history.append(record)
            print(json.dumps(record), flush=True)
        if (
            args.op_rollout
            and not args.aligned_targets
            and args.op_refresh_every
            and (step + 1) % args.op_refresh_every == 0
        ):
            print(f"=== streaming rollout refresh @ step {step + 1} ===", flush=True)
            op_examples = collect_op_examples(
                args, wrapper, model, teacher_wrapper, op_utterances, restore_alpha=alpha
            )
            train_examples = interleave_with_ratio(standard_examples, op_examples, args.op_mix_frac)
            print(
                "OPREFRESH "
                + json.dumps(
                    {"step": step + 1, "op_examples": len(op_examples), "mix": len(train_examples)}
                ),
                flush=True,
            )
        if args.stream_eval_every and (step + 1) % args.stream_eval_every == 0:
            model.eval()
            with torch.no_grad():
                stream = evaluate_streaming(
                    wrapper, model, eval_utts, args.stream_eval_n,
                    chunk_ms=args.op_chunk_ms,
                )
            model.train()
            set_precision(model, args.bit_width, alpha)
            ev = {
                "step": step + 1,
                "stream_wer": stream["wer"],
                "stream_cer": stream["cer"],
                "stream_n": stream["n"],
            }
            history.append(ev)
            print("STREAM " + json.dumps(ev), flush=True)
        if args.eval_every and (step + 1) % args.eval_every == 0:
            prec = evaluate_precisions(
                wrapper, model, eval_utts, args.eval_n,
                bit_width=args.bit_width, both=bool(args.eval_both),
                include_fp16=bool(args.eval_fp16),
            )
            ev = {"step": step + 1, "alpha": alpha,
                  "fp16_wer": prec["fp16"]["wer"] if "fp16" in prec else None,
                  "binary_wer": prec["binary"]["wer"] if "binary" in prec else None,
                  "ternary_wer": prec["ternary"]["wer"] if "ternary" in prec else None,
                  # non-ternary widths (e.g. 2-bit) are recorded under their own
                  # key by evaluate_precisions; surface them so a 2-bit run's
                  # offline readout is not silently missing from the EVAL line.
                  "quant_bw": args.bit_width,
                  "quant_wer": (prec.get(f"bw{args.bit_width:g}") or prec.get("ternary") or {}).get("wer")}
            history.append(ev)
            print("EVAL " + json.dumps(ev), flush=True)
            set_precision(model, args.bit_width, alpha)  # restore training precision
        if args.save_dir and args.save_every and (step + 1) % args.save_every == 0:
            save_checkpoint(Path(args.save_dir) / "last.pt", model, step + 1, args)
        # Step-numbered checkpoints run on their OWN cadence, deliberately not
        # nested inside `--save-every`: nesting them meant the two periods had to
        # coincide, so a run with --save-every 250 and --keep-step-checkpoints 400
        # only ever wrote at multiples of 2000 (2026-10-19: two 1.7B runs were
        # thought to lack the flag entirely; they had it, and the cadences simply
        # never met).
        step_path = step_checkpoint_path(args.save_dir, step + 1, args.keep_step_checkpoints)
        if step_path is not None and not step_path.exists():
            save_checkpoint(step_path, model, step + 1, args)
# Wrap up: persist the history, the timings, the final checkpoint, and run
# the end-of-run evaluation. The run's JSON is the artifact of record.
    results["history"] = history
    results["qat_seconds"] = time.perf_counter() - started
    if args.op_rollout or args.aligned_targets:
        results["on_policy_examples"] = len(op_examples)
        results["on_policy_utterances"] = len(op_utterances)
        results["mix_size"] = len(train_examples)
    if args.save_dir:
        save_checkpoint(Path(args.save_dir) / "last.pt", model, args.steps, args)
    if qec is not None:
        qec.close()  # eval is hook-free; remove the training-only observer

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
    if args.qec_weight > 0:
        # Never let a QEC arm overwrite the control's results JSON.
        tag += f"_qec{args.qec_weight:g}_s{args.qec_stride}"
    if args.cotrain_fp_every > 0:
        tag += f"_ct{args.cotrain_fp_every}"
    if fp16_layers:
        digest = hashlib.sha1("\n".join(sorted(fp16_layers)).encode()).hexdigest()[:8]
        tag += f"_mix{len(fp16_layers)}_{digest}"
    (out_dir / f"stage1_{tag}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", out_dir / f"stage1_{tag}.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
