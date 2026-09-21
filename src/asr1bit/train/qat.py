"""Progressive quantization-aware training loop (Stage 1, T1.4).

Alpha ramp 0 -> 1 blends full-precision weights toward fully quantized weights,
with fp32 shadow weights + STE (see :class:`~asr1bit.bitlinear.BitLinear`).

Heavy integrations (bitsandbytes 8-bit Adam, peft LoRA) are imported lazily so
the helper logic is unit-testable without them.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from asr1bit.bitlinear import BitLinear


def progressive_alpha(
    step: int, total_steps: int, start: float = 0.0, end: float = 1.0
) -> float:
    """Linear alpha ramp from ``start`` to ``end`` over ``total_steps``."""
    if total_steps <= 0:
        return float(end)
    fraction = min(max(step, 0), total_steps) / total_steps
    return float(start + (end - start) * fraction)


def quantization_delay_alpha(step: int, delay_steps: int) -> float:
    """Quantization delay: pure fp (alpha=0) for ``delay_steps``, then alpha=1.

    Safer than the blended ramp: with an identity-STE, blending in the forward
    pass does not match the gradient and is a confirmed corruption trigger
    (see docs/LAB_NOTES.md). Delaying avoids any intermediate blend.
    """
    return 0.0 if step < delay_steps else 1.0


def set_qat_alpha(model: nn.Module, alpha: float) -> int:
    """Set the progressive alpha on every :class:`BitLinear`; returns the count."""
    count = 0
    for module in model.modules():
        if isinstance(module, BitLinear):
            module.alpha = float(alpha)
            count += 1
    return count


def set_activation_quant(model: nn.Module, enabled: bool) -> int:
    """Enable/disable INT8 activation quantization on all BitLinear layers."""
    count = 0
    for module in model.modules():
        if isinstance(module, BitLinear):
            module.quantize_activations = bool(enabled)
            count += 1
    return count


def non_bitlinear_linear_names(model: nn.Module) -> list[str]:
    """Names of ``nn.Linear`` modules that were *not* replaced by BitLinear."""
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and not isinstance(module, BitLinear)
    ]


def lora_targets(model: nn.Module, exclude: Sequence[str] = ("lm_head",)) -> list[str]:
    """Base module names for LoRA on the non-binary (non-BitLinear) linears."""
    names = set()
    for name in non_bitlinear_linear_names(model):
        base = name.rpartition(".")[2] or name
        if base not in exclude:
            names.add(base)
    return sorted(names)


def enable_gradient_checkpointing(model: nn.Module) -> None:
    """Enable gradient checkpointing if supported and disable the KV cache."""
    enable = getattr(model, "gradient_checkpointing_enable", None)
    if callable(enable):
        enable()
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = False


def apply_lora(
    model: nn.Module,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Sequence[str] | None = None,
) -> nn.Module:
    """Wrap the non-binary linear layers with LoRA adapters (lazy peft import)."""
    from peft import LoraConfig, get_peft_model

    targets = list(target_modules) if target_modules else lora_targets(model)
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def freeze_non_bitlinear(model: nn.Module) -> int:
    """Freeze every parameter except BitLinear shadow weights (QAT-only).

    Used when LoRA is unavailable so a QAT run updates only the quantized
    projections instead of the whole model (which diverges). Returns the number
    of trainable parameters remaining.
    """
    for param in model.parameters():
        param.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, BitLinear):
            module.weight.requires_grad_(True)
            if module.bias is not None:
                module.bias.requires_grad_(True)
    return trainable_parameter_count(model)


def build_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    use_8bit: bool = True,
) -> torch.optim.Optimizer:
    """Optimizer over trainable params (8-bit Adam when available)."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("model has no trainable parameters")
    if use_8bit:
        try:
            import bitsandbytes as bnb

            return bnb.optim.Adam8bit(params, lr=lr, weight_decay=weight_decay)
        except Exception as exc:  # noqa: BLE001 - fall back to AdamW
            warnings.warn(f"bitsandbytes unavailable ({exc}); using AdamW", stacklevel=2)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def make_sft_labels(
    input_ids: torch.Tensor,
    prefix_lens: Sequence[int],
    pad_token_id: int | None = None,
) -> torch.Tensor:
    """Mask the prompt (audio + ``language ...<asr_text>`` prefix) out of labels.

    Labels are ``1`` (i.e. not ``-100``) only over the transcription tokens,
    matching Qwen's official ``finetuning/qwen3_asr_sft.py`` convention.
    """
    labels = input_ids.clone()
    for i, prefix_len in enumerate(prefix_lens):
        labels[i, : int(prefix_len)] = -100
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100
    return labels


def grad_norms(model: nn.Module) -> dict[str, float]:
    """Per-parameter gradient L2 norms (only params that have a gradient)."""
    norms: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = float(param.grad.detach().norm())
    return norms


def check_finite_grads(model: nn.Module) -> None:
    """Raise ``FloatingPointError`` if any gradient is non-finite."""
    for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            raise FloatingPointError(f"non-finite gradient in parameter: {name}")


def train_step(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float = 1.0,
) -> float:
    """One optimization step; returns the loss value."""
    optimizer.zero_grad()
    logits = model(inputs)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return float(loss.detach())


@dataclass
class QATConfig:
    """Settings for a progressive QAT run."""

    steps: int = 200
    lr: float = 1e-4
    alpha_warmup_frac: float = 0.3
    grad_accum: int = 4
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    max_grad_norm: float = 1.0
    use_8bit: bool = True
    seed: int = 0
    log_every: int = 10
    history: list[dict[str, float]] = field(default_factory=list)


def run_qat(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    config: QATConfig,
    optimizer: torch.optim.Optimizer | None = None,
) -> list[dict[str, float]]:
    """Run progressive QAT over ``batches``; returns the loss history.

    Each item of a batch is a list of scalar losses (for gradient accumulation).
    ``batches`` yields ``(inputs, labels)`` tensors for a single forward loss.
    """
    if optimizer is None:
        optimizer = build_optimizer(model, lr=config.lr, use_8bit=config.use_8bit)

    warmup = max(1, int(config.steps * config.alpha_warmup_frac))
    history: list[dict[str, float]] = []
    data = list(batches)
    if not data:
        return history

    accumulation: list[float] = []
    for step in range(config.steps):
        alpha = progressive_alpha(step, warmup)
        set_qat_alpha(model, alpha)
        inputs, labels = data[step % len(data)]
        loss = train_step(model, inputs, labels, optimizer, config.max_grad_norm)
        accumulation.append(loss)
        if (step + 1) % config.grad_accum == 0:
            mean_loss = sum(accumulation) / len(accumulation)
            accumulation = []
            record = {"step": float(step + 1), "alpha": alpha, "loss": mean_loss}
            history.append(record)
            config.history.append(record)
            if config.log_every and (len(history) % config.log_every == 0):
                print(f"step {step + 1}/{config.steps} alpha={alpha:.3f} loss={mean_loss:.4f}", flush=True)
    return history


def trainable_parameter_count(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def state_dict_size_mb(state: dict[str, Any]) -> float:
    """Approximate size of a state dict in MB."""
    total = 0
    for value in state.values():
        if isinstance(value, torch.Tensor):
            total += value.numel() * value.element_size()
    return total / 1e6
