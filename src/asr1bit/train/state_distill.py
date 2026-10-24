"""Streaming-state distillation: per-chunk hidden-state matching.

The existing distillation term (:func:`asr1bit.train.qat.distillation_loss`)
matches output logits on whole-utterance label positions. It cannot penalise a
drifting streaming state until the next tokens are already wrong. This module
adds the optional, disabled-by-default per-chunk term

    L_state = mean_{valid positions} || LN(h_student) - LN(h_teacher) ||^2

where both states are captured at the *same* decoder layer on the *same* batch.
Each aligned batch is already one streaming state: audio truncated at the chunk
boundary, a stable prefix, and only the words heard in the chunk as the target.
So the loss sees exactly the information the inference algorithm has at that
step — no future audio, no future text.

Design constraints (docs/streaming_state_distillation_design.md):

- teacher frozen, forward under ``no_grad``, states detached;
- teacher and student layer indices identical;
- positions restricted to supervised target tokens (``labels != -100``),
  non-padding positions (``attention_mask != 0``) and samples explicitly
  flagged as streaming states (``state_mask``);
- LayerNorm on both sides removes scale mismatch (same architecture, no
  projection needed);
- weight 0 removes the term exactly, leaving the existing path untouched.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "HiddenStateCapture",
    "combine_losses",
    "decoder_layer",
    "decoder_layer_count",
    "resolve_layer",
    "streaming_state_loss",
]


def _layer_list(model: nn.Module) -> nn.ModuleList:
    """The decoder layer list of a Qwen3-ASR student/teacher.

    Accepts the wrapped ASR model (``.thinker.model.layers``) and plain
    conditional-generation models (``.model.layers``).
    """
    thinker = getattr(model, "thinker", None)
    if thinker is not None:
        inner = getattr(thinker, "model", None)
        if inner is not None and hasattr(inner, "layers"):
            return inner.layers
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner.layers
    raise AttributeError(
        "model has no decoder layer list (expected .thinker.model.layers or .model.layers)"
    )


def decoder_layer_count(model: nn.Module) -> int:
    """Number of decoder layers in ``model``."""
    return len(_layer_list(model))


def resolve_layer(model: nn.Module, layer: int) -> int:
    """Resolve a possibly-negative layer index to a concrete one.

    ``layer < 0`` selects the middle decoder layer, the design's default: deep
    enough to carry content, shallow enough to stay close to the audio
    conditioning signal.
    """
    count = decoder_layer_count(model)
    if count == 0:
        raise ValueError("model has no decoder layers")
    if layer < 0:
        return count // 2
    if layer >= count:
        raise ValueError(f"state-distill layer {layer} out of range (model has {count} layers)")
    return int(layer)


def decoder_layer(model: nn.Module, layer: int) -> nn.Module:
    """The decoder layer module at ``layer`` (non-negative index)."""
    layers = _layer_list(model)
    return layers[layer]


class HiddenStateCapture:
    """Capture a module's forward output for the duration of a ``with`` block.

    Only the *last* forward is kept (training runs one forward per batch); the
    hook is always removed, including on exceptions.
    """

    def __init__(self, module: nn.Module):
        self.module = module
        self.value: torch.Tensor | None = None
        self._handle = None

    def _hook(self, _module, _inputs, output) -> None:
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(hidden, torch.Tensor):
            raise TypeError(f"expected a tensor output, got {type(hidden)!r}")
        self.value = hidden

    def __enter__(self) -> HiddenStateCapture:  # noqa: PYI034 - returns self, py3.10 target
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc) -> bool:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


def streaming_state_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    sample_mask: torch.Tensor | None = None,
    *,
    normalize: bool = True,
    normalize_eps: float = 1e-6,
) -> torch.Tensor:
    """Masked MSE between student and teacher hidden states.

    Args:
        student_hidden: ``(batch, seq, hidden)`` student states (grad-carrying).
        teacher_hidden: same shape, teacher states (detached by the caller).
        labels: ``(batch, seq)`` with ``-100`` at prompt/pad positions. A
            position participates only if its label is a supervised target
            token, i.e. the state is one the streaming objective asks the model
            to produce.
        attention_mask: optional ``(batch, seq)`` real-token mask.
        sample_mask: optional ``(batch,)`` or ``(batch, 1)`` mask selecting
            streaming-state samples (1) from ordinary full-utterance samples
            (0). A sample with no aligned target contributes nothing.
        normalize: apply LayerNorm over the hidden dimension to both sides, so
            the loss measures representation direction/magnitude in a
            scale-free space (the teacher is fp16 and the student's decoder is
            quantized, so raw scales differ).
        normalize_eps: LayerNorm epsilon.

    Returns:
        Scalar loss. When no position is valid the result is a zero tensor
        connected to ``student_hidden`` so ``backward()`` still works.
    """
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError(
            f"state shapes differ: student {tuple(student_hidden.shape)} vs "
            f"teacher {tuple(teacher_hidden.shape)}"
        )
    if labels.shape != student_hidden.shape[:2]:
        raise ValueError(
            f"labels shape {tuple(labels.shape)} does not match states "
            f"{tuple(student_hidden.shape[:2])}"
        )

    student = student_hidden
    teacher = teacher_hidden.detach()
    if normalize:
        student = F.layer_norm(student, (student.shape[-1],), eps=normalize_eps)
        teacher = F.layer_norm(teacher, (teacher.shape[-1],), eps=normalize_eps)

    valid = labels != -100
    if attention_mask is not None:
        valid = valid & (attention_mask != 0)
    if sample_mask is not None:
        mask = sample_mask.reshape(-1, 1).to(device=student.device, dtype=valid.dtype)
        valid = valid & (mask != 0)

    if not bool(valid.any()):
        return student.sum() * 0.0

    squared = (student - teacher).pow(2).mean(dim=-1)
    return squared[valid].mean()


def combine_losses(
    ce: torch.Tensor,
    kl: torch.Tensor | None,
    state: torch.Tensor | None,
    *,
    distill_weight: float,
    state_weight: float,
) -> torch.Tensor:
    """The documented objective ``(1-w)CE + w*KL + lambda_state*L_state``.

    With both weights zero this returns ``ce`` itself (same tensor), which is
    the compatibility guarantee for the existing training path.
    """
    if distill_weight <= 0:
        total = ce
    elif kl is None:
        raise ValueError("distill_weight > 0 requires a KL term")
    else:
        total = (1.0 - float(distill_weight)) * ce + float(distill_weight) * kl
    if state_weight > 0:
        if state is None:
            raise ValueError("state_weight > 0 requires a state loss")
        total = total + float(state_weight) * state
    return total
