"""Output-aware quantization-error compensation (QEC).

Why this module exists: the project's failed quantizer experiments show that
*weight-space* reconstruction does not predict ASR WER (g64 improved
reconstruction but lost 0.5-2.5 WER; the 2-bit quantizer had lower
reconstruction error and lost 13 WER points). The damage that matters is the
difference between the ternary layer's *function* and the fp shadow's function
on the activations the model actually sees.

QEC adds a training-only regularizer, per BitLinear layer:

    L_qec = || Q(W) x - W x ||^2 / (|| W x ||^2 + eps)

where ``x`` is the (already activation-quantized) input the layer received.
The target ``W x`` is detached, so the gradient flows only through the
quantized path and pushes the shadow weights toward a ternary function that
reproduces the full-precision response. At inference the regularizer does not
exist: the deployed model is exactly the same BitLinear as before, so the
ternary representation, packed size and streaming algorithm are unchanged.

The controller is disabled by default in the trainer (``--qec-weight 0``),
skips eval-mode forwards, and adds no parameters, so checkpoints remain
compatible with the baseline.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from asr1bit.bitlinear import BitLinear, absmax_quantize_activations

__all__ = ["QECController", "relative_output_error"]


def relative_output_error(
    quantized_out: torch.Tensor,
    full_precision_out: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Scale-free MSE between a quantized layer's output and its fp target.

    Both tensors must have the same shape. The target is detached: QEC is a
    regularizer on the quantized function, not an objective that moves the
    target toward the quantized output.
    """
    if quantized_out.shape != full_precision_out.shape:
        raise ValueError(
            f"output shapes differ: {tuple(quantized_out.shape)} vs "
            f"{tuple(full_precision_out.shape)}"
        )
    target = full_precision_out.detach()
    denominator = target.pow(2).mean().clamp(min=eps)
    return (quantized_out - target).pow(2).mean() / denominator


class QECController:
    """Accumulate per-layer output errors during a training forward.

    Args:
        model: model containing :class:`BitLinear` modules (usually the QAT
            student after ``apply_layer_policy``).
        stride: apply the regularizer to every ``stride``-th BitLinear (1 =
            all layers). A stride > 1 is a compute-saving ablation, not the
            primary configuration.
        eps: denominator floor for the relative error.

    The controller registers forward hooks; call :meth:`reset` before each
    training forward and :meth:`loss` after it (before ``backward``). Hooks are
    removed by :meth:`close`.
    """

    def __init__(self, model: nn.Module, *, stride: int = 1, eps: float = 1e-6):
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.eps = float(eps)
        self.stride = int(stride)
        self.enabled = True
        self._losses: list[torch.Tensor] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        layers = [module for module in model.modules() if isinstance(module, BitLinear)]
        for index, module in enumerate(layers):
            if index % self.stride == 0:
                self._handles.append(module.register_forward_hook(self._hook))
        if not self._handles:
            raise ValueError("QEC found no BitLinear modules in the model")

    def _hook(self, module: BitLinear, inputs: tuple, output: torch.Tensor) -> None:
        if not self.enabled or not module.training or module.alpha <= 0:
            return
        x = inputs[0]
        if module.quantize_activations:
            # The module's forward uses the fake-quantized input; the fp target
            # must use exactly the same input, or QEC would measure activation
            # quantization instead of weight quantization.
            x = absmax_quantize_activations(x)
        full_precision = F.linear(x, module.weight, module.bias)
        self._losses.append(relative_output_error(output, full_precision, eps=self.eps))

    def reset(self) -> None:
        """Clear the losses from the previous step."""
        self._losses.clear()

    def loss(self) -> torch.Tensor | None:
        """Mean relative output error over the hooked layers, or None."""
        if not self._losses:
            return None
        return torch.stack(self._losses).mean()

    def close(self) -> None:
        """Remove all hooks and disable the controller."""
        self.enabled = False
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._losses.clear()

    def __enter__(self) -> QECController:  # noqa: PYI034 - returns self, py3.10 target
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


def hooked_layers(controller: QECController) -> Iterable[torch.utils.hooks.RemovableHandle]:
    """Expose the registered handles (used by tests to count layers)."""
    return tuple(controller._handles)
