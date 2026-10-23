"""BitLinear: drop-in replacement for ``nn.Linear`` with a bit-width switch.

Stage 1, T1.2. Holds fp32 shadow weights updated via a straight-through
estimator (STE), quantizes weights on the forward pass, and optionally
quantizes activations to INT8 (absmax, per-token).

``learnable_scale=True`` adds a BitNet b1.58-style learnable dequantization
gain (per-tensor or per-group, log-parameterized so it stays strictly positive,
initialized to ``mean|W|``). The rounding grid uses the detached scale while
gradients flow exactly to the scale parameter.

Locked interface: ``BitLinear(in_dim, out_dim, bit_width, quantize_activations)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from asr1bit.quant import (
    dequantize_weights,
    quantize_2bit,
    quantize_binary,
    quantize_ternary,
)

_TERNARY_WIDTHS = (1.58, "ternary")
_TWO_BIT_WIDTHS = (2.0, "2bit")


def absmax_quantize_activations(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Per-token absmax INT8 quantization (straight-through friendly)."""
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=-1, keepdim=True) / qmax
    scale = scale.clamp(min=torch.finfo(x.dtype).tiny)
    return torch.clamp(torch.round(x / scale), -qmax, qmax) * scale


class BitLinear(nn.Module):
    """Linear layer whose weights are binary (``bit_width=1``) or ternary
    (``bit_width=1.58``) on the forward pass, with an fp32 shadow weight."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bit_width: float = 1,
        quantize_activations: bool = True,
        group_size: int | None = None,
        bias: bool = False,
        learnable_scale: bool = False,
        stochastic: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bit_width = bit_width
        self.quantize_activations = quantize_activations
        self.group_size = group_size
        self.learnable_scale = learnable_scale
        self.stochastic = stochastic
        # Progressive QAT ramp: 0 = full precision, 1 = fully quantized.
        self.alpha = 1.0
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if learnable_scale:
            # Log-parameterized scale: ``weight_scale`` holds log(gamma), so the
            # effective scale stays strictly positive and updates are
            # multiplicative (BitNet-style tensor-wise learnable scale).
            self.weight_scale = nn.Parameter(torch.zeros(self._scale_shape()))
            self._init_weight_scale()
        else:
            self.register_parameter("weight_scale", None)

    def _scale_shape(self) -> tuple[int, ...]:
        """Shape of the learnable scale: scalar per-tensor, ``(groups, 1)`` grouped."""
        if self.group_size is None:
            return ()
        if self.weight.numel() % self.group_size != 0:
            raise ValueError(
                f"group_size {self.group_size} must divide weight numel {self.weight.numel()}"
            )
        return (self.weight.numel() // self.group_size, 1)

    def gamma(self) -> torch.Tensor:
        """Effective positive dequantization scale (exp of the log parameter)."""
        return self.weight_scale.exp().clamp(min=1e-8, max=1.0)

    def _init_weight_scale(self) -> None:
        """Re-initialize the log-scale so ``gamma`` starts at the observed ``mean|W|``."""
        if self.weight_scale is None:
            return
        with torch.no_grad():
            weight = self.weight.detach()
            if self.group_size is None:
                value = weight.abs().mean().reshape(())
            else:
                value = weight.reshape(-1, self.group_size).abs().mean(dim=1, keepdim=True)
            self.weight_scale.copy_(value.clamp(min=1e-8).log())

    def _quantized_weight(self) -> torch.Tensor:
        if self.learnable_scale:
            gamma = self.gamma()
            round_stochastic = self.stochastic and self.training
            if self.bit_width in _TERNARY_WIDTHS:
                qw, _ = quantize_ternary(
                    self.weight, self.group_size, gamma=gamma.detach(), stochastic=round_stochastic
                )
            elif self.bit_width in _TWO_BIT_WIDTHS:
                qw, _ = quantize_2bit(self.weight, self.group_size, gamma=gamma.detach())
            else:
                qw, _ = quantize_binary(self.weight, self.group_size, stochastic=round_stochastic)
            dequantized = dequantize_weights(qw, gamma, self.group_size)
        else:
            round_stochastic = self.stochastic and self.training
            if self.bit_width in _TERNARY_WIDTHS:
                qw, scale = quantize_ternary(self.weight, self.group_size, stochastic=round_stochastic)
            elif self.bit_width in _TWO_BIT_WIDTHS:
                qw, scale = quantize_2bit(self.weight, self.group_size)
            else:
                qw, scale = quantize_binary(self.weight, self.group_size, stochastic=round_stochastic)
            dequantized = dequantize_weights(qw, scale, self.group_size)
        # Progressive ramp: alpha=0 is full precision, alpha=1 fully quantized.
        blended = (1.0 - self.alpha) * self.weight + self.alpha * dequantized
        # STE: forward uses the blended value, gradient flows to the shadow.
        output = self.weight + (blended - self.weight).detach()
        if self.learnable_scale:
            # Zero in forward, exact gradient for the scale in backward.
            scale_term = self.alpha * dequantized
            output = output + (scale_term - scale_term.detach())
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantize_activations:
            quantized = absmax_quantize_activations(x)
            x = x + (quantized - x).detach()
        return F.linear(x, self._quantized_weight(), self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs) -> BitLinear:
        """Build a BitLinear copying ``linear``'s weights/bias on its device."""
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            **kwargs,
        ).to(device=linear.weight.device)
        return layer.load_from_linear(linear)

    def load_from_linear(self, linear: nn.Linear) -> BitLinear:
        """Copy weights (and bias) from ``linear`` into this layer."""
        with torch.no_grad():
            self.weight.copy_(linear.weight.detach().float())
            if self.bias is not None and linear.bias is not None:
                self.bias.copy_(linear.bias.detach().float())
        self._init_weight_scale()
        return self

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bit_width={self.bit_width}, quantize_activations={self.quantize_activations}, "
            f"group_size={self.group_size}, learnable_scale={self.learnable_scale}"
        )
