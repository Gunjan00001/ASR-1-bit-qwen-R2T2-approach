"""BitLinear: drop-in replacement for ``nn.Linear`` with a bit-width switch.

Stage 1, T1.2. Holds fp32 shadow weights updated via a straight-through
estimator (STE), quantizes weights on the forward pass, and optionally
quantizes activations to INT8 (absmax, per-token).

Locked interface: ``BitLinear(in_dim, out_dim, bit_width, quantize_activations)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from asr1bit.quant import dequantize_weights, quantize_binary, quantize_ternary

_TERNARY_WIDTHS = (1.58, "ternary")


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
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bit_width = bit_width
        self.quantize_activations = quantize_activations
        self.group_size = group_size
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def _quantized_weight(self) -> torch.Tensor:
        if self.bit_width in _TERNARY_WIDTHS:
            qw, scale = quantize_ternary(self.weight, self.group_size)
        else:
            qw, scale = quantize_binary(self.weight, self.group_size)
        dequantized = dequantize_weights(qw, scale, self.group_size)
        # STE: forward uses the quantized value, gradient flows to the shadow.
        return self.weight + (dequantized - self.weight).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantize_activations:
            quantized = absmax_quantize_activations(x)
            x = x + (quantized - x).detach()
        return F.linear(x, self._quantized_weight(), self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs) -> BitLinear:
        """Build a BitLinear copying ``linear``'s weights/bias."""
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            **kwargs,
        )
        return layer.load_from_linear(linear)

    def load_from_linear(self, linear: nn.Linear) -> BitLinear:
        """Copy weights (and bias) from ``linear`` into this layer."""
        with torch.no_grad():
            self.weight.copy_(linear.weight.detach().float())
            if self.bias is not None and linear.bias is not None:
                self.bias.copy_(linear.bias.detach().float())
        return self

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bit_width={self.bit_width}, quantize_activations={self.quantize_activations}, "
            f"group_size={self.group_size}"
        )
