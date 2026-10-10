"""Weight quantizers for 1-bit / ternary linear layers (Stage 1, T1.1).

Locked interfaces::

    quantize_binary(weight, group_size=None)            -> (qW, scale)
    quantize_ternary(weight, group_size=None, gamma=None) -> (qW, scale)
    dequantize_weights(qW, scale, group_size=None)      -> weight

Binary:  ``q = sign(W - mean(W))``, ``scale = mean|W|`` (sign(>=0) -> +1).
Ternary: ``q = RoundClip(W / gamma, -1, 1)``, ``gamma = mean|W|`` (or a
caller-supplied ``gamma`` so :class:`~asr1bit.bitlinear.BitLinear` can use a
learnable per-tensor/per-group scale).

Group-wise scaling (Bonsai-style) reduces along the last dimension in blocks of
``group_size``; ``scale`` then has shape ``(num_groups, 1)`` and broadcasts when
the weight is viewed as ``(-1, group_size)``. ``group_size=None`` is per-tensor.
"""

from __future__ import annotations

import torch


def _group_view(weight: torch.Tensor, group_size: int | None) -> tuple[torch.Tensor, int]:
    if group_size is None:
        return weight.reshape(1, -1), weight.numel()
    if weight.shape[-1] % group_size != 0:
        raise ValueError(
            f"group_size {group_size} must divide last dim {weight.shape[-1]}"
        )
    return weight.reshape(-1, group_size), group_size


def quantize_binary(weight: torch.Tensor, group_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``weight`` to ``{-1, +1}`` with a centered-sign rule.

    Returns the quantized weights (same shape as ``weight``) and the scale
    (scalar for per-tensor, ``(num_groups, 1)`` for group-wise).
    """
    original_shape = weight.shape
    grouped, _ = _group_view(weight, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    centered = grouped - mean
    q = torch.where(centered >= 0, 1.0, -1.0).to(weight.dtype)
    scale = grouped.abs().mean(dim=1, keepdim=True).to(weight.dtype)
    if group_size is None:
        return q.reshape(original_shape), scale.reshape(())
    return q.reshape(original_shape), scale


def quantize_ternary(
    weight: torch.Tensor,
    group_size: int | None = None,
    gamma: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``weight`` to ``{-1, 0, +1}`` via RoundClip.

    Args:
        weight: Weight matrix, any shape.
        group_size: Optional block size along the last dimension.
        gamma: Optional rounding scale (scalar or ``(num_groups, 1)``). When
            omitted the observed ``mean|W|`` per tensor/group is used. A
            learnable scale passes ``gamma.detach()`` here for the rounding
            grid while keeping the parameter as the dequantization gain.

    Returns:
        ``(q, scale)`` where ``q`` holds the ``{-1, 0, +1}`` values and
        ``scale`` is the gamma actually used.
    """
    original_shape = weight.shape
    grouped, _ = _group_view(weight, group_size)
    if gamma is None:
        gamma_eff = grouped.abs().mean(dim=1, keepdim=True).to(weight.dtype)
    else:
        gamma_eff = gamma.to(device=grouped.device, dtype=weight.dtype).reshape(-1, 1)
        if gamma_eff.shape[0] != grouped.shape[0]:
            raise ValueError(
                f"gamma has {gamma_eff.shape[0]} scales but weight has {grouped.shape[0]} groups"
            )
    q = torch.clamp(
        torch.round(grouped / gamma_eff.clamp(min=torch.finfo(weight.dtype).tiny)), -1.0, 1.0
    )
    q = torch.where(gamma_eff == 0, torch.zeros_like(q), q).to(weight.dtype)
    if group_size is None:
        return q.reshape(original_shape), gamma_eff.reshape(())
    return q.reshape(original_shape), gamma_eff


def dequantize_weights(
    qw: torch.Tensor, scale: torch.Tensor, group_size: int | None = None
) -> torch.Tensor:
    """Reconstruct fp weights from quantized values and scale."""
    if group_size is None:
        return qw * scale
    return (qw.reshape(-1, group_size) * scale).reshape(qw.shape)
