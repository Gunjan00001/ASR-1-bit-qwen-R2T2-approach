"""Bonsai-inspired structured rotation for ternary quantization (experimental).

Kept separate from the baseline quantizer on purpose: nothing here is imported
by ``quant.py`` or ``bitlinear.py``, so the shipped model path stays untouched.

Math: for a linear layer ``y = x W^T``, choose an orthogonal ``R`` acting on the
input features and use ``W' = W R^T``. Then ``x' W'^T = x R (W R^T)^T =
x R R^T W^T = x W^T``: transforming the activations with ``R`` and the weights
with ``R^T`` leaves the function identical, while the *quantizer* sees a
different weight matrix. The hypothesis under test is that the rotated matrix
is friendlier to ternary rounding (fewer outliers per group), not that the
transform changes the model.

The transform is blockwise and deterministic: a normalized Sylvester Hadamard
matrix per block, pre-multiplied by a fixed +/-1 diagonal drawn from a seeded
generator. Dimensions that are not a multiple of the block size keep their
remainder features unrotated (identity), which preserves exact equivalence.
"""

from __future__ import annotations

import torch

__all__ = [
    "hadamard",
    "rotate_activations",
    "rotate_weight",
    "signed_rotation",
    "unrotate_activations",
]


def hadamard(n: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Normalized Sylvester Hadamard matrix (``n`` must be a power of two)."""
    if n < 1 or n & (n - 1):
        raise ValueError(f"hadamard size must be a power of two, got {n}")
    h = torch.ones(1, 1, dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h / (n ** 0.5)


def signed_rotation(n: int, seed: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Orthogonal ``diag(signs) @ H`` with a deterministic, seeded sign pattern."""
    h = hadamard(n, dtype=dtype)
    generator = torch.Generator().manual_seed(seed)
    signs = torch.randint(0, 2, (n,), generator=generator, dtype=torch.int64)
    diagonal = torch.where(signs > 0, 1.0, -1.0).to(dtype)
    return diagonal.unsqueeze(1) * h


def _blocks(length: int, block: int) -> list[tuple[int, int]]:
    """Contiguous [start, end) blocks; the remainder is left unrotated."""
    if block <= 0:
        raise ValueError("block must be positive")
    spans = [(i, min(i + block, length)) for i in range(0, length, block)]
    return [(a, b) for a, b in spans if b - a == block]


def rotate_weight(weight: torch.Tensor, block: int, seed: int) -> tuple[torch.Tensor, dict]:
    """Rotate ``weight`` (``out x in``) along its input dimension, blockwise.

    Returns the rotated weight and the metadata needed to reproduce the
    transform (``block``, ``seed``, ``in_features``) — store it with any
    checkpoint that uses this path.
    """
    if weight.dim() != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(weight.shape)}")
    in_features = weight.shape[1]
    rotated = weight.clone()
    for start, end in _blocks(in_features, block):
        r = signed_rotation(block, seed, dtype=weight.dtype)
        rotated[:, start:end] = weight[:, start:end] @ r.T
    return rotated, {"block": block, "seed": seed, "in_features": in_features}


def rotate_activations(x: torch.Tensor, meta: dict) -> torch.Tensor:
    """Apply the matching transform to inputs: ``x' = x R^T`` per block.

    The pairing must mirror :func:`rotate_weight` (``W' = W R^T``): then
    ``x' W'^T = x R^T R W^T = x W^T``. Using ``x R`` instead silently changes
    the function, which the equivalence test catches.
    """
    in_features = x.shape[-1]
    if in_features != meta["in_features"]:
        raise ValueError(f"activation width {in_features} != weight {meta['in_features']}")
    rotated = x.clone()
    for start, end in _blocks(in_features, meta["block"]):
        r = signed_rotation(meta["block"], meta["seed"], dtype=x.dtype)
        rotated[..., start:end] = x[..., start:end] @ r.T
    return rotated


def unrotate_activations(x_rot: torch.Tensor, meta: dict) -> torch.Tensor:
    """Inverse of :func:`rotate_activations` (``R`` is orthogonal)."""
    in_features = x_rot.shape[-1]
    if in_features != meta["in_features"]:
        raise ValueError(f"activation width {in_features} != weight {meta['in_features']}")
    restored = x_rot.clone()
    for start, end in _blocks(in_features, meta["block"]):
        r = signed_rotation(meta["block"], meta["seed"], dtype=x_rot.dtype)
        restored[..., start:end] = x_rot[..., start:end] @ r
    return restored
