"""Weight quantizers for 1-bit / ternary linear layers (Stage 1, T1.1).

Locked interfaces:
    quantize_binary(weight)  -> (qW, scale)
    quantize_ternary(weight) -> (qW, scale)

Binary:  sign(W - mean(W)) * mean|W|
Ternary: RoundClip(W / gamma, -1, 1), gamma = mean|W|
"""


def quantize_binary(weight):
    """Quantize weights to {-1, +1} with a per-tensor scale."""
    raise NotImplementedError("Stage 1, T1.1")


def quantize_ternary(weight):
    """Quantize weights to {-1, 0, +1} with a per-tensor scale."""
    raise NotImplementedError("Stage 1, T1.1")
