"""BitLinear: drop-in replacement for ``nn.Linear`` with a bit-width switch.

Stage 1, T1.2. Holds fp32 shadow weights updated via a straight-through
estimator, quantizes weights on the forward pass, and optionally quantizes
activations to INT8 (absmax, per-token).

Locked interface:
    BitLinear(in_dim, out_dim, bit_width, quantize_activations)
"""


class BitLinear:
    """Placeholder; real implementation replaces ``nn.Linear`` (Stage 1, T1.2)."""

    def __init__(self, in_dim, out_dim, bit_width=1, quantize_activations=True):
        raise NotImplementedError("Stage 1, T1.2")
