"""Pack 1-bit/2-bit weights and export to GGUF (Stage 5, T5.1-T5.2).

Locked interface:
    export_gguf(model, out_path)
"""


def pack_binary(weight):
    """Pack {-1, +1} weights to 1 bit per parameter."""
    raise NotImplementedError("Stage 5, T5.1")


def pack_ternary(weight):
    """Pack {-1, 0, +1} weights to 2 bits per parameter."""
    raise NotImplementedError("Stage 5, T5.1")


def export_gguf(model, out_path):
    """Export a quantized model to GGUF for CPU inference."""
    raise NotImplementedError("Stage 5, T5.2")
