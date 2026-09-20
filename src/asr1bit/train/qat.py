"""Progressive quantization-aware training loop (Stage 1, T1.4).

Alpha ramp 0 -> 1 blending full-precision weights toward fully quantized
weights, with fp32 shadow weights + STE.
"""
