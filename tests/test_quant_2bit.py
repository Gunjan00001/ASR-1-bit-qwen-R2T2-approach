import torch

from asr1bit.bitlinear import BitLinear
from asr1bit.quant import dequantize_weights, quantize_2bit, quantize_ternary


def test_levels_are_four_odd_values():
    w = torch.randn(8, 16)
    q, _ = quantize_2bit(w)
    assert set(q.unique().tolist()) <= {-3.0, -1.0, 1.0, 3.0}
    assert len(q.unique()) > 1


def test_two_bit_beats_ternary_error():
    torch.manual_seed(0)
    w = torch.randn(64, 128)
    q2, s2 = quantize_2bit(w)
    q1, s1 = quantize_ternary(w)
    err2 = (dequantize_weights(q2, s2) - w).norm() / w.norm()
    err1 = (dequantize_weights(q1, s1) - w).norm() / w.norm()
    assert err2 < err1


def test_bitlinear_two_bit_forward_runs_and_quantizes():
    torch.manual_seed(0)
    layer = BitLinear(16, 8, bit_width=2.0)
    out = layer(torch.randn(4, 16))
    assert out.shape == (4, 8)
    qw = layer._quantized_weight()
    assert torch.isfinite(qw).all()
    ternary = BitLinear(16, 8, bit_width=1.58)
    ternary.load_state_dict(layer.state_dict())
    assert not torch.allclose(qw, ternary._quantized_weight())
