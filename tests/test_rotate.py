import torch

from asr1bit.quant import dequantize_weights, quantize_ternary
from asr1bit.rotate import (
    hadamard,
    rotate_activations,
    rotate_weight,
    signed_rotation,
    unrotate_activations,
)


def test_hadamard_is_orthogonal():
    for n in (1, 4, 64):
        h = hadamard(n)
        assert torch.allclose(h @ h.T, torch.eye(n), atol=1e-5)


def test_rotation_is_deterministic_and_seeded():
    a = signed_rotation(64, seed=7)
    b = signed_rotation(64, seed=7)
    c = signed_rotation(64, seed=8)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert torch.allclose(a @ a.T, torch.eye(64), atol=1e-5)


def test_unquantized_forward_equivalence_including_remainder():
    torch.manual_seed(0)
    for in_features, block in ((1024, 256), (1000, 256), (100, 64)):
        w = torch.randn(32, in_features)
        x = torch.randn(4, in_features)
        w_rot, meta = rotate_weight(w, block, seed=11)
        y = x @ w.T
        y_rot = rotate_activations(x, meta) @ w_rot.T
        assert torch.allclose(y, y_rot, atol=1e-4), (in_features, block)


def test_unrotate_round_trip():
    x = torch.randn(3, 1000)
    _, meta = rotate_weight(torch.randn(8, 1000), 256, seed=3)
    assert torch.allclose(unrotate_activations(rotate_activations(x, meta), meta), x, atol=1e-5)


def test_quantized_levels_stay_ternary_after_rotation():
    w = torch.randn(16, 512)
    w_rot, _ = rotate_weight(w, 256, seed=5)
    q, _ = quantize_ternary(w_rot)
    assert set(q.unique().tolist()) <= {-1.0, 0.0, 1.0}


def test_rotation_reduces_ternary_error_on_outlier_weights():
    """Rotation spreads per-row outliers, so fewer weights clip at the same scale."""
    torch.manual_seed(0)
    w = torch.randn(32, 1024)
    w[:, ::128] *= 8.0  # spiky columns: a stand-in for real outlier channels
    w_rot, _ = rotate_weight(w, 1024, seed=13)
    def rel_error(matrix):
        q, s = quantize_ternary(matrix)
        return ((dequantize_weights(q, s) - matrix).norm() / matrix.norm()).item()
    assert rel_error(w_rot) < rel_error(w)
