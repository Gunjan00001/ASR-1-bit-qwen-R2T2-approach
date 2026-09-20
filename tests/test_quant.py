"""Tests for weight quantizers (Stage 1, T1.1).

Binary:  q = sign(W - mean(W)) * mean|W|          (sign(>=0) -> +1)
Ternary: q = RoundClip(W / gamma, -1, 1), gamma = mean|W|
Both support optional per-group scaling along the last dimension.
"""

import pytest
import torch

from asr1bit.quant import (
    dequantize_weights,
    quantize_binary,
    quantize_ternary,
)


class TestBinary:
    def test_sign_and_scale_roundtrip(self):
        w = torch.tensor([[2.0, -2.0]])
        q, scale = quantize_binary(w)
        assert torch.equal(q, torch.tensor([[1.0, -1.0]]))
        assert torch.allclose(scale, torch.tensor(2.0))
        assert torch.allclose(dequantize_weights(q, scale), w)

    def test_centered_sign_zero_maps_to_plus_one(self):
        w = torch.tensor([1.0, 2.0, 3.0])  # mean 2, centered [-1, 0, 1]
        q, scale = quantize_binary(w)
        assert torch.equal(q, torch.tensor([-1.0, 1.0, 1.0]))
        assert torch.allclose(scale, torch.tensor(2.0))

    def test_output_dtype_is_float32(self):
        w = torch.randn(4, 4, dtype=torch.float32)
        q, scale = quantize_binary(w)
        assert q.dtype == torch.float32
        assert scale.dtype == torch.float32

    def test_groupwise_shapes_and_roundtrip(self):
        w = torch.tensor([[2.0, -2.0, 10.0, -10.0]])
        q, scale = quantize_binary(w, group_size=2)
        assert q.shape == w.shape
        assert scale.shape == (2, 1)
        assert torch.equal(q, torch.tensor([[1.0, -1.0, 1.0, -1.0]]))
        assert torch.allclose(scale.reshape(-1), torch.tensor([2.0, 10.0]))
        assert torch.allclose(dequantize_weights(q, scale, group_size=2), w)

    def test_groupwise_beats_pertensor_on_scaled_groups(self):
        w = torch.tensor([[0.1, -0.1, 100.0, -100.0]])
        _, s_tensor = quantize_binary(w)
        _, s_group = quantize_binary(w, group_size=2)
        # per-tensor scale is dominated by the large group -> small group lost
        assert float(s_tensor) > float(s_group.reshape(-1)[0])

    def test_group_size_must_divide_last_dim(self):
        with pytest.raises(ValueError):
            quantize_binary(torch.zeros(1, 3), group_size=2)


class TestTernary:
    def test_roundclip(self):
        w = torch.tensor([0.9, -0.05, 1.2])
        q, gamma = quantize_ternary(w)
        expected_gamma = w.abs().mean()
        assert torch.allclose(gamma, expected_gamma)
        assert torch.equal(q, torch.tensor([1.0, 0.0, 1.0]))

    def test_values_are_in_ternary_set(self):
        w = torch.randn(64)
        q, _ = quantize_ternary(w)
        assert set(q.unique().tolist()) <= {-1.0, 0.0, 1.0}

    def test_zero_weights(self):
        q, gamma = quantize_ternary(torch.zeros(4))
        assert torch.count_nonzero(q) == 0
        assert float(gamma) == 0.0

    def test_groupwise_shapes(self):
        w = torch.tensor([[0.2, -0.1, 3.0, -3.2]])
        q, gamma = quantize_ternary(w, group_size=2)
        assert q.shape == w.shape
        assert gamma.shape == (2, 1)
        assert set(q.unique().tolist()) <= {-1.0, 0.0, 1.0}

    def test_group_size_must_divide_last_dim(self):
        with pytest.raises(ValueError):
            quantize_ternary(torch.zeros(1, 3), group_size=2)


class TestDequantize:
    def test_pertensor(self):
        q = torch.tensor([1.0, -1.0])
        scale = torch.tensor(3.0)
        assert torch.allclose(dequantize_weights(q, scale), torch.tensor([3.0, -3.0]))

    def test_groupwise(self):
        q = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
        scale = torch.tensor([[2.0], [10.0]])
        out = dequantize_weights(q, scale, group_size=2)
        assert torch.allclose(out, torch.tensor([[2.0, -2.0, 10.0, -10.0]]))
