"""Tests for BitLinear (Stage 1, T1.2)."""

import torch
from torch import nn

from asr1bit.bitlinear import BitLinear, absmax_quantize_activations


class TestConstruction:
    def test_is_nn_module_with_shadow_weight(self):
        layer = BitLinear(8, 4)
        assert isinstance(layer, nn.Module)
        assert layer.weight.shape == (4, 8)
        assert layer.weight.dtype == torch.float32
        assert isinstance(layer.weight, nn.Parameter)

    def test_optional_bias(self):
        assert BitLinear(8, 4, bias=False).bias is None
        assert BitLinear(8, 4, bias=True).bias.shape == (4,)

    def test_forward_shape(self):
        layer = BitLinear(8, 4)
        assert layer(torch.randn(3, 8)).shape == (3, 4)


class TestLoadFromLinear:
    def test_classmethod_copies_weights(self):
        linear = nn.Linear(6, 5)
        layer = BitLinear.from_linear(linear)
        assert torch.equal(layer.weight, linear.weight.float())
        assert layer.in_features == 6
        assert layer.out_features == 5

    def test_instance_method_copies_weights(self):
        linear = nn.Linear(6, 5)
        layer = BitLinear(6, 5)
        returned = layer.load_from_linear(linear)
        assert returned is layer
        assert torch.equal(layer.weight, linear.weight.float())


class TestSteGradients:
    def test_weight_grad_flows_to_shadow(self):
        layer = BitLinear(8, 4)
        layer(torch.randn(2, 8)).sum().backward()
        assert layer.weight.grad is not None
        assert layer.weight.grad.abs().sum() > 0

    def test_input_grad_flows_with_activation_quant(self):
        layer = BitLinear(8, 4, quantize_activations=True)
        x = torch.randn(2, 8, requires_grad=True)
        layer(x).sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestBitWidth:
    def test_binary_and_ternary_differ(self):
        torch.manual_seed(0)
        layer = BitLinear(16, 8, bit_width=1, quantize_activations=False)
        x = torch.randn(2, 16)
        binary = layer(x)
        layer.bit_width = 1.58
        ternary = layer(x)
        assert not torch.allclose(binary, ternary)

    def test_output_is_finite(self):
        for width in (1, 1.58):
            out = BitLinear(16, 8, bit_width=width)(torch.randn(4, 16))
            assert torch.isfinite(out).all()


class TestActivationQuantization:
    def test_absmax_range(self):
        x = torch.tensor([[0.5, -1.0, 0.25]])
        q = absmax_quantize_activations(x)
        assert q.shape == x.shape
        assert torch.all(q.abs() <= x.abs().max() + 1e-6)

    def test_quantized_activations_change_output(self):
        torch.manual_seed(0)
        layer = BitLinear(16, 8, quantize_activations=True)
        x = torch.randn(2, 16)
        with_quant = layer(x)
        layer.quantize_activations = False
        without_quant = layer(x)
        assert not torch.allclose(with_quant, without_quant)


class TestGroupSize:
    def test_groupwise_forward_runs(self):
        layer = BitLinear(8, 4, group_size=4)
        assert layer(torch.randn(2, 8)).shape == (2, 4)
