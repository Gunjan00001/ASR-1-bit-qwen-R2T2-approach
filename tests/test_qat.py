"""Tests for the progressive QAT loop helpers (Stage 1, T1.4)."""

import types

import torch
from torch import nn

from asr1bit.bitlinear import BitLinear
from asr1bit.replace import apply_layer_policy
from asr1bit.train.qat import (
    build_optimizer,
    check_finite_grads,
    enable_gradient_checkpointing,
    freeze_non_bitlinear,
    grad_norms,
    non_bitlinear_linear_names,
    progressive_alpha,
    set_activation_quant,
    set_qat_alpha,
    train_step,
)


class _FakeASR(nn.Module):
    """Small stand-in mirroring Qwen3-ASR submodule names."""

    def __init__(self, dim=8):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Block(dim) for _ in range(2)])
        self.audio_tower = nn.Module()
        self.audio_tower.layers = nn.ModuleList([_Block(dim)])
        self.lm_head = nn.Linear(dim, 100)


class _Block(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.self_attn = nn.Module()
        for proj in ("q", "k", "v", "o"):
            setattr(self.self_attn, f"{proj}_proj", nn.Linear(dim, dim))
        self.mlp = nn.Module()
        for proj in ("gate", "up", "down"):
            setattr(self.mlp, f"{proj}_proj", nn.Linear(dim, dim))


class TestProgressiveAlpha:
    def test_endpoints(self):
        assert progressive_alpha(0, 100) == 0.0
        assert progressive_alpha(100, 100) == 1.0

    def test_midpoint(self):
        assert progressive_alpha(50, 100) == 0.5

    def test_clamps_past_total(self):
        assert progressive_alpha(200, 100) == 1.0

    def test_custom_range(self):
        assert progressive_alpha(0, 10, start=0.2, end=0.8) == 0.2
        assert progressive_alpha(10, 10, start=0.2, end=0.8) == 0.8

    def test_monotonic(self):
        values = [progressive_alpha(step, 10) for step in range(11)]
        assert values == sorted(values)


class TestSetQatAlpha:
    def test_sets_alpha_on_all_bitlinear(self):
        model = _FakeASR()
        apply_layer_policy(model, "decoder_attn")
        count = set_qat_alpha(model, 0.5)
        assert count == 8
        for module in model.modules():
            if isinstance(module, BitLinear):
                assert module.alpha == 0.5


class TestNonBitlinearLinearNames:
    def test_excludes_replaced_decoder_attn(self):
        model = _FakeASR()
        apply_layer_policy(model, "decoder_attn")
        names = non_bitlinear_linear_names(model)
        assert "model.layers.0.self_attn.q_proj" not in names
        assert "model.layers.0.mlp.gate_proj" in names
        assert "audio_tower.layers.0.self_attn.q_proj" in names
        assert "lm_head" in names


class TestSetActivationQuant:
    def test_toggles_all_bitlinear(self):
        model = _FakeASR()
        apply_layer_policy(model, "decoder_attn")
        assert set_activation_quant(model, False) == 8
        for module in model.modules():
            if isinstance(module, BitLinear):
                assert module.quantize_activations is False
        set_activation_quant(model, True)
        for module in model.modules():
            if isinstance(module, BitLinear):
                assert module.quantize_activations is True


class TestFreezeNonBitlinear:
    def test_only_bitlinear_shadows_trainable(self):
        model = _FakeASR()
        apply_layer_policy(model, "decoder_attn")
        count = freeze_non_bitlinear(model)
        assert count > 0
        trainable = set()
        for name, module in model.named_modules():
            if isinstance(module, BitLinear):
                trainable.add(f"{name}.weight")
                if module.bias is not None:
                    trainable.add(f"{name}.bias")
        for name, param in model.named_parameters():
            assert param.requires_grad == (name in trainable), name

    def test_trainable_count_smaller_than_total(self):
        model = _FakeASR()
        apply_layer_policy(model, "decoder_attn")
        total = sum(p.numel() for p in model.parameters())
        assert freeze_non_bitlinear(model) < total


class TestGradientCheckpointing:
    def test_enables_and_disables_cache(self):
        model = _FakeASR()
        model.config = types.SimpleNamespace(use_cache=True)
        calls = []
        model.gradient_checkpointing_enable = lambda: calls.append(True)
        enable_gradient_checkpointing(model)
        assert calls == [True]
        assert model.config.use_cache is False

    def test_missing_method_is_noop(self):
        enable_gradient_checkpointing(nn.Linear(2, 2))


class TestBuildOptimizer:
    def test_only_trainable_params(self):
        model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
        model[1].weight.requires_grad_(False)
        model[1].bias.requires_grad_(False)
        optimizer = build_optimizer(model, lr=1e-3, use_8bit=False)
        params = [p for group in optimizer.param_groups for p in group["params"]]
        assert len(params) == 2  # first layer weight+bias only


class TestGradDiagnostics:
    def test_grad_norms_after_backward(self):
        layer = nn.Linear(4, 3)
        layer(torch.randn(2, 4)).sum().backward()
        norms = grad_norms(layer)
        assert "weight" in norms and "bias" in norms
        assert all(norm >= 0 for norm in norms.values())

    def test_check_finite_grads_raises_on_nan(self):
        import pytest

        layer = nn.Linear(4, 3)
        layer(torch.randn(2, 4)).sum().backward()
        layer.weight.grad[0, 0] = float("nan")
        with pytest.raises(FloatingPointError):
            check_finite_grads(layer)

    def test_check_finite_grads_passes_normally(self):
        layer = nn.Linear(4, 3)
        layer(torch.randn(2, 4)).sum().backward()
        check_finite_grads(layer)


class TestTrainStep:
    def test_loss_decreases(self):
        torch.manual_seed(0)
        model = nn.Linear(4, 2)
        x = torch.randn(16, 4)
        labels = (x[:, 0] > 0).long()
        optimizer = build_optimizer(model, lr=0.1, use_8bit=False)
        losses = [train_step(model, x, labels, optimizer) for _ in range(20)]
        assert losses[-1] < losses[0]
