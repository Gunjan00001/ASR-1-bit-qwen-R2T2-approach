"""Tests for the layer policy (Stage 1, T1.3)."""

import pytest
import torch
from torch import nn

from asr1bit.bitlinear import BitLinear
from asr1bit.replace import apply_layer_policy, count_bitlinear


class _Block(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(dim, dim)
        self.self_attn.k_proj = nn.Linear(dim, dim)
        self.self_attn.v_proj = nn.Linear(dim, dim)
        self.self_attn.o_proj = nn.Linear(dim, dim)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(dim, dim)
        self.mlp.up_proj = nn.Linear(dim, dim)
        self.mlp.down_proj = nn.Linear(dim, dim)


class _Tower(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.layers = nn.ModuleList([_Block(dim) for _ in range(2)])


class _FakeASR(nn.Module):
    """Mirrors Qwen3-ASR submodule names at small scale."""

    def __init__(self, dim=8):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(100, dim)
        self.model.layers = nn.ModuleList([_Block(dim) for _ in range(2)])
        self.model.norm = nn.LayerNorm(dim)
        self.audio_tower = _Tower(dim)
        self.lm_head = nn.Linear(dim, 100)


def _names(model, cls):
    return {name for name, module in model.named_modules() if isinstance(module, cls)}


class TestDecoderAttnPolicy:
    def test_replaces_qkvo_in_decoder_only(self):
        model = _FakeASR()
        apply_layer_policy(model, "decoder_attn")
        bitlinear = _names(model, BitLinear)
        assert bitlinear == {
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.1.self_attn.q_proj",
            "model.layers.1.self_attn.k_proj",
            "model.layers.1.self_attn.v_proj",
            "model.layers.1.self_attn.o_proj",
        }

    def test_keeps_encoder_ffn_head_embeddings_norms(self):
        model = _FakeASR()
        apply_layer_policy(model, "decoder_attn")
        assert isinstance(model.audio_tower.layers[0].self_attn.q_proj, nn.Linear)
        assert not isinstance(model.audio_tower.layers[0].self_attn.q_proj, BitLinear)
        assert type(model.model.layers[0].mlp.gate_proj) is nn.Linear
        assert type(model.lm_head) is nn.Linear
        assert isinstance(model.model.embed_tokens, nn.Embedding)
        assert isinstance(model.model.norm, nn.LayerNorm)

    def test_weights_are_preserved(self):
        model = _FakeASR()
        before = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
        apply_layer_policy(model, "decoder_attn")
        after = model.model.layers[0].self_attn.q_proj.weight.detach()
        assert torch.equal(before, after)

    def test_returns_same_model_and_count(self):
        model = _FakeASR()
        returned = apply_layer_policy(model, "decoder_attn")
        assert returned is model
        assert count_bitlinear(model) == 8


class TestOtherPolicies:
    def test_decoder_attn_ffn_adds_ffn(self):
        model = _FakeASR()
        apply_layer_policy(model, "decoder_attn_ffn")
        assert isinstance(model.model.layers[0].mlp.gate_proj, BitLinear)
        assert count_bitlinear(model) == 8 + 6

    def test_none_policy_replaces_nothing(self):
        model = _FakeASR()
        apply_layer_policy(model, "none")
        assert count_bitlinear(model) == 0

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError):
            apply_layer_policy(_FakeASR(), "does-not-exist")


class TestOptions:
    def test_bit_width_and_group_size_forwarded(self):
        model = _FakeASR()
        apply_layer_policy(model, "decoder_attn", bit_width=1.58, group_size=4)
        layer = model.model.layers[0].self_attn.q_proj
        assert layer.bit_width == 1.58
        assert layer.group_size == 4
