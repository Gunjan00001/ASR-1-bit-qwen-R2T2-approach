"""Tests for the sensitivity-guided mixed-precision layer policy."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from asr1bit.bitlinear import BitLinear
from asr1bit.replace import apply_mixed_precision, count_bitlinear

TRAINER = Path(__file__).resolve().parents[1] / "scripts" / "run_stage1_qat.py"
EVALUATOR = Path(__file__).resolve().parents[1] / "scripts" / "eval_streaming.py"


class _Block(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.self_attn = nn.Module()
        for proj in ("q", "k", "v", "o"):
            setattr(self.self_attn, f"{proj}_proj", nn.Linear(dim, dim))
        self.mlp = nn.Module()
        for proj in ("gate", "up", "down"):
            setattr(self.mlp, f"{proj}_proj", nn.Linear(dim, dim))


class _FakeASR(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Block(dim) for _ in range(2)])
        self.audio_tower = nn.Module()
        self.audio_tower.layers = nn.ModuleList([_Block(dim)])


def _k_names():
    return {
        "model.layers.0.self_attn.k_proj",
        "model.layers.1.self_attn.k_proj",
    }


class TestMixedPolicy:
    def test_keeps_selected_layers_fp16_and_quantizes_the_rest(self):
        model = _FakeASR()
        apply_mixed_precision(model, _k_names(), bit_width=1.58)
        assert type(model.model.layers[0].self_attn.k_proj) is nn.Linear
        assert type(model.model.layers[1].self_attn.k_proj) is nn.Linear
        for layer in model.model.layers:
            assert isinstance(layer.self_attn.q_proj, BitLinear)
            assert isinstance(layer.self_attn.v_proj, BitLinear)
            assert isinstance(layer.self_attn.o_proj, BitLinear)
            assert isinstance(layer.mlp.gate_proj, BitLinear)
            assert isinstance(layer.mlp.up_proj, BitLinear)
            assert isinstance(layer.mlp.down_proj, BitLinear)
        assert count_bitlinear(model) == 12  # 2 layers x 6 non-k projections
        assert model.fp16_layers == sorted(_k_names())

    def test_encoder_and_unknown_names_are_rejected(self):
        with pytest.raises(ValueError):
            apply_mixed_precision(
                _FakeASR(), {"model.layers.0.self_attn.k_proj", "nope"}
            )
        with pytest.raises(ValueError):
            apply_mixed_precision(
                _FakeASR(), {"audio_tower.layers.0.self_attn.q_proj"}
            )

    def test_empty_list_is_rejected(self):
        with pytest.raises(ValueError):
            apply_mixed_precision(_FakeASR(), set())

    def test_excluded_weights_are_preserved(self):
        model = _FakeASR()
        before = model.model.layers[0].self_attn.k_proj.weight.detach().clone()
        apply_mixed_precision(model, _k_names(), bit_width=1.58)
        assert torch.equal(model.model.layers[0].self_attn.k_proj.weight, before)

    def test_quantized_shadows_are_initialized_from_the_pretrained_weights(self):
        model = _FakeASR()
        q_before = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
        apply_mixed_precision(model, _k_names(), bit_width=1.58)
        assert torch.equal(model.model.layers[0].self_attn.q_proj.weight, q_before)


def _load(path: Path, name: str):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCli:
    def test_trainer_accepts_fp16_layers_file(self, monkeypatch):
        module = _load(TRAINER, "run_stage1_qat_mixed_test")
        monkeypatch.setattr(sys, "argv", [
            "run_stage1_qat.py", "--steps", "1",
            "--fp16-layers-file", "configs/mixed_k_fp16.json",
        ])
        args = module.parse_args()
        assert args.fp16_layers_file == "configs/mixed_k_fp16.json"

    def test_evaluator_accepts_fp16_layers_file(self, monkeypatch):
        module = _load(EVALUATOR, "eval_streaming_mixed_test")
        monkeypatch.setattr(sys, "argv", [
            "eval_streaming.py", "--fp16-layers-file", "configs/mixed_k_fp16.json",
        ])
        args = module.parse_args()
        assert args.fp16_layers_file == "configs/mixed_k_fp16.json"
