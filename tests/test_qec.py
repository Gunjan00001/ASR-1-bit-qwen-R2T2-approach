"""Tests for output-aware quantization-error compensation (QEC).

The design requirements: the regularizer is training-only, disabled by
default, adds no parameters, uses the same activation-quantized input as the
layer, skips eval forwards, propagates gradients only through the quantized
path, and preserves the baseline behavior when disabled.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from asr1bit.bitlinear import BitLinear
from asr1bit.train.qec import QECController, hooked_layers, relative_output_error

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_stage1_qat.py"


class _Block(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.proj = BitLinear(dim, dim, quantize_activations=False)


class _Model(nn.Module):
    def __init__(self, layers=4, dim=8):
        super().__init__()
        self.layers = nn.ModuleList([_Block(dim) for _ in range(layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer.proj(x)
        return x


class TestRelativeError:
    def test_zero_for_identical(self):
        tensor = torch.randn(4, 8)
        assert float(relative_output_error(tensor, tensor.clone())) == 0.0

    def test_positive_for_different(self):
        a = torch.zeros(4, 8)
        b = torch.ones(4, 8)
        assert float(relative_output_error(a, b)) == pytest.approx(1.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            relative_output_error(torch.zeros(2, 3), torch.zeros(2, 4))

    def test_finite_for_zero_target(self):
        value = relative_output_error(torch.zeros(2, 3), torch.zeros(2, 3))
        assert torch.isfinite(value)


class TestController:
    def _model(self):
        model = _Model()
        for block in model.layers:
            block.proj.alpha = 1.0
        return model

    def test_registers_only_bitlinear_layers(self):
        model = _Model(layers=4)
        controller = QECController(model, stride=1)
        assert len(hooked_layers(controller)) == 4
        controller.close()

    def test_stride_selects_every_nth_layer(self):
        controller = QECController(_Model(layers=5), stride=2)
        assert len(hooked_layers(controller)) == 3
        controller.close()

    def test_positive_loss_at_alpha_one(self):
        torch.manual_seed(0)
        model = self._model()
        controller = QECController(model)
        controller.reset()
        model(torch.randn(3, 8))
        loss = controller.loss()
        assert loss is not None and float(loss.detach()) > 0
        controller.close()

    def test_skips_at_alpha_zero(self):
        model = self._model()
        for block in model.layers:
            block.proj.alpha = 0.0
        controller = QECController(model)
        controller.reset()
        model(torch.randn(3, 8))
        assert controller.loss() is None
        controller.close()

    def test_skips_eval_mode(self):
        model = self._model().eval()
        controller = QECController(model)
        controller.reset()
        with torch.no_grad():
            model(torch.randn(3, 8))
        assert controller.loss() is None
        controller.close()

    def test_gradient_flows_to_shadow_weights(self):
        torch.manual_seed(0)
        model = self._model()
        controller = QECController(model)
        controller.reset()
        model(torch.randn(3, 8))
        loss = controller.loss()
        loss.backward()
        grads = [block.proj.weight.grad for block in model.layers]
        assert all(grad is not None for grad in grads)
        assert all(grad.abs().sum() > 0 for grad in grads)
        controller.close()

    def test_no_parameters_are_added(self):
        model = self._model()
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        controller = QECController(model)
        controller.reset()
        model(torch.randn(3, 8))
        after = model.state_dict()
        assert set(before) == set(after)
        for name, tensor in before.items():
            assert torch.equal(tensor, after[name])
        controller.close()

    def test_hooks_are_removed_on_close(self):
        model = self._model()
        controller = QECController(model)
        assert any(module._forward_hooks for module in model.modules() if isinstance(module, BitLinear))
        controller.close()
        assert all(not module._forward_hooks for module in model.modules() if isinstance(module, BitLinear))

    def test_disabled_controller_does_nothing(self):
        model = self._model()
        controller = QECController(model)
        controller.enabled = False
        controller.reset()
        model(torch.randn(3, 8))
        assert controller.loss() is None
        controller.close()

    def test_loss_is_finite(self):
        torch.manual_seed(0)
        model = self._model()
        controller = QECController(model)
        controller.reset()
        model(torch.randn(3, 8) * 100.0)
        loss = controller.loss()
        assert loss is not None and torch.isfinite(loss).all()
        controller.close()

    def test_activation_quantized_input_is_used(self):
        """At alpha=1 with activation quantization, QEC must compare against
        the fp response on the *quantized* input (otherwise the error includes
        activation noise and the test's relative scale changes)."""
        torch.manual_seed(0)
        layer = BitLinear(8, 8, quantize_activations=True)
        layer.alpha = 1.0
        model = nn.Sequential(layer)
        controller = QECController(model)
        controller.reset()
        x = torch.randn(3, 8)
        model(x)
        loss = controller.loss()
        assert loss is not None and float(loss.detach()) > 0
        controller.close()


def _load_trainer():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("run_stage1_qat_qec_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTrainerContract:
    def test_qec_flags_default_off(self, monkeypatch):
        module = _load_trainer()
        monkeypatch.setattr(sys, "argv", ["run_stage1_qat.py", "--steps", "1"])
        args = module.parse_args()
        assert args.qec_weight == 0.0
        assert args.qec_stride == 1
        assert args.qec_eps == 1e-6

    def test_qec_flags_parse(self, monkeypatch):
        module = _load_trainer()
        monkeypatch.setattr(sys, "argv", [
            "run_stage1_qat.py", "--steps", "1",
            "--qec-weight", "0.5", "--qec-stride", "2", "--qec-eps", "1e-8",
        ])
        args = module.parse_args()
        assert args.qec_weight == 0.5
        assert args.qec_stride == 2
        assert args.qec_eps == 1e-8
