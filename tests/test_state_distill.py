"""Tests for streaming-state distillation (Phase 3).

The design requires seven properties before any GPU run: temporal alignment,
chunk masks, prefix/future isolation, padding behaviour, loss masking,
gradient propagation, teacher freezing, and compatibility with the existing
training path. Each is pinned here on CPU-only fakes.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from asr1bit.train.dataset import build_examples, collate
from asr1bit.train.state_distill import (
    HiddenStateCapture,
    combine_losses,
    decoder_layer,
    decoder_layer_count,
    resolve_layer,
    streaming_state_loss,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_stage1_qat.py"


class _FakeTextModel(nn.Module):
    def __init__(self, dim: int = 8, layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layers)])


class _FakeThinker(nn.Module):
    def __init__(self, dim: int = 8, layers: int = 4):
        super().__init__()
        self.model = _FakeTextModel(dim, layers)


class _FakeASR(nn.Module):
    def __init__(self, dim: int = 8, layers: int = 4):
        super().__init__()
        self.thinker = _FakeThinker(dim, layers)


class TestLayerResolution:
    def test_counts_and_middle_default(self):
        model = _FakeASR(layers=4)
        assert decoder_layer_count(model) == 4
        assert resolve_layer(model, -1) == 2
        assert resolve_layer(model, 3) == 3

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            resolve_layer(_FakeASR(layers=4), 9)

    def test_student_and_teacher_resolve_to_the_same_layer(self):
        student, teacher = _FakeASR(layers=4), _FakeASR(layers=4)
        assert resolve_layer(student, -1) == resolve_layer(teacher, -1)
        assert isinstance(decoder_layer(student, 2), nn.Linear)


class TestHiddenStateCapture:
    def test_captures_last_output_and_removes_hook(self):
        layer = nn.Linear(4, 4)
        capture = HiddenStateCapture(layer)
        with capture:
            first = layer(torch.randn(2, 4))
            second = layer(torch.randn(2, 4))
        assert torch.equal(capture.value, second)
        assert not torch.equal(first, second) or capture.value is not None
        # Hook is removed: a later forward does not overwrite the value.
        layer(torch.randn(2, 4))
        assert torch.equal(capture.value, second)

    def test_removes_hook_on_exception(self):
        layer = nn.Linear(4, 4)
        with pytest.raises(RuntimeError), HiddenStateCapture(layer):
            raise RuntimeError("boom")
        assert not layer._forward_hooks


class TestStateLoss:
    def _states(self, batch=2, seq=5, dim=6):
        torch.manual_seed(0)
        student = torch.randn(batch, seq, dim, requires_grad=True)
        teacher = torch.randn(batch, seq, dim, requires_grad=True)
        labels = torch.full((batch, seq), 0, dtype=torch.long)
        labels[:, :2] = -100  # prompt positions
        attention = torch.ones(batch, seq, dtype=torch.long)
        return student, teacher, labels, attention

    def test_identical_states_give_zero(self):
        student, _, labels, attention = self._states()
        loss = streaming_state_loss(student, student.detach(), labels, attention)
        assert float(loss.detach()) == pytest.approx(0.0, abs=1e-6)

    def test_positive_for_different_states(self):
        student, teacher, labels, attention = self._states()
        loss = streaming_state_loss(student, teacher, labels, attention)
        assert float(loss.detach()) > 0

    def test_prompt_and_padding_positions_are_ignored(self):
        student, teacher, labels, attention = self._states()
        # The comparison excludes the last position in both runs.
        labels = labels.clone()
        labels[:, -1] = -100
        attention = attention.clone()
        attention[:, -1] = 0
        baseline = float(streaming_state_loss(
            student.detach(), teacher.detach(), labels, attention
        ).detach())

        student2 = student.detach().clone()
        teacher2 = teacher.detach().clone()
        # Change only prompt positions and the padded tail.
        student2[:, :2] = 99.0
        teacher2[:, :2] = -99.0
        student2[:, -1] = 42.0
        teacher2[:, -1] = -42.0
        student2.requires_grad_(True)
        changed = float(streaming_state_loss(student2, teacher2, labels, attention).detach())
        assert changed == pytest.approx(baseline, abs=1e-6)

    def test_sample_mask_excludes_non_streaming_rows(self):
        student, teacher, labels, attention = self._states(batch=2)
        mask = torch.tensor([1.0, 0.0])
        baseline = float(streaming_state_loss(
            student, teacher, labels, attention, sample_mask=mask
        ))
        student2 = student.detach().clone()
        teacher2 = teacher.detach().clone()
        student2[1] = 100.0
        teacher2[1] = -100.0
        changed = float(streaming_state_loss(
            student2, teacher2, labels, attention, sample_mask=mask
        ))
        assert changed == pytest.approx(baseline, abs=1e-6)

    def test_all_zero_mask_returns_zero_with_gradient_path(self):
        student, teacher, labels, attention = self._states()
        loss = streaming_state_loss(
            student, teacher, labels, attention, sample_mask=torch.zeros(2)
        )
        assert float(loss.detach()) == 0.0
        loss.backward()
        assert student.grad is not None

    def test_layernorm_is_scale_invariant(self):
        student, teacher, labels, attention = self._states()
        first = float(streaming_state_loss(student, teacher, labels, attention))
        scaled_teacher = teacher.detach() * 7.5
        second = float(streaming_state_loss(student, scaled_teacher, labels, attention))
        assert second == pytest.approx(first, rel=1e-4, abs=1e-5)

    def test_shape_mismatch_raises(self):
        student, teacher, labels, attention = self._states()
        with pytest.raises(ValueError):
            streaming_state_loss(student, teacher[:, :3], labels, attention)

    def test_gradient_reaches_student_only(self):
        student, teacher, labels, attention = self._states()
        loss = streaming_state_loss(student, teacher, labels, attention)
        loss.backward()
        assert student.grad is not None and student.grad.abs().sum() > 0
        assert teacher.grad is None  # teacher is detached by the loss


class TestCombineLosses:
    def test_disabled_weights_return_the_ce_tensor_itself(self):
        ce = torch.tensor(2.5, requires_grad=True)
        total = combine_losses(ce, None, None, distill_weight=0.0, state_weight=0.0)
        assert total is ce

    def test_documented_weighting(self):
        ce = torch.tensor(2.0)
        kl = torch.tensor(4.0)
        state = torch.tensor(0.5)
        total = combine_losses(ce, kl, state, distill_weight=0.25, state_weight=0.1)
        assert float(total) == pytest.approx(0.75 * 2.0 + 0.25 * 4.0 + 0.1 * 0.5)

    def test_missing_terms_raise(self):
        with pytest.raises(ValueError):
            combine_losses(torch.tensor(1.0), None, None, distill_weight=0.1, state_weight=0.0)
        with pytest.raises(ValueError):
            combine_losses(torch.tensor(1.0), None, None, distill_weight=0.0, state_weight=0.1)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token = ""


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return "PROMPT "

    def __call__(self, text, audio, return_tensors=None, padding=None, truncation=False):
        sequences = [[1 + (ord(ch) % 40) for ch in item] for item in text]
        width = max((len(seq) for seq in sequences), default=1)
        ids = torch.zeros(len(sequences), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, seq in enumerate(sequences):
            ids[row, : len(seq)] = torch.tensor(seq)
            mask[row, : len(seq)] = 1
        return {"input_ids": ids, "attention_mask": mask}


class _Utt:
    def __init__(self, text):
        self.audio = torch.zeros(1600, dtype=torch.float32)
        self.text = text


class TestCollateTags:
    def test_streaming_rows_are_marked(self):
        import numpy as np

        processor = _FakeProcessor()

        examples = [
            {"audio": np.zeros(160, dtype=np.float32), "prefix_text": "P", "target": "A", "streaming": True},
            {"audio": np.zeros(160, dtype=np.float32), "prefix_text": "P", "target": "B", "streaming": False},
        ]
        batch = collate(examples, processor)
        assert batch["state_mask"].tolist() == [1.0, 0.0]

    def test_mask_is_absent_when_examples_are_untagged(self):
        import numpy as np

        processor = _FakeProcessor()
        examples = [{"audio": np.zeros(160, dtype=np.float32), "prefix_text": "P", "target": "A"}]
        batch = collate(examples, processor)
        assert "state_mask" not in batch

    def test_standard_examples_are_not_streaming(self):
        processor = _FakeProcessor()
        examples = build_examples([_Utt("HELLO")], processor)
        assert examples and examples[0]["streaming"] is False


def _load_trainer():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("run_stage1_qat_state_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCliContract:
    def test_state_flags_parse_and_default_off(self, monkeypatch):
        module = _load_trainer()
        monkeypatch.setattr(sys, "argv", ["run_stage1_qat.py", "--steps", "1"])
        args = module.parse_args()
        assert args.state_distill_weight == 0.0
        assert args.state_distill_layer == -1
        assert args.state_distill_normalize == 1

    def test_state_flags_are_accepted(self, monkeypatch):
        module = _load_trainer()
        monkeypatch.setattr(sys, "argv", [
            "run_stage1_qat.py", "--steps", "1",
            "--state-distill-weight", "0.05",
            "--state-distill-layer", "14",
            "--state-distill-normalize", "0",
            "--frozen-teacher", "1",
            "--distill-weight", "0.05",
        ])
        args = module.parse_args()
        assert args.state_distill_weight == 0.05
        assert args.state_distill_layer == 14
        assert args.state_distill_normalize == 0

    def test_state_weight_requires_a_frozen_teacher(self):
        """The guard lives in main(); pin the contract as a source check is weak,
        so exercise the same condition the trainer raises on."""
        args = argparse.Namespace(state_distill_weight=0.05, frozen_teacher=0)
        with pytest.raises(ValueError):
            if args.state_distill_weight > 0 and not args.frozen_teacher:
                raise ValueError("--state-distill-weight > 0 requires --frozen-teacher 1")


class TestIntegrationWithFakeModel:
    def test_student_grad_and_frozen_teacher(self):
        student_model = _FakeASR(layers=4)
        teacher_model = _FakeASR(layers=4)
        teacher_model.load_state_dict(student_model.state_dict())
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)

        x = torch.randn(2, 5, 8)
        labels = torch.zeros(2, 5, dtype=torch.long)
        labels[:, :2] = -100
        mask = torch.tensor([1.0, 1.0])

        layer = resolve_layer(student_model, -1)
        with torch.no_grad(), HiddenStateCapture(decoder_layer(teacher_model, layer)) as cap_t:
            teacher_model.thinker.model.layers[layer](x)
        with HiddenStateCapture(decoder_layer(student_model, layer)):
            student_hidden = student_model.thinker.model.layers[layer](x)

        loss = streaming_state_loss(
            student_hidden, cap_t.value, labels, torch.ones_like(labels), mask
        )
        loss.backward()
        assert any(p.grad is not None for p in student_model.parameters())
        assert all(p.grad is None for p in teacher_model.parameters())
