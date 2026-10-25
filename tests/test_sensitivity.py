"""Tests for the task-aware layer-sensitivity analyzer."""

from __future__ import annotations

import torch
from torch import nn

from asr1bit.analysis.sensitivity import (
    SensitivityAnalyzer,
    aggregate_by_component,
)
from asr1bit.bitlinear import BitLinear


class _Block(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = BitLinear(dim, dim, quantize_activations=False)
        self.mlp = nn.Module()
        self.mlp.gate_proj = BitLinear(dim, dim, quantize_activations=False)

    def forward(self, x):
        return self.mlp.gate_proj(self.self_attn.q_proj(x))


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block(), _Block()])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _run(alpha=1.0):
    torch.manual_seed(0)
    model = _Model()
    for module in model.modules():
        if isinstance(module, BitLinear):
            module.alpha = alpha
    analyzer = SensitivityAnalyzer(model)
    analyzer.reset()
    x = torch.randn(3, 8)
    loss = model(x).pow(2).mean()
    analyzer.score(loss)
    rows = analyzer.results()
    analyzer.close()
    return model, rows


class TestAnalyzer:
    def test_reports_every_bitlinear_layer(self):
        _, rows = _run()
        assert len(rows) == 4
        assert {row.component for row in rows} == {"q", "ffn_in"}

    def test_scores_are_finite_and_nonnegative(self):
        _, rows = _run()
        assert all(torch.isfinite(torch.tensor(row.task_impact)) for row in rows)
        assert all(row.task_impact >= 0 for row in rows)
        assert all(row.local_relative_error >= 0 for row in rows)

    def test_alpha_zero_has_zero_error(self):
        _, rows = _run(alpha=0.0)
        assert all(row.local_relative_error < 1e-8 for row in rows)
        assert all(row.task_impact < 1e-8 for row in rows)

    def test_impact_shares_sum_to_one(self):
        _, rows = _run()
        assert abs(sum(row.task_impact_share for row in rows) - 1.0) < 1e-6

    def test_impact_shares_sum_to_one_over_multiple_batches(self):
        torch.manual_seed(0)
        model = _Model()
        for module in model.modules():
            if isinstance(module, BitLinear):
                module.alpha = 1.0
        analyzer = SensitivityAnalyzer(model)
        for _ in range(3):
            analyzer.reset()
            loss = model(torch.randn(2, 8)).pow(2).mean()
            analyzer.score(loss)
        rows = analyzer.results()
        analyzer.close()
        assert abs(sum(row.task_impact_share for row in rows) - 1.0) < 1e-6
        assert all(row.batches == 3 for row in rows)

    def test_sorted_by_task_impact(self):
        _, rows = _run()
        assert [row.task_impact for row in rows] == sorted(
            [row.task_impact for row in rows], reverse=True
        )

    def test_does_not_populate_parameter_gradients(self):
        model, _ = _run()
        assert all(parameter.grad is None for parameter in model.parameters())

    def test_state_dict_is_unchanged(self):
        model = _Model()
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        analyzer = SensitivityAnalyzer(model)
        analyzer.reset()
        loss = model(torch.randn(2, 8)).pow(2).mean()
        analyzer.score(loss)
        analyzer.close()
        after = model.state_dict()
        assert set(before) == set(after)
        for name, tensor in before.items():
            assert torch.equal(tensor, after[name])

    def test_close_removes_hooks(self):
        model = _Model()
        analyzer = SensitivityAnalyzer(model)
        assert any(module._forward_hooks for module in model.modules() if isinstance(module, BitLinear))
        analyzer.close()
        assert all(not module._forward_hooks for module in model.modules() if isinstance(module, BitLinear))

    def test_aggregate_by_component(self):
        _, rows = _run()
        aggregate = aggregate_by_component(rows)
        assert set(aggregate) == {"q", "ffn_in"}
        assert aggregate["q"]["layers"] == 2
        assert aggregate["ffn_in"]["params"] == 2 * 8 * 8
        assert abs(sum(bucket["task_impact_share"] for bucket in aggregate.values()) - 1.0) < 1e-6
