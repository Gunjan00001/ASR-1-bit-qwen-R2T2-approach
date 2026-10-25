"""Task-aware layer sensitivity for ternary BitLinear projections.

Why this module exists: the project's earlier shadow-substitution screen was
invalid (flipping a trained ternary layer to its fp16 shadow destroys the
model; it measured the precision-switch mismatch, not sensitivity). Local
weight/response reconstruction is also a poor proxy: g64 improved it and lost
WER. This module instead estimates, for each quantized projection, the
first-order effect of its quantization error on the *task loss*:

    impact_l = mean_tokens | grad_l · (Q(W_l)x_l - W_l x_l) |

where ``grad_l`` is the gradient of the training loss with respect to the
layer's output and the error is the ternary-vs-fp output difference on the
same (activation-quantized) input. It is a local linearization on a small
calibration set, not a WER measurement; it is the cheap screening stage
required before a mixed-precision candidate is trained and officially
evaluated.

The analyzer is read-only: it adds forward hooks, never changes parameters,
and removes the hooks on close.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from asr1bit.bitlinear import BitLinear, absmax_quantize_activations
from asr1bit.replace import component_of

__all__ = ["LayerSensitivity", "SensitivityAnalyzer", "aggregate_by_component"]


@dataclass
class LayerSensitivity:
    """One projection's screening measurements."""

    name: str
    params: int
    component: str
    local_relative_error: float
    task_impact: float
    task_impact_share: float
    task_impact_per_param: float
    batches: int


class SensitivityAnalyzer:
    """Capture per-layer quantization errors and their output gradients."""

    def __init__(self, model: nn.Module, *, eps: float = 1e-6):
        self.eps = float(eps)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._entries: list[tuple[str, int, torch.Tensor, torch.Tensor]] = []
        self._sums: dict[str, dict[str, float]] = {}
        for name, module in model.named_modules():
            if isinstance(module, BitLinear):
                def hook(_module, inputs, output, name=name, module=module):
                    self._hook(name, module, inputs, output)

                self._handles.append(module.register_forward_hook(hook))
        if not self._handles:
            raise ValueError("no BitLinear modules found")

    def _hook(self, name: str, module: BitLinear, inputs, output) -> None:
        x = inputs[0]
        if module.quantize_activations:
            x = absmax_quantize_activations(x)
        with torch.no_grad():
            full_precision = F.linear(x, module.weight, module.bias)
        if not isinstance(output, torch.Tensor):
            raise TypeError("BitLinear output is not a tensor")
        self._entries.append((name, module.weight.numel(), output, full_precision.detach()))

    def __enter__(self) -> SensitivityAnalyzer:  # noqa: PYI034 - returns self, py3.10 target
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    def reset(self) -> None:
        """Forget the previous forward's tensors (call before each batch)."""
        self._entries.clear()

    def score(self, loss: torch.Tensor) -> None:
        """Backpropagate ``loss`` to every captured layer output and accumulate."""
        if not self._entries:
            return
        outputs = [entry[2] for entry in self._entries]
        grads = torch.autograd.grad(
            loss,
            outputs,
            retain_graph=False,
            allow_unused=True,
        )
        for (name, params, output, full_precision), grad in zip(self._entries, grads):
            if grad is None:
                continue
            error = output.detach() - full_precision
            # First-order loss change if this layer's error were removed:
            # -g . error. Per-token, then absolute mean over tokens/batch.
            impact = (grad * error).sum(dim=-1).abs().mean().item()
            fp_energy = full_precision.pow(2).mean().clamp(min=self.eps)
            local = (error.pow(2).mean() / fp_energy).item()
            row = self._sums.setdefault(
                name, {"params": float(params), "impact": 0.0, "local": 0.0, "batches": 0.0}
            )
            row["impact"] += impact
            row["local"] += local
            row["batches"] += 1.0
        self._entries.clear()

    def results(self) -> list[LayerSensitivity]:
        """Per-layer measurements, sorted by absolute task impact (descending)."""
        if not self._sums:
            return []
        averaged = {
            name: row["impact"] / max(1.0, row["batches"])
            for name, row in self._sums.items()
        }
        total_impact = sum(averaged.values())
        rows: list[LayerSensitivity] = []
        for name, row in self._sums.items():
            batches = max(1.0, row["batches"])
            impact = averaged[name]
            params = int(row["params"])
            rows.append(
                LayerSensitivity(
                    name=name,
                    params=params,
                    component=component_of(name) or "other",
                    local_relative_error=row["local"] / batches,
                    task_impact=impact,
                    task_impact_share=(impact / total_impact) if total_impact > 0 else 0.0,
                    task_impact_per_param=impact / params if params else 0.0,
                    batches=int(row["batches"]),
                )
            )
        rows.sort(key=lambda item: item.task_impact, reverse=True)
        return rows

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._entries.clear()


def aggregate_by_component(rows: Sequence[LayerSensitivity]) -> dict[str, dict[str, float]]:
    """Aggregate sensitivity by projection type (q/k/v/o/gate/up/down)."""
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        bucket = out.setdefault(
            row.component, {"layers": 0, "params": 0, "task_impact": 0.0, "local_relative_error": 0.0}
        )
        bucket["layers"] += 1
        bucket["params"] += row.params
        bucket["task_impact"] += row.task_impact
        bucket["local_relative_error"] += row.local_relative_error
    total_impact = sum(bucket["task_impact"] for bucket in out.values())
    for bucket in out.values():
        layers = max(1, int(bucket["layers"]))
        bucket["local_relative_error"] /= layers
        bucket["task_impact_share"] = (
            bucket["task_impact"] / total_impact if total_impact > 0 else 0.0
        )
    return out
