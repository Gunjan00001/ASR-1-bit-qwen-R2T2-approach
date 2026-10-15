"""Checkpoint bookkeeping for QAT runs (save / load / selectable states).

Why this module exists: a run's *trajectory* matters more than its final state.
Three separate experiments in this project showed the best streaming state
appearing mid-run and then degrading, and one arm lost its best checkpoint to a
resume that overwrote ``last.pt``. Keeping the save/load/selection logic in one
small, tested place is what makes "evaluate every state and pick the best"
possible (see ``--keep-step-checkpoints`` in scripts/run_stage1_qat.py).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def step_checkpoint_path(save_dir: str | Path, step: int, every: int) -> Path | None:
    """Step-numbered checkpoint path when ``step`` is a multiple of ``every``.

    Keeps the training trajectory's best states selectable afterwards (a
    ``last.pt``-only run loses every earlier state; the binary arm's best
    streaming state was lost that way on 2026-10-15).
    """
    if not every or every <= 0 or step <= 0 or step % every != 0:
        return None
    return Path(save_dir) / f"step{step}.pt"


def shadow_state(model) -> dict:
    """Trainable tensor snapshot (BitLinear shadows and learnable scales)."""
    return {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}


def save_checkpoint(path: Path, model, step: int, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "config": vars(args), "shadows": shadow_state(model)}, path)
    print(f"saved checkpoint {path}", flush=True)


def load_checkpoint(path: str | Path, model) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    shadows = payload.get("shadows", {})
    restored = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in shadows and shadows[name].shape == param.shape:
                param.copy_(shadows[name])
                restored += 1
    print(f"resumed {restored} tensors from {path} (step {payload.get('step')})", flush=True)
    return int(payload.get("step", 0))
