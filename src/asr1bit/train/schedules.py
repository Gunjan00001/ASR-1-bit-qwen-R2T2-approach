"""Learning-rate schedule construction (warmup + cosine).

Why: every run's LR curve is part of the experimental record — a "miracle"
improvement in this project once turned out to be a resume restarting the
schedule rather than the model learning anything, so the schedule is built here,
in one explicit, inspectable place.
"""
from __future__ import annotations

import argparse
import math


def build_lr_lambda(args: argparse.Namespace):
    """Warmup then constant or cosine decay to zero over ``--steps``."""
    warmup = max(1, args.warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        if args.lr_schedule == "cosine":
            progress = (step - warmup) / max(1, args.steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return 1.0

    return lr_lambda
