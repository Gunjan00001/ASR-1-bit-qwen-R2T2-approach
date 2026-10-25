"""Precision co-training schedule for ternary QAT.

Why this module exists: the project's fp16 readout drifts during QAT (the
shadow weights move away from the pretrained basin), and the frozen-teacher
anchor only slows that drift. Co-training periodically refreshes the
full-precision path by running selected optimization steps at ``alpha=0``
while the remaining steps run at the ternary operating point. The deployed
model is still ternary: the schedule changes only which forward the optimizer
sees, never the inference representation.

The schedule is deliberately simple and deterministic: after the initial
quantization delay, every ``fp_every``-th step (1-indexed) is a full-precision
step. ``fp_every=4`` therefore spends 25% of the post-delay budget on the fp
path and 75% on the ternary path. The LR schedule, data order, teacher and
budget are the baseline's; the single experimental variable is ``fp_every``.
"""

from __future__ import annotations

__all__ = ["cotrain_alpha"]


def cotrain_alpha(
    step: int,
    scheduled_alpha: float,
    *,
    fp_every: int = 0,
    start_step: int = 0,
) -> float:
    """Return the alpha to use for ``step`` under the co-training schedule.

    Args:
        step: Zero-based training step.
        scheduled_alpha: The alpha the baseline recipe would use (from the
            delay/ramp mode).
        fp_every: Every N-th step (1-indexed) is full precision. ``0`` or a
            negative value disables the schedule and returns
            ``scheduled_alpha`` unchanged.
        start_step: First step at which co-training may override the baseline
            schedule. The quantization delay is left untouched.

    Returns:
        ``0.0`` on a co-training step, otherwise ``scheduled_alpha``.
    """
    if fp_every <= 0 or step < start_step:
        return scheduled_alpha
    return 0.0 if (step + 1) % fp_every == 0 else scheduled_alpha
