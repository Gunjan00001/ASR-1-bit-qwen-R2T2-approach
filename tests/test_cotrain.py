"""Tests for the precision co-training schedule."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from asr1bit.train.cotrain import cotrain_alpha

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_stage1_qat.py"


class TestSchedule:
    def test_disabled_returns_scheduled_alpha(self):
        for step in range(10):
            assert cotrain_alpha(step, 1.0, fp_every=0) == 1.0
            assert cotrain_alpha(step, 0.5, fp_every=-3) == 0.5

    def test_every_fourth_step_is_full_precision(self):
        values = [cotrain_alpha(step, 1.0, fp_every=4) for step in range(8)]
        assert values == [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]

    def test_start_step_leaves_quantization_delay_untouched(self):
        # Steps 0-1 are the baseline's own fp delay; co-training starts at 2.
        values = [
            cotrain_alpha(step, 0.0 if step < 2 else 1.0, fp_every=2, start_step=2)
            for step in range(6)
        ]
        assert values == [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]

    def test_ternary_steps_keep_the_scheduled_alpha(self):
        assert cotrain_alpha(0, 0.5, fp_every=4, start_step=0) == 0.5
        assert cotrain_alpha(1, 0.5, fp_every=4, start_step=0) == 0.5
        assert cotrain_alpha(3, 0.5, fp_every=4, start_step=0) == 0.0

    def test_does_not_change_fp_steps_that_are_already_zero(self):
        assert cotrain_alpha(3, 0.0, fp_every=4) == 0.0

    def test_deterministic(self):
        first = [cotrain_alpha(step, 1.0, fp_every=3) for step in range(100)]
        second = [cotrain_alpha(step, 1.0, fp_every=3) for step in range(100)]
        assert first == second


def _load_trainer():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("run_stage1_qat_cotrain_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTrainerContract:
    def test_flag_defaults_off(self, monkeypatch):
        module = _load_trainer()
        monkeypatch.setattr(sys, "argv", ["run_stage1_qat.py", "--steps", "1"])
        args = module.parse_args()
        assert args.cotrain_fp_every == 0

    def test_flag_parses(self, monkeypatch):
        module = _load_trainer()
        monkeypatch.setattr(sys, "argv", [
            "run_stage1_qat.py", "--steps", "1", "--cotrain-fp-every", "4",
        ])
        args = module.parse_args()
        assert args.cotrain_fp_every == 4
