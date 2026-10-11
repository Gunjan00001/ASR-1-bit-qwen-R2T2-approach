"""Glue test for the trainer's on-policy rollout (scripts/run_stage1_qat.py).

Exercises ``rollout_onpolicy_examples`` with a fake wrapper (fake processor +
fake inner model) so the wiring — staleness of the batch decode fn, prompt
building, sample selection, example targets — is verified before the GPU job.

Note: the script is loaded by path because it is a CLI entry point, not an
importable package module.
"""

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from asr1bit.data.load import Utterance

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_stage1_qat.py"


def _load_trainer():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("run_stage1_qat_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeProcessor:
    def __init__(self, continuation=" he nods his consent now please"):
        self.tokenizer = _FakeTokenizer()
        self.calls = []
        self.continuation = continuation

    def __call__(self, text, audio, return_tensors=None, padding=None):
        self.calls.append({"text": list(text), "audio": list(audio)})
        return _FakeInputs(len(list(text)))

    def batch_decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return [self.continuation] * int(np.asarray(ids).shape[0])


class _FakeTokenizer:
    eos_token_id = 2

    def encode(self, text):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)


class _FakeInputs:
    def __init__(self, batch):
        self._batch = batch

    def to(self, *args, **kwargs):
        return self

    def __getitem__(self, key):
        return self

    def keys(self):
        return ["input_ids"]

    @property
    def shape(self):
        return (self._batch, 5)


class _FakeInner:
    device = "cpu"
    dtype = "float32"
    eos_id = 2

    def __init__(self):
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        batch = kwargs["input_ids"].shape[0]
        sequences = np.zeros((batch, 8), dtype=int)
        sequences[:, -1] = self.eos_id
        return types.SimpleNamespace(sequences=sequences)


class _FakeWrapper:
    def __init__(self, continuation=" he nods his consent now please"):
        self.processor = _FakeProcessor(continuation)
        self.model = _FakeInner()
        self.max_new_tokens = 32

    def _build_text_prompt(self, context="", force_language=None):
        return "PROMPT "


def _args(**overrides):
    values = {
        "op_distill": 0,
        "op_max_new_tokens": 16,
        "op_chunk_ms": 1000,
        "op_batch_size": 2,
        "op_max_rounds": 4,
        "op_chunk_stride": 1,
        "op_max_samples": 100,
        "op_warmup_eos": 1,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _utterance(text, seconds):
    samples = int(seconds * 16000)
    return Utterance(
        id=text,
        audio=np.linspace(-0.5, 0.5, samples, dtype=np.float32),
        text=text,
        reference=text.lower(),
        duration=seconds,
    )


class TestRolloutOnPolicyExamples:
    def test_collects_examples_for_every_kept_state(self):
        module = _load_trainer()
        wrapper = _FakeWrapper()
        utterances = [_utterance("HE NODS HIS CONSENT NOW PLEASE", 2.5)]
        examples = module.rollout_onpolicy_examples(_args(), wrapper, utterances)
        # 2.5 s at 1 s chunks -> three chunk states; warm-up chunks train EOS,
        # the third continues the model's own rolled-back prefix ("he").
        assert len(examples) == 3
        assert examples[0]["target"] == ""
        assert examples[1]["target"] == ""
        assert examples[2]["prefix_text"] == "PROMPT he"
        assert examples[2]["target"] == " NODS HIS CONSENT NOW PLEASE"
        assert wrapper.processor.calls  # the fake model was actually driven

    def test_max_rounds_truncates_long_utterances(self):
        module = _load_trainer()
        wrapper = _FakeWrapper()
        utterances = [_utterance("HE NODS", 10.0)]
        examples = module.rollout_onpolicy_examples(
            _args(op_max_rounds=3, op_chunk_stride=3), wrapper, utterances
        )
        assert len(examples) == 3  # warm-up 0/1 + the truncated last round

    def test_pool_is_capped_deterministically(self):
        module = _load_trainer()
        wrapper = _FakeWrapper()
        utterances = [_utterance("HE NODS " * 20, 30.0)]
        capped = module.rollout_onpolicy_examples(
            _args(op_max_samples=4, op_chunk_stride=1), wrapper, utterances
        )
        assert len(capped) == 4
        again = module.rollout_onpolicy_examples(
            _args(op_max_samples=4, op_chunk_stride=1), wrapper, utterances
        )
        assert [e["prefix_text"] for e in capped] == [e["prefix_text"] for e in again]

    def test_keep_policy_covers_warmup_stride_and_last_round(self):
        module = _load_trainer()
        keep = module.build_op_keep(_args(op_chunk_stride=3, op_max_rounds=8))
        assert [chunk for chunk in range(8) if keep(chunk, False)] == [0, 1, 3, 6, 7]
        assert keep(4, True)  # a genuinely final chunk is always kept

    def test_distill_mode_clones_the_teacher_continuation(self):
        module = _load_trainer()
        student = _FakeWrapper()
        teacher = _FakeWrapper(continuation=" he nods his consent")
        utterances = [_utterance("HE NODS HIS CONSENT", 2.5)]
        examples = module.rollout_onpolicy_examples(
            _args(op_distill=1), student, utterances, teacher_wrapper=teacher
        )
        assert examples
        assert all(e["target"] == " he nods his consent" for e in examples)
        # The student model is only used for the prompt/codec, never generation.
        assert student.model.calls == 0
        assert teacher.model.calls > 0

    def test_distill_mode_requires_a_teacher(self):
        module = _load_trainer()
        wrapper = _FakeWrapper()
        with pytest.raises(ValueError):
            module.rollout_onpolicy_examples(
                _args(op_distill=1), wrapper, [_utterance("HE NODS", 2.5)]
            )
