import argparse

import numpy as np
import torch

from asr1bit.train.dataset import load_train_utterances


def _cache(path, n):
    audio = torch.from_numpy(np.zeros(1600, dtype=np.float32))
    payload = {
        "meta": {"kept": n},
        "utterances": [{"id": f"u{i}", "text": f"text {i}", "audio": audio} for i in range(n)],
    }
    torch.save(payload, path)
    return path


def _args(path, train_n):
    return argparse.Namespace(train_cache=str(path), train_n=train_n)


def test_default_cap_is_train_n(tmp_path):
    args = _args(_cache(tmp_path / "c.pt", 5), train_n=2)
    assert len(load_train_utterances(args)) == 2


def test_limit_overrides_train_n(tmp_path):
    args = _args(_cache(tmp_path / "c.pt", 5), train_n=2)
    assert len(load_train_utterances(args, limit=4)) == 4


def test_limit_cannot_exceed_the_cache(tmp_path):
    args = _args(_cache(tmp_path / "c.pt", 3), train_n=1)
    assert len(load_train_utterances(args, limit=9)) == 3
