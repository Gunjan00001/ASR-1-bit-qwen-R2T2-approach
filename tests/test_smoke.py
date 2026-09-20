"""Smoke test: package imports and version is set."""

import asr1bit


def test_version():
    assert asr1bit.__version__ == "0.1.0"
