"""Forced alignment builder (Stage 4, T4.1).

Uses Qwen3-ForcedAligner-0.6B to produce word/token timestamps, which are
then consumed by ``data/prefix.py`` to build stable-prefix training targets.
"""
