"""Config-matrix evaluation harness -> result tables (Stage 0 + 6).

Locked interface:
    run_baseline(model_id, mode, chunk_ms) -> WERReport
"""


class WERReport:
    """Container for WER, latency, RTF, and size results."""


def run_baseline(model_id, mode, chunk_ms):
    """Run a baseline/streaming config and return a ``WERReport``."""
    raise NotImplementedError("Stage 0, T0.3-T0.4")
