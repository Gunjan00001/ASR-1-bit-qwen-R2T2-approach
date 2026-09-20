"""Streaming decode + Longest Stable Prefix (LSP) emission policy.

Stage 0 (chunked decode) and Stage 4 (T4.4).

Locked interfaces:
    LSPState
    emit_stable(state) -> (committed, pending)
"""


class LSPState:
    """Streaming state: audio buffer, pending tokens, committed transcript."""


def emit_stable(state):
    """Return ``(committed, pending)`` where committed text never changes."""
    raise NotImplementedError("Stage 4, T4.4")
