"""Gold alignment: how much of the transcript the model's own text already covers.

Why this exists: on-policy training needs to know which words are *new* for a
given chunk. We align the model's committed text against the reference
transcript (edit distance) and return the unfinished remainder as the target.
If the model already covered everything, the correct target is the empty
string: "stop, nothing new here". That empty-target case is what makes the
model able to stay silent - the single most important behaviour in streaming.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

from asr1bit.text import normalize_text


def alignment_key(word: str) -> str:
    """Case/punctuation-insensitive key for one word (1:1 with the raw word)."""
    return normalize_text(word, mode="librispeech")


def gold_coverage(
    own_words: Sequence[str],
    gold_words: Sequence[str],
    *,
    key: Callable[[str], str] | None = None,
) -> int:
    """Number of leading reference words the model's own text covers.

    Unit-cost edit-distance alignment between the model's own words and the
    reference; the returned coverage is the last reference index aligned to a
    matching word (+1), so repetitions and insertions in the own text do not
    inflate it. Returns 0 when nothing matches.
    """
    own = list(own_words)
    gold = list(gold_words)
    n, m = len(own), len(gold)
    if n == 0 or m == 0:
        return 0
    if key is not None:
        own = [key(word) for word in own]
        gold = [key(word) for word in gold]

    def matches(a: str, b: str) -> bool:
        return a == b and (key is None or a != "")

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        row, previous = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            cost = 0 if matches(own[i - 1], gold[j - 1]) else 1
            row[j] = min(previous[j] + 1, row[j - 1] + 1, previous[j - 1] + cost)

    i, j = n, m
    covered = 0
    while i > 0 and j > 0:
        if matches(own[i - 1], gold[j - 1]) and dp[i][j] == dp[i - 1][j - 1]:
            covered = max(covered, j)
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
        elif dp[i][j] == dp[i][j - 1] + 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    return covered
