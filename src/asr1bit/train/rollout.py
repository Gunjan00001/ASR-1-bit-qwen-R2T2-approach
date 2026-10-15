"""Batched on-policy rollout: replay the inference algorithm during training.

Why this exists: streaming behaviour is defined by the *official* algorithm -
feed an audio prefix, ask for the text so far, repeat. Training on examples that
do not come from that process leaves a distribution gap the model cannot cross
(measured: 203.5 WER under the official algorithm while offline was fine). This
module runs the real algorithm over training utterances, but smartly: it decodes
every chunk of every utterance in one batched ``generate`` call, which turns a
process that would take hours into minutes.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from asr1bit.qwen.stream import (
    SAMPLE_RATE,
    TokenCodec,
    parse_stream_output,
    rollback_prefix,
)
from asr1bit.train.samples import StreamingSample, split_result

# The shape of the batched decoder the backend provides: (prompts, audios) ->
# decoded text per prompt. Defined here because this module is its only user.
BatchDecodeFn = Callable[[Sequence[str], Sequence[np.ndarray]], Sequence[str]]


def collect_streaming_samples(
    decode_batch: BatchDecodeFn,
    codec: TokenCodec,
    prompt_raw: str,
    audios: Sequence[np.ndarray],
    *,
    chunk_size_sec: float = 0.32,
    unfixed_chunk_num: int = 2,
    unfixed_token_num: int = 5,
    sample_rate: int = SAMPLE_RATE,
    keep: Callable[[int, bool], bool] | None = None,
    batch_size: int = 4,
    max_rounds: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[list[StreamingSample]]:
    """Replay the official streaming algorithm and record on-policy states.

    Args:
        decode_batch: ``(prompts, audios) -> continuations``. Each continuation
            is the text generated *after* its prompt; audios are the accumulated
            chunks (what the official algorithm re-feeds).
        codec: Tokenizer used for the official prefix rollback.
        prompt_raw: Chat-template prompt without any prefix (the official
            ``_build_text_prompt(context="", force_language=None)``).
        audios: One waveform per utterance (16 kHz float32).
        keep: ``(chunk_id, final) -> bool`` selection of states to record.
            Defaults to keeping every state (memory-heavy; callers normally
            subsample).
        batch_size: Utterances decoded per model call.
        max_rounds: Optional cap on the number of chunk rounds.
        progress: ``(completed_utterances, total)`` callback per round.

    Returns:
        Per-utterance lists of :class:`StreamingSample` (input order preserved).
    """
    chunk_samples = max(1, round(chunk_size_sec * sample_rate))
    keep = keep or (lambda chunk_id, final: True)

    states = []
    for index, audio in enumerate(audios):
        array = np.asarray(audio, dtype=np.float32).reshape(-1)
        states.append(
            {"index": index, "audio": array, "position": 0, "chunk_id": 0, "raw": ""}
        )
    results: list[list[StreamingSample]] = [[] for _ in states]

    round_id = 0
    while True:
        active = [state for state in states if state["position"] < state["audio"].shape[0]]
        if not active or (max_rounds is not None and round_id >= max_rounds):
            break

        records: list[dict] = []
        for state in active:
            start = state["position"]
            end = min(start + chunk_samples, state["audio"].shape[0])
            prefix = (
                ""
                if state["chunk_id"] < unfixed_chunk_num
                else rollback_prefix(codec, state["raw"], unfixed_token_num)
            )
            records.append(
                {
                    "state": state,
                    "prefix": prefix,
                    "prompt": prompt_raw + prefix,
                    "feed": state["audio"][start:end],
                    "end": end,
                    "final": end >= state["audio"].shape[0],
                }
            )

        for batch_start in range(0, len(records), batch_size):
            sub_records = records[batch_start : batch_start + batch_size]
            results_batch = decode_batch(
                [record["prompt"] for record in sub_records],
                [record["feed"] for record in sub_records],
            )
            for record, result in zip(sub_records, results_batch):
                state = record["state"]
                prefix = record["prefix"]
                continuation, eos = split_result(result)
                if keep(state["chunk_id"], record["final"]):
                    results[state["index"]].append(
                        StreamingSample(
                            audio=state["audio"][: record["end"]].copy(),
                            prompt=record["prompt"],
                            own_text=parse_stream_output(prefix)[1],
                            chunk_id=state["chunk_id"],
                            audio_sec=record["end"] / float(sample_rate),
                            final=record["final"],
                            continuation=continuation,
                            eos=eos,
                        )
                    )
                state["raw"] = prefix + (continuation or "")
                state["position"] = record["end"]
                state["chunk_id"] += 1

        round_id += 1
        if progress is not None:
            done = sum(1 for state in states if state["position"] >= state["audio"].shape[0])
            progress(done, len(states))
    return results
