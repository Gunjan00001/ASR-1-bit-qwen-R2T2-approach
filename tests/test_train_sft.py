"""Tests for the on-policy streaming SFT helpers (Stage 4, T4.3).

The rollout replay, the gold alignment and the offline/on-policy mixing are all
CPU-testable: the model is replaced by a scripted ``decode_batch``.
"""

import numpy as np

from asr1bit.qwen.stream import WhitespaceTokenCodec
from asr1bit.train.examples import build_distill_examples, build_onpolicy_examples
from asr1bit.train.filters import is_degenerate
from asr1bit.train.gold import gold_coverage
from asr1bit.train.rollout import collect_streaming_samples
from asr1bit.train.samples import (
    StreamingSample,
    continuation,
    interleave_with_ratio,
    prompt_tail,
    select_evenly,
)


class _ScriptedBatchDecoder:
    """Returns a scripted continuation per call and records the calls."""

    def __init__(self, outputs=None):
        self.calls = []
        self._outputs = list(outputs or [])

    def __call__(self, prompts, audios):
        self.calls.append(
            {
                "prompts": list(prompts),
                "audio_lengths": [int(np.asarray(a).shape[0]) for a in audios],
            }
        )
        index = len(self.calls) - 1
        text = self._outputs[index] if index < len(self._outputs) else " x y z"
        return [text] * len(prompts)


def _tone(samples: int, sample_rate: int = 1000) -> np.ndarray:
    return np.linspace(0.0, 1.0, samples, dtype=np.float32)


class TestGoldCoverage:
    def test_exact_prefix(self):
        assert gold_coverage(["a", "b", "c"], ["a", "b", "c", "d"]) == 3

    def test_repetition_tail_does_not_inflate_coverage(self):
        assert gold_coverage(["he", "nods", "his", "his", "his"], ["he", "nods", "his", "consent"]) == 3

    def test_full_transcript_plus_loop_covers_everything(self):
        own = ["he", "nods", "his", "consent", "ill", "ill"]
        assert gold_coverage(own, ["he", "nods", "his", "consent"]) == 4

    def test_skipped_reference_word_still_counts_as_reached(self):
        # The alignment inserts the missing word (cost 1) and matches "consent";
        # the target is then "" (stop) rather than a re-emission that would make
        # the transcript worse.
        assert gold_coverage(["he", "nods", "consent"], ["he", "nods", "his", "consent"]) == 4

    def test_error_in_the_middle(self):
        assert gold_coverage(["a", "x", "c"], ["a", "b", "c", "d"]) == 3

    def test_garbage_covers_nothing(self):
        assert gold_coverage(["zzz", "zzz"], ["a", "b"]) == 0

    def test_empty_inputs(self):
        assert gold_coverage([], ["a", "b"]) == 0
        assert gold_coverage(["a"], []) == 0

    def test_full_match(self):
        assert gold_coverage(["a", "b"], ["a", "b"]) == 2

    def test_key_is_case_and_punctuation_insensitive(self):
        def key(word):
            return word.lower().strip(".")

        assert gold_coverage(["WORDS.", "WAS"], ["words", "was", "it"], key=key) == 2

    def test_empty_key_never_matches(self):
        def key(word):
            return word.lower().strip("-")

        assert gold_coverage(["--"], ["a"], key=key) == 0

    def test_words_out_of_order_are_handled_by_the_alignment(self):
        # An extra leading word is an insertion; the rest still lines up.
        assert gold_coverage(["oh", "he", "nods", "his"], ["he", "nods", "his", "consent"]) == 3


class TestContinuation:
    def test_empty_remainder_is_eos_only(self):
        assert continuation("PROMPT", []) == ""

    def test_empty_prompt_has_no_leading_space(self):
        assert continuation("", ["words", "was"]) == "words was"

    def test_prompt_ending_with_space_is_not_doubled(self):
        assert continuation("PROMPT ", ["words", "was"]) == "words was"

    def test_prompt_ending_on_a_word_gets_a_separator(self):
        assert continuation("PROMPT WORDS", ["was", "it"]) == " was it"

    def test_completes_a_partial_word(self):
        assert continuation("PROMPT WORDS WAS IT THEIR COLOU", ["COLOURS"]) == "RS"

    def test_completes_a_partial_word_then_continues(self):
        assert continuation("PROMPT COLOU", ["COLOURS", "NOW"]) == "RS NOW"

    def test_partial_word_that_does_not_match_still_separates(self):
        assert continuation("PROMPT COLOU", ["THEIR"]) == " THEIR"


class TestCollectStreamingSamples:
    def _collect(self, audios, **kwargs):
        decoder = _ScriptedBatchDecoder()
        kwargs.setdefault("chunk_size_sec", 1.0)
        kwargs.setdefault("sample_rate", 1000)
        kwargs.setdefault("unfixed_chunk_num", 2)
        kwargs.setdefault("unfixed_token_num", 1)
        prompt_raw = kwargs.pop("prompt_raw", "PROMPT ")
        samples = collect_streaming_samples(
            decoder,
            WhitespaceTokenCodec(),
            prompt_raw,
            audios,
            **kwargs,
        )
        return decoder, samples

    def test_prompts_use_the_official_rollback_chain(self):
        decoder, _ = self._collect([_tone(3000)])
        # chunk 0 and 1 are warm-up (empty prefix); chunk 2 continues the
        # rolled-back own text from chunk 1 (" x y z" -> "x y").
        assert decoder.calls[0]["prompts"] == ["PROMPT "]
        assert decoder.calls[1]["prompts"] == ["PROMPT "]
        assert decoder.calls[2]["prompts"] == ["PROMPT x y"]

    def test_audio_heard_grows_by_one_chunk(self):
        _, samples = self._collect([_tone(3000)])
        assert [s.audio.shape[0] for s in samples[0]] == [1000, 2000, 3000]
        assert [s.audio_sec for s in samples[0]] == [1.0, 2.0, 3.0]

    def test_final_flag_marks_the_tail_chunk(self):
        _, samples = self._collect([_tone(2500)])
        assert [s.final for s in samples[0]] == [False, False, True]

    def test_samples_preserve_utterance_order(self):
        _, samples = self._collect([_tone(2100), _tone(1000)])
        assert len(samples) == 2
        assert len(samples[0]) == 3
        assert len(samples[1]) == 1
        assert samples[1][0].final

    def test_keep_filters_states(self):
        _, samples = self._collect(
            [_tone(3000)], keep=lambda chunk_id, final: final or chunk_id == 0
        )
        assert [s.chunk_id for s in samples[0]] == [0, 2]

    def test_batch_size_groups_calls(self):
        decoder, _ = self._collect([_tone(2000), _tone(2000)], batch_size=2)
        assert len(decoder.calls) == 2  # both rounds, one call each
        assert decoder.calls[0]["audio_lengths"] == [1000, 1000]
        decoder, _ = self._collect([_tone(2000), _tone(2000)], batch_size=1)
        assert len(decoder.calls) == 4

    def test_max_rounds_caps_the_replay(self):
        _, samples = self._collect([_tone(5000)], max_rounds=2)
        assert len(samples[0]) == 2

    def test_progress_reports_completed_utterances(self):
        seen = []
        self._collect([_tone(2000), _tone(1000)], progress=lambda done, total: seen.append((done, total)))
        assert seen[-1] == (2, 2)
        assert all(total == 2 for _, total in seen)

    def test_own_text_is_the_parsed_prefix(self):
        decoder = _ScriptedBatchDecoder(
            outputs=["language English<asr_text>he nods", " his consent"]
        )
        samples = collect_streaming_samples(
            decoder,
            WhitespaceTokenCodec(),
            "PROMPT ",
            [_tone(3000)],
            chunk_size_sec=1.0,
            sample_rate=1000,
            unfixed_chunk_num=0,
            unfixed_token_num=1,
        )
        # chunk 0 sees an empty prefix; chunk 1 continues "he" (one token was
        # rolled back from "…he nods"); chunk 2 continues "he his".
        assert samples[0][0].own_text == ""
        assert samples[0][1].own_text == "he"
        assert samples[0][2].own_text == "he his"

    def test_tuple_results_record_continuation_and_eos(self):
        class _TupleDecoder:
            def __call__(self, prompts, audios):
                return [(" he nods", True) for _ in prompts]

        samples = collect_streaming_samples(
            _TupleDecoder(),
            WhitespaceTokenCodec(),
            "PROMPT ",
            [_tone(3000)],
            chunk_size_sec=1.0,
            sample_rate=1000,
            unfixed_chunk_num=2,
            unfixed_token_num=1,
        )
        assert samples[0][0].continuation == " he nods"
        assert samples[0][0].eos is True
        # The prefix chain still advances on the text part ("he nods" -> rollback
        # one token -> "he") for the third chunk.
        assert samples[0][2].prompt == "PROMPT he"


class TestPromptTail:
    def test_strips_the_own_text(self):
        sample = StreamingSample(
            audio=np.zeros(10, dtype=np.float32),
            prompt="PROMPT he nods",
            own_text="he nods",
            chunk_id=3,
            audio_sec=1.0,
        )
        assert prompt_tail(sample) == "PROMPT "

    def test_empty_own_text_returns_the_prompt(self):
        sample = StreamingSample(
            audio=np.zeros(10, dtype=np.float32),
            prompt="PROMPT",
            own_text="",
            chunk_id=0,
            audio_sec=1.0,
        )
        assert prompt_tail(sample) == "PROMPT"


class TestBuildOnpolicyExamples:
    def _sample(self, chunk_id, own_text, prompt=None, final=False):
        tail = prompt if prompt is not None else f"PROMPT {own_text}"
        return StreamingSample(
            audio=np.zeros(100, dtype=np.float32),
            prompt=tail,
            own_text=own_text,
            chunk_id=chunk_id,
            audio_sec=0.5 * (chunk_id + 1),
            final=final,
        )

    def test_warmup_chunks_target_eos(self):
        samples = [[self._sample(0, ""), self._sample(1, "")]]
        examples = build_onpolicy_examples(samples, ["WORDS WAS IT"])
        assert [e["target"] for e in examples] == ["", ""]

    def test_warmup_target_can_be_disabled(self):
        samples = [[self._sample(0, "", prompt="PROMPT")]]
        examples = build_onpolicy_examples(
            samples, ["WORDS WAS IT"], warmup_target_eos=False
        )
        assert examples[0]["target"] == "WORDS WAS IT"

    def test_target_is_the_transcript_remainder(self):
        samples = [[self._sample(5, "WORDS WAS IT")]]
        examples = build_onpolicy_examples(samples, ["WORDS WAS IT THEIR COLOURS"])
        assert examples[0]["target"] == " THEIR COLOURS"

    def test_complete_own_text_targets_eos(self):
        samples = [[self._sample(5, "WORDS WAS IT THEIR COLOURS")]]
        examples = build_onpolicy_examples(samples, ["WORDS WAS IT THEIR COLOURS"])
        assert examples[0]["target"] == ""

    def test_loop_prefix_target_is_the_remainder_when_unfiltered(self):
        # The unfiltered behaviour (phase-1 recipe) fit these loop-prefix →
        # remainder pairs, which taught the loop back; it is kept for the
        # record and is off by default (docs/LAB_NOTES.md 2026-10-12).
        samples = [[self._sample(5, "HE NODS HIS HIS HIS HIS")]]
        examples = build_onpolicy_examples(
            samples, ["HE NODS HIS CONSENT"], drop_degenerate=False
        )
        assert examples[0]["target"] == " CONSENT"

    def test_partial_word_is_completed(self):
        samples = [[self._sample(5, "WORDS WAS IT THEIR COLOU")]]
        examples = build_onpolicy_examples(samples, ["WORDS WAS IT THEIR COLOURS"])
        assert examples[0]["target"] == "RS"

    def test_prefix_text_is_the_exact_prompt(self):
        samples = [[self._sample(5, "WORDS WAS", prompt="PROMPT WORDS WAS")]]
        examples = build_onpolicy_examples(samples, ["WORDS WAS IT"])
        assert examples[0]["prefix_text"] == "PROMPT WORDS WAS"
        assert examples[0]["audio"].shape == (100,)

    def test_flattens_across_utterances_in_order(self):
        samples = [
            [self._sample(0, "")],
            [self._sample(5, "A")],
        ]
        examples = build_onpolicy_examples(samples, ["A B", "A C"])
        assert len(examples) == 2
        assert examples[1]["target"] == " C"


class TestDistillExamples:
    def _sample(self, chunk_id, continuation_text, eos=True, prompt=None, own_text=""):
        return StreamingSample(
            audio=np.zeros(8, dtype=np.float32),
            prompt=prompt if prompt is not None else f"PROMPT {continuation_text}",
            own_text=own_text,
            chunk_id=chunk_id,
            audio_sec=0.32 * (chunk_id + 1),
            continuation=continuation_text,
            eos=eos,
        )

    def test_target_is_the_teacher_continuation_verbatim(self):
        sample = self._sample(4, " he nods his", prompt="PROMPT he")
        examples = build_distill_examples([[sample]])
        assert len(examples) == 1
        assert examples[0]["prefix_text"] == "PROMPT he"
        assert examples[0]["target"] == " he nods his"
        assert examples[0]["audio"] is sample.audio

    def test_generations_without_eos_are_dropped(self):
        truncated = self._sample(2, " assistant assistant", eos=False)
        assert build_distill_examples([[truncated]]) == []

    def test_empty_warmup_continuation_is_kept(self):
        # The teacher answers a near-empty chunk with EOS and no text: the empty
        # target (plus the collator's EOS) is exactly the stop signal we need.
        sample = self._sample(0, "", eos=True, prompt="PROMPT")
        examples = build_distill_examples([[sample]])
        assert len(examples) == 1
        assert examples[0]["target"] == ""

    def test_metadata_is_stripped_from_the_target(self):
        sample = self._sample(2, "language English<asr_text>If.", eos=True, prompt="PROMPT\n")
        examples = build_distill_examples([[sample]])
        assert examples[0]["target"] == "If."

    def test_empty_metadata_target_becomes_eos_only(self):
        sample = self._sample(0, "language None<asr_text>", eos=True, prompt="PROMPT\n")
        examples = build_distill_examples([[sample]])
        assert examples[0]["target"] == ""

    def test_plain_continuation_is_untouched(self):
        sample = self._sample(4, " it only were not so.", eos=True, prompt="PROMPT\nIf.")
        examples = build_distill_examples([[sample]])
        assert examples[0]["target"] == " it only were not so."

    def test_metadata_is_stripped_from_the_recorded_prompt(self):
        prompt_raw = "TEMPLATE\n"
        sample = self._sample(
            4, " were not so.", eos=True, prompt="TEMPLATE\nlanguage English<asr_text>If it only"
        )
        examples = build_distill_examples([[sample]], prompt_raw=prompt_raw)
        assert examples[0]["prefix_text"] == "TEMPLATE\nIf it only"
        assert examples[0]["target"] == " were not so."

    def test_prompt_is_kept_when_prompt_raw_is_unknown(self):
        sample = self._sample(
            4, " x", eos=True, prompt="TEMPLATE\nlanguage English<asr_text>If"
        )
        examples = build_distill_examples([[sample]])
        assert examples[0]["prefix_text"] == "TEMPLATE\nlanguage English<asr_text>If"

    def test_metadata_handling_can_be_disabled(self):
        sample = self._sample(2, "language English<asr_text>If.", eos=True, prompt="PROMPT\n")
        examples = build_distill_examples([[sample]], strip_metadata=False)
        assert examples[0]["target"] == "language English<asr_text>If."


class TestDegenerateDetection:
    def test_flags_subword_loops(self):
        assert is_degenerate("language'S'S'S'S'S'S")
        assert is_degenerate("WORDS WAS IT'S'T'T'T'T")

    def test_flags_word_loops(self):
        assert is_degenerate("HE NODS HIS HIS HIS HIS")
        assert is_degenerate("THE WHOLE PROCEEDING WAS STANDSTANDSTANDSTAND")

    def test_flags_periodic_units(self):
        assert is_degenerate("abcabcabcabc")

    def test_few_repeats_are_tolerated(self):
        # Up to three trailing repeats can be legitimate speech; the detector
        # needs ``min_repeats`` (default 4).
        assert not is_degenerate("hello hello hello")
        assert not is_degenerate("HI HI")

    def test_normal_text_is_not_degenerate(self):
        assert not is_degenerate("WORDS WAS IT THEIR COLOURS")
        assert not is_degenerate("he nods his consent")
        assert not is_degenerate("")

    def test_single_character_is_not_flagged(self):
        assert not is_degenerate("A")


class TestDegenerateFilter:
    def _sample(self, chunk_id, own_text, prompt=None):
        return StreamingSample(
            audio=np.zeros(4, dtype=np.float32),
            prompt=prompt if prompt is not None else f"PROMPT {own_text}",
            own_text=own_text,
            chunk_id=chunk_id,
            audio_sec=0.32 * (chunk_id + 1),
        )

    def test_looping_own_text_samples_are_dropped(self):
        samples = [
            [
                self._sample(5, "HE NODS HIS HIS HIS HIS"),
                self._sample(6, "HE NODS HIS CONSENT"),
            ]
        ]
        kept = build_onpolicy_examples(samples, ["HE NODS HIS CONSENT"])
        assert len(kept) == 1
        assert kept[0]["target"] == ""

    def test_filter_can_be_disabled(self):
        samples = [[self._sample(5, "HE NODS HIS HIS HIS HIS")]]
        kept = build_onpolicy_examples(
            samples, ["HE NODS HIS CONSENT"], drop_degenerate=False
        )
        assert len(kept) == 1

    def test_warmup_samples_are_never_dropped(self):
        samples = [[self._sample(0, "", prompt="PROMPT")]]
        kept = build_onpolicy_examples(samples, ["HE NODS"])
        assert len(kept) == 1
        assert kept[0]["target"] == ""


class TestInterleaveWithRatio:
    def _items(self, prefix, n):
        return [{"id": f"{prefix}{i}"} for i in range(n)]

    def test_half_half_alternates(self):
        out = interleave_with_ratio(self._items("s", 4), self._items("o", 4), 0.5)
        assert [item["id"] for item in out] == ["s0", "o0", "s1", "o1", "s2", "o2", "s3", "o3"]

    def test_zero_fraction_returns_standard(self):
        out = interleave_with_ratio(self._items("s", 3), self._items("o", 3), 0.0)
        assert [item["id"] for item in out] == ["s0", "s1", "s2"]

    def test_full_fraction_returns_onpolicy(self):
        out = interleave_with_ratio(self._items("s", 3), self._items("o", 3), 1.0)
        assert [item["id"] for item in out] == ["o0", "o1", "o2"]

    def test_quarter_fraction_respects_ratio(self):
        # Pools sized for the ratio (|op| = 0.25/(1-0.25) * |std|): both are
        # consumed exactly and the mix lands on the target share.
        out = interleave_with_ratio(self._items("s", 12), self._items("o", 4), 0.25)
        assert len(out) == 16
        assert len([item for item in out if item["id"].startswith("o")]) == 4

    def test_all_onpolicy_samples_are_kept_when_the_pool_is_larger(self):
        # The ratio is a target, not a cap: no on-policy sample is dropped, so
        # an oversized pool raises the on-policy share instead.
        out = interleave_with_ratio(self._items("s", 12), self._items("o", 12), 0.25)
        assert len(out) == 24
        assert len([item for item in out if item["id"].startswith("o")]) == 12

    def test_exhausted_onpolicy_falls_back_to_standard(self):
        out = interleave_with_ratio(self._items("s", 4), self._items("o", 1), 0.9)
        assert len(out) == 5
        assert [item["id"] for item in out] == ["s0", "o0", "s1", "s2", "s3"]

    def test_no_onpolicy_samples(self):
        assert interleave_with_ratio(self._items("s", 2), [], 0.5) == self._items("s", 2)


class TestSelectEvenly:
    def test_returns_everything_when_under_the_cap(self):
        items = [1, 2, 3]
        assert select_evenly(items, 5) == items

    def test_zero_or_negative_cap(self):
        assert select_evenly([1, 2, 3], 0) == []

    def test_spreads_deterministically(self):
        picked = select_evenly(list(range(10)), 4)
        assert picked == [0, 2, 5, 7]
        assert select_evenly(list(range(10)), 4) == picked


class TestWordsCache:
    """The aligner-words cache: a hit must skip the aligner entirely."""

    def test_roundtrip_and_alignment_skip(self, tmp_path):
        from asr1bit.train.aligned import build_aligned_examples
        from asr1bit.train.aligned_cache import WordsCache, utts_key

        class U:
            def __init__(self, text, duration):
                self.text, self.duration = text, duration
                self.audio = [0.0] * 16000  # sliceable, one second

        utts = [U("HELLO THERE", 1.0), U("SECOND ONE", 1.0)]
        calls = {"n": 0}

        from types import SimpleNamespace

        def fake_align(audio, text, aligner=None):
            calls["n"] += 1
            return [SimpleNamespace(text="HELLO", start=0.0, end=0.4),
                    SimpleNamespace(text="THERE", start=0.4, end=0.9)]

        # the builder imports align_words *inside* the function from its source
        # module, so the patch must land there
        import asr1bit.data.align as align_mod
        orig = align_mod.align_words
        align_mod.align_words = fake_align
        try:
            cache = WordsCache(utts_key(utts, "m"), directory=tmp_path)
            first = build_aligned_examples(utts, processor=_FakeProcessor(), aligner=object(),
                                           words_cache=cache)
            assert calls["n"] == 2 and first
            cache.save()

            cache2 = WordsCache(utts_key(utts, "m"), directory=tmp_path)
            second = build_aligned_examples(utts, processor=_FakeProcessor(), aligner=object(),
                                            words_cache=cache2)
            assert calls["n"] == 2, "a cache hit must not call the aligner again"
            assert len(second) == len(first)
        finally:
            align_mod.align_words = orig


class _FakeProcessor:
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return "<|prompt|>"
