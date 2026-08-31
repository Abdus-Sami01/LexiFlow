import numpy as np
import pytest

from lexiflow import export
from lexiflow.asr.backends import ScriptedBackend
from lexiflow.asr.engine import TranscriptionEngine
from lexiflow.audio.segmenter import SpeechSegment
from lexiflow.audio.speaker import (
    SpeakerTracker,
    attribute_words,
    voice_embedding,
    word_turns,
)
from lexiflow.config import ASRConfig, DiarizationConfig, LexiFlowConfig

RATE = 16_000


def voice(fundamental, seconds=1.0, rate=RATE):
    times = np.arange(int(rate * seconds)) / rate
    harmonics = sum(np.sin(2 * np.pi * fundamental * k * times) / k for k in range(1, 12))
    return (harmonics * 0.2).astype(np.float32)


@pytest.fixture()
def two_voices():
    return np.concatenate([voice(110.0, 1.5), voice(260.0, 1.5)])


@pytest.fixture()
def tracker(two_voices):
    tracker = SpeakerTracker(min_seconds=0.2)
    tracker.assign(two_voices[: RATE + RATE // 2], RATE)
    tracker.assign(two_voices[RATE + RATE // 2 :], RATE)
    return tracker


def words_over(count, span=3.0, origin=0.0):
    step = span / count
    return [
        {
            "start": origin + index * step,
            "end": origin + (index + 1) * step,
            "text": f"w{index}",
        }
        for index in range(count)
    ]


def test_classify_does_not_invent_or_adapt_clusters(tracker, two_voices):
    before = [profile.centroid.copy() for profile in tracker.profiles()]
    embedding = voice_embedding(two_voices[:RATE], RATE)

    assignment = tracker.classify(embedding)
    assert assignment is not None
    assert assignment.is_new is False
    assert tracker.speaker_count == 2
    assert all(
        np.allclose(old, new.centroid)
        for old, new in zip(before, tracker.profiles())
    )


def test_classify_on_an_empty_tracker_returns_nothing():
    assert SpeakerTracker().classify(voice_embedding(voice(120.0), RATE)) is None


def test_words_are_labelled_with_the_voice_underneath_them(tracker, two_voices):
    labelled = attribute_words(words_over(12), two_voices, RATE, tracker)

    assert len(labelled) == 12
    labels = [word["speaker"] for word in labelled]
    assert len(set(labels)) == 2
    assert labels[0] != labels[-1]
    assert all(word["speaker_confidence"] >= 0.0 for word in labelled)


def test_the_switch_lands_near_the_real_boundary(tracker, two_voices):
    labelled = attribute_words(words_over(12), two_voices, RATE, tracker)
    labels = [word["speaker"] for word in labelled]
    switches = [index for index in range(1, len(labels)) if labels[index] != labels[index - 1]]
    assert len(switches) == 1
    assert 4 <= switches[0] <= 8


def test_absolute_word_timings_are_anchored_by_origin(tracker, two_voices):
    shifted = attribute_words(
        words_over(12, origin=1_000.0), two_voices, RATE, tracker, origin=1_000.0
    )
    plain = attribute_words(words_over(12), two_voices, RATE, tracker)
    assert [word["speaker"] for word in shifted] == [word["speaker"] for word in plain]


def test_a_lone_disagreeing_word_is_smoothed_away(tracker, two_voices):
    labelled = attribute_words(words_over(20), two_voices, RATE, tracker)
    labels = [word["speaker"] for word in labelled]
    for index in range(1, len(labels) - 1):
        if labels[index - 1] == labels[index + 1]:
            assert labels[index] == labels[index - 1]


def test_unconfident_words_fall_back_to_the_segment_label(tracker, two_voices):
    labelled = attribute_words(
        words_over(6), two_voices, RATE, tracker, min_confidence=2.0, fallback="Speaker Z"
    )
    assert {word["speaker"] for word in labelled} == {"Speaker Z"}


def test_no_words_and_no_audio_are_both_survivable(tracker):
    assert attribute_words([], voice(120.0), RATE, tracker) == []
    empty = np.zeros(0, dtype=np.float32)
    assert attribute_words(words_over(3), empty, RATE, tracker)[0].get("speaker") is None


def test_turns_collapse_runs_of_one_speaker():
    words = [
        {"start": 0.0, "end": 0.5, "text": "hello", "speaker": "Speaker A"},
        {"start": 0.5, "end": 1.0, "text": "there", "speaker": "Speaker A"},
        {"start": 1.0, "end": 1.5, "text": "hi", "speaker": "Speaker B"},
    ]
    turns = word_turns(words)
    assert [turn["speaker"] for turn in turns] == ["Speaker A", "Speaker B"]
    assert turns[0]["text"] == "hello there"
    assert turns[0]["end"] == 1.0


def test_turns_skip_blank_words():
    assert word_turns([{"text": "  ", "speaker": "Speaker A"}]) == []


def scripted_engine(word_level=True):
    asr = ASRConfig(backend="scripted", word_timestamps=True, warmup=False)
    backend = ScriptedBackend(["one two three four"], asr)
    return TranscriptionEngine(
        asr,
        backend=backend,
        diarization=DiarizationConfig(word_level=word_level, min_seconds=0.2),
    )


def segment_for(audio):
    return SpeechSegment(
        audio=audio,
        sample_rate=RATE,
        started_at=100.0,
        ended_at=100.0 + audio.size / RATE,
        index=1,
        is_final=True,
        reason="silence",
        peak_rms=0.2,
    )


def test_the_engine_labels_words_when_asked(two_voices):
    engine = scripted_engine()
    utterance = engine.transcribe_segment(segment_for(two_voices))
    words = [word for span in utterance.spans for word in span.get("words") or []]
    assert words
    assert all("speaker" in word for word in words)


def test_the_engine_leaves_words_alone_by_default(two_voices):
    engine = scripted_engine(word_level=False)
    utterance = engine.transcribe_segment(segment_for(two_voices))
    words = [word for span in utterance.spans for word in span.get("words") or []]
    assert all("speaker" not in word for word in words)


def test_word_cues_carry_their_own_speaker():
    row = _Row(
        spans=[
            {
                "start": 0.0,
                "end": 1.0,
                "text": "hello there",
                "words": [
                    {"start": 0.0, "end": 0.5, "text": "hello", "speaker": "Speaker A"},
                    {"start": 0.5, "end": 1.0, "text": "there", "speaker": "Speaker B"},
                ],
            }
        ]
    )
    cues = export.to_cues([row], granularity="word")
    assert [cue.speaker for cue in cues] == ["Speaker A", "Speaker B"]


def test_a_word_without_a_label_keeps_the_row_speaker():
    row = _Row(
        speaker="Speaker A",
        spans=[
            {
                "start": 0.0,
                "end": 1.0,
                "text": "hello",
                "words": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            }
        ],
    )
    assert export.to_cues([row], granularity="word")[0].speaker == "Speaker A"


class _Row:
    def __init__(self, text="hello there", speaker=None, spans=None):
        self.seq = 1
        self.text = text
        self.speaker = speaker
        self.started_at = 0.0
        self.ended_at = 1.0
        self.translation = None
        self.spans = spans or []


MIXED_ROW = {
    "seq": 1,
    "text": "hello there",
    "speaker": "Speaker A",
    "spans": [
        {
            "words": [
                {"text": "hello", "speaker": "Speaker A"},
                {"text": "there", "speaker": "Speaker B"},
            ]
        }
    ],
}


def test_markdown_splits_a_line_that_holds_two_voices():
    body = export.to_markdown({"transcript": [MIXED_ROW]})
    assert "- **Speaker A** · hello" in body
    assert "- **Speaker B** · there" in body


def test_markdown_leaves_a_single_voice_line_whole():
    row = dict(MIXED_ROW)
    row["spans"] = [{"words": [{"text": "hello there", "speaker": "Speaker A"}]}]
    body = export.to_markdown({"transcript": [row]})
    assert "- **Speaker A** · hello there" in body


def test_markdown_is_unchanged_without_word_labels():
    row = {"seq": 1, "text": "hello there", "speaker": "Speaker A"}
    assert "- **Speaker A** · hello there" in export.to_markdown({"transcript": [row]})


def test_word_level_without_word_timestamps_is_a_config_error():
    config = LexiFlowConfig()
    config.diarization.word_level = True
    config.asr.word_timestamps = False
    assert any("word_timestamps" in problem for problem in config.validate())

    config.asr.word_timestamps = True
    assert config.validate() == []


def test_a_non_positive_word_window_is_a_config_error():
    config = LexiFlowConfig()
    config.diarization.word_window_seconds = 0.0
    assert any("word_window_seconds" in problem for problem in config.validate())
