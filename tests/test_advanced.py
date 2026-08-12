import time

import numpy as np
import pytest

from lexiflow.asr.backends import ScriptedBackend
from lexiflow.audio.segmenter import SpeechSegmenter
from lexiflow.audio.speaker import SpeakerTracker, mfcc, voice_embedding
from lexiflow.config import LexiFlowConfig, SegmenterConfig
from lexiflow.nlp.pipeline import AnalyticsEngine
from lexiflow.nlp.summarize import (
    DigestBuilder,
    KeyphraseRanker,
    TextRankSummarizer,
    TopicTracker,
    split_sentences,
    tokenize,
)
from lexiflow.pipeline import LexiFlowPipeline
from lexiflow.state.store import SessionStore

SAMPLE_RATE = 16_000

MEETING = [
    "The ingestion rewrite is done and the ring buffer is holding up under load.",
    "We cut the ring buffer latency down and the audio pipeline finally feels smooth.",
    "Ring buffer memory usage stays flat even after an hour of continuous capture.",
    "Let's talk about the hiring plan for the design team next quarter.",
    "The design team needs two more product designers before the hiring freeze.",
    "Hiring approvals for design have to clear finance before the freeze lands.",
]


def synthetic_voice(fundamental, seconds=2.0, seed=0, sample_rate=SAMPLE_RATE):
    generator = np.random.default_rng(seed)
    times = np.arange(int(sample_rate * seconds)) / sample_rate
    harmonics = sum(np.sin(2 * np.pi * fundamental * k * times) / k for k in range(1, 12))
    return (harmonics * 0.2 + generator.normal(0, 0.01, times.size)).astype(np.float32)


@pytest.fixture()
def config(tmp_path):
    settings = LexiFlowConfig()
    settings.state.database_path = tmp_path / "advanced.db"
    settings.segmenter.min_segment_seconds = 0.3
    settings.segmenter.silence_hangover_seconds = 0.2
    settings.segmenter.max_segment_seconds = 20.0
    settings.segmenter.partial_interval_seconds = 0.5
    settings.segmenter.partial_min_seconds = 0.5
    settings.asr.warmup = False
    return settings


def test_mfcc_shape_and_stability():
    cepstra = mfcc(synthetic_voice(140, seconds=1.0, seed=3))
    assert cepstra.shape[1] == 13
    assert cepstra.shape[0] > 50
    assert np.isfinite(cepstra).all()


def test_mfcc_rejects_short_audio():
    assert mfcc(np.zeros(100, dtype=np.float32)).shape[0] == 0
    assert voice_embedding(np.zeros(100, dtype=np.float32)) is None


def test_voice_embedding_is_unit_length():
    embedding = voice_embedding(synthetic_voice(180, seed=5))
    assert embedding is not None
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)


def test_speaker_tracker_separates_two_voices():
    tracker = SpeakerTracker()
    labels = [
        tracker.assign(synthetic_voice(fundamental, seed=seed)).label
        for fundamental, seed in [(110, 1), (240, 2), (112, 3), (238, 4), (109, 5)]
    ]
    assert tracker.speaker_count == 2
    assert labels[0] == labels[2] == labels[4]
    assert labels[1] == labels[3]
    assert labels[0] != labels[1]


def test_speaker_tracker_respects_max_speakers():
    tracker = SpeakerTracker(max_speakers=1)
    tracker.assign(synthetic_voice(110, seed=1))
    assignment = tracker.assign(synthetic_voice(300, seed=2))
    assert tracker.speaker_count == 1
    assert assignment.is_new is False


def test_speaker_tracker_ignores_short_audio():
    tracker = SpeakerTracker(min_seconds=1.0)
    assert tracker.assign(synthetic_voice(140, seconds=0.4, seed=1)) is None
    assert tracker.speaker_count == 0


def test_speaker_profiles_accumulate_time():
    tracker = SpeakerTracker()
    tracker.assign(synthetic_voice(110, seconds=2.0, seed=1))
    tracker.assign(synthetic_voice(111, seconds=2.0, seed=2))
    profile = tracker.profiles()[0]
    assert profile.segments == 2
    assert profile.total_seconds == pytest.approx(4.0, abs=0.05)


def test_segmenter_emits_partials_before_the_pause():
    settings = SegmenterConfig(
        min_segment_seconds=0.3,
        silence_hangover_seconds=0.3,
        max_segment_seconds=20.0,
        partial_interval_seconds=0.5,
        partial_min_seconds=0.5,
    )
    segmenter = SpeechSegmenter(settings)
    segments = list(segmenter.push(synthetic_voice(150, seconds=3.0, seed=2)))
    assert any(not segment.is_final for segment in segments)
    assert all(segment.reason == "partial" for segment in segments if not segment.is_final)


def test_segmenter_partials_can_be_disabled():
    settings = SegmenterConfig(max_segment_seconds=20.0, emit_partials=False)
    segmenter = SpeechSegmenter(settings)
    assert list(segmenter.push(synthetic_voice(150, seconds=3.0, seed=2))) == []


def test_tokenize_drops_stopwords():
    tokens = tokenize("The deadline is on Friday and it is very important")
    assert "the" not in tokens
    assert "deadline" in tokens and "friday" in tokens


def test_split_sentences():
    assert len(split_sentences("One thing. Two things! Three? ")) == 3


def test_keyphrase_ranker_finds_multiword_phrases():
    phrases = [item.text for item in KeyphraseRanker().rank(" ".join(MEETING), limit=8)]
    assert any("ring buffer" in phrase for phrase in phrases)
    assert any("design team" in phrase or "hiring" in phrase for phrase in phrases)


def test_keyphrase_ranker_handles_empty_text():
    assert KeyphraseRanker().rank("") == []


def test_textrank_prefers_central_sentences():
    ranked = TextRankSummarizer().rank(MEETING)
    assert len(ranked) == len(MEETING)
    assert ranked[0].score >= ranked[-1].score
    assert TextRankSummarizer().summarize(MEETING, limit=3)[0].position < ranked[-1].position + 1


def test_textrank_handles_single_and_empty_input():
    assert TextRankSummarizer().rank([]) == []
    assert TextRankSummarizer().rank(["only one"])[0].score == 1.0


def test_topic_tracker_detects_subject_change():
    tracker = TopicTracker(window=3, threshold=0.2, min_tokens=4)
    shifts = [tracker.push(line) for line in MEETING]
    assert any(shift is not None for shift in shifts)
    detected = next(shift for shift in shifts if shift is not None)
    assert detected.similarity < 0.2
    assert tracker.shifts


def test_topic_tracker_stays_quiet_on_one_subject():
    tracker = TopicTracker(window=2, threshold=0.05, min_tokens=4)
    line = "ring buffer latency stays flat under load"
    assert all(tracker.push(line) is None for _ in range(8))


def test_digest_builder_produces_summary_and_rate():
    digest = DigestBuilder().build(MEETING, audio_seconds=60.0)
    assert digest.summary
    assert digest.keyphrases
    assert digest.word_count > 0
    assert digest.speaking_rate == pytest.approx(digest.word_count, abs=0.01)
    assert "## Summary" in digest.as_markdown()


def test_digest_builder_handles_empty_input():
    digest = DigestBuilder().build([])
    assert digest.summary == []
    assert "nothing captured yet" in digest.as_markdown()


def test_analytics_engine_reports_topic_shift(config):
    engine = AnalyticsEngine(config.nlp)
    engine.config.topic_window = 3
    engine.topics = TopicTracker(window=3, threshold=0.2, min_tokens=4)
    insights = [engine.analyse(line) for line in MEETING]
    assert any(insight.topic_shift is not None for insight in insights)
    assert engine.stats.topic_shifts >= 1


def test_store_tracks_speaker_share(config):
    store = SessionStore(config.state)
    analytics = AnalyticsEngine(config.nlp)
    store.record(
        "first line", analytics.analyse("first line"), speaker="Speaker A", audio_seconds=3.0
    )
    store.record(
        "second line", analytics.analyse("second line"), speaker="Speaker B", audio_seconds=1.0
    )
    rows = store.speakers()
    assert [row["label"] for row in rows] == ["Speaker A", "Speaker B"]
    assert rows[0]["share"] == pytest.approx(0.75)
    assert store.metrics()["speakers"] == 2
    store.close()


def test_store_partial_is_replaced_by_final(config):
    store = SessionStore(config.state)
    store.set_partial("half a sen", "Speaker A")
    assert store.partial()["text"] == "half a sen"
    store.record("half a sentence finished")
    assert store.partial() is None
    store.close()


def test_store_cross_session_search(config):
    first = SessionStore(config.state)
    analytics = AnalyticsEngine(config.nlp)
    first.seed(["the quarterly budget review is on Monday"], analytics)
    first.close()

    second = SessionStore(config.state)
    hits = second.search_all_sessions("budget")
    assert hits and hits[0]["text"].startswith("the quarterly budget")
    assert hits[0]["session_id"] == first.session_id
    assert second.search_all_sessions("") == []

    recovered = second.load_session(first.session_id)
    assert recovered[0]["text"] == "the quarterly budget review is on Monday"
    assert second.latest_session_with_transcript() == first.session_id
    second.close()


def test_store_session_search_returns_real_rows(config):
    store = SessionStore(config.state)
    analytics = AnalyticsEngine(config.nlp)
    store.seed(["the certificate expires on Friday"], analytics)
    hit = store.search("certificate")[0]
    assert hit["seq"] == 1
    assert "certificate" in hit["text"]
    store.close()


def test_store_digest_round_trip(config):
    store = SessionStore(config.state)
    analytics = AnalyticsEngine(config.nlp)
    store.seed(MEETING, analytics)
    digest = store.digest(analytics)
    assert digest.summary
    assert digest.keyphrases
    store.close()


def test_pipeline_streams_partials_and_speakers(config):
    backend = ScriptedBackend(["alpha line", "beta line", "gamma line"], config.asr)
    pipeline = LexiFlowPipeline(config, backend=backend)
    pipeline.start(open_microphone=False)

    for fundamental, seed in [(110, 11), (240, 12)]:
        pipeline.feed(synthetic_voice(fundamental, seconds=3.0, seed=seed))
        pipeline.feed(np.zeros(SAMPLE_RATE, dtype=np.float32))
        time.sleep(0.4)

    deadline = time.time() + 6.0
    while time.time() < deadline and len(pipeline.store.transcript()) < 2:
        time.sleep(0.05)

    pipeline.stop()
    health = pipeline.health()
    assert health.errors == []
    assert health.partials_out >= 1
    assert health.speakers == 2
    assert {item.speaker for item in pipeline.store.transcript()} == {"Speaker A", "Speaker B"}
    assert pipeline.snapshot()["speakers"]
    pipeline.close()


def test_pipeline_digest_from_injected_text(config):
    pipeline = LexiFlowPipeline(config, backend=ScriptedBackend([], config.asr))
    for line in MEETING:
        pipeline.submit_text(line)
    digest = pipeline.digest()
    assert digest.summary
    assert digest.word_count > 20
    pipeline.close()


def test_diarization_can_be_switched_off(config):
    config.diarization.enabled = False
    pipeline = LexiFlowPipeline(config, backend=ScriptedBackend(["quiet"], config.asr))
    assert pipeline.transcription.speakers is None
    assert pipeline.health().speakers == 0
    pipeline.close()


class SlowBackend(ScriptedBackend):
    def transcribe(self, audio, sample_rate=SAMPLE_RATE):
        time.sleep(0.4)
        return super().transcribe(audio, sample_rate)


def test_drain_waits_for_a_slow_backend(config):
    config.segmenter.emit_partials = False
    backend = SlowBackend(["first line", "second line", "third line"], config.asr)
    pipeline = LexiFlowPipeline(config, backend=backend)
    pipeline.start(open_microphone=False)

    for seed in (21, 22, 23):
        pipeline.feed(synthetic_voice(130 + seed, seconds=1.0, seed=seed))
        pipeline.feed(np.zeros(SAMPLE_RATE // 2, dtype=np.float32))

    assert pipeline.drain(timeout=20.0) is True
    assert len(pipeline.store.transcript()) == 3
    assert pipeline.health().segment_queue == 0
    pipeline.close()


def test_drain_reports_false_when_it_times_out(config):
    config.segmenter.emit_partials = False
    pipeline = LexiFlowPipeline(config, backend=SlowBackend(["slow"], config.asr))
    pipeline.start(open_microphone=False)
    for seed in (31, 32, 33, 34):
        pipeline.feed(synthetic_voice(150 + seed, seconds=1.0, seed=seed))
        pipeline.feed(np.zeros(SAMPLE_RATE // 2, dtype=np.float32))
    assert pipeline.drain(timeout=0.2) is False
    pipeline.close()
