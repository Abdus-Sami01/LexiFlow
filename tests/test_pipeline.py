import time

import numpy as np
import pytest

from lexiflow.asr.backends import ScriptedBackend, available_backends, create_backend
from lexiflow.asr.engine import TranscriptionEngine
from lexiflow.asr.hardware import build_command, compiler_flags, detect_hardware
from lexiflow.audio.segmenter import SpeechSegment
from lexiflow.config import LexiFlowConfig
from lexiflow.nlp.pipeline import AnalyticsEngine
from lexiflow.pipeline import LexiFlowPipeline
from lexiflow.state.store import SessionStore


@pytest.fixture()
def config(tmp_path):
    settings = LexiFlowConfig()
    settings.state.database_path = tmp_path / "test.db"
    settings.segmenter.min_segment_seconds = 0.2
    settings.segmenter.silence_hangover_seconds = 0.2
    settings.asr.warmup = False
    return settings


def test_hardware_detection_reports_flags():
    profile = detect_hardware()
    assert profile.cpu_count >= 1
    flags = compiler_flags(profile)
    assert flags["CMAKE_BUILD_TYPE"] == "Release"
    assert "cmake --build" in build_command(profile=profile)


def test_null_backend_is_always_available():
    assert "null" in available_backends()
    backend = create_backend()
    assert backend.name in available_backends()


def test_scripted_backend_cycles_lines():
    backend = ScriptedBackend(["one", "two"]).load()
    audio = np.zeros(16_000, dtype=np.float32)
    assert backend.transcribe(audio).text == "one"
    assert backend.transcribe(audio).text == "two"
    assert backend.transcribe(audio).text == "one"


def test_transcription_engine_skips_empty_results(config):
    engine = TranscriptionEngine(config.asr, ScriptedBackend([""], config.asr))
    segment = SpeechSegment(np.zeros(1_600, dtype=np.float32), 16_000, time.time(), time.time(), 0)
    assert engine.transcribe_segment(segment) is None
    assert engine.stats.empty_results == 1


def test_transcription_engine_produces_utterance(config):
    engine = TranscriptionEngine(config.asr, ScriptedBackend(["hello there"], config.asr))
    segment = SpeechSegment(np.zeros(16_000, dtype=np.float32), 16_000, time.time(), time.time(), 0)
    utterance = engine.transcribe_segment(segment)
    assert utterance.text == "hello there"
    assert utterance.audio_seconds == pytest.approx(1.0)


def test_store_records_actions_and_search(config):
    store = SessionStore(config.state)
    analytics = AnalyticsEngine(config.nlp)
    store.seed(
        [
            "Remind me to renew the certificate before Friday.",
            "The rollout went great and everyone is happy.",
        ],
        analytics,
    )
    assert len(store.transcript()) == 2
    assert store.actions(include_done=False)
    assert store.search("certificate")
    assert store.search("") == []
    store.close()


def test_store_toggle_action_persists(config):
    store = SessionStore(config.state)
    analytics = AnalyticsEngine(config.nlp)
    store.seed(["Remind me to book the venue."], analytics)
    action = store.actions()[0]
    assert store.toggle_action(action.identifier).done is True
    assert store.toggle_action(action.identifier).done is False
    assert store.toggle_action("missing") is None
    store.close()


def test_store_metrics_and_export(config):
    store = SessionStore(config.state)
    analytics = AnalyticsEngine(config.nlp)
    store.seed(["We decided to ship on Monday."], analytics)
    metrics = store.metrics()
    assert metrics["utterances"] == 1
    payload = store.export()
    assert payload["transcript"][0]["text"].startswith("We decided")
    assert payload["session"]["id"] == store.session_id
    store.close()


def test_store_notifies_subscribers(config):
    store = SessionStore(config.state)
    received = []
    unsubscribe = store.subscribe(lambda event, payload: received.append(event))
    store.record("plain line")
    unsubscribe()
    store.record("second line")
    assert received == ["transcript"]
    store.close()


def test_pipeline_threads_move_audio_to_insight(config):
    backend = ScriptedBackend(["Remind me to file the report by Friday."], config.asr)
    pipeline = LexiFlowPipeline(config, backend=backend)
    pipeline.start(open_microphone=False)

    rng = np.random.default_rng(7)
    pipeline.feed(rng.normal(0, 0.3, 16_000).astype(np.float32))
    pipeline.feed(np.zeros(16_000, dtype=np.float32))

    deadline = time.time() + 5.0
    while time.time() < deadline and not pipeline.store.transcript():
        time.sleep(0.05)

    pipeline.stop()
    transcript = pipeline.store.transcript()
    assert transcript, "pipeline produced no transcript"
    assert "file the report" in transcript[0].text
    assert pipeline.store.actions()
    health = pipeline.health()
    assert health.errors == []
    pipeline.close()


def test_pipeline_submit_text_bypasses_audio(config):
    pipeline = LexiFlowPipeline(config, backend=ScriptedBackend([], config.asr))
    pipeline.submit_text("The deadline is tomorrow and I am worried.")
    snapshot = pipeline.snapshot()
    assert snapshot["transcript"][0]["compound"] < 0
    assert snapshot["actions"]
    pipeline.close()


def test_config_json_roundtrip(config):
    restored = LexiFlowConfig.from_dict(config.to_dict())
    assert restored.audio.target_sample_rate == 16_000
    assert restored.state.database_path == config.state.database_path
