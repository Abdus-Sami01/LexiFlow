import socket
import time

import numpy as np
import pytest

from lexiflow.asr.backends import ScriptedBackend
from lexiflow.config import LexiFlowConfig
from lexiflow.nlp.pipeline import AnalyticsEngine
from lexiflow.pipeline import LexiFlowPipeline

SAMPLE_RATE = 16_000


class NetworkUsed(AssertionError):
    """Raised the moment anything in the pipeline reaches for a socket."""


@pytest.fixture()
def no_network(monkeypatch):
    """Make every outbound path explode, so a stray call cannot pass silently."""
    attempts = []

    def refuse(*args, **kwargs):
        attempts.append(args)
        raise NetworkUsed(f"network access attempted: {args}")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    return attempts


def voice(fundamental, seconds=2.0, seed=0):
    generator = np.random.default_rng(seed)
    times = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    harmonics = sum(np.sin(2 * np.pi * fundamental * k * times) / k for k in range(1, 12))
    return (harmonics * 0.2 + generator.normal(0, 0.01, times.size)).astype(np.float32)


def offline_config(tmp_path):
    config = LexiFlowConfig()
    config.state.database_path = tmp_path / "offline.db"
    config.segmenter.min_segment_seconds = 0.4
    config.segmenter.silence_hangover_seconds = 0.3
    config.asr.warmup = False
    return config


def test_analytics_never_touches_the_network(no_network):
    engine = AnalyticsEngine()
    for line in [
        "Remind me to email finance before Friday.",
        "La fecha límite es el viernes y estoy preocupado.",
        "Die Frist ist Freitag und ich bin besorgt.",
    ]:
        engine.analyse(line)
    engine.digest(["Remind me to email finance before Friday."], 10.0)
    assert no_network == []


def test_full_pipeline_never_touches_the_network(tmp_path, no_network):
    config = offline_config(tmp_path)
    backend = ScriptedBackend(["Remind me to file the report by Friday."], config.asr)
    pipeline = LexiFlowPipeline(config, backend=backend)
    pipeline.start(open_microphone=False)

    pipeline.feed(voice(150, seconds=2.0, seed=1))
    pipeline.feed(np.zeros(SAMPLE_RATE, dtype=np.float32))
    assert pipeline.drain(timeout=15.0) is True
    pipeline.stop()

    assert pipeline.store.transcript()
    assert pipeline.health().errors == []
    assert no_network == []
    pipeline.close()


def test_exports_and_search_stay_offline(tmp_path, no_network):
    from lexiflow import export

    config = offline_config(tmp_path)
    pipeline = LexiFlowPipeline(config, backend=ScriptedBackend([], config.asr))
    pipeline.submit_text("Remind me to renew the certificate before Friday.")

    rows = pipeline.store.transcript()
    payload = pipeline.store.export()
    for fmt in sorted(export.FORMATS):
        assert export.render(fmt, rows, payload, pipeline.digest())
    assert pipeline.store.search("certificate")
    assert pipeline.store.search_all_sessions("certificate")
    assert no_network == []
    pipeline.close()


def test_nothing_is_written_outside_the_state_directory(tmp_path):
    config = offline_config(tmp_path)
    pipeline = LexiFlowPipeline(config, backend=ScriptedBackend([], config.asr))
    pipeline.submit_text("Remind me to check the disk usage tomorrow.")
    pipeline.close()

    written = {path.name for path in tmp_path.iterdir()}
    assert "offline.db" in written
    assert all(name.startswith("offline.db") for name in written)


def test_in_memory_mode_writes_nothing_at_all(tmp_path):
    config = offline_config(tmp_path)
    config.state.persist = False
    pipeline = LexiFlowPipeline(config, backend=ScriptedBackend([], config.asr))
    pipeline.submit_text("Remind me to rotate the logs on Monday.")

    assert pipeline.store.transcript()
    assert pipeline.store.actions()
    assert list(tmp_path.iterdir()) == []
    pipeline.close()


def test_no_audio_is_ever_written_to_disk(tmp_path):
    config = offline_config(tmp_path)
    pipeline = LexiFlowPipeline(config, backend=ScriptedBackend(["spoken"], config.asr))
    pipeline.start(open_microphone=False)
    pipeline.feed(voice(150, seconds=2.0, seed=2))
    pipeline.feed(np.zeros(SAMPLE_RATE, dtype=np.float32))
    pipeline.drain(timeout=10.0)
    pipeline.stop()

    audio_files = [
        path
        for path in tmp_path.rglob("*")
        if path.suffix.lower() in {".wav", ".raw", ".pcm", ".flac", ".mp3", ".ogg"}
    ]
    assert audio_files == []
    pipeline.close()


def test_ring_buffer_memory_is_fixed_regardless_of_stream_length(tmp_path):
    config = offline_config(tmp_path)
    config.audio.ring_buffer_seconds = 5.0
    pipeline = LexiFlowPipeline(config, backend=ScriptedBackend([], config.asr))
    pipeline.start(open_microphone=False)

    capacity = pipeline.stream.buffer.capacity
    nbytes = pipeline.stream.buffer._buffer.nbytes
    for _ in range(40):
        pipeline.feed(voice(150, seconds=0.5, seed=3))

    assert pipeline.stream.buffer.capacity == capacity
    assert pipeline.stream.buffer._buffer.nbytes == nbytes
    assert len(pipeline.stream.buffer) <= capacity
    assert pipeline.stream.buffer.stats().dropped > 0
    pipeline.stop()
    pipeline.close()


def test_transcript_retention_is_bounded(tmp_path):
    config = offline_config(tmp_path)
    config.state.max_transcript_items = 25
    config.state.persist = False
    pipeline = LexiFlowPipeline(config, backend=ScriptedBackend([], config.asr))

    for index in range(120):
        pipeline.submit_text(f"line number {index} about the ring buffer")

    assert len(pipeline.store.transcript()) == 25
    assert pipeline.store.metrics()["utterances"] == 25
    pipeline.close()


def soak_pipeline(tmp_path, name):
    config = offline_config(tmp_path)
    config.state.database_path = tmp_path / name
    config.state.persist = False
    config.segmenter.emit_partials = False
    config.diarization.enabled = False
    return LexiFlowPipeline(config, backend=ScriptedBackend(["a spoken line"], config.asr))


def test_a_long_session_loses_nothing_when_the_reader_keeps_up(tmp_path):
    """Two minutes of speech, paced so the ring buffer is never overrun."""
    pipeline = soak_pipeline(tmp_path, "paced.db")
    pipeline.start(open_microphone=False)

    utterances = 40
    for index in range(utterances):
        pipeline.feed(voice(140 + (index % 5), seconds=2.0, seed=index))
        pipeline.feed(np.zeros(SAMPLE_RATE // 2, dtype=np.float32))
        time.sleep(0.02)

    assert pipeline.drain(timeout=60.0) is True
    pipeline.stop()

    health = pipeline.health()
    assert health.errors == []
    assert health.segment_queue == 0
    assert health.utterances_out == utterances
    assert pipeline.stream.buffer.stats().dropped == 0
    pipeline.close()


def test_overload_drops_audio_instead_of_growing_memory(tmp_path):
    """Push far more than the buffer holds: it must shed frames, not allocate."""
    pipeline = soak_pipeline(tmp_path, "overload.db")
    pipeline.config.audio.ring_buffer_seconds = 10.0
    pipeline.stream.buffer = type(pipeline.stream.buffer)(SAMPLE_RATE * 10)
    pipeline.start(open_microphone=False)

    nbytes = pipeline.stream.buffer._buffer.nbytes
    started = time.time()
    for index in range(60):
        pipeline.feed(voice(150, seconds=2.0, seed=index))

    pipeline.drain(timeout=30.0)
    pipeline.stop()

    health = pipeline.health()
    assert health.errors == []
    assert pipeline.stream.buffer._buffer.nbytes == nbytes
    assert pipeline.stream.buffer.stats().dropped > 0
    assert health.utterances_out >= 1
    assert time.time() - started < 60
    pipeline.close()
