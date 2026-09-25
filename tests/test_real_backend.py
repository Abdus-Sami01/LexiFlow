"""The real Whisper runtime, end to end, with a model built here instead of downloaded.

Every other test runs against the scripted backend, which proves the wiring but never
touches a real inference engine. These do: real CTranslate2, real beam search, real
word timings. They are slow and opt-in, so run them with LEXIFLOW_REAL_BACKEND=1.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from lexiflow.asr import models  # noqa: E402
from lexiflow.asr.backends import create_backend  # noqa: E402
from lexiflow.config import ASRConfig, LexiFlowConfig  # noqa: E402
from lexiflow.pipeline import LexiFlowPipeline  # noqa: E402
from lexiflow.selftest import two_speaker_audio  # noqa: E402

RATE = 16_000

pytestmark = pytest.mark.skipif(
    os.environ.get("LEXIFLOW_REAL_BACKEND") != "1",
    reason="slow: set LEXIFLOW_REAL_BACKEND=1 to run the real inference engine",
)


def requires(module: str) -> None:
    pytest.importorskip(module, reason=f"{module} is not installed")


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    requires("ctranslate2")
    requires("faster_whisper")
    import tiny_whisper

    return tiny_whisper.build(tmp_path_factory.mktemp("weights") / "ct2-whisper")


@pytest.fixture()
def asr_config(model):
    return ASRConfig(
        backend="faster_whisper",
        model_path=str(model),
        warmup=False,
        beam_size=1,
        word_timestamps=True,
    )


def speech(seconds=2.0, fundamental=150.0):
    times = np.arange(int(RATE * seconds)) / RATE
    harmonics = sum(np.sin(2 * np.pi * fundamental * k * times) / k for k in range(1, 12))
    return (harmonics * 0.2).astype(np.float32)


def test_the_built_model_looks_like_a_ctranslate2_model(model):
    assert models.is_ctranslate2_model(Path(model))
    assert models.model_format(str(model)) == "ctranslate2"
    assert models.resolve(str(model)) == str(model)
    for name in ("model.bin", "config.json", "tokenizer.json", "vocabulary.json"):
        assert (Path(model) / name).is_file(), name


def test_the_real_backend_loads_and_transcribes(asr_config):
    backend = create_backend(asr_config)
    backend.load()

    assert backend.name == "faster_whisper"
    assert backend.is_loaded
    assert backend.model_format == "ctranslate2"

    result = backend.transcribe(speech(), RATE)
    assert result.backend == "faster_whisper"
    assert result.language
    assert result.audio_seconds == pytest.approx(2.0, abs=0.05)
    assert result.inference_seconds > 0.0


def test_the_real_backend_returns_the_span_shape_the_pipeline_expects(asr_config):
    backend = create_backend(asr_config)
    backend.load()
    result = backend.transcribe(speech(), RATE)

    assert result.segments
    for span in result.segments:
        assert {"start", "end", "text", "words"} <= set(span)
        assert span["end"] >= span["start"]
        for word in span["words"]:
            assert {"start", "end", "text"} <= set(word)


def test_a_ggml_path_is_refused_by_the_ctranslate2_backend(tmp_path):
    requires("faster_whisper")
    weights = tmp_path / "ggml-base.en.bin"
    weights.write_bytes(b"\x00" * 32)

    backend = create_backend(ASRConfig(backend="faster_whisper", model_path=str(weights)))
    with pytest.raises(Exception):
        backend.load()


def test_the_download_guard_still_holds_with_the_library_installed():
    requires("faster_whisper")
    from lexiflow.asr.backends import BackendUnavailable, FasterWhisperBackend

    backend = FasterWhisperBackend(ASRConfig(backend="faster_whisper", model_name="base.en"))
    with pytest.raises(BackendUnavailable) as raised:
        backend.load()
    assert "allow_downloads" in str(raised.value)


def test_the_whole_pipeline_runs_on_the_real_engine(tmp_path, model):
    config = LexiFlowConfig()
    config.state.database_path = tmp_path / "real.db"
    config.asr.backend = "faster_whisper"
    config.asr.model_path = str(model)
    config.asr.warmup = False
    config.asr.beam_size = 1
    config.segmenter.emit_partials = False

    audio = two_speaker_audio()
    pipeline = LexiFlowPipeline(config)
    try:
        pipeline.start(open_microphone=False)
        block = config.audio.block_size
        for offset in range(0, audio.size, block):
            pipeline.feed(audio[offset : offset + block])
        pipeline.drain(timeout=600.0)
        pipeline.stop()

        health = pipeline.health()
        assert health.errors == []
        assert health.segments_in == 2
        assert health.asr_realtime_factor > 0.0
        assert len(pipeline.store.transcript()) == 2
        assert health.speakers == 2
    finally:
        pipeline.close()
