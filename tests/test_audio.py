import numpy as np
import pytest

from lexiflow.audio.conversion import prepare_for_whisper, resample_linear, rms, to_float32, to_mono
from lexiflow.audio.ring_buffer import AudioRingBuffer, RingBufferOverflow
from lexiflow.audio.segmenter import SpeechSegmenter
from lexiflow.config import SegmenterConfig


def test_ring_buffer_roundtrip():
    buffer = AudioRingBuffer(100)
    buffer.write(np.arange(30, dtype=np.float32))
    chunk = buffer.read(30)
    assert np.allclose(chunk, np.arange(30))
    assert len(buffer) == 0


def test_ring_buffer_wraps_and_drops_oldest():
    buffer = AudioRingBuffer(10)
    buffer.write(np.arange(8, dtype=np.float32))
    buffer.write(np.arange(8, 16, dtype=np.float32))
    remaining = buffer.read(10)
    assert remaining.size == 10
    assert np.allclose(remaining, np.arange(6, 16))
    assert buffer.stats().dropped == 6


def test_ring_buffer_strict_overflow():
    buffer = AudioRingBuffer(4)
    buffer.write(np.ones(4, dtype=np.float32))
    with pytest.raises(RingBufferOverflow):
        buffer.write(np.ones(2, dtype=np.float32), strict=True)


def test_ring_buffer_peek_does_not_consume():
    buffer = AudioRingBuffer(16)
    buffer.write(np.arange(10, dtype=np.float32))
    assert np.allclose(buffer.peek(4), np.arange(6, 10))
    assert len(buffer) == 10


def test_ring_buffer_read_timeout_returns_empty():
    buffer = AudioRingBuffer(16)
    assert buffer.read(4, timeout=0.01).size == 0


def test_int16_normalisation():
    raw = np.array([-32768, 0, 32767], dtype=np.int16)
    converted = to_float32(raw)
    assert converted.dtype == np.float32
    assert converted.min() >= -1.0 and converted.max() <= 1.0


def test_stereo_downmix():
    stereo = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
    assert np.allclose(to_mono(stereo, 2), [0.0, 0.0])


def test_resample_to_16k():
    source_rate, seconds = 44_100, 0.5
    tone = np.sin(
        2 * np.pi * 440 * np.arange(int(source_rate * seconds)) / source_rate
    ).astype(np.float32)
    resampled = resample_linear(tone, source_rate, 16_000)
    assert abs(resampled.size - 8_000) <= 1


def test_prepare_for_whisper_from_bytes():
    raw = (np.ones(2_000, dtype=np.int16) * 1000).tobytes()
    prepared = prepare_for_whisper(raw, 32_000, channels=2, dtype="int16")
    assert prepared.dtype == np.float32
    assert abs(prepared.size - 500) <= 1


def _speech(seconds, rate=16_000, amplitude=0.3):
    samples = int(rate * seconds)
    noise = np.random.default_rng(1).normal(0, amplitude, samples)
    return noise.astype(np.float32)


def test_segmenter_emits_on_silence():
    config = SegmenterConfig(min_segment_seconds=0.3, silence_hangover_seconds=0.2)
    segmenter = SpeechSegmenter(config)
    segments = list(segmenter.push(_speech(1.5)))
    segments += list(segmenter.push(np.zeros(16_000, dtype=np.float32)))
    assert segments
    assert segments[0].duration >= 0.3
    assert segments[0].reason == "silence"


def test_segmenter_caps_long_speech():
    config = SegmenterConfig(max_segment_seconds=1.0, min_segment_seconds=0.2)
    segmenter = SpeechSegmenter(config)
    segments = list(segmenter.push(_speech(3.0)))
    assert len(segments) >= 2
    assert all(segment.reason == "max_length" for segment in segments)


def test_segmenter_ignores_pure_silence():
    segmenter = SpeechSegmenter()
    assert list(segmenter.push(np.zeros(16_000, dtype=np.float32))) == []
    assert segmenter.flush() is None


def test_rms_of_silence_is_zero():
    assert rms(np.zeros(100, dtype=np.float32)) == 0.0
