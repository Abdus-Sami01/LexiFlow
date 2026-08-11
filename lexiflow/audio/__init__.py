"""Phase 1: real-time stream ingestion and signal processing."""

from .capture import (
    AudioBackendUnavailable,
    CaptureStats,
    MicrophoneStream,
    SegmentProducer,
    list_input_devices,
)
from .conversion import (
    WHISPER_SAMPLE_RATE,
    dbfs,
    peak,
    prepare_for_whisper,
    resample_linear,
    rms,
    to_float32,
    to_mono,
)
from .ring_buffer import AudioRingBuffer, RingBufferOverflow, RingBufferStats
from .segmenter import SpeechSegment, SpeechSegmenter

__all__ = [
    "AudioBackendUnavailable",
    "AudioRingBuffer",
    "CaptureStats",
    "MicrophoneStream",
    "RingBufferOverflow",
    "RingBufferStats",
    "SegmentProducer",
    "SpeechSegment",
    "SpeechSegmenter",
    "WHISPER_SAMPLE_RATE",
    "dbfs",
    "list_input_devices",
    "peak",
    "prepare_for_whisper",
    "resample_linear",
    "rms",
    "to_float32",
    "to_mono",
]
