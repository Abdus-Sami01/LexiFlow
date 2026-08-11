"""Phase 2: CPU inference acceleration through native Whisper bindings."""

from .backends import (
    BackendUnavailable,
    NullBackend,
    ScriptedBackend,
    TranscriptionResult,
    WhisperBackend,
    available_backends,
    backend_report,
    create_backend,
)
from .engine import EngineStats, TranscriptionConsumer, TranscriptionEngine, Utterance
from .hardware import HardwareProfile, build_command, compiler_flags, describe, detect_hardware

__all__ = [
    "BackendUnavailable",
    "EngineStats",
    "HardwareProfile",
    "NullBackend",
    "ScriptedBackend",
    "TranscriptionConsumer",
    "TranscriptionEngine",
    "TranscriptionResult",
    "Utterance",
    "WhisperBackend",
    "available_backends",
    "backend_report",
    "build_command",
    "compiler_flags",
    "create_backend",
    "describe",
    "detect_hardware",
]
