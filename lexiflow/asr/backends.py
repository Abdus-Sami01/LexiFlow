"""Native Whisper bindings that take a pointer to our audio, not a file path."""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np

from ..config import ASRConfig
from .hardware import detect_hardware

_REGISTRY: Dict[str, Type["WhisperBackend"]] = {}


class BackendUnavailable(RuntimeError):
    """Raised when a backend's native library or model file is missing."""


@dataclass
class TranscriptionResult:
    text: str
    language: str = "en"
    segments: List[Dict[str, Any]] = field(default_factory=list)
    inference_seconds: float = 0.0
    audio_seconds: float = 0.0
    backend: str = "unknown"

    @property
    def realtime_factor(self) -> float:
        if self.audio_seconds <= 0.0:
            return 0.0
        return self.inference_seconds / self.audio_seconds


def register_backend(name: str) -> Callable[[Type["WhisperBackend"]], Type["WhisperBackend"]]:
    def decorator(klass: Type["WhisperBackend"]) -> Type["WhisperBackend"]:
        klass.name = name
        _REGISTRY[name] = klass
        return klass

    return decorator


class WhisperBackend:
    """Common surface every local Whisper implementation exposes."""

    name = "base"
    priority = 100

    def __init__(self, config: Optional[ASRConfig] = None) -> None:
        self.config = config or ASRConfig()
        self._loaded = False

    @classmethod
    def is_available(cls) -> bool:
        raise NotImplementedError

    def load(self) -> "WhisperBackend":
        raise NotImplementedError

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> TranscriptionResult:
        raise NotImplementedError

    def unload(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def warmup(self, seconds: float = 0.5) -> None:
        silence = np.zeros(int(16_000 * seconds), dtype=np.float32)
        try:
            self.transcribe(silence)
        except Exception:
            pass

    @staticmethod
    def _as_float32(audio: np.ndarray) -> np.ndarray:
        array = np.asarray(audio, dtype=np.float32).reshape(-1)
        return np.ascontiguousarray(array)


def _try_import(module: str):
    try:
        return importlib.import_module(module)
    except Exception:
        return None


@register_backend("pywhispercpp")
class PyWhisperCppBackend(WhisperBackend):
    """whisper.cpp through pywhispercpp; the array is passed straight to C++."""

    priority = 10

    @classmethod
    def is_available(cls) -> bool:
        return _try_import("pywhispercpp.model") is not None

    def load(self) -> "PyWhisperCppBackend":
        module = _try_import("pywhispercpp.model")
        if module is None:
            raise BackendUnavailable("pywhispercpp is not installed")
        options = {
            "n_threads": self.config.resolved_threads(),
            "language": self.config.language,
            "translate": self.config.translate,
            "print_progress": False,
            "print_realtime": False,
            "single_segment": self.config.single_segment,
            "no_context": self.config.no_context,
        }
        self._model = module.Model(self.config.model_path or self.config.model_name, **options)
        self._loaded = True
        return self

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> TranscriptionResult:
        if not self._loaded:
            self.load()
        array = self._as_float32(audio)
        started = time.perf_counter()
        raw_segments = self._model.transcribe(array)
        elapsed = time.perf_counter() - started
        segments = [
            {
                "start": getattr(item, "t0", 0) / 100.0,
                "end": getattr(item, "t1", 0) / 100.0,
                "text": getattr(item, "text", "").strip(),
            }
            for item in raw_segments
        ]
        return TranscriptionResult(
            text=" ".join(part["text"] for part in segments).strip(),
            language=self.config.language,
            segments=segments,
            inference_seconds=elapsed,
            audio_seconds=array.size / float(sample_rate),
            backend=self.name,
        )


@register_backend("whisper_cpp_python")
class WhisperCppPythonBackend(WhisperBackend):
    """whisper-cpp-python ctypes bindings against a locally compiled libwhisper."""

    priority = 20

    @classmethod
    def is_available(cls) -> bool:
        return _try_import("whisper_cpp_python") is not None

    def load(self) -> "WhisperCppPythonBackend":
        module = _try_import("whisper_cpp_python")
        if module is None:
            raise BackendUnavailable("whisper-cpp-python is not installed")
        if not self.config.model_path:
            raise BackendUnavailable("whisper_cpp_python requires an explicit ggml model_path")
        self._model = module.Whisper(
            model_path=self.config.model_path, n_threads=self.config.resolved_threads()
        )
        self._loaded = True
        return self

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> TranscriptionResult:
        if not self._loaded:
            self.load()
        array = self._as_float32(audio)
        started = time.perf_counter()
        raw = self._model.transcribe(array, language=self.config.language)
        elapsed = time.perf_counter() - started
        if isinstance(raw, dict):
            text = str(raw.get("text", "")).strip()
            segments = list(raw.get("segments", []) or [])
        else:
            text, segments = str(raw).strip(), []
        return TranscriptionResult(
            text=text,
            language=self.config.language,
            segments=segments,
            inference_seconds=elapsed,
            audio_seconds=array.size / float(sample_rate),
            backend=self.name,
        )


@register_backend("faster_whisper")
class FasterWhisperBackend(WhisperBackend):
    """CTranslate2 build; a strong fallback when whisper.cpp is not compiled."""

    priority = 30

    @classmethod
    def is_available(cls) -> bool:
        return _try_import("faster_whisper") is not None

    def load(self) -> "FasterWhisperBackend":
        module = _try_import("faster_whisper")
        if module is None:
            raise BackendUnavailable("faster-whisper is not installed")
        self._model = module.WhisperModel(
            self.config.model_path or self.config.model_name,
            device="cpu",
            compute_type="int8",
            cpu_threads=self.config.resolved_threads(),
        )
        self._loaded = True
        return self

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> TranscriptionResult:
        if not self._loaded:
            self.load()
        array = self._as_float32(audio)
        started = time.perf_counter()
        raw_segments, info = self._model.transcribe(
            array,
            language=None if self.config.language == "auto" else self.config.language,
            beam_size=self.config.beam_size,
            vad_filter=False,
        )
        segments = [
            {"start": item.start, "end": item.end, "text": item.text.strip()}
            for item in raw_segments
        ]
        elapsed = time.perf_counter() - started
        return TranscriptionResult(
            text=" ".join(part["text"] for part in segments).strip(),
            language=getattr(info, "language", self.config.language),
            segments=segments,
            inference_seconds=elapsed,
            audio_seconds=array.size / float(sample_rate),
            backend=self.name,
        )


@register_backend("null")
class NullBackend(WhisperBackend):
    """Keeps the pipeline runnable on a machine with no model installed."""

    priority = 900

    @classmethod
    def is_available(cls) -> bool:
        return True

    def load(self) -> "NullBackend":
        self._loaded = True
        return self

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> TranscriptionResult:
        array = self._as_float32(audio)
        return TranscriptionResult(
            text="",
            language=self.config.language,
            inference_seconds=0.0,
            audio_seconds=array.size / float(sample_rate),
            backend=self.name,
        )


class ScriptedBackend(WhisperBackend):
    """Deterministic backend used by the test suite and the offline demo."""

    name = "scripted"
    priority = 1_000

    def __init__(self, lines: Optional[List[str]] = None, config: Optional[ASRConfig] = None):
        super().__init__(config)
        self._lines = list(lines or [])
        self._cursor = 0

    @classmethod
    def is_available(cls) -> bool:
        return True

    def load(self) -> "ScriptedBackend":
        self._loaded = True
        return self

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> TranscriptionResult:
        array = self._as_float32(audio)
        text = self._lines[self._cursor % len(self._lines)] if self._lines else ""
        self._cursor += 1
        return TranscriptionResult(
            text=text,
            language=self.config.language,
            inference_seconds=0.0,
            audio_seconds=array.size / float(sample_rate),
            backend=self.name,
        )


def available_backends() -> List[str]:
    return [
        name
        for name, klass in sorted(_REGISTRY.items(), key=lambda item: item[1].priority)
        if klass.is_available()
    ]


def create_backend(config: Optional[ASRConfig] = None) -> WhisperBackend:
    """Pick the fastest importable backend, honouring an explicit override."""
    config = config or ASRConfig()
    if config.backend and config.backend != "auto":
        klass = _REGISTRY.get(config.backend)
        if klass is None:
            raise BackendUnavailable(f"unknown backend '{config.backend}'")
        if not klass.is_available():
            raise BackendUnavailable(f"backend '{config.backend}' is not importable")
        return klass(config)

    for _, klass in sorted(_REGISTRY.items(), key=lambda item: item[1].priority):
        if klass.is_available():
            return klass(config)
    raise BackendUnavailable("no Whisper backend is available")


def backend_report() -> Dict[str, Any]:
    profile = detect_hardware()
    return {
        "hardware": {
            "system": profile.system,
            "machine": profile.machine,
            "physical_cores": profile.physical_cores,
            "apple_silicon": profile.is_apple_silicon,
            "avx2": profile.has("avx2"),
            "avx512": profile.has("avx512f"),
        },
        "available": available_backends(),
    }
