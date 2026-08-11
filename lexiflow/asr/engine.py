"""Thread 2 of the pipeline: pull segments, run local Whisper, push text."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from ..audio.segmenter import SpeechSegment
from ..config import ASRConfig
from .backends import TranscriptionResult, WhisperBackend, create_backend


@dataclass
class Utterance:
    """A finished piece of transcript, ready for the analytics consumer."""

    text: str
    started_at: float
    ended_at: float
    index: int
    audio_seconds: float
    inference_seconds: float
    backend: str
    language: str = "en"
    metadata: dict = field(default_factory=dict)

    @property
    def realtime_factor(self) -> float:
        if self.audio_seconds <= 0.0:
            return 0.0
        return self.inference_seconds / self.audio_seconds


@dataclass
class EngineStats:
    segments_in: int = 0
    utterances_out: int = 0
    empty_results: int = 0
    errors: int = 0
    audio_seconds: float = 0.0
    inference_seconds: float = 0.0

    @property
    def realtime_factor(self) -> float:
        if self.audio_seconds <= 0.0:
            return 0.0
        return self.inference_seconds / self.audio_seconds


class TranscriptionEngine:
    """Thin, reusable wrapper so the backend can be swapped at runtime."""

    def __init__(
        self, config: Optional[ASRConfig] = None, backend: Optional[WhisperBackend] = None
    ) -> None:
        self.config = config or ASRConfig()
        self.backend = backend or create_backend(self.config)
        self.stats = EngineStats()
        self._lock = threading.Lock()

    def ensure_loaded(self) -> None:
        with self._lock:
            if not self.backend.is_loaded:
                self.backend.load()
                if self.config.warmup:
                    self.backend.warmup()

    def transcribe_segment(self, segment: SpeechSegment) -> Optional[Utterance]:
        self.ensure_loaded()
        self.stats.segments_in += 1
        started = time.perf_counter()
        with self._lock:
            result: TranscriptionResult = self.backend.transcribe(
                segment.audio, segment.sample_rate
            )
        wall_clock = time.perf_counter() - started
        self.stats.audio_seconds += result.audio_seconds
        self.stats.inference_seconds += result.inference_seconds or wall_clock

        text = (result.text or "").strip()
        if not text:
            self.stats.empty_results += 1
            return None

        self.stats.utterances_out += 1
        return Utterance(
            text=text,
            started_at=segment.started_at,
            ended_at=segment.ended_at,
            index=segment.index,
            audio_seconds=result.audio_seconds,
            inference_seconds=result.inference_seconds or wall_clock,
            backend=result.backend,
            language=result.language,
            metadata={"segment_reason": segment.reason, "peak_rms": segment.peak_rms},
        )


class TranscriptionConsumer(threading.Thread):
    """Reads the segment queue, writes the utterance queue, never blocks the mic."""

    def __init__(
        self,
        engine: TranscriptionEngine,
        source: "queue.Queue[Optional[SpeechSegment]]",
        sink: "queue.Queue[Optional[Utterance]]",
        name: str = "lexiflow-asr",
    ) -> None:
        super().__init__(name=name, daemon=True)
        self.engine = engine
        self.source = source
        self.sink = sink
        self.error: Optional[BaseException] = None
        self._stop_event = threading.Event()

    def run(self) -> None:
        try:
            while not self._stop_event.is_set():
                segment = self.source.get()
                if segment is None:
                    break
                try:
                    utterance = self.engine.transcribe_segment(segment)
                except Exception:
                    self.engine.stats.errors += 1
                    continue
                if utterance is not None:
                    self.sink.put(utterance)
        except BaseException as exc:  # pragma: no cover - defensive
            self.error = exc
        finally:
            self.sink.put(None)

    def stop(self) -> None:
        self._stop_event.set()
        self.source.put(None)
