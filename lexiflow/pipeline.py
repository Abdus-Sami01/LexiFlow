"""The producer/consumer orchestrator that holds all five phases together."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .asr.backends import WhisperBackend
from .asr.engine import TranscriptionConsumer, TranscriptionEngine, Utterance
from .audio.capture import MicrophoneStream, SegmentProducer
from .audio.segmenter import SpeechSegment
from .config import LexiFlowConfig
from .nlp.pipeline import AnalyticsEngine
from .observability import FAILURES, record_failure
from .state.consumer import AnalyticsConsumer
from .state.store import SessionStore


@dataclass
class PipelineHealth:
    running: bool
    segment_queue: int
    utterance_queue: int
    captured_seconds: float
    segments_in: int
    utterances_out: int
    asr_realtime_factor: float
    analytics_average_ms: float
    partials_out: int
    dropped_partials: int
    speaker_splits: int
    speakers: int
    speech_translations: int
    keeping_up: bool
    errors: List[str]
    failures: int = 0
    failures_by_component: Dict[str, int] = field(default_factory=dict)


class LexiFlowPipeline:
    """Start, observe and stop the whole local engine through one object."""

    def __init__(
        self,
        config: Optional[LexiFlowConfig] = None,
        store: Optional[SessionStore] = None,
        backend: Optional[WhisperBackend] = None,
    ) -> None:
        self.config = config or LexiFlowConfig()
        self.store = store or SessionStore(self.config.state)
        self.analytics = AnalyticsEngine(self.config.nlp, self.config.translation)
        self.transcription = TranscriptionEngine(
            self.config.asr, backend, self.config.diarization, self.config.translation
        )
        self.stream = MicrophoneStream(self.config.audio, on_level=self._on_level)

        self._segment_queue: "queue.Queue[Optional[SpeechSegment]]" = queue.Queue(
            maxsize=self.config.asr.max_queue_size
        )
        self._utterance_queue: "queue.Queue[Optional[Utterance]]" = queue.Queue(
            maxsize=self.config.nlp.max_queue_size
        )

        self._producer: Optional[SegmentProducer] = None
        self._asr_consumer: Optional[TranscriptionConsumer] = None
        self._analytics_consumer: Optional[AnalyticsConsumer] = None
        self._running = threading.Event()
        self._level = 0.0
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    @property
    def input_level(self) -> float:
        with self._lock:
            return self._level

    def _on_level(self, level: float) -> None:
        with self._lock:
            self._level = level

    def start(self, open_microphone: bool = True) -> "LexiFlowPipeline":
        if self._running.is_set():
            return self

        self.transcription.ensure_loaded()
        self._load_speaker_profiles()
        if open_microphone:
            self.stream.start()
        else:
            self.stream.start_virtual()

        self._producer = SegmentProducer(
            self.stream,
            self._segment_queue,
            self.config.segmenter,
            diarization=self.config.diarization,
            partial_gate=self._partial_gate,
        )
        self._asr_consumer = TranscriptionConsumer(
            self.transcription, self._segment_queue, self._utterance_queue
        )
        self._analytics_consumer = AnalyticsConsumer(
            self.analytics, self.store, self._utterance_queue
        )

        for worker in (self._producer, self._asr_consumer, self._analytics_consumer):
            worker.start()

        self._running.set()
        self.store.update_metrics(
            started_at=time.time(),
            asr_backend=self.transcription.backend.name,
            analytics_backends=self.analytics.backends,
        )
        return self

    def _load_speaker_profiles(self) -> int:
        path = self.config.diarization.profile_path
        if path is None or self.transcription.speakers is None:
            return 0
        return self.transcription.speakers.load(path)

    def _save_speaker_profiles(self) -> None:
        path = self.config.diarization.profile_path
        if path is None or self.transcription.speakers is None:
            return
        try:
            self.transcription.speakers.save(path)
        except OSError as error:
            record_failure("pipeline.speaker_profiles", error, path=str(path))

    def rename_speaker(self, label: str, name: str) -> bool:
        """Enrol a cluster under a real name and keep it for the next session."""
        tracker = self.transcription.speakers
        if tracker is None or not tracker.rename(label, name):
            return False
        self.store.rename_speaker(label, name)
        self._save_speaker_profiles()
        return True

    def _partial_gate(self) -> bool:
        """Stop paying for partials as soon as inference stops keeping up."""
        if not self.config.segmenter.emit_partials:
            return False
        stats = self.transcription.stats
        if stats.audio_seconds <= 0.0:
            return True
        return stats.realtime_factor <= self.config.asr.max_realtime_factor

    def feed(self, samples: np.ndarray, source_rate: Optional[int] = None) -> int:
        """Push audio in from a file, a socket or a test, bypassing the mic."""
        return self.stream.feed(samples, source_rate)

    def submit_text(self, text: str) -> None:
        """Inject a transcript line directly, skipping capture and inference."""
        insight = self.analytics.analyse(text)
        self.store.record(text, insight)

    def stop(self, timeout: float = 5.0) -> None:
        if not self._running.is_set():
            return
        self._running.clear()

        if self._producer is not None:
            self._producer.stop()
            self._producer.join(timeout=timeout)
        if self._asr_consumer is not None:
            self._asr_consumer.join(timeout=timeout)
        if self._analytics_consumer is not None:
            self._analytics_consumer.join(timeout=timeout)

        self._save_speaker_profiles()
        self.store.update_metrics(stopped_at=time.time())

    def close(self) -> None:
        self.stop()
        self.store.close()

    def drain(self, timeout: float = 30.0, settle_checks: int = 3) -> bool:
        """Block until every queued segment has been transcribed and analysed."""
        deadline = time.time() + timeout
        settled = 0
        while time.time() < deadline:
            if self._is_idle():
                settled += 1
                if settled >= settle_checks:
                    return True
            else:
                settled = 0
            time.sleep(0.05)
        return self._is_idle()

    def _is_idle(self) -> bool:
        if not self._segment_queue.empty() or not self._utterance_queue.empty():
            return False
        if self._asr_consumer is not None and self._asr_consumer.busy:
            return False
        if self._analytics_consumer is not None and self._analytics_consumer.busy:
            return False
        return True

    def health(self) -> PipelineHealth:
        errors: List[str] = []
        for worker in (self._producer, self._asr_consumer, self._analytics_consumer):
            if worker is not None and getattr(worker, "error", None) is not None:
                errors.append(f"{worker.name}: {worker.error}")
        return PipelineHealth(
            running=self.is_running,
            segment_queue=self._segment_queue.qsize(),
            utterance_queue=self._utterance_queue.qsize(),
            captured_seconds=round(self.stream.stats.frames / 16_000.0, 2),
            segments_in=self.transcription.stats.segments_in,
            utterances_out=self.transcription.stats.utterances_out,
            asr_realtime_factor=round(self.transcription.stats.realtime_factor, 3),
            analytics_average_ms=round(self.analytics.stats.average_ms, 3),
            partials_out=self.transcription.stats.partials_out,
            dropped_partials=self._producer.dropped_partials if self._producer else 0,
            speaker_splits=self._producer.speaker_splits if self._producer else 0,
            keeping_up=self.transcription.stats.realtime_factor
            <= self.config.asr.max_realtime_factor,
            speakers=self.transcription.speakers.speaker_count
            if self.transcription.speakers
            else 0,
            speech_translations=self.transcription.stats.speech_translations,
            errors=errors,
            failures=FAILURES.total,
            failures_by_component=FAILURES.counts(),
        )

    def snapshot(self, transcript_limit: int = 200) -> Dict[str, Any]:
        """Everything the dashboard needs in a single lock-free-for-the-caller read."""
        health = self.health()
        return {
            "health": health.__dict__,
            "metrics": self.store.metrics(),
            "input_level": self.input_level,
            "transcript": [item.as_dict() for item in self.store.transcript(transcript_limit)],
            "actions": [item.as_dict() for item in self.store.actions()],
            "entities": self.store.entity_counts(),
            "sentiment": self.store.sentiment_timeline(),
            "speakers": self.store.speakers(),
            "topics": self.store.topics(),
            "partial": self.store.partial(),
            "failures": FAILURES.summary(),
            "translation": self.analytics.translator.stats()
            if self.analytics.translator
            else None,
        }

    def digest(self, limit: Optional[int] = None):
        """Extractive summary, keyphrases and topic shifts for the session so far."""
        return self.store.digest(self.analytics, limit)

    def subscribe(self, listener: Callable[[str, Any], None]) -> Callable[[], None]:
        return self.store.subscribe(listener)

    def __enter__(self) -> "LexiFlowPipeline":
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.close()
