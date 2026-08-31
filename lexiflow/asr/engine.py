"""Thread 2 of the pipeline: pull segments, run local Whisper, push text."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

from ..audio.segmenter import SpeechSegment
from ..audio.speaker import SpeakerTracker, attribute_words
from ..config import ASRConfig, DiarizationConfig, TranslationConfig
from ..observability import record_failure
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
    is_final: bool = True
    speaker: Optional[str] = None
    speaker_confidence: float = 0.0
    spans: List[dict] = field(default_factory=list)
    translation: Optional[str] = None
    translation_engine: Optional[str] = None
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
    partials_out: int = 0
    empty_results: int = 0
    errors: int = 0
    speech_translations: int = 0
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
        self,
        config: Optional[ASRConfig] = None,
        backend: Optional[WhisperBackend] = None,
        diarization: Optional[DiarizationConfig] = None,
        translation: Optional[TranslationConfig] = None,
    ) -> None:
        self.config = config or ASRConfig()
        self.backend = backend or create_backend(self.config)
        self.diarization = diarization or DiarizationConfig()
        self.translation = translation or TranslationConfig()
        self.speakers = (
            SpeakerTracker(
                similarity_threshold=self.diarization.similarity_threshold,
                max_speakers=self.diarization.max_speakers,
                min_seconds=self.diarization.min_seconds,
                adaptation_rate=self.diarization.adaptation_rate,
            )
            if self.diarization.enabled
            else None
        )
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
        if segment.is_final:
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

        if segment.is_final:
            self.stats.utterances_out += 1
        else:
            self.stats.partials_out += 1

        speaker, confidence = self._attribute(segment)
        spoken, engine = self._speech_translation(segment, result.language)
        return Utterance(
            translation=spoken,
            translation_engine=engine,
            spans=self._label_words(self._absolute_spans(result, segment), segment, speaker),
            text=text,
            started_at=segment.started_at,
            ended_at=segment.ended_at,
            index=segment.index,
            audio_seconds=result.audio_seconds,
            inference_seconds=result.inference_seconds or wall_clock,
            backend=result.backend,
            language=result.language,
            is_final=segment.is_final,
            speaker=speaker,
            speaker_confidence=confidence,
            metadata={"segment_reason": segment.reason, "peak_rms": segment.peak_rms},
        )

    def _speech_translation(
        self, segment: SpeechSegment, language: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Whisper translates straight from audio, which beats translating our own text."""
        if not (self.translation.enabled and self.translation.speech_translation):
            return None, None
        if self.translation.target_language != "en" or language == "en":
            return None, None
        if not segment.is_final or not self.backend.supports_translation:
            return None, None

        try:
            with self._lock:
                translated = self.backend.transcribe(
                    segment.audio, segment.sample_rate, task="translate"
                )
        except Exception as error:
            record_failure("asr.translate", error)
            self.stats.errors += 1
            return None, None

        text = (translated.text or "").strip()
        if not text:
            return None, None
        self.stats.speech_translations += 1
        return text, f"whisper:{self.backend.name}"

    @staticmethod
    def _absolute_spans(result: TranscriptionResult, segment: SpeechSegment) -> List[dict]:
        """Re-anchor the backend's relative timings onto the segment's wall clock."""
        origin = segment.started_at
        spans: List[dict] = []
        for part in result.segments or []:
            start = float(part.get("start") or 0.0)
            end = float(part.get("end") or start)
            words = [
                {
                    "start": origin + float(word.get("start") or 0.0),
                    "end": origin + float(word.get("end") or 0.0),
                    "text": word.get("text", ""),
                }
                for word in (part.get("words") or [])
                if word.get("text")
            ]
            spans.append(
                {
                    "start": origin + start,
                    "end": origin + end,
                    "text": (part.get("text") or "").strip(),
                    "words": words,
                }
            )
        return spans

    def _label_words(
        self, spans: List[dict], segment: SpeechSegment, fallback: Optional[str]
    ) -> List[dict]:
        """Per-word attribution, so one segment can carry two voices instead of an average."""
        if self.speakers is None or not self.diarization.word_level or not segment.is_final:
            return spans
        for span in spans:
            words = span.get("words") or []
            if not words:
                continue
            try:
                span["words"] = attribute_words(
                    words,
                    segment.audio,
                    segment.sample_rate,
                    self.speakers,
                    origin=segment.started_at,
                    window_seconds=self.diarization.word_window_seconds,
                    min_confidence=self.diarization.word_min_confidence,
                    fallback=fallback,
                )
            except Exception as error:
                record_failure("asr.word_speakers", error)
        return spans

    def _attribute(self, segment: SpeechSegment) -> tuple[Optional[str], float]:
        if self.speakers is None or not segment.is_final:
            return None, 0.0
        assignment = self.speakers.assign(
            segment.audio, segment.sample_rate, timestamp=segment.started_at
        )
        if assignment is None:
            return None, 0.0
        return assignment.label, assignment.confidence


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
        self.busy = False
        self._stop_event = threading.Event()

    def run(self) -> None:
        try:
            while not self._stop_event.is_set():
                segment = self.source.get()
                if segment is None:
                    break
                self.busy = True
                try:
                    utterance = self.engine.transcribe_segment(segment)
                except Exception as error:
                    record_failure("asr.transcribe", error)
                    self.engine.stats.errors += 1
                    continue
                finally:
                    self.busy = False
                if utterance is not None:
                    self.sink.put(utterance)
        except BaseException as exc:  # pragma: no cover - defensive
            self.error = exc
        finally:
            self.sink.put(None)

    def stop(self) -> None:
        self._stop_event.set()
        self.source.put(None)
