"""Thread 3: turn transcript into insight and commit it to shared state."""

from __future__ import annotations

import queue
import threading
from typing import Callable, Optional

from ..asr.engine import Utterance
from ..nlp.pipeline import AnalyticsEngine
from ..observability import record_failure
from .store import SessionStore


class AnalyticsConsumer(threading.Thread):
    """Drains the utterance queue so the ASR thread is never held up."""

    def __init__(
        self,
        engine: AnalyticsEngine,
        store: SessionStore,
        source: "queue.Queue[Optional[Utterance]]",
        name: str = "lexiflow-analytics",
        scrub: Optional[Callable[[str], str]] = None,
    ) -> None:
        super().__init__(name=name, daemon=True)
        self.engine = engine
        self.store = store
        self.source = source
        self.scrub = scrub or (lambda text: text)
        self.error: Optional[BaseException] = None
        self.busy = False
        self._stop_event = threading.Event()

    def run(self) -> None:
        try:
            while not self._stop_event.is_set():
                utterance = self.source.get()
                if utterance is None:
                    break
                self.busy = True
                try:
                    self._handle(utterance)
                finally:
                    self.busy = False
        except BaseException as exc:  # pragma: no cover - defensive
            self.error = exc

    def _handle(self, utterance: Utterance) -> None:
        if not utterance.is_final:
            self.store.set_partial(utterance.text, utterance.speaker)
            return
        try:
            insight = self.engine.analyse(utterance.text, utterance.translation)
        except Exception as error:
            record_failure("analytics", error)
            insight = None
        self.store.record(
            self.scrub(utterance.text),
            insight,
            started_at=utterance.started_at,
            ended_at=utterance.ended_at,
            backend=utterance.backend,
            audio_seconds=utterance.audio_seconds,
            inference_seconds=utterance.inference_seconds,
            speaker=utterance.speaker,
            speaker_confidence=utterance.speaker_confidence,
            spans=utterance.spans,
            translation=utterance.translation,
        )
        self.store.update_metrics(
            asr_realtime_factor=round(utterance.realtime_factor, 3),
            analytics_average_ms=round(self.engine.stats.average_ms, 3),
            analytics_backends=self.engine.backends,
        )

    def stop(self) -> None:
        self._stop_event.set()
        self.source.put(None)
