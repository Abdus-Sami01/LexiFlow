"""Energy based speech segmentation with an adaptive noise floor.

Whisper is happiest with utterance sized chunks. Feeding it fixed five second
windows chops words in half; feeding it a whole minute blows the latency
budget. This segmenter watches the short term RMS against a slowly adapting
noise floor and emits a segment when the speaker pauses, when the maximum
segment length is reached, or when the stream is flushed.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterator, List, Optional

import numpy as np

from ..config import SegmenterConfig
from .conversion import rms


@dataclass
class SpeechSegment:
    """One utterance sized slab of 16 kHz mono float32 audio."""

    audio: np.ndarray
    sample_rate: int
    started_at: float
    ended_at: float
    index: int
    peak_rms: float = 0.0
    reason: str = "silence"
    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.audio.size / float(self.sample_rate) if self.sample_rate else 0.0


class SpeechSegmenter:
    """Push frames in, pull :class:`SpeechSegment` objects out."""

    def __init__(self, config: Optional[SegmenterConfig] = None, sample_rate: int = 16_000) -> None:
        self.config = config or SegmenterConfig()
        self.sample_rate = sample_rate
        self._frame_size = max(1, int(sample_rate * self.config.frame_duration_ms / 1000))
        self._pre_roll_frames = max(
            1, int(self.config.pre_roll_seconds * 1000 / self.config.frame_duration_ms)
        )
        self._hangover_frames = max(
            1, int(self.config.silence_hangover_seconds * 1000 / self.config.frame_duration_ms)
        )
        self._pre_roll: Deque[np.ndarray] = deque(maxlen=self._pre_roll_frames)
        self._pending: List[np.ndarray] = []
        self._residual = np.zeros(0, dtype=np.float32)
        self._noise_floor = self.config.absolute_silence_rms
        self._speech_frames = 0
        self._silence_frames = 0
        self._in_speech = False
        self._segment_index = 0
        self._segment_started_at = 0.0
        self._segment_peak = 0.0

    @property
    def frame_size(self) -> int:
        return self._frame_size

    @property
    def noise_floor(self) -> float:
        return self._noise_floor

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def push(self, samples: np.ndarray) -> Iterator[SpeechSegment]:
        """Consume arbitrary length audio and yield completed segments."""
        block = np.asarray(samples, dtype=np.float32).reshape(-1)
        if block.size:
            self._residual = (
                np.concatenate((self._residual, block)) if self._residual.size else block
            )

        while self._residual.size >= self._frame_size:
            frame = self._residual[: self._frame_size]
            self._residual = self._residual[self._frame_size :]
            segment = self._consume_frame(frame)
            if segment is not None:
                yield segment

    def flush(self, reason: str = "flush") -> Optional[SpeechSegment]:
        """Emit whatever is buffered, used on shutdown or on a manual stop."""
        if self._in_speech and self._residual.size:
            self._pending.append(self._residual)
        self._residual = np.zeros(0, dtype=np.float32)
        if not self._in_speech or not self._pending:
            self._reset_segment_state()
            return None
        return self._emit(reason)

    def _consume_frame(self, frame: np.ndarray) -> Optional[SpeechSegment]:
        level = rms(frame)
        threshold = max(
            self.config.absolute_silence_rms,
            self._noise_floor * self.config.speech_trigger_ratio,
        )
        is_speech = level >= threshold

        if not is_speech:
            alpha = self.config.noise_floor_alpha
            self._noise_floor = alpha * self._noise_floor + (1.0 - alpha) * level

        if not self._in_speech:
            self._pre_roll.append(frame)
            if is_speech:
                self._speech_frames += 1
                if self._speech_frames >= self.config.speech_start_frames:
                    self._open_segment()
            else:
                self._speech_frames = 0
            return None

        self._pending.append(frame)
        self._segment_peak = max(self._segment_peak, level)

        if is_speech:
            self._silence_frames = 0
        else:
            self._silence_frames += 1

        if self._buffered_seconds() >= self.config.max_segment_seconds:
            return self._emit("max_length")

        if self._silence_frames >= self._hangover_frames:
            if self._buffered_seconds() >= self.config.min_segment_seconds:
                return self._emit("silence")
            self._discard_segment()
        return None

    def _open_segment(self) -> None:
        self._in_speech = True
        self._silence_frames = 0
        self._segment_peak = 0.0
        self._segment_started_at = time.time()
        self._pending = list(self._pre_roll)
        self._pre_roll.clear()

    def _buffered_seconds(self) -> float:
        frames = sum(chunk.size for chunk in self._pending)
        return frames / float(self.sample_rate)

    def _emit(self, reason: str) -> SpeechSegment:
        audio = np.concatenate(self._pending) if self._pending else np.zeros(0, dtype=np.float32)
        segment = SpeechSegment(
            audio=audio,
            sample_rate=self.sample_rate,
            started_at=self._segment_started_at or time.time(),
            ended_at=time.time(),
            index=self._segment_index,
            peak_rms=self._segment_peak,
            reason=reason,
            metadata={"noise_floor": self._noise_floor},
        )
        self._segment_index += 1
        self._reset_segment_state()
        return segment

    def _discard_segment(self) -> None:
        self._reset_segment_state()

    def _reset_segment_state(self) -> None:
        self._pending = []
        self._pre_roll.clear()
        self._in_speech = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._segment_peak = 0.0
