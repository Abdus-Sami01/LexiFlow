"""Microphone capture: raw bytes from the sound card straight into RAM."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, List, Optional

import numpy as np

from ..config import AudioConfig, DiarizationConfig, SegmenterConfig
from .conversion import prepare_for_whisper, rms
from .ring_buffer import AudioRingBuffer
from .segmenter import SpeechSegment, SpeechSegmenter, split_on_speaker_change


class AudioBackendUnavailable(RuntimeError):
    """Raised when no supported capture library is importable."""


@dataclass
class CaptureStats:
    blocks: int = 0
    frames: int = 0
    overflows: int = 0
    last_rms: float = 0.0
    started_at: float = 0.0

    @property
    def seconds_captured(self) -> float:
        return self.frames / 16_000.0


def _import_sounddevice():
    try:
        import sounddevice  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on host audio stack
        raise AudioBackendUnavailable(
            "sounddevice is required for live capture: pip install 'lexiflow[audio]'"
        ) from exc
    return sounddevice


def list_input_devices() -> List[dict]:
    """Enumerate every device that can record, for the CLI and the dashboard."""
    sd = _import_sounddevice()
    devices = []
    for index, info in enumerate(sd.query_devices()):
        if int(info.get("max_input_channels", 0)) <= 0:
            continue
        devices.append(
            {
                "index": index,
                "name": info.get("name", f"device-{index}"),
                "channels": int(info["max_input_channels"]),
                "default_samplerate": int(info.get("default_samplerate", 0) or 0),
                "hostapi": info.get("hostapi"),
            }
        )
    return devices


class MicrophoneStream:
    """Thread-safe producer that fills an :class:`AudioRingBuffer`."""

    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        on_level: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.config = config or AudioConfig()
        self.stats = CaptureStats()
        self.buffer = AudioRingBuffer(self.config.ring_buffer_frames)
        self._on_level = on_level
        self._stream: Any = None
        self._running = threading.Event()
        self._source_rate = self.config.capture_sample_rate or self.config.target_sample_rate
        self._source_channels = self.config.capture_channels or self.config.target_channels

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    @property
    def source_sample_rate(self) -> int:
        return self._source_rate

    def start(self) -> "MicrophoneStream":
        if self._running.is_set():
            return self
        sd = _import_sounddevice()
        device_info = sd.query_devices(self.config.device, "input")
        self._source_rate = self.config.capture_sample_rate or int(
            device_info.get("default_samplerate") or self.config.target_sample_rate
        )
        self._source_channels = self.config.capture_channels or min(
            int(device_info.get("max_input_channels", 1)) or 1, self.config.target_channels
        )
        block = max(1, int(self._source_rate * self.config.block_duration_ms / 1000))

        self._stream = sd.InputStream(
            samplerate=self._source_rate,
            channels=self._source_channels,
            dtype=self.config.dtype,
            blocksize=block,
            device=self.config.device,
            callback=self._callback,
        )
        self._stream.start()
        self._running.set()
        self.stats.started_at = time.time()
        return self

    def start_virtual(self) -> "MicrophoneStream":
        """Open the stream in fed-audio mode: no device, same downstream contract."""
        self._running.set()
        self.stats.started_at = time.time()
        return self

    def stop(self) -> None:
        self._running.clear()
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        self.buffer.close()

    def feed(self, samples: np.ndarray, source_rate: Optional[int] = None) -> int:
        """Inject audio directly; used by file replay and by the test suite."""
        converted = prepare_for_whisper(
            samples,
            source_rate or self.config.target_sample_rate,
            channels=1,
            dtype=self.config.dtype,
            target_rate=self.config.target_sample_rate,
        )
        written = self.buffer.write(converted)
        self.stats.blocks += 1
        self.stats.frames += written
        self.stats.last_rms = rms(converted)
        if self._on_level is not None:
            self._on_level(self.stats.last_rms)
        return written

    def _callback(self, indata, frames, time_info, status) -> None:  # pragma: no cover - realtime
        if status:
            self.stats.overflows += 1
        converted = prepare_for_whisper(
            indata,
            self._source_rate,
            channels=self._source_channels,
            dtype=self.config.dtype,
            target_rate=self.config.target_sample_rate,
        )
        written = self.buffer.write(converted)
        self.stats.blocks += 1
        self.stats.frames += written
        self.stats.last_rms = rms(converted)
        if self._on_level is not None:
            try:
                self._on_level(self.stats.last_rms)
            except Exception:
                pass

    def read_blocks(self, timeout: float = 0.5) -> Iterator[np.ndarray]:
        """Yield contiguous audio blocks until the stream is stopped."""
        block_size = self.config.block_size
        while self._running.is_set() or len(self.buffer):
            chunk = self.buffer.read(block_size, timeout=timeout)
            if chunk.size:
                yield chunk
            elif self.buffer.is_closed:
                break

    def segments(
        self, segmenter_config: Optional[SegmenterConfig] = None, timeout: float = 0.5
    ) -> Iterator[SpeechSegment]:
        """The headline generator: microphone in, utterances out."""
        segmenter = SpeechSegmenter(segmenter_config, self.config.target_sample_rate)
        for chunk in self.read_blocks(timeout=timeout):
            for segment in segmenter.push(chunk):
                yield segment
        tail = segmenter.flush()
        if tail is not None:
            yield tail

    def __enter__(self) -> "MicrophoneStream":
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.stop()


class SegmentProducer(threading.Thread):
    """Thread 1 of the producer/consumer design: audio in, segments queued."""

    def __init__(
        self,
        stream: MicrophoneStream,
        output: "queue.Queue[Optional[SpeechSegment]]",
        segmenter_config: Optional[SegmenterConfig] = None,
        name: str = "lexiflow-producer",
        diarization: Optional[DiarizationConfig] = None,
        partial_gate: Optional[Callable[[], bool]] = None,
    ) -> None:
        super().__init__(name=name, daemon=True)
        self.stream = stream
        self.output = output
        self.segmenter_config = segmenter_config
        self.diarization = diarization
        self.partial_gate = partial_gate
        self.dropped_partials = 0
        self.speaker_splits = 0
        self.error: Optional[BaseException] = None
        self._stop_event = threading.Event()

    def run(self) -> None:
        try:
            for segment in self.stream.segments(self.segmenter_config):
                if self._stop_event.is_set():
                    break
                if segment.is_final:
                    for part in self._maybe_split(segment):
                        self.output.put(part)
                    continue
                if self.partial_gate is not None and not self.partial_gate():
                    self.dropped_partials += 1
                    continue
                try:
                    if self.output.qsize() == 0:
                        self.output.put_nowait(segment)
                    else:
                        self.dropped_partials += 1
                except queue.Full:
                    self.dropped_partials += 1
        except BaseException as exc:  # pragma: no cover - defensive
            self.error = exc
        finally:
            self.output.put(None)

    def _maybe_split(self, segment: SpeechSegment) -> List[SpeechSegment]:
        if self.diarization is None or not self.diarization.split_on_change:
            return [segment]
        parts = split_on_speaker_change(
            segment,
            threshold=self.diarization.change_threshold,
            window_seconds=self.diarization.change_window_seconds,
        )
        if len(parts) > 1:
            self.speaker_splits += 1
        return parts

    def stop(self) -> None:
        self._stop_event.set()
        self.stream.stop()
