"""Speaker turn attribution from MFCC centroids, computed with numpy alone."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

PRE_EMPHASIS = 0.97
FRAME_LENGTH_MS = 25
FRAME_STRIDE_MS = 10
FFT_SIZE = 512
MEL_FILTERS = 26
CEPSTRAL_COEFFICIENTS = 13
LIFTER = 22


def hz_to_mel(hertz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hertz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


@lru_cache(maxsize=8)
def mel_filterbank(sample_rate: int, filters: int = MEL_FILTERS, fft_size: int = FFT_SIZE):
    low_mel, high_mel = hz_to_mel(0.0), hz_to_mel(sample_rate / 2.0)
    points = mel_to_hz(np.linspace(low_mel, high_mel, filters + 2))
    bins = np.floor((fft_size + 1) * points / sample_rate).astype(int)
    bank = np.zeros((filters, fft_size // 2 + 1), dtype=np.float64)
    for index in range(1, filters + 1):
        left, center, right = bins[index - 1], bins[index], bins[index + 1]
        if center == left:
            center = left + 1
        if right <= center:
            right = center + 1
        right = min(right, bank.shape[1] - 1)
        center = min(center, right - 1) if right - 1 > left else center
        for position in range(left, min(center, bank.shape[1])):
            bank[index - 1, position] = (position - left) / max(1, center - left)
        for position in range(center, min(right, bank.shape[1])):
            bank[index - 1, position] = (right - position) / max(1, right - center)
    return bank


@lru_cache(maxsize=4)
def dct_matrix(coefficients: int, filters: int):
    grid = np.arange(filters, dtype=np.float64)
    matrix = np.array(
        [np.cos(np.pi * order / filters * (grid + 0.5)) for order in range(coefficients)]
    )
    matrix *= np.sqrt(2.0 / filters)
    return matrix


def mfcc(audio: np.ndarray, sample_rate: int = 16_000) -> np.ndarray:
    """Standard MFCC frontend: pre-emphasis, mel filterbank, log, DCT, liftering."""
    signal = np.asarray(audio, dtype=np.float64).reshape(-1)
    if signal.size < FFT_SIZE:
        return np.zeros((0, CEPSTRAL_COEFFICIENTS), dtype=np.float64)

    emphasised = np.append(signal[0], signal[1:] - PRE_EMPHASIS * signal[:-1])
    frame_length = int(round(sample_rate * FRAME_LENGTH_MS / 1000))
    frame_stride = int(round(sample_rate * FRAME_STRIDE_MS / 1000))
    frame_count = 1 + max(0, (emphasised.size - frame_length) // frame_stride)
    if frame_count <= 0:
        return np.zeros((0, CEPSTRAL_COEFFICIENTS), dtype=np.float64)

    indices = (
        np.tile(np.arange(frame_length), (frame_count, 1))
        + np.arange(0, frame_count * frame_stride, frame_stride)[:, None]
    )
    frames = emphasised[indices] * np.hamming(frame_length)
    power = np.square(np.abs(np.fft.rfft(frames, FFT_SIZE))) / FFT_SIZE

    energies = np.dot(power, mel_filterbank(sample_rate).T)
    energies = np.where(energies <= 1e-10, 1e-10, energies)
    cepstra = np.dot(np.log(energies), dct_matrix(CEPSTRAL_COEFFICIENTS, MEL_FILTERS).T)

    order = np.arange(CEPSTRAL_COEFFICIENTS)
    cepstra *= 1.0 + (LIFTER / 2.0) * np.sin(np.pi * order / LIFTER)
    return cepstra


def voice_embedding(audio: np.ndarray, sample_rate: int = 16_000) -> Optional[np.ndarray]:
    """Mean and standard deviation of the cepstra, mean-normalised then unit length."""
    cepstra = mfcc(audio, sample_rate)
    if cepstra.shape[0] < 3:
        return None
    normalised = cepstra - cepstra.mean(axis=0, keepdims=True)
    embedding = np.concatenate((cepstra.mean(axis=0), normalised.std(axis=0)))
    norm = np.linalg.norm(embedding)
    if norm <= 0.0 or not np.isfinite(norm):
        return None
    return (embedding / norm).astype(np.float32)


@dataclass
class SpeakerProfile:
    label: str
    centroid: np.ndarray
    segments: int = 1
    total_seconds: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0

    def as_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "segments": self.segments,
            "total_seconds": round(self.total_seconds, 2),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }


@dataclass
class SpeakerAssignment:
    label: str
    similarity: float
    is_new: bool
    confidence: float = 0.0
    metadata: Dict[str, float] = field(default_factory=dict)


class SpeakerTracker:
    """Online agglomerative clustering: no pretrained model, no training step."""

    def __init__(
        self,
        similarity_threshold: float = 0.72,
        max_speakers: int = 8,
        min_seconds: float = 0.6,
        adaptation_rate: float = 0.25,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_speakers = max_speakers
        self.min_seconds = min_seconds
        self.adaptation_rate = adaptation_rate
        self._profiles: Dict[str, SpeakerProfile] = {}
        self._lock = threading.RLock()
        self._counter = 0

    @property
    def speaker_count(self) -> int:
        with self._lock:
            return len(self._profiles)

    def profiles(self) -> List[SpeakerProfile]:
        with self._lock:
            return sorted(self._profiles.values(), key=lambda item: item.label)

    def reset(self) -> None:
        with self._lock:
            self._profiles.clear()
            self._counter = 0

    def assign(
        self, audio: np.ndarray, sample_rate: int = 16_000, timestamp: float = 0.0
    ) -> Optional[SpeakerAssignment]:
        duration = np.asarray(audio).size / float(sample_rate or 1)
        if duration < self.min_seconds:
            return None
        embedding = voice_embedding(audio, sample_rate)
        if embedding is None:
            return None
        return self.assign_embedding(embedding, duration, timestamp)

    def assign_embedding(
        self, embedding: np.ndarray, duration: float = 0.0, timestamp: float = 0.0
    ) -> SpeakerAssignment:
        with self._lock:
            best_label, best_similarity, runner_up = self._nearest(embedding)

            if best_label is None or (
                best_similarity < self.similarity_threshold
                and len(self._profiles) < self.max_speakers
            ):
                self._counter += 1
                label = f"Speaker {chr(ord('A') + (self._counter - 1) % 26)}"
                self._profiles[label] = SpeakerProfile(
                    label=label,
                    centroid=embedding.copy(),
                    total_seconds=duration,
                    first_seen=timestamp,
                    last_seen=timestamp,
                )
                return SpeakerAssignment(label, best_similarity, True, confidence=1.0)

            profile = self._profiles[best_label]
            profile.centroid = self._blend(profile.centroid, embedding)
            profile.segments += 1
            profile.total_seconds += duration
            profile.last_seen = timestamp
            margin = best_similarity - runner_up if runner_up > 0 else best_similarity
            return SpeakerAssignment(
                label=best_label,
                similarity=best_similarity,
                is_new=False,
                confidence=float(min(1.0, max(0.0, margin * 2.0))),
                metadata={"runner_up": runner_up},
            )

    def _nearest(self, embedding: np.ndarray) -> Tuple[Optional[str], float, float]:
        scores = [
            (label, float(np.dot(profile.centroid, embedding)))
            for label, profile in self._profiles.items()
        ]
        if not scores:
            return None, 0.0, 0.0
        scores.sort(key=lambda item: -item[1])
        runner_up = scores[1][1] if len(scores) > 1 else 0.0
        return scores[0][0], scores[0][1], runner_up

    def _blend(self, centroid: np.ndarray, embedding: np.ndarray) -> np.ndarray:
        merged = (1.0 - self.adaptation_rate) * centroid + self.adaptation_rate * embedding
        norm = np.linalg.norm(merged)
        return (merged / norm).astype(np.float32) if norm > 0 else centroid
