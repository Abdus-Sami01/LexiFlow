"""On-the-fly format conversion between whatever the sound card hands us and"""

from __future__ import annotations

from typing import Union

import numpy as np

INT16_SCALE = 32768.0
INT32_SCALE = 2147483648.0
WHISPER_SAMPLE_RATE = 16_000

ArrayLike = Union[np.ndarray, bytes, bytearray, memoryview]


def decode_raw(buffer: ArrayLike, dtype: str) -> np.ndarray:
    """Turn a raw byte payload from a PortAudio style callback into an array."""
    if isinstance(buffer, np.ndarray):
        return buffer
    return np.frombuffer(bytes(buffer), dtype=np.dtype(dtype))


def to_float32(samples: np.ndarray) -> np.ndarray:
    """Normalise any common PCM integer layout into [-1.0, 1.0] float32."""
    array = np.asarray(samples)
    if array.dtype == np.float32:
        return array
    if array.dtype == np.float64:
        return array.astype(np.float32, copy=False)
    if array.dtype == np.int16:
        return (array.astype(np.float32) / INT16_SCALE).astype(np.float32, copy=False)
    if array.dtype == np.int32:
        return (array.astype(np.float32) / INT32_SCALE).astype(np.float32, copy=False)
    if array.dtype == np.uint8:
        return ((array.astype(np.float32) - 128.0) / 128.0).astype(np.float32, copy=False)
    return array.astype(np.float32, copy=False)


def to_mono(samples: np.ndarray, channels: int) -> np.ndarray:
    """Average interleaved channels down to a single stream."""
    array = np.asarray(samples, dtype=np.float32)
    if channels <= 1:
        return array.reshape(-1)
    if array.ndim == 2:
        return array.mean(axis=1, dtype=np.float32)
    usable = (array.size // channels) * channels
    if usable == 0:
        return np.zeros(0, dtype=np.float32)
    return array[:usable].reshape(-1, channels).mean(axis=1, dtype=np.float32)


def resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Band-limited enough linear resampler; cheap and good for speech."""
    array = np.asarray(samples, dtype=np.float32).reshape(-1)
    if source_rate == target_rate or array.size == 0:
        return array
    duration = array.size / float(source_rate)
    target_length = int(round(duration * target_rate))
    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)
    source_positions = np.arange(array.size, dtype=np.float64)
    target_positions = np.linspace(0.0, array.size - 1, target_length, dtype=np.float64)
    return np.interp(target_positions, source_positions, array).astype(np.float32, copy=False)


def prepare_for_whisper(
    buffer: ArrayLike,
    source_rate: int,
    channels: int = 1,
    dtype: str = "float32",
    target_rate: int = WHISPER_SAMPLE_RATE,
) -> np.ndarray:
    """Full decode → float32 → mono → 16 kHz path in one call."""
    decoded = decode_raw(buffer, dtype)
    mono = to_mono(to_float32(decoded), channels)
    return resample_linear(mono, source_rate, target_rate)


def rms(samples: np.ndarray) -> float:
    array = np.asarray(samples, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(array, dtype=np.float64))))


def peak(samples: np.ndarray) -> float:
    array = np.asarray(samples, dtype=np.float32).reshape(-1)
    return float(np.max(np.abs(array))) if array.size else 0.0


def dbfs(samples: np.ndarray, floor_db: float = -90.0) -> float:
    level = rms(samples)
    if level <= 0.0:
        return floor_db
    return max(floor_db, float(20.0 * np.log10(level)))
