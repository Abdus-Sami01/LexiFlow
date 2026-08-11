"""A lock guarded, fixed capacity ring buffer for mono float32 audio."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np


class RingBufferOverflow(RuntimeError):
    """Raised only when a caller explicitly asks for strict overflow behaviour."""


@dataclass
class RingBufferStats:
    capacity: int
    available: int
    written: int
    dropped: int
    read: int

    @property
    def fill_ratio(self) -> float:
        return self.available / self.capacity if self.capacity else 0.0


class AudioRingBuffer:
    """Single-producer / single-consumer friendly, but safe for many of each."""

    def __init__(self, capacity_frames: int, dtype: str = "float32") -> None:
        if capacity_frames <= 0:
            raise ValueError("capacity_frames must be positive")
        self._capacity = int(capacity_frames)
        self._dtype = np.dtype(dtype)
        self._buffer = np.zeros(self._capacity, dtype=self._dtype)
        self._write_index = 0
        self._available = 0
        self._total_written = 0
        self._total_dropped = 0
        self._total_read = 0
        self._condition = threading.Condition(threading.Lock())
        self._closed = False

    @property
    def capacity(self) -> int:
        return self._capacity

    def stats(self) -> RingBufferStats:
        with self._condition:
            return RingBufferStats(
                capacity=self._capacity,
                available=self._available,
                written=self._total_written,
                dropped=self._total_dropped,
                read=self._total_read,
            )

    def write(self, samples: np.ndarray, strict: bool = False) -> int:
        """Append samples, overwriting the oldest audio when the buffer is full."""
        block = np.asarray(samples, dtype=self._dtype).reshape(-1)
        if block.size == 0:
            return 0
        if block.size > self._capacity:
            self._total_dropped += block.size - self._capacity
            block = block[-self._capacity :]

        with self._condition:
            if self._closed:
                return 0
            overflow = max(0, self._available + block.size - self._capacity)
            if overflow and strict:
                raise RingBufferOverflow(
                    f"{overflow} frames would be discarded (capacity={self._capacity})"
                )

            end = self._write_index + block.size
            if end <= self._capacity:
                self._buffer[self._write_index : end] = block
            else:
                head = self._capacity - self._write_index
                self._buffer[self._write_index :] = block[:head]
                self._buffer[: block.size - head] = block[head:]

            self._write_index = end % self._capacity
            self._available = min(self._capacity, self._available + block.size)
            self._total_written += block.size
            self._total_dropped += overflow
            self._condition.notify_all()
            return block.size

    def read(self, frames: int, timeout: Optional[float] = None) -> np.ndarray:
        """Pop up to ``frames`` samples, blocking until some audio is available."""
        if frames <= 0:
            return np.zeros(0, dtype=self._dtype)

        with self._condition:
            if not self._wait_for_data(timeout):
                return np.zeros(0, dtype=self._dtype)
            take = min(frames, self._available)
            start = (self._write_index - self._available) % self._capacity
            end = start + take
            if end <= self._capacity:
                chunk = self._buffer[start:end].copy()
            else:
                head = self._capacity - start
                chunk = np.concatenate(
                    (self._buffer[start:].copy(), self._buffer[: take - head].copy())
                )
            self._available -= take
            self._total_read += take
            return chunk

    def read_exact(self, frames: int, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """Return exactly ``frames`` samples or ``None`` if the buffer closed first."""
        collected = []
        remaining = frames
        while remaining > 0:
            chunk = self.read(remaining, timeout=timeout)
            if chunk.size == 0:
                if self.is_closed:
                    return None
                if timeout is not None:
                    return None
                continue
            collected.append(chunk)
            remaining -= chunk.size
        return np.concatenate(collected) if collected else np.zeros(0, dtype=self._dtype)

    def peek(self, frames: int) -> np.ndarray:
        """Look at the most recent ``frames`` samples without consuming them."""
        with self._condition:
            take = min(frames, self._available)
            if take == 0:
                return np.zeros(0, dtype=self._dtype)
            start = (self._write_index - take) % self._capacity
            end = start + take
            if end <= self._capacity:
                return self._buffer[start:end].copy()
            head = self._capacity - start
            return np.concatenate(
                (self._buffer[start:].copy(), self._buffer[: take - head].copy())
            )

    def clear(self) -> None:
        with self._condition:
            self._available = 0
            self._write_index = 0
            self._buffer.fill(0)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    @property
    def is_closed(self) -> bool:
        with self._condition:
            return self._closed

    def _wait_for_data(self, timeout: Optional[float]) -> bool:
        if self._available:
            return True
        if self._closed:
            return False
        return self._condition.wait_for(
            lambda: self._available > 0 or self._closed, timeout=timeout
        ) and self._available > 0

    def __len__(self) -> int:
        with self._condition:
            return self._available
