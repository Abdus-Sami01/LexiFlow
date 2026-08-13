"""One place to record the failures the pipeline is designed to survive.

Every stage here is deliberately fault tolerant: a SQLite write that fails must
not kill the microphone thread, and a translator that throws must not lose the
transcript. That resilience used to be spelled ``except: pass``, which meant a
disk full at minute forty looked exactly like a healthy session. Now the same
failures are swallowed by the pipeline but counted here, so the dashboards and
``doctor`` can say what went wrong.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

LOGGER_NAME = "lexiflow"
HANDLER_NAME = "lexiflow-stderr"
DEFAULT_HISTORY = 50


@dataclass
class Failure:
    component: str
    message: str
    kind: str
    at: float = field(default_factory=time.time)
    detail: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "message": self.message,
            "kind": self.kind,
            "at": self.at,
            "detail": dict(self.detail),
        }

    def __str__(self) -> str:
        return f"{self.component}: {self.kind}: {self.message}"


class FailureLog:
    """A bounded, thread-safe record of everything that quietly went wrong."""

    def __init__(self, history: int = DEFAULT_HISTORY) -> None:
        self.history = max(1, history)
        self._entries: List[Failure] = []
        self._counts: Counter[str] = Counter()
        self._lock = threading.Lock()

    def record(
        self,
        component: str,
        error: BaseException | str,
        detail: Optional[Dict[str, Any]] = None,
    ) -> Failure:
        kind = type(error).__name__ if isinstance(error, BaseException) else "error"
        failure = Failure(component, str(error), kind, detail=dict(detail or {}))
        with self._lock:
            self._entries.append(failure)
            if len(self._entries) > self.history:
                self._entries.pop(0)
            self._counts[component] += 1
        logging.getLogger(LOGGER_NAME).warning("%s", failure)
        return failure

    def recent(self, limit: Optional[int] = None) -> List[Failure]:
        with self._lock:
            entries = list(self._entries)
        return entries[-limit:] if limit else entries

    def counts(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._counts)

    @property
    def total(self) -> int:
        with self._lock:
            return sum(self._counts.values())

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._counts.clear()

    def summary(self, limit: int = 5) -> Dict[str, Any]:
        return {
            "total": self.total,
            "by_component": self.counts(),
            "recent": [item.as_dict() for item in self.recent(limit)],
        }


FAILURES = FailureLog()


def record_failure(
    component: str, error: BaseException | str, **detail: Any
) -> Failure:
    """Shorthand used at every ``except`` site that intentionally continues."""
    return FAILURES.record(component, error, detail or None)


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    return logging.getLogger(name if name.startswith(LOGGER_NAME) else f"{LOGGER_NAME}.{name}")


def configure_logging(verbose: bool = False, quiet: bool = False) -> logging.Logger:
    """Attach a single stderr handler; libraries should never own the root logger."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.ERROR if quiet else logging.DEBUG if verbose else logging.WARNING)
    if not any(handler.get_name() == HANDLER_NAME for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.set_name(HANDLER_NAME)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger


logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())
