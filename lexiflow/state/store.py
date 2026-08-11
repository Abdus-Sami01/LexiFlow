"""Thread-safe application state plus optional SQLite persistence and search.

Readers (the dashboard) and writers (the analytics consumer) live on different
threads, so every mutation goes through one re-entrant lock and every read
returns a snapshot copy. The SQLite mirror is opened per thread and uses an
FTS5 index when the interpreter was built with it, falling back to a LIKE scan
otherwise.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional

from ..config import StateConfig
from ..nlp.pipeline import Insight

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    name TEXT,
    started_at REAL,
    ended_at REAL
);
CREATE TABLE IF NOT EXISTS transcript (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    text TEXT NOT NULL,
    started_at REAL,
    ended_at REAL,
    backend TEXT,
    audio_seconds REAL,
    inference_seconds REAL,
    compound REAL,
    label TEXT
);
CREATE INDEX IF NOT EXISTS transcript_session_idx ON transcript(session_id, seq);
CREATE TABLE IF NOT EXISTS action_items (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    kind TEXT,
    text TEXT,
    rule TEXT,
    due TEXT,
    priority INTEGER,
    confidence REAL,
    done INTEGER DEFAULT 0,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    text TEXT,
    kind TEXT,
    label TEXT,
    seen_at REAL
);
CREATE INDEX IF NOT EXISTS entities_session_idx ON entities(session_id, kind);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS transcript_fts
USING fts5(text, session_id UNINDEXED, seq UNINDEXED, content='');
"""


@dataclass
class TranscriptItem:
    seq: int
    text: str
    started_at: float
    ended_at: float
    backend: str = "unknown"
    audio_seconds: float = 0.0
    inference_seconds: float = 0.0
    compound: float = 0.0
    label: str = "neutral"
    entities: List[Dict[str, Any]] = field(default_factory=list)
    extractions: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "text": self.text,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "backend": self.backend,
            "audio_seconds": self.audio_seconds,
            "inference_seconds": self.inference_seconds,
            "compound": self.compound,
            "label": self.label,
            "entities": list(self.entities),
            "extractions": list(self.extractions),
        }


@dataclass
class ActionItem:
    identifier: str
    kind: str
    text: str
    rule: str
    priority: int = 1
    confidence: float = 0.8
    due: Optional[str] = None
    done: bool = False
    created_at: float = field(default_factory=time.time)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.identifier,
            "kind": self.kind,
            "text": self.text,
            "rule": self.rule,
            "priority": self.priority,
            "confidence": self.confidence,
            "due": self.due,
            "done": self.done,
            "created_at": self.created_at,
        }


class SessionStore:
    """The single source of truth every thread reads from and writes to."""

    def __init__(self, config: Optional[StateConfig] = None) -> None:
        self.config = config or StateConfig()
        self.session_id = uuid.uuid4().hex[:16]
        self.session_name = self.config.session_name or time.strftime("session-%Y%m%d-%H%M%S")
        self.started_at = time.time()

        self._lock = threading.RLock()
        self._transcript: Deque[TranscriptItem] = deque(maxlen=self.config.max_transcript_items)
        self._actions: Dict[str, ActionItem] = {}
        self._entity_counts: Dict[str, Dict[str, int]] = {}
        self._sentiment_timeline: Deque[Dict[str, float]] = deque(
            maxlen=self.config.max_events
        )
        self._metrics: Dict[str, Any] = {}
        self._listeners: List[Callable[[str, Any], None]] = []
        self._sequence = 0
        self._local = threading.local()
        self._fts_enabled = False

        if self.config.persist:
            self._initialise_database()

    @property
    def database_path(self) -> Path:
        return Path(self.config.database_path)

    def _initialise_database(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connection()
        with connection:
            connection.executescript(SCHEMA)
            try:
                connection.executescript(FTS_SCHEMA)
                self._fts_enabled = True
            except sqlite3.OperationalError:
                self._fts_enabled = False
            connection.execute(
                "INSERT OR REPLACE INTO sessions(id, name, started_at, ended_at) VALUES (?,?,?,?)",
                (self.session_id, self.session_name, self.started_at, None),
            )

    def _connection(self) -> sqlite3.Connection:
        existing = getattr(self._local, "connection", None)
        if existing is not None:
            return existing
        connection = sqlite3.connect(str(self.database_path), timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        self._local.connection = connection
        return connection

    def subscribe(self, listener: Callable[[str, Any], None]) -> Callable[[], None]:
        with self._lock:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return unsubscribe

    def _emit(self, event: str, payload: Any) -> None:
        for listener in list(self._listeners):
            try:
                listener(event, payload)
            except Exception:
                continue

    def record(self, text: str, insight: Optional[Insight] = None, **kwargs: Any) -> TranscriptItem:
        """Append one utterance and everything the analytics layer found in it."""
        now = time.time()
        sentiment = insight.sentiment if insight else None

        with self._lock:
            self._sequence += 1
            item = TranscriptItem(
                seq=self._sequence,
                text=text,
                started_at=float(kwargs.get("started_at", now)),
                ended_at=float(kwargs.get("ended_at", now)),
                backend=str(kwargs.get("backend", "unknown")),
                audio_seconds=float(kwargs.get("audio_seconds", 0.0)),
                inference_seconds=float(kwargs.get("inference_seconds", 0.0)),
                compound=float(sentiment.compound) if sentiment else 0.0,
                label=sentiment.label if sentiment else "neutral",
                entities=[entity.as_dict() for entity in (insight.entities if insight else [])],
                extractions=[
                    extraction.as_dict() for extraction in (insight.extractions if insight else [])
                ],
            )
            self._transcript.append(item)
            self._sentiment_timeline.append(
                {
                    "seq": item.seq,
                    "at": item.ended_at,
                    "compound": item.compound,
                    "rolling": float(insight.rolling_sentiment) if insight else 0.0,
                    "momentum": float(insight.sentiment_momentum) if insight else 0.0,
                }
            )

            if insight is not None:
                for extraction in insight.extractions:
                    if extraction.kind in {"action_item", "deadline", "blocker", "decision"}:
                        self._actions[extraction.identifier] = ActionItem(
                            identifier=extraction.identifier,
                            kind=extraction.kind,
                            text=extraction.text,
                            rule=extraction.rule,
                            priority=extraction.priority,
                            confidence=extraction.confidence,
                            due=extraction.due,
                        )
                for entity in insight.entities:
                    bucket = self._entity_counts.setdefault(entity.kind, {})
                    bucket[entity.text] = bucket.get(entity.text, 0) + 1

        if self.config.persist:
            self._persist(item, insight)
        self._emit("transcript", item)
        return item

    def _persist(self, item: TranscriptItem, insight: Optional[Insight]) -> None:
        connection = self._connection()
        try:
            with connection:
                connection.execute(
                    "INSERT INTO transcript(session_id, seq, text, started_at, ended_at, backend,"
                    " audio_seconds, inference_seconds, compound, label)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        self.session_id,
                        item.seq,
                        item.text,
                        item.started_at,
                        item.ended_at,
                        item.backend,
                        item.audio_seconds,
                        item.inference_seconds,
                        item.compound,
                        item.label,
                    ),
                )
                if self._fts_enabled:
                    connection.execute(
                        "INSERT INTO transcript_fts(text, session_id, seq) VALUES (?,?,?)",
                        (item.text, self.session_id, item.seq),
                    )
                if insight is not None:
                    connection.executemany(
                        "INSERT OR REPLACE INTO action_items"
                        "(id, session_id, kind, text, rule, due, priority, confidence, done,"
                        " created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        [
                            (
                                extraction.identifier,
                                self.session_id,
                                extraction.kind,
                                extraction.text,
                                extraction.rule,
                                extraction.due,
                                extraction.priority,
                                extraction.confidence,
                                0,
                                time.time(),
                            )
                            for extraction in insight.extractions
                            if extraction.kind
                            in {"action_item", "deadline", "blocker", "decision"}
                        ],
                    )
                    connection.executemany(
                        "INSERT INTO entities(session_id, text, kind, label, seen_at)"
                        " VALUES (?,?,?,?,?)",
                        [
                            (self.session_id, entity.text, entity.kind, entity.label, time.time())
                            for entity in insight.entities
                        ],
                    )
        except sqlite3.Error:
            pass

    def toggle_action(self, identifier: str, done: Optional[bool] = None) -> Optional[ActionItem]:
        with self._lock:
            action = self._actions.get(identifier)
            if action is None:
                return None
            action.done = (not action.done) if done is None else bool(done)
            snapshot = ActionItem(**action.__dict__)

        if self.config.persist:
            try:
                connection = self._connection()
                with connection:
                    connection.execute(
                        "UPDATE action_items SET done=? WHERE id=?",
                        (1 if snapshot.done else 0, identifier),
                    )
            except sqlite3.Error:
                pass
        self._emit("action", snapshot)
        return snapshot

    def transcript(self, limit: Optional[int] = None) -> List[TranscriptItem]:
        with self._lock:
            items = list(self._transcript)
        return items[-limit:] if limit else items

    def actions(self, include_done: bool = True) -> List[ActionItem]:
        with self._lock:
            items = list(self._actions.values())
        if not include_done:
            items = [item for item in items if not item.done]
        items.sort(key=lambda item: (-item.priority, item.created_at))
        return items

    def entity_counts(self, kind: Optional[str] = None) -> Dict[str, Dict[str, int]]:
        with self._lock:
            if kind is not None:
                return {kind: dict(self._entity_counts.get(kind, {}))}
            return {key: dict(value) for key, value in self._entity_counts.items()}

    def sentiment_timeline(self, limit: Optional[int] = None) -> List[Dict[str, float]]:
        with self._lock:
            points = list(self._sentiment_timeline)
        return points[-limit:] if limit else points

    def update_metrics(self, **values: Any) -> None:
        with self._lock:
            self._metrics.update(values)

    def metrics(self) -> Dict[str, Any]:
        with self._lock:
            payload = dict(self._metrics)
            payload.update(
                {
                    "session_id": self.session_id,
                    "session_name": self.session_name,
                    "uptime_seconds": round(time.time() - self.started_at, 2),
                    "utterances": len(self._transcript),
                    "open_actions": sum(1 for item in self._actions.values() if not item.done),
                    "total_actions": len(self._actions),
                }
            )
        return payload

    def search(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Instant local search: FTS5 first, in-memory substring scan as backup."""
        needle = (query or "").strip()
        if not needle:
            return []

        if self.config.persist and self._fts_enabled:
            try:
                connection = self._connection()
                rows = connection.execute(
                    "SELECT seq, text FROM transcript_fts WHERE transcript_fts MATCH ?"
                    " AND session_id = ? ORDER BY seq DESC LIMIT ?",
                    (self._as_match_query(needle), self.session_id, limit),
                ).fetchall()
                if rows:
                    return [{"seq": row["seq"], "text": row["text"]} for row in rows]
            except sqlite3.Error:
                pass

        lowered = needle.lower()
        with self._lock:
            matches = [
                {"seq": item.seq, "text": item.text}
                for item in self._transcript
                if lowered in item.text.lower()
            ]
        return matches[-limit:][::-1]

    @staticmethod
    def _as_match_query(query: str) -> str:
        tokens = [token for token in query.replace('"', " ").split() if token]
        return " ".join(f'"{token}"*' for token in tokens) if tokens else query

    def export(self) -> Dict[str, Any]:
        return {
            "session": {
                "id": self.session_id,
                "name": self.session_name,
                "started_at": self.started_at,
            },
            "metrics": self.metrics(),
            "transcript": [item.as_dict() for item in self.transcript()],
            "actions": [item.as_dict() for item in self.actions()],
            "entities": self.entity_counts(),
            "sentiment": self.sentiment_timeline(),
        }

    def export_json(self, path: Optional[Path] = None, indent: int = 2) -> str:
        payload = json.dumps(self.export(), indent=indent, default=str)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(payload, encoding="utf-8")
        return payload

    def past_sessions(self, limit: int = 25) -> List[Dict[str, Any]]:
        if not self.config.persist:
            return []
        try:
            connection = self._connection()
            rows = connection.execute(
                "SELECT id, name, started_at, ended_at FROM sessions"
                " ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error:
            return []

    def close(self) -> None:
        if self.config.persist:
            try:
                connection = self._connection()
                with connection:
                    connection.execute(
                        "UPDATE sessions SET ended_at=? WHERE id=?",
                        (time.time(), self.session_id),
                    )
                connection.close()
            except sqlite3.Error:
                pass
            self._local.connection = None

    def seed(self, lines: Iterable[str], analytics) -> None:
        """Replay text through the analytics engine; handy for demos and tests."""
        for line in lines:
            self.record(line, analytics.analyse(line))
