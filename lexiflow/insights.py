"""Look across sessions, not just within one.

A single meeting's action items are easy. The useful question is the one that
spans meetings: what did we commit to three weeks ago and never close, which
blocker keeps coming back, and who has been at the centre of it. Everything
here is ordinary SQL over the sessions already on disk.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence

from .nlp.summarize import tokenize

STALE_AFTER_DAYS = 7.0
SIMILARITY = 0.82
TRACKED_KINDS = ("action_item", "deadline", "blocker", "decision")


def _normalise(text: str) -> str:
    return " ".join(sorted(tokenize(text))) or re.sub(r"\s+", " ", (text or "").lower()).strip()


def _alike(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    return SequenceMatcher(None, left, right).ratio() >= SIMILARITY


@dataclass
class OpenItem:
    """One unfinished commitment, with how long it has been unfinished."""

    identifier: str
    kind: str
    text: str
    priority: int
    due: Optional[str]
    session_id: str
    session_name: str
    created_at: float
    occurrences: int = 1

    @property
    def age_days(self) -> float:
        return max(0.0, (time.time() - self.created_at) / 86_400.0)

    @property
    def stale(self) -> bool:
        return self.age_days >= STALE_AFTER_DAYS

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.identifier,
            "kind": self.kind,
            "text": self.text,
            "priority": self.priority,
            "due": self.due,
            "session": self.session_name,
            "age_days": round(self.age_days, 1),
            "stale": self.stale,
            "occurrences": self.occurrences,
        }


@dataclass
class RecurringItem:
    """The same commitment or blocker, raised in more than one session."""

    text: str
    kind: str
    occurrences: int
    sessions: List[str] = field(default_factory=list)
    variants: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "kind": self.kind,
            "occurrences": self.occurrences,
            "sessions": list(self.sessions),
            "variants": list(self.variants),
        }


@dataclass
class Review:
    sessions: List[Dict[str, Any]] = field(default_factory=list)
    open_items: List[OpenItem] = field(default_factory=list)
    recurring: List[RecurringItem] = field(default_factory=list)
    people: List[Dict[str, Any]] = field(default_factory=list)
    sentiment_trend: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def stale_items(self) -> List[OpenItem]:
        return [item for item in self.open_items if item.stale]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sessions": list(self.sessions),
            "open_items": [item.as_dict() for item in self.open_items],
            "recurring": [item.as_dict() for item in self.recurring],
            "people": list(self.people),
            "sentiment_trend": list(self.sentiment_trend),
        }

    def as_markdown(self) -> str:
        lines = ["# Across your sessions", ""]
        lines.append(
            f"{len(self.sessions)} session(s) · {len(self.open_items)} open item(s) · "
            f"{len(self.stale_items)} older than {STALE_AFTER_DAYS:.0f} days"
        )

        lines.extend(["", "## Still open", ""])
        if self.open_items:
            for item in self.open_items:
                age = f"{item.age_days:.0f}d"
                due = f" · due {item.due}" if item.due else ""
                flag = " **stale**" if item.stale else ""
                repeats = f" · raised {item.occurrences}x" if item.occurrences > 1 else ""
                kind = "" if item.kind == "action_item" else f"{item.kind.replace('_', ' ')}: "
                lines.append(
                    f"- [ ] {kind}{item.text}{due}{repeats} "
                    f"_({item.session_name}, {age} ago)_{flag}"
                )
        else:
            lines.append("- nothing outstanding")

        if self.recurring:
            lines.extend(["", "## Keeps coming up", ""])
            for item in self.recurring:
                lines.append(
                    f"- {item.text} — {item.occurrences} times across "
                    f"{len(item.sessions)} sessions"
                )

        if self.people:
            lines.extend(["", "## Named most often", ""])
            for person in self.people:
                lines.append(
                    f"- {person['text']} · {person['mentions']} mentions in "
                    f"{person['sessions']} sessions"
                )

        if self.sentiment_trend:
            lines.extend(["", "## Mood by session", "", "| session | utterances | sentiment |"])
            lines.append("| --- | --- | --- |")
            for row in self.sentiment_trend:
                lines.append(
                    f"| {row['name']} | {row['utterances']} | {row['average_sentiment']:+.2f} |"
                )

        lines.append("")
        return "\n".join(lines)


def open_items(store: Any, limit: int = 200, collapse: bool = True) -> List[OpenItem]:
    """Newest first, with the same commitment from several sessions shown once."""
    items: List[OpenItem] = []
    signatures: List[str] = []

    for row in store.all_actions(limit=limit, open_only=True):
        if row["kind"] not in TRACKED_KINDS:
            continue
        item = OpenItem(
            identifier=row["id"],
            kind=row["kind"],
            text=row["text"],
            priority=int(row["priority"] or 1),
            due=row["due"],
            session_id=row["session_id"],
            session_name=row["session_name"] or row["session_id"],
            created_at=float(row["created_at"] or row["session_started"] or time.time()),
        )

        if collapse:
            signature = _normalise(item.text)
            match = next(
                (
                    existing
                    for existing, known in zip(items, signatures)
                    if existing.kind == item.kind and _alike(known, signature)
                ),
                None,
            )
            if match is not None:
                match.occurrences += 1
                match.created_at = min(match.created_at, item.created_at)
                continue
            signatures.append(signature)

        items.append(item)

    items.sort(key=lambda item: (-item.priority, -item.age_days))
    return items


def recurring_items(
    store: Any, minimum_sessions: int = 2, kinds: Sequence[str] = TRACKED_KINDS, limit: int = 400
) -> List[RecurringItem]:
    """Group near-identical items so a repeated blocker shows up as one row."""
    groups: List[Dict[str, Any]] = []
    for row in store.all_actions(limit=limit):
        if row["kind"] not in kinds:
            continue
        signature = _normalise(row["text"])
        if not signature:
            continue
        for group in groups:
            if group["kind"] == row["kind"] and _alike(group["signature"], signature):
                group["occurrences"] += 1
                group["sessions"].add(row["session_name"] or row["session_id"])
                group["variants"].add(row["text"])
                break
        else:
            groups.append(
                {
                    "signature": signature,
                    "kind": row["kind"],
                    "text": row["text"],
                    "occurrences": 1,
                    "sessions": {row["session_name"] or row["session_id"]},
                    "variants": {row["text"]},
                }
            )

    found = [
        RecurringItem(
            text=group["text"],
            kind=group["kind"],
            occurrences=group["occurrences"],
            sessions=sorted(group["sessions"]),
            variants=sorted(group["variants"]),
        )
        for group in groups
        if len(group["sessions"]) >= minimum_sessions
    ]
    found.sort(key=lambda item: (-len(item.sessions), -item.occurrences))
    return found


def build(store: Any, sessions: int = 20, people: int = 8) -> Review:
    summaries = store.session_summaries(limit=sessions)
    trend = [
        {
            "id": row["id"],
            "name": row["name"] or row["id"],
            "utterances": int(row["utterances"] or 0),
            "average_sentiment": float(row["average_sentiment"] or 0.0),
            "started_at": row["started_at"],
        }
        for row in summaries
    ]
    return Review(
        sessions=summaries,
        open_items=open_items(store),
        recurring=recurring_items(store),
        people=store.entity_totals(limit=people, kind="person"),
        sentiment_trend=list(reversed(trend)),
    )
