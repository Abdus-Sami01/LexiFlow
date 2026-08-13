"""Strip identifying detail out of a transcript before it leaves the machine.

Recording locally is only half the privacy story: the moment a transcript is
exported it can be pasted anywhere. This module removes the parts that identify
people, using the same entity extraction the analytics already run plus a set
of high-confidence patterns for the things regexes are genuinely good at.

Pseudonyms are stable within a session, so "Sarah Chen" becomes PERSON_1
everywhere and the document still reads as a conversation between people rather
than a wall of black boxes.
"""

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern, Sequence, Tuple

MASK_CHARACTER = "█"

PATTERNS: Dict[str, Pattern[str]] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"),
    "url": re.compile(r"\bhttps?://\S+", re.IGNORECASE),
    "phone": re.compile(
        r"(?<![\w.])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?"
        r"\d{3,4}[\s.-]\d{3,4}(?:[\s.-]\d{2,4})?(?![\w.])"
    ),
    "card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),
    "ip": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "postcode": re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}

ENTITY_KINDS = {"person", "organization", "location", "contact", "link"}
DEFAULT_KINDS = ("email", "phone", "card", "iban", "ssn", "person")


@dataclass
class Redaction:
    """One removal, kept so a caller can audit what was taken out."""

    kind: str
    original: str
    replacement: str
    start: int
    end: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "replacement": self.replacement,
            "start": self.start,
            "end": self.end,
            "length": len(self.original),
        }


@dataclass
class RedactionResult:
    text: str
    removals: List[Redaction] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.removals)

    def counts(self) -> Dict[str, int]:
        tally: Dict[str, int] = {}
        for item in self.removals:
            tally[item.kind] = tally.get(item.kind, 0) + 1
        return tally


class Redactor:
    """Pattern hits plus entity hits, resolved into one non-overlapping pass."""

    def __init__(
        self,
        kinds: Sequence[str] = DEFAULT_KINDS,
        mode: str = "pseudonym",
        entities: Optional[Any] = None,
        salt: str = "",
    ) -> None:
        self.kinds = tuple(kinds)
        self.mode = mode
        self.entities = entities
        self.salt = salt
        self._aliases: Dict[Tuple[str, str], str] = {}
        self._counters: Dict[str, int] = {}
        self._lock = threading.Lock()

    def alias_for(self, kind: str, value: str) -> str:
        """Same input, same pseudonym, for as long as this redactor lives."""
        key = (kind, value.strip().lower())
        with self._lock:
            existing = self._aliases.get(key)
            if existing is not None:
                return existing
            if self.mode == "hash":
                digest = hashlib.sha256((self.salt + key[1]).encode("utf-8")).hexdigest()[:8]
                alias = f"[{kind.upper()}:{digest}]"
            elif self.mode == "mask":
                alias = MASK_CHARACTER * max(4, min(len(value), 12))
            elif self.mode == "label":
                alias = f"[{kind.upper()}]"
            else:
                self._counters[kind] = self._counters.get(kind, 0) + 1
                alias = f"[{kind.upper()}_{self._counters[kind]}]"
            self._aliases[key] = alias
            return alias

    def aliases(self) -> Dict[str, str]:
        with self._lock:
            return {f"{kind}:{value}": alias for (kind, value), alias in self._aliases.items()}

    def _pattern_spans(self, text: str) -> List[Tuple[int, int, str, str]]:
        found: List[Tuple[int, int, str, str]] = []
        for kind, pattern in PATTERNS.items():
            if kind not in self.kinds:
                continue
            for match in pattern.finditer(text):
                value = match.group(0).strip()
                if kind == "card" and len(re.sub(r"\D", "", value)) < 13:
                    continue
                found.append((match.start(), match.end(), kind, value))
        return found

    def _entity_spans(self, text: str) -> List[Tuple[int, int, str, str]]:
        wanted = {kind for kind in self.kinds if kind in ENTITY_KINDS}
        if not wanted or self.entities is None:
            return []
        found = []
        for entity in self.entities.extract(text):
            if entity.kind in wanted:
                found.append((entity.start, entity.end, entity.kind, entity.text))
        return found

    def redact(self, text: str) -> RedactionResult:
        if not text or not text.strip():
            return RedactionResult(text or "")

        spans = self._pattern_spans(text) + self._entity_spans(text)
        if not spans:
            return RedactionResult(text)

        spans.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        chosen: List[Tuple[int, int, str, str]] = []
        for span in spans:
            if chosen and span[0] < chosen[-1][1]:
                continue
            chosen.append(span)

        pieces: List[str] = []
        removals: List[Redaction] = []
        cursor = 0
        for start, end, kind, value in chosen:
            alias = self.alias_for(kind, value)
            pieces.append(text[cursor:start])
            pieces.append(alias)
            removals.append(Redaction(kind, value, alias, start, end))
            cursor = end
        pieces.append(text[cursor:])
        return RedactionResult("".join(pieces), removals)

    def redact_rows(self, rows: Sequence[Any]) -> List[Any]:
        """Return shallow copies with text, translation and spans rewritten."""
        redacted = []
        for row in rows:
            clone = _Redacted(row)
            clone.text = self.redact(row.text).text
            translation = getattr(row, "translation", None)
            clone.translation = self.redact(translation).text if translation else None
            clone.spans = [
                {**span, "text": self.redact(str(span.get("text", ""))).text}
                for span in (getattr(row, "spans", None) or [])
            ]
            redacted.append(clone)
        return redacted

    def redact_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Scrub an exported session dict in place of the original."""
        cleaned = dict(payload)
        cleaned["transcript"] = [
            {
                **row,
                "text": self.redact(row.get("text", "")).text,
                "translation": (
                    self.redact(row["translation"]).text if row.get("translation") else None
                ),
            }
            for row in payload.get("transcript") or []
        ]
        cleaned["actions"] = [
            {**action, "text": self.redact(action.get("text", "")).text}
            for action in payload.get("actions") or []
        ]
        cleaned["entities"] = {
            kind: self._scrub_counts(kind, counts)
            for kind, counts in (payload.get("entities") or {}).items()
        }
        return cleaned

    def _scrub_counts(self, kind: str, counts: Dict[str, int]) -> Dict[str, int]:
        """Entity buckets are named by extractor kind, not by pattern kind."""
        scrubbed: Dict[str, int] = {}
        for name, count in counts.items():
            if kind in self.kinds:
                key = self.alias_for(kind, name)
            else:
                key = self.redact(name).text
            scrubbed[key] = scrubbed.get(key, 0) + count
        return scrubbed


class _Redacted:
    """A stand-in row that keeps timing but carries scrubbed text."""

    def __init__(self, source: Any) -> None:
        self.seq = getattr(source, "seq", 0)
        self.started_at = source.started_at
        self.ended_at = source.ended_at
        self.speaker = getattr(source, "speaker", None)
        self.compound = getattr(source, "compound", 0.0)
        self.label = getattr(source, "label", "neutral")
        self.text = source.text
        self.translation = getattr(source, "translation", None)
        self.spans: List[Dict[str, Any]] = []

    def as_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "text": self.text,
            "speaker": self.speaker,
            "translation": self.translation,
            "compound": self.compound,
            "label": self.label,
        }


def build(config: Optional[Any] = None, entities: Optional[Any] = None) -> Redactor:
    from .config import RedactionConfig

    settings = config or RedactionConfig()
    return Redactor(
        kinds=tuple(settings.kinds),
        mode=settings.mode,
        entities=entities,
        salt=settings.salt,
    )
