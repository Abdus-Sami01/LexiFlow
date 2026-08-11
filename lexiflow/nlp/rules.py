"""Deterministic pattern extraction for action items, deadlines and questions.

Regex beats a transformer here on both accuracy and latency: the phrasing
people use to assign work is small and highly conventional, and matching it
costs microseconds instead of milliseconds.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Pattern, Tuple

PRIORITY_HINTS = {
    "critical": 3, "urgent": 3, "asap": 3, "immediately": 3, "blocker": 3,
    "important": 2, "priority": 2, "soon": 2, "today": 2, "tonight": 2,
    "whenever": 0, "eventually": 0, "sometime": 0, "later": 0,
}

RELATIVE_DATE_PATTERN = re.compile(
    r"\b("
    r"today|tonight|tomorrow|yesterday|"
    r"next\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"this\s+(?:week|month|afternoon|evening|morning|monday|tuesday|wednesday|thursday|friday)|"
    r"end\s+of\s+(?:day|week|month|quarter|sprint)|"
    r"eod|eow|cob|"
    r"in\s+\d+\s+(?:minutes?|hours?|days?|weeks?|months?)|"
    r"by\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)|"
    r"(?:mon|tues|wednes|thurs|fri|satur|sun)day|"
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    r")\b",
    re.IGNORECASE,
)

NUMERIC_DATE_PATTERN = re.compile(r"\b\d{1,4}[/-]\d{1,2}(?:[/-]\d{1,4})?\b")
TIME_PATTERN = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b|\b\d{1,2}:\d{2}\b", re.IGNORECASE)


@dataclass
class Extraction:
    """One rule hit, carrying enough context to render it in the dashboard."""

    kind: str
    text: str
    rule: str
    span: Tuple[int, int]
    confidence: float = 0.8
    priority: int = 1
    due: Optional[str] = None
    identifier: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    metadata: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "id": self.identifier,
            "kind": self.kind,
            "text": self.text,
            "rule": self.rule,
            "span": list(self.span),
            "confidence": self.confidence,
            "priority": self.priority,
            "due": self.due,
            "metadata": dict(self.metadata),
        }


@dataclass
class RuleSpec:
    name: str
    kind: str
    pattern: Pattern[str]
    confidence: float = 0.8
    capture: int = 1


def _compile(expression: str) -> Pattern[str]:
    return re.compile(expression, re.IGNORECASE)


DEFAULT_RULES: List[RuleSpec] = [
    RuleSpec(
        "reminder",
        "action_item",
        _compile(r"\bremind\s+(?:me|us|him|her|them)\s+to\s+(.+?)(?=[.?!;]|$)"),
        0.95,
    ),
    RuleSpec(
        "assignment",
        "action_item",
        _compile(
            r"\b(?:can|could|will|would)\s+you\s+(?:please\s+)?(.+?)(?=[.?!;]|$)"
        ),
        0.8,
    ),
    RuleSpec(
        "commitment",
        "action_item",
        _compile(r"\bi(?:'ll| will| am going to| gonna)\s+(.+?)(?=[.?!;]|$)"),
        0.85,
    ),
    RuleSpec(
        "team_commitment",
        "action_item",
        _compile(r"\bwe\s+(?:need|have)\s+to\s+(.+?)(?=[.?!;]|$)"),
        0.85,
    ),
    RuleSpec(
        "directive",
        "action_item",
        _compile(r"\b(?:make sure|be sure|don't forget|do not forget)\s+to\s+(.+?)(?=[.?!;]|$)"),
        0.9,
    ),
    RuleSpec(
        "follow_up",
        "action_item",
        _compile(r"\b(?:follow up|circle back|check back)\s+(?:on|with)\s+(.+?)(?=[.?!;]|$)"),
        0.85,
    ),
    RuleSpec(
        "action_item_phrase",
        "action_item",
        _compile(r"\baction item(?:s)?\s*(?:is|are|:)?\s*(.+?)(?=[.?!;]|$)"),
        0.9,
    ),
    RuleSpec(
        "task_assignment",
        "action_item",
        _compile(r"\b(?:let's|lets)\s+(.+?)(?=[.?!;]|$)"),
        0.7,
    ),
    RuleSpec(
        "deadline",
        "deadline",
        _compile(r"\bdeadline\s+(?:is|was|will be)?\s*(.+?)(?=[.?!;,]|\s+and\b|\s+but\b|$)"),
        0.95,
    ),
    RuleSpec(
        "due_by",
        "deadline",
        _compile(
            r"\b(?:due|needs to be done|has to be ready)\s+(?:by|on|before)\s+(.+?)(?=[.?!;]|$)"
        ),
        0.9,
    ),
    RuleSpec(
        "ship_by",
        "deadline",
        _compile(r"\b(?:ship|deliver|submit|send)\s+(?:it|this|that)?\s*by\s+(.+?)(?=[.?!;]|$)"),
        0.85,
    ),
    RuleSpec(
        "decision",
        "decision",
        _compile(r"\bwe(?:'ve| have)?\s+decided\s+(?:to|that)\s+(.+?)(?=[.?!;]|$)"),
        0.9,
    ),
    RuleSpec(
        "agreement",
        "decision",
        _compile(
            r"\b(?:we agreed|agreement is|consensus is)\s+(?:to|that|on)?\s*(.+?)(?=[.?!;]|$)"
        ),
        0.85,
    ),
    RuleSpec(
        "blocker",
        "blocker",
        _compile(r"\b(?:blocked (?:on|by)|blocker is|stuck on)\s+(.+?)(?=[.?!;]|$)"),
        0.9,
    ),
    RuleSpec(
        "risk",
        "risk",
        _compile(r"\b(?:risk is|concern is|worried about|problem is)\s+(.+?)(?=[.?!;]|$)"),
        0.8,
    ),
    RuleSpec(
        "question",
        "question",
        _compile(r"((?:^|[.!?]\s*)(?:who|what|when|where|why|how|which)\b[^.?!]*\?)"),
        0.75,
    ),
    RuleSpec(
        "metric",
        "metric",
        _compile(r"\b((?:\$|€|£)?\d[\d,.]*\s*(?:%|percent|k|m|bn|million|billion|users|dollars)?)\b"),
        0.6,
    ),
]


def infer_priority(text: str) -> int:
    lowered = text.lower()
    priority = 1
    for hint, level in PRIORITY_HINTS.items():
        if hint in lowered:
            priority = max(priority, level) if level >= 1 else min(priority, level)
    return priority


def find_due_date(text: str) -> Optional[str]:
    for pattern in (RELATIVE_DATE_PATTERN, NUMERIC_DATE_PATTERN, TIME_PATTERN):
        match = pattern.search(text)
        if match:
            return match.group(0).strip()
    return None


def _clean(fragment: str) -> str:
    cleaned = re.sub(r"\s+", " ", fragment).strip(" ,;:-")
    return cleaned


class RuleEngine:
    """Runs every rule over a line and returns de-duplicated extractions."""

    def __init__(self, rules: Optional[List[RuleSpec]] = None, min_length: int = 3) -> None:
        self.rules = list(rules or DEFAULT_RULES)
        self.min_length = min_length

    def add_rule(self, spec: RuleSpec) -> None:
        self.rules.append(spec)

    def extract(self, text: str) -> List[Extraction]:
        if not text or not text.strip():
            return []

        results: List[Extraction] = []
        seen: set[Tuple[str, str]] = set()

        for rule in self.rules:
            for match in rule.pattern.finditer(text):
                group_index = rule.capture if rule.pattern.groups >= rule.capture else 0
                fragment = _clean(match.group(group_index) or "")
                if len(fragment) < self.min_length:
                    continue
                key = (rule.kind, fragment.lower())
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    Extraction(
                        kind=rule.kind,
                        text=fragment,
                        rule=rule.name,
                        span=(match.start(group_index), match.end(group_index)),
                        confidence=rule.confidence,
                        priority=infer_priority(match.group(0)),
                        due=find_due_date(match.group(0)),
                    )
                )

        results.sort(key=lambda item: (-item.confidence, item.span[0]))
        return self._suppress_nested(results)

    @staticmethod
    def _suppress_nested(items: List[Extraction]) -> List[Extraction]:
        """Drop a hit that is fully contained in a longer hit of the same kind."""
        kept: List[Extraction] = []
        for candidate in sorted(items, key=lambda item: item.span[0] - item.span[1]):
            contained = any(
                other.kind == candidate.kind
                and other.span[0] <= candidate.span[0]
                and other.span[1] >= candidate.span[1]
                for other in kept
            )
            if not contained:
                kept.append(candidate)
        kept.sort(key=lambda item: (-item.confidence, item.span[0]))
        return kept
