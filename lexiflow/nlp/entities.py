"""Named entity extraction with a spaCy small model, or regex when it is absent."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from ..observability import get_logger

INTERESTING_LABELS = {
    "PERSON": "person",
    "ORG": "organization",
    "GPE": "location",
    "LOC": "location",
    "FAC": "location",
    "DATE": "date",
    "TIME": "time",
    "MONEY": "money",
    "PERCENT": "percent",
    "PRODUCT": "product",
    "EVENT": "event",
    "NORP": "group",
    "CARDINAL": "number",
}

TITLE_PATTERN = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)")
PERSON_PATTERN = re.compile(r"\b(?=([A-Z][a-z]{2,})\s+([A-Z][a-z]{2,})\b)")

NON_NAME_LEADERS = frozenset(
    """
    email call text remind ask tell send ship review check follow meet let please can could
    would should will shall the this that these those there here what when where which who
    why how our your their his her its every each some many most next last first second
    good great morning afternoon evening hello thanks thank sorry maybe perhaps also but and
    before after during until while because since though although however therefore
    monday tuesday wednesday thursday friday saturday sunday today tomorrow yesterday
    january february march april may june july august september october november december
    """.split()
)
ORG_SUFFIX_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*)*)\s+"
    r"(Inc|LLC|Ltd|Corp|Corporation|Company|GmbH|PLC|Group|Labs|Technologies|Systems)\b"
)
MONEY_PATTERN = re.compile(r"[$€£]\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:k|m|bn|million|billion))?", re.I)
PERCENT_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s?(?:%|percent)\b", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
URL_PATTERN = re.compile(r"\bhttps?://\S+\b")


@dataclass
class Entity:
    text: str
    label: str
    kind: str
    start: int
    end: int
    source: str = "rules"

    def as_dict(self) -> Dict[str, object]:
        return {
            "text": self.text,
            "label": self.label,
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
            "source": self.source,
        }


@lru_cache(maxsize=4)
def _load_spacy(model_name: str):
    try:
        import spacy
    except Exception:
        return None
    try:
        return spacy.load(model_name, disable=["lemmatizer", "textcat"])
    except Exception as error:
        get_logger("entities").debug("spacy model %s unavailable: %s", model_name, error)
        return None


class EntityExtractor:
    """spaCy when present, deterministic gazetteer fallback when it is not."""

    def __init__(self, model_name: str = "en_core_web_sm", enable_spacy: bool = True) -> None:
        self.model_name = model_name
        self._nlp = _load_spacy(model_name) if enable_spacy else None

    @property
    def backend(self) -> str:
        return f"spacy:{self.model_name}" if self._nlp is not None else "regex"

    def extract(self, text: str) -> List[Entity]:
        if not text or not text.strip():
            return []
        entities = self._spacy_entities(text) if self._nlp is not None else []
        entities.extend(self._regex_entities(text, skip_spans=[(e.start, e.end) for e in entities]))
        entities.sort(key=lambda item: item.start)
        return entities

    def _spacy_entities(self, text: str) -> List[Entity]:
        document = self._nlp(text)
        found: List[Entity] = []
        for span in document.ents:
            kind = INTERESTING_LABELS.get(span.label_)
            if kind is None:
                continue
            found.append(
                Entity(
                    text=span.text,
                    label=span.label_,
                    kind=kind,
                    start=span.start_char,
                    end=span.end_char,
                    source="spacy",
                )
            )
        return found

    def _regex_entities(
        self, text: str, skip_spans: Optional[List[Tuple[int, int]]] = None
    ) -> List[Entity]:
        skip = skip_spans or []

        def overlaps(start: int, end: int) -> bool:
            return any(start < other_end and end > other_start for other_start, other_end in skip)

        found: List[Entity] = []
        patterns = (
            (ORG_SUFFIX_PATTERN, "ORG", "organization", 0),
            (TITLE_PATTERN, "PERSON", "person", 1),
            (MONEY_PATTERN, "MONEY", "money", 0),
            (PERCENT_PATTERN, "PERCENT", "percent", 0),
            (EMAIL_PATTERN, "EMAIL", "contact", 0),
            (URL_PATTERN, "URL", "link", 0),
        )
        for pattern, label, kind, group in patterns:
            for match in pattern.finditer(text):
                start, end = match.span(group)
                if overlaps(start, end):
                    continue
                found.append(
                    Entity(match.group(group).strip(), label, kind, start, end, source="regex")
                )
                skip.append((start, end))

        for match in PERSON_PATTERN.finditer(text):
            start, end = match.start(1), match.end(2)
            if overlaps(start, end) or match.group(1).lower() in NON_NAME_LEADERS:
                continue
            found.append(Entity(text[start:end], "PERSON", "person", start, end, source="regex"))
            skip.append((start, end))

        return found
