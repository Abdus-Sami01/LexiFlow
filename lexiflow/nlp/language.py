"""Detect the spoken language so the English-only analytics can step aside."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from . import multilingual

WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

ENGLISH_MARKERS: Set[str] = {
    "the", "and", "that", "have", "for", "not", "with", "you", "this", "but", "his", "from",
    "they", "say", "her", "she", "will", "one", "all", "would", "there", "their", "what",
    "about", "which", "when", "make", "like", "time", "just", "know", "take", "into", "your",
    "some", "could", "them", "than", "then", "look", "only", "come", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even",
    "want", "because", "these", "give", "day", "most", "need", "should", "does", "going",
}

DETECT_ONLY: Dict[str, Set[str]] = {
    "pl": {
        "nie", "jest", "się", "że", "tak", "ale", "jak", "czy", "już", "tylko", "bardzo",
        "dzisiaj", "jutro", "teraz", "praca", "problem", "dobrze", "dziękuję", "proszę",
        "musimy", "możesz", "trzeba", "wszystko", "jeszcze", "zawsze", "nigdy", "ponieważ",
    },
    "sv": {
        "och", "att", "det", "som", "för", "inte", "med", "har", "jag", "vi", "men", "kan",
        "ska", "idag", "imorgon", "arbete", "problem", "tack", "mycket", "alltid", "aldrig",
        "eftersom", "kanske", "behöver", "bara", "redan", "sedan", "hela",
    },
}

MARKERS: Dict[str, Set[str]] = {
    "en": ENGLISH_MARKERS,
    **{code: set(words) for code, words in multilingual.MARKERS.items()},
    **DETECT_ONLY,
}

DIACRITIC_HINTS: Dict[str, str] = {**multilingual.DIACRITICS, "pl": "ąćęłńóśźż", "sv": "åäö"}

ANALYTICS_LANGUAGES = frozenset({"en"}) | multilingual.SUPPORTED
MIN_TOKENS_FOR_CONFIDENCE = 4


@dataclass
class LanguageGuess:
    code: str
    confidence: float
    supported: bool
    scores: Dict[str, float]

    def as_dict(self) -> Dict[str, object]:
        return {
            "code": self.code,
            "confidence": round(self.confidence, 4),
            "supported": self.supported,
        }


def detect(text: str, default: str = "en") -> LanguageGuess:
    """Score stopword overlap plus diacritic hints; cheap and good enough for routing."""
    tokens = [match.group(0).lower() for match in WORD_RE.finditer(text or "")]
    if not tokens:
        return LanguageGuess(default, 0.0, default in ANALYTICS_LANGUAGES, {})

    lowered = (text or "").lower()
    scores: Dict[str, float] = {}
    for code, markers in MARKERS.items():
        hits = sum(1 for token in tokens if token in markers)
        score = hits / len(tokens)
        bonus = sum(0.04 for character in DIACRITIC_HINTS.get(code, "") if character in lowered)
        scores[code] = score + min(0.2, bonus)

    best = max(scores, key=lambda code: scores[code])
    ranked = sorted(scores.values(), reverse=True)
    margin = ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0)

    if len(tokens) < MIN_TOKENS_FOR_CONFIDENCE or scores[best] == 0.0:
        return LanguageGuess(default, 0.0, default in ANALYTICS_LANGUAGES, scores)

    confidence = min(1.0, scores[best] + margin)
    return LanguageGuess(best, confidence, best in ANALYTICS_LANGUAGES, scores)


class LanguageRouter:
    """Sticky detection: one noisy line should not flip the whole session."""

    def __init__(self, default: str = "en", window: int = 5, switch_margin: float = 0.15) -> None:
        self.default = default
        self.window = max(1, window)
        self.switch_margin = switch_margin
        self.current = default
        self._recent: List[LanguageGuess] = []

    def observe(self, text: str) -> LanguageGuess:
        guess = detect(text, self.default)
        self._recent.append(guess)
        if len(self._recent) > self.window:
            self._recent.pop(0)

        tally: Dict[str, float] = {}
        for item in self._recent:
            tally[item.code] = tally.get(item.code, 0.0) + item.confidence

        if tally:
            leader = max(tally, key=lambda code: tally[code])
            if leader != self.current:
                incumbent = tally.get(self.current, 0.0)
                if tally[leader] - incumbent >= self.switch_margin:
                    self.current = leader
        return LanguageGuess(
            self.current,
            guess.confidence,
            self.current in ANALYTICS_LANGUAGES,
            guess.scores,
        )

    def reset(self, code: Optional[str] = None) -> None:
        self.current = code or self.default
        self._recent.clear()
