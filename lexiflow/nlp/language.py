"""Detect the spoken language so the English-only analytics can step aside."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

MARKERS: Dict[str, Set[str]] = {
    "en": {
        "the", "and", "that", "have", "for", "not", "with", "you", "this", "but", "his", "from",
        "they", "say", "her", "she", "will", "one", "all", "would", "there", "their", "what",
        "about", "which", "when", "make", "like", "time", "just", "know", "take", "into", "your",
        "some", "could", "them", "than", "then", "look", "only", "come", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even",
        "want", "because", "these", "give", "day", "most", "need", "should", "does", "going",
    },
    "es": {
        "que", "de", "no", "la", "el", "es", "en", "lo", "un", "por", "qué", "una", "los", "con",
        "para", "está", "esto", "del", "las", "muy", "más", "pero", "todo", "bien", "sí", "aquí",
        "ahora", "cuando", "porque", "hacer", "puede", "tiene", "vamos", "también", "hasta",
        "desde", "sobre", "entre", "nosotros", "ustedes", "ellos", "este", "esa", "ese", "cómo",
        "dónde", "quién", "gracias", "señor", "nada", "algo", "otro", "tiempo", "año", "día",
    },
    "fr": {
        "que", "de", "je", "est", "pas", "le", "vous", "la", "tu", "il", "et", "les", "des", "en",
        "un", "une", "ce", "qui", "nous", "sur", "pour", "dans", "avec", "mais", "tout", "plus",
        "bien", "être", "avoir", "faire", "comme", "aussi", "très", "quand", "parce", "alors",
        "donc", "chose", "temps", "jour", "année", "merci", "oui", "non", "peut", "cette", "ces",
        "leur", "sans", "sous", "entre", "après", "avant", "encore", "toujours", "jamais",
    },
    "de": {
        "der", "die", "und", "ich", "das", "nicht", "sie", "ist", "es", "den", "zu", "wir", "mit",
        "ein", "eine", "auf", "für", "aber", "auch", "als", "war", "hat", "dass", "sich", "von",
        "dem", "noch", "wie", "über", "nur", "muss", "kann", "sehr", "schon", "immer", "jetzt",
        "hier", "dann", "weil", "wenn", "oder", "mehr", "einen", "seine", "ihre", "unser",
        "danke", "bitte", "heute", "morgen", "jahr", "zeit", "arbeit", "machen", "haben",
    },
    "it": {
        "che", "di", "non", "il", "la", "un", "per", "sono", "una", "mi", "con", "ma", "come",
        "questo", "bene", "più", "anche", "molto", "quando", "perché", "cosa", "tutto", "solo",
        "adesso", "grazie", "sì", "noi", "loro", "essere", "fare", "avere", "dove", "chi",
        "tempo", "anno", "giorno", "lavoro", "sempre", "mai", "ancora", "dopo", "prima",
    },
    "pt": {
        "que", "não", "de", "para", "com", "uma", "você", "por", "mais", "isso", "está", "muito",
        "como", "mas", "quando", "porque", "então", "aqui", "agora", "obrigado", "sim", "nós",
        "eles", "fazer", "ter", "ser", "tempo", "ano", "dia", "trabalho", "sempre", "nunca",
        "ainda", "depois", "antes", "tudo", "nada", "algo", "outro", "esse", "essa",
    },
    "nl": {
        "de", "het", "een", "ik", "je", "niet", "dat", "en", "van", "is", "we", "op", "voor",
        "met", "maar", "ook", "als", "zijn", "hebben", "worden", "kunnen", "moeten", "heel",
        "altijd", "nooit", "vandaag", "morgen", "jaar", "tijd", "werk", "dank", "graag",
        "omdat", "wanneer", "waar", "wie", "hoe", "nog", "alleen", "samen",
    },
}

DIACRITIC_HINTS: Dict[str, str] = {
    "es": "ñáéíóúü¿¡",
    "fr": "àâçéèêëîïôûùüÿœ",
    "de": "äöüß",
    "pt": "ãõáâçéêíóôú",
    "it": "àèéìòù",
    "nl": "ëïĳ",
}

ANALYTICS_LANGUAGES = frozenset({"en", "es", "fr", "de", "it", "pt"})
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
