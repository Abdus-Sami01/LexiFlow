"""Phase 3 glue: one call turns a line of transcript into structured insight."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import NLPConfig
from .entities import Entity, EntityExtractor
from .language import ANALYTICS_LANGUAGES, LanguageGuess, LanguageRouter
from .multilingual import rules_for
from .rules import DEFAULT_RULES, Extraction, RuleEngine
from .sentiment import SentimentEngine, SentimentScore
from .summarize import ConversationDigest, DigestBuilder, TopicShift, TopicTracker


@dataclass
class Insight:
    """Everything the analytics layer knows about a single utterance."""

    text: str
    sentiment: Optional[SentimentScore] = None
    entities: List[Entity] = field(default_factory=list)
    extractions: List[Extraction] = field(default_factory=list)
    rolling_sentiment: float = 0.0
    sentiment_momentum: float = 0.0
    language: str = "en"
    language_confidence: float = 0.0
    analytics_applied: bool = True
    topic_shift: Optional[TopicShift] = None
    elapsed_ms: float = 0.0
    created_at: float = field(default_factory=time.time)

    @property
    def action_items(self) -> List[Extraction]:
        return [item for item in self.extractions if item.kind == "action_item"]

    @property
    def deadlines(self) -> List[Extraction]:
        return [item for item in self.extractions if item.kind == "deadline"]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "sentiment": self.sentiment.as_dict() if self.sentiment else None,
            "entities": [item.as_dict() for item in self.entities],
            "extractions": [item.as_dict() for item in self.extractions],
            "rolling_sentiment": self.rolling_sentiment,
            "sentiment_momentum": self.sentiment_momentum,
            "language": self.language,
            "language_confidence": self.language_confidence,
            "analytics_applied": self.analytics_applied,
            "topic_shift": self.topic_shift.as_dict() if self.topic_shift else None,
            "elapsed_ms": self.elapsed_ms,
            "created_at": self.created_at,
        }


@dataclass
class AnalyticsStats:
    processed: int = 0
    total_ms: float = 0.0
    action_items: int = 0
    entities: int = 0
    topic_shifts: int = 0
    skipped_language: int = 0

    @property
    def average_ms(self) -> float:
        return self.total_ms / self.processed if self.processed else 0.0


class AnalyticsEngine:
    """Rules, tiny model and arithmetic sentiment, in that order of priority."""

    def __init__(self, config: Optional[NLPConfig] = None) -> None:
        self.config = config or NLPConfig()
        self.rules = RuleEngine() if self.config.enable_rules else None
        self._rule_engines: Dict[str, RuleEngine] = {"en": self.rules} if self.rules else {}
        self.languages = (
            LanguageRouter(default=self.config.default_language)
            if self.config.detect_language
            else None
        )
        self.entities = EntityExtractor(self.config.spacy_model, self.config.enable_spacy)
        self.sentiment = (
            SentimentEngine(window=self.config.sentiment_window)
            if self.config.enable_sentiment
            else None
        )
        self.topics = (
            TopicTracker(window=self.config.topic_window, threshold=self.config.topic_threshold)
            if self.config.enable_topics
            else None
        )
        self.digests = DigestBuilder(
            summary_limit=self.config.summary_sentences,
            keyphrase_limit=self.config.keyphrase_limit,
        )
        self.stats = AnalyticsStats()

    @property
    def backends(self) -> Dict[str, str]:
        return {
            "rules": "enabled" if self.rules else "disabled",
            "entities": self.entities.backend,
            "sentiment": self.sentiment.engine_name if self.sentiment else "disabled",
            "topics": "enabled" if self.topics else "disabled",
            "language": self.languages.current if self.languages else "en (fixed)",
        }

    def _rules_for(self, language: str) -> Optional[RuleEngine]:
        """English rules always apply; a supported language adds its own pack on top."""
        if not self.config.enable_rules:
            return None
        if language not in self._rule_engines:
            extra = rules_for(language)
            if not extra:
                return self._rule_engines.get("en")
            self._rule_engines[language] = RuleEngine(list(DEFAULT_RULES) + extra)
        return self._rule_engines[language]

    def digest(self, lines: List[str], audio_seconds: float = 0.0) -> ConversationDigest:
        return self.digests.build(
            lines,
            audio_seconds,
            topics=self.topics.shifts if self.topics else [],
            language=self.languages.current if self.languages else "en",
        )

    def analyse(self, text: str) -> Insight:
        started = time.perf_counter()
        cleaned = (text or "").strip()

        guess = (
            self.languages.observe(cleaned)
            if self.languages and cleaned
            else LanguageGuess(self.config.default_language, 0.0, True, {})
        )
        supported = guess.code in ANALYTICS_LANGUAGES

        rules = self._rules_for(guess.code) if supported else None
        extractions = rules.extract(cleaned) if rules else []
        entities = self.entities.extract(cleaned)
        score = (
            self.sentiment.score(cleaned, guess.code) if self.sentiment and supported else None
        )
        shift = self.topics.push(cleaned, guess.code) if self.topics and cleaned else None

        insight = Insight(
            text=cleaned,
            sentiment=score,
            entities=entities,
            extractions=extractions,
            rolling_sentiment=self.sentiment.rolling_average if self.sentiment else 0.0,
            sentiment_momentum=self.sentiment.momentum() if self.sentiment else 0.0,
            topic_shift=shift,
            language=guess.code,
            language_confidence=guess.confidence,
            analytics_applied=supported,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

        self.stats.processed += 1
        self.stats.total_ms += insight.elapsed_ms
        self.stats.action_items += len(insight.action_items)
        self.stats.entities += len(entities)
        self.stats.topic_shifts += 1 if shift else 0
        self.stats.skipped_language += 0 if supported else 1
        return insight
