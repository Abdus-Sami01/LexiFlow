"""Phase 3 glue: one call turns a line of transcript into structured insight."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import NLPConfig, TranslationConfig
from .entities import Entity, EntityExtractor
from .language import ANALYTICS_LANGUAGES, LanguageGuess, LanguageRouter
from .multilingual import rules_for
from .rules import DEFAULT_RULES, Extraction, RuleEngine
from .sentiment import SentimentEngine, SentimentScore
from .summarize import ConversationDigest, DigestBuilder, TopicShift, TopicTracker
from .translate import TranslationEngine


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
    analysed_language: str = "en"
    translation: Optional[str] = None
    translation_engine: Optional[str] = None
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
            "analysed_language": self.analysed_language,
            "translation": self.translation,
            "translation_engine": self.translation_engine,
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
    analysed_via_translation: int = 0

    @property
    def average_ms(self) -> float:
        return self.total_ms / self.processed if self.processed else 0.0


class AnalyticsEngine:
    """Rules, tiny model and arithmetic sentiment, in that order of priority."""

    def __init__(
        self,
        config: Optional[NLPConfig] = None,
        translation: Optional[TranslationConfig] = None,
    ) -> None:
        self.config = config or NLPConfig()
        self.translation_config = translation or TranslationConfig()
        self.translator = (
            TranslationEngine(self.translation_config) if self.translation_config.enabled else None
        )
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
            "translation": self.translator.engine_name if self.translator else "disabled",
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

    def _translate(
        self, text: str, language: str, provided: Optional[str]
    ) -> tuple[Optional[str], Optional[str]]:
        """Prefer a translation the ASR already produced over paying for another pass."""
        if not text or not self.translation_config.enabled:
            return None, None
        if provided:
            return provided, "whisper"
        if self.translator is None:
            return None, None
        result = self.translator.translate(text, language)
        return (result.text, result.engine) if result else (None, None)

    def _pick_subject(
        self, text: str, language: str, supported: bool, translation: Optional[str]
    ) -> tuple[Optional[str], str, bool]:
        """Analyse the original when we can, the translation when we cannot."""
        if supported:
            return text, language, False
        target = self.translation_config.target_language
        if (
            self.translation_config.enabled
            and self.translation_config.analyse_translation
            and translation
            and target in ANALYTICS_LANGUAGES
        ):
            return translation, target, True
        return None, language, False

    def digest(self, lines: List[str], audio_seconds: float = 0.0) -> ConversationDigest:
        return self.digests.build(
            lines,
            audio_seconds,
            topics=self.topics.shifts if self.topics else [],
            language=self.languages.current if self.languages else "en",
        )

    def analyse(self, text: str, translation: Optional[str] = None) -> Insight:
        started = time.perf_counter()
        cleaned = (text or "").strip()

        guess = (
            self.languages.observe(cleaned)
            if self.languages and cleaned
            else LanguageGuess(self.config.default_language, 0.0, True, {})
        )
        supported = guess.code in ANALYTICS_LANGUAGES
        rendered, engine = self._translate(cleaned, guess.code, translation)

        subject, analysed_language, via_translation = self._pick_subject(
            cleaned, guess.code, supported, rendered
        )

        rules = self._rules_for(analysed_language) if subject is not None else None
        extractions = rules.extract(subject) if rules else []
        entities = self.entities.extract(subject or cleaned)
        score = (
            self.sentiment.score(subject, analysed_language)
            if self.sentiment and subject is not None
            else None
        )
        shift = (
            self.topics.push(subject, analysed_language) if self.topics and subject else None
        )

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
            analytics_applied=subject is not None,
            analysed_language=analysed_language,
            translation=rendered,
            translation_engine=engine,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

        self.stats.processed += 1
        self.stats.total_ms += insight.elapsed_ms
        self.stats.action_items += len(insight.action_items)
        self.stats.entities += len(entities)
        self.stats.topic_shifts += 1 if shift else 0
        self.stats.skipped_language += 0 if subject is not None else 1
        self.stats.analysed_via_translation += 1 if via_translation else 0
        return insight
