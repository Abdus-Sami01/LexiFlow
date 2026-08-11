"""Phase 3 glue: one call turns a line of transcript into structured insight."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import NLPConfig
from .entities import Entity, EntityExtractor
from .rules import Extraction, RuleEngine
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

    @property
    def average_ms(self) -> float:
        return self.total_ms / self.processed if self.processed else 0.0


class AnalyticsEngine:
    """Rules, tiny model and arithmetic sentiment, in that order of priority."""

    def __init__(self, config: Optional[NLPConfig] = None) -> None:
        self.config = config or NLPConfig()
        self.rules = RuleEngine() if self.config.enable_rules else None
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
        }

    def digest(self, lines: List[str], audio_seconds: float = 0.0) -> ConversationDigest:
        return self.digests.build(
            lines, audio_seconds, topics=self.topics.shifts if self.topics else []
        )

    def analyse(self, text: str) -> Insight:
        started = time.perf_counter()
        cleaned = (text or "").strip()

        extractions = self.rules.extract(cleaned) if self.rules else []
        entities = self.entities.extract(cleaned)
        score = self.sentiment.score(cleaned) if self.sentiment else None
        shift = self.topics.push(cleaned) if self.topics and cleaned else None

        insight = Insight(
            text=cleaned,
            sentiment=score,
            entities=entities,
            extractions=extractions,
            rolling_sentiment=self.sentiment.rolling_average if self.sentiment else 0.0,
            sentiment_momentum=self.sentiment.momentum() if self.sentiment else 0.0,
            topic_shift=shift,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

        self.stats.processed += 1
        self.stats.total_ms += insight.elapsed_ms
        self.stats.action_items += len(insight.action_items)
        self.stats.entities += len(entities)
        self.stats.topic_shifts += 1 if shift else 0
        return insight
