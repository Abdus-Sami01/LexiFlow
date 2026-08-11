"""Phase 3: zero-cost NLP analytics."""

from .entities import Entity, EntityExtractor
from .pipeline import AnalyticsEngine, AnalyticsStats, Insight
from .rules import DEFAULT_RULES, Extraction, RuleEngine, RuleSpec, find_due_date, infer_priority
from .sentiment import LexiconSentimentAnalyzer, SentimentEngine, SentimentScore, label_for

__all__ = [
    "AnalyticsEngine",
    "AnalyticsStats",
    "DEFAULT_RULES",
    "Entity",
    "EntityExtractor",
    "Extraction",
    "Insight",
    "LexiconSentimentAnalyzer",
    "RuleEngine",
    "RuleSpec",
    "SentimentEngine",
    "SentimentScore",
    "find_due_date",
    "infer_priority",
    "label_for",
]
