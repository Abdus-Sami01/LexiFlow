from lexiflow.nlp.pipeline import AnalyticsEngine
from lexiflow.nlp.rules import RuleEngine, find_due_date, infer_priority
from lexiflow.nlp.sentiment import LexiconSentimentAnalyzer, SentimentEngine


def _texts(extractions, kind):
    return [item.text for item in extractions if item.kind == kind]


def test_reminder_rule():
    engine = RuleEngine()
    hits = engine.extract("Remind me to send the invoice to Sarah tomorrow.")
    assert "send the invoice to Sarah tomorrow" in _texts(hits, "action_item")


def test_deadline_rule_and_due_date():
    engine = RuleEngine()
    hits = engine.extract("The deadline is Friday, no excuses.")
    deadlines = [item for item in hits if item.kind == "deadline"]
    assert deadlines
    assert deadlines[0].due.lower() == "friday"


def test_blocker_and_decision_rules():
    engine = RuleEngine()
    assert _texts(engine.extract("I am blocked on the audio driver."), "blocker")
    assert _texts(engine.extract("We decided to ship the rewrite first."), "decision")


def test_priority_inference():
    assert infer_priority("this is urgent") == 3
    assert infer_priority("important cleanup") == 2
    assert infer_priority("plain task") == 1


def test_find_due_date_variants():
    assert find_due_date("do it by 5pm")
    assert find_due_date("ship by 2026-04-20")
    assert find_due_date("end of week") == "end of week"
    assert find_due_date("no temporal marker") is None


def test_lexicon_sentiment_polarity():
    analyzer = LexiconSentimentAnalyzer()
    assert analyzer.polarity_scores("This is absolutely fantastic work").compound > 0.4
    assert analyzer.polarity_scores("This is a terrible, broken disaster").compound < -0.4
    assert analyzer.polarity_scores("the meeting is at three").compound == 0.0


def test_lexicon_sentiment_negation_flips_sign():
    analyzer = LexiconSentimentAnalyzer()
    positive = analyzer.polarity_scores("the build is good").compound
    negated = analyzer.polarity_scores("the build is not good").compound
    assert positive > 0 > negated


def test_sentiment_engine_tracks_momentum():
    engine = SentimentEngine(prefer_vader=False, window=6)
    for line in ["awful", "terrible", "bad", "great", "fantastic", "wonderful"]:
        engine.score(line)
    assert engine.momentum() > 0
    assert len(engine.history()) == 6


def test_analytics_engine_end_to_end():
    engine = AnalyticsEngine()
    insight = engine.analyse(
        "Remind me to email Sarah Chen at Northwind Systems, the deadline is Friday."
    )
    assert insight.action_items
    assert insight.deadlines
    assert insight.entities
    assert insight.elapsed_ms >= 0.0
    assert engine.stats.processed == 1


def test_analytics_engine_handles_empty_input():
    engine = AnalyticsEngine()
    insight = engine.analyse("   ")
    assert insight.text == ""
    assert insight.extractions == []
