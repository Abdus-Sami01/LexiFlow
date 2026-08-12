import numpy as np
import pytest

from lexiflow.asr.backends import ScriptedBackend
from lexiflow.audio.segmenter import SpeechSegment, split_on_speaker_change
from lexiflow.audio.speaker import SpeakerTracker, change_score, find_change_point
from lexiflow.config import LexiFlowConfig
from lexiflow.nlp.language import ANALYTICS_LANGUAGES, LanguageRouter, detect
from lexiflow.nlp.multilingual import lexicon_for, rules_for
from lexiflow.nlp.pipeline import AnalyticsEngine
from lexiflow.nlp.sentiment import SentimentEngine
from lexiflow.nlp.summarize import compress, stopwords_for, tokenize
from lexiflow.pipeline import LexiFlowPipeline

SAMPLE_RATE = 16_000

SAMPLES = {
    "en": "The deadline is Friday and I am worried we are going to slip again.",
    "es": "La fecha límite es el viernes y estoy muy preocupado por el retraso.",
    "fr": "La date limite est vendredi et je suis très inquiet du retard.",
    "de": "Die Frist ist Freitag und ich bin sehr besorgt wegen der Verzögerung.",
}


def voice(fundamental, seconds=2.0, seed=0):
    generator = np.random.default_rng(seed)
    times = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    harmonics = sum(np.sin(2 * np.pi * fundamental * k * times) / k for k in range(1, 12))
    return (harmonics * 0.2 + generator.normal(0, 0.01, times.size)).astype(np.float32)


@pytest.mark.parametrize("code", sorted(SAMPLES))
def test_detects_each_supported_language(code):
    guess = detect(SAMPLES[code])
    assert guess.code == code
    assert guess.supported is True
    assert guess.confidence > 0.0


def test_detection_falls_back_on_too_little_text():
    guess = detect("hola")
    assert guess.confidence == 0.0
    assert guess.code == "en"


def test_unsupported_language_is_flagged():
    guess = detect("Ik denk dat we het vandaag samen moeten doen omdat het niet werkt")
    assert guess.code == "nl"
    assert guess.supported is False
    assert "nl" not in ANALYTICS_LANGUAGES


def test_router_is_sticky_across_a_noisy_line():
    router = LanguageRouter()
    for _ in range(3):
        router.observe(SAMPLES["fr"])
    assert router.current == "fr"
    router.observe("ok")
    assert router.current == "fr"


def test_router_switches_when_the_language_really_changes():
    router = LanguageRouter()
    router.observe(SAMPLES["en"])
    for _ in range(4):
        router.observe(SAMPLES["de"])
    assert router.current == "de"


@pytest.mark.parametrize("code", ["es", "fr", "de"])
def test_multilingual_sentiment_has_both_polarities(code):
    engine = SentimentEngine(prefer_vader=False, language=code)
    negative = engine.score(SAMPLES[code], code)
    assert negative.compound < -0.2
    assert lexicon_for(code)


def test_multilingual_positive_sentiment():
    engine = SentimentEngine(prefer_vader=False, language="es")
    assert engine.score("El resultado es excelente y estoy muy contento", "es").compound > 0.4


@pytest.mark.parametrize("code", ["es", "fr", "de"])
def test_each_pack_has_rules(code):
    kinds = {spec.kind for spec in rules_for(code)}
    assert {"action_item", "deadline", "blocker", "decision"} <= kinds


def test_analytics_extracts_from_spanish():
    engine = AnalyticsEngine()
    insight = engine.analyse(
        "Recuérdame que envíe la hoja de precios a Sara antes del viernes."
    )
    assert insight.language == "es"
    assert insight.analytics_applied is True
    assert insight.action_items


def test_analytics_extracts_from_french_and_german():
    engine = AnalyticsEngine()
    french = engine.analyse("Je suis bloqué sur le pilote audio, c est un problème terrible.")
    assert french.language == "fr"
    assert any(item.kind == "blocker" for item in french.extractions)

    engine = AnalyticsEngine()
    german = engine.analyse("Wir haben entschieden, dass wir zuerst das Rewrite ausliefern.")
    assert german.language == "de"
    assert any(item.kind == "decision" for item in german.extractions)


def test_analytics_steps_aside_for_unsupported_languages():
    engine = AnalyticsEngine()
    text = "Ik denk dat we het vandaag samen moeten doen omdat het niet werkt en"
    insight = engine.analyse(text)
    assert insight.language == "nl"
    assert insight.analytics_applied is False
    assert insight.extractions == []
    assert insight.sentiment is None
    assert engine.stats.skipped_language == 1


def test_detection_can_be_switched_off():
    config = LexiFlowConfig()
    config.nlp.detect_language = False
    engine = AnalyticsEngine(config.nlp)
    insight = engine.analyse(SAMPLES["es"])
    assert insight.language == "en"
    assert insight.analytics_applied is True


def test_tokenizer_keeps_accented_words():
    assert "límite" in tokenize("la fecha límite es el viernes", language="es")
    assert "verzögerung" in tokenize("die Verzögerung ist ein Problem", language="de")


def test_language_stopwords_extend_the_english_set():
    spanish = stopwords_for("es")
    assert "the" in spanish and "porque" in spanish
    assert stopwords_for("en") == stopwords_for("en")


def test_compress_strips_filler_without_inventing_words():
    assert compress("So, you know, we basically need to ship it") == "We need to ship it"
    assert compress("The deadline is Friday.") == "The deadline is Friday."
    assert compress("") == ""


def test_change_point_found_between_two_voices():
    joined = np.concatenate([voice(110, 2.0, 1), voice(240, 2.0, 2)])
    index = find_change_point(joined)
    assert index is not None
    assert abs(index - SAMPLE_RATE * 2) <= SAMPLE_RATE * 0.3
    assert change_score(joined) > 0.3


def test_no_change_point_in_a_single_voice():
    assert find_change_point(voice(110, 4.0, 3)) is None
    assert change_score(voice(110, 4.0, 3)) < 0.1


def test_split_produces_two_segments():
    joined = np.concatenate([voice(110, 2.0, 1), voice(240, 2.0, 2)])
    segment = SpeechSegment(joined, SAMPLE_RATE, 100.0, 104.0, 0)
    parts = split_on_speaker_change(segment)
    assert len(parts) == 2
    assert parts[0].reason == "speaker_change"
    assert parts[0].ended_at == parts[1].started_at
    assert parts[0].duration + parts[1].duration == pytest.approx(segment.duration, abs=0.01)


def test_split_leaves_a_single_voice_alone():
    segment = SpeechSegment(voice(110, 4.0, 3), SAMPLE_RATE, 100.0, 104.0, 0)
    assert len(split_on_speaker_change(segment)) == 1


def test_split_ignores_short_segments():
    segment = SpeechSegment(voice(110, 1.0, 3), SAMPLE_RATE, 0.0, 1.0, 0)
    assert len(split_on_speaker_change(segment)) == 1


def test_speaker_enrolment_round_trips(tmp_path):
    tracker = SpeakerTracker()
    tracker.assign(voice(110, 2.0, 1))
    label = tracker.profiles()[0].label
    assert tracker.rename(label, "Amara") is True
    assert tracker.rename("nobody", "x") is False

    path = tracker.save(tmp_path / "voices.json")
    restored = SpeakerTracker()
    assert restored.load(path) == 1
    assert restored.profiles()[0].label == "Amara"

    again = restored.assign(voice(111, 2.0, 9))
    assert again.label == "Amara"
    assert again.is_new is False


def test_loading_missing_or_broken_profiles_is_safe(tmp_path):
    tracker = SpeakerTracker()
    assert tracker.load(tmp_path / "absent.json") == 0
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert tracker.load(broken) == 0


def test_pipeline_splits_and_renames(tmp_path):
    config = LexiFlowConfig()
    config.state.database_path = tmp_path / "split.db"
    config.diarization.profile_path = tmp_path / "voices.json"
    config.segmenter.min_segment_seconds = 0.4
    config.segmenter.silence_hangover_seconds = 0.3
    config.segmenter.max_segment_seconds = 30.0
    config.segmenter.emit_partials = False
    config.asr.warmup = False

    backend = ScriptedBackend(["first half", "second half"], config.asr)
    pipeline = LexiFlowPipeline(config, backend=backend)
    pipeline.start(open_microphone=False)
    pipeline.feed(np.concatenate([voice(110, 2.5, 1), voice(240, 2.5, 2)]))
    pipeline.feed(np.zeros(SAMPLE_RATE, dtype=np.float32))
    assert pipeline.drain(timeout=15.0) is True
    pipeline.stop()

    health = pipeline.health()
    assert health.speaker_splits == 1
    assert health.speakers == 2
    assert len(pipeline.store.transcript()) == 2

    first = pipeline.store.transcript()[0].speaker
    assert pipeline.rename_speaker(first, "Amara") is True
    assert pipeline.store.transcript()[0].speaker == "Amara"
    assert (tmp_path / "voices.json").is_file()
    pipeline.close()


def test_partial_gate_closes_when_inference_falls_behind(tmp_path):
    config = LexiFlowConfig()
    config.state.database_path = tmp_path / "gate.db"
    config.asr.warmup = False
    pipeline = LexiFlowPipeline(config, backend=ScriptedBackend([], config.asr))

    assert pipeline._partial_gate() is True
    pipeline.transcription.stats.audio_seconds = 10.0
    pipeline.transcription.stats.inference_seconds = 9.0
    assert pipeline._partial_gate() is False
    assert pipeline.health().keeping_up is False

    pipeline.transcription.stats.inference_seconds = 1.0
    assert pipeline._partial_gate() is True
    pipeline.close()


def test_partial_gate_respects_the_config_switch(tmp_path):
    config = LexiFlowConfig()
    config.state.database_path = tmp_path / "gate2.db"
    config.segmenter.emit_partials = False
    config.asr.warmup = False
    pipeline = LexiFlowPipeline(config, backend=ScriptedBackend([], config.asr))
    assert pipeline._partial_gate() is False
    pipeline.close()
