import time

import numpy as np
import pytest

from lexiflow.asr.backends import ScriptedBackend
from lexiflow.config import LexiFlowConfig, NLPConfig, TranslationConfig
from lexiflow.nlp import translate as translation_module
from lexiflow.nlp.pipeline import AnalyticsEngine
from lexiflow.nlp.translate import (
    IdentityTranslator,
    TranslationEngine,
    Translator,
    available_translators,
    create_translator,
    report,
)
from lexiflow.pipeline import LexiFlowPipeline

SAMPLE_RATE = 16_000
POLISH = "Nie wiem czy to jest dobrze ale musimy to zrobić dzisiaj ponieważ nie działa"

DICTIONARY = {
    ("es", "en"): {
        "la fecha límite es el viernes": "the deadline is Friday",
        "hola mundo": "hello world",
        "recuérdame que envíe el informe": "remind me to send the report",
    },
    ("en", "es"): {"hello world": "hola mundo"},
    ("pl", "en"): {
        "nie wiem czy to jest dobrze ale musimy to zrobić dzisiaj ponieważ nie działa": (
            "remind me to finish the report before Friday"
        ),
        "nie wiem czy to jest dobrze": "remind me to email finance on Friday",
    },
}


class DictionaryTranslator(Translator):
    """A deterministic stand-in so the wiring can be tested without model weights."""

    name = "dictionary"
    priority = 5

    def __init__(self, config=None):
        super().__init__(config)
        self.calls = 0

    @classmethod
    def is_available(cls):
        return True

    def supports(self, source, target):
        return source == target or (source, target) in DICTIONARY

    def translate(self, text, source, target):
        self.calls += 1
        if source == target:
            return text
        return DICTIONARY[(source, target)][text.lower()]


def voice(fundamental, seconds=2.0, seed=0):
    generator = np.random.default_rng(seed)
    times = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    harmonics = sum(np.sin(2 * np.pi * fundamental * k * times) / k for k in range(1, 12))
    return (harmonics * 0.2 + generator.normal(0, 0.01, times.size)).astype(np.float32)


def enabled(**overrides):
    return TranslationConfig(enabled=True, **overrides)


def test_identity_translator_is_always_present():
    assert "identity" in available_translators()
    assert IdentityTranslator().supports("en", "en") is True
    assert IdentityTranslator().supports("es", "en") is False


def test_create_translator_honours_an_explicit_backend():
    assert create_translator(TranslationConfig(backend="identity")).name == "identity"
    with pytest.raises(translation_module.TranslationUnavailable):
        create_translator(TranslationConfig(backend="nope"))


def test_report_describes_the_local_setup():
    payload = report().as_dict()
    assert "backend" in payload and isinstance(payload["available"], list)


def test_engine_translates_and_caches():
    translator = DictionaryTranslator()
    engine = TranslationEngine(enabled(), translator)

    first = engine.translate("la fecha límite es el viernes", "es")
    assert first.text == "the deadline is Friday"
    assert first.cached is False
    assert first.engine == "dictionary"

    second = engine.translate("la fecha límite es el viernes", "es")
    assert second.cached is True
    assert translator.calls == 1
    assert engine.translated == 1


def test_engine_skips_same_language_and_empty_text():
    engine = TranslationEngine(enabled(), DictionaryTranslator())
    assert engine.translate("hello", "en") is None
    assert engine.translate("   ", "es") is None
    assert engine.translate("hola mundo", "es", "es") is None


def test_engine_reports_failure_instead_of_raising():
    engine = TranslationEngine(enabled(), DictionaryTranslator())
    assert engine.translate("desconocido", "es") is None
    assert engine.failures == 1


def test_engine_survives_a_translator_that_throws():
    class Broken(DictionaryTranslator):
        def supports(self, source, target):
            return True

        def translate(self, text, source, target):
            raise RuntimeError("model exploded")

    engine = TranslationEngine(enabled(), Broken())
    assert engine.translate("hola mundo", "es") is None
    assert engine.failures == 1


def test_engine_respects_a_non_english_target():
    engine = TranslationEngine(enabled(target_language="es"), DictionaryTranslator())
    result = engine.translate("hello world", "en")
    assert result.text == "hola mundo"
    assert result.target == "es"


def test_engine_cache_is_bounded():
    engine = TranslationEngine(enabled(cache_size=1), DictionaryTranslator())
    engine.translate("hola mundo", "es")
    engine.translate("la fecha límite es el viernes", "es")
    assert engine.translated == 2
    assert engine.stats()["engine"] == "dictionary"


def test_analytics_translates_a_supported_language_for_display():
    engine = AnalyticsEngine(NLPConfig(), enabled())
    engine.translator = TranslationEngine(enabled(), DictionaryTranslator())
    insight = engine.analyse("Recuérdame que envíe el informe")
    assert insight.language == "es"
    assert insight.analysed_language == "es"
    assert insight.translation == "remind me to send the report"
    assert insight.action_items


def test_analytics_falls_back_to_the_translation_for_unsupported_languages():
    engine = AnalyticsEngine(NLPConfig(), enabled())
    engine.translator = TranslationEngine(enabled(), DictionaryTranslator())
    insight = engine.analyse(POLISH)

    assert insight.language == "pl"
    assert insight.analysed_language == "en"
    assert insight.analytics_applied is True
    assert insight.action_items
    assert engine.stats.analysed_via_translation == 1


def test_analytics_prefers_a_translation_the_asr_already_made():
    translator = DictionaryTranslator()
    engine = AnalyticsEngine(NLPConfig(), enabled())
    engine.translator = TranslationEngine(enabled(), translator)
    insight = engine.analyse(
        POLISH, translation="remind me to email finance on Friday"
    )
    assert insight.translation_engine == "whisper"
    assert translator.calls == 0
    assert insight.action_items


def test_analytics_leaves_unsupported_languages_alone_when_disabled():
    config = TranslationConfig(enabled=True, analyse_translation=False)
    engine = AnalyticsEngine(NLPConfig(), config)
    engine.translator = TranslationEngine(enabled(), DictionaryTranslator())
    insight = engine.analyse(POLISH)
    assert insight.analytics_applied is False
    assert insight.translation is not None


def test_backends_report_translation_state():
    engine = AnalyticsEngine(NLPConfig(), enabled())
    assert engine.backends["translation"] != "disabled"
    assert AnalyticsEngine().backends["translation"] == "disabled"


def test_scripted_backend_serves_the_translate_task():
    backend = ScriptedBackend(["hola mundo"], translations=["hello world"]).load()
    audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    assert backend.supports_translation is True
    assert backend.transcribe(audio).text == "hola mundo"
    assert backend.transcribe(audio, task="translate").text == "hello world"


def test_backend_without_translations_declines():
    backend = ScriptedBackend(["only this"]).load()
    assert backend.supports_translation is False


def test_pipeline_stores_speech_translations(tmp_path):
    config = LexiFlowConfig()
    config.state.database_path = tmp_path / "translate.db"
    config.translation.enabled = True
    config.translation.target_language = "en"
    config.segmenter.min_segment_seconds = 0.4
    config.segmenter.silence_hangover_seconds = 0.3
    config.segmenter.emit_partials = False
    config.asr.warmup = False
    config.asr.language = "es"

    backend = ScriptedBackend(
        ["la fecha límite es el viernes"],
        config.asr,
        translations=["remind me to send the report before Friday"],
    )
    pipeline = LexiFlowPipeline(config, backend=backend)
    pipeline.start(open_microphone=False)
    pipeline.feed(voice(150, seconds=2.0, seed=4))
    pipeline.feed(np.zeros(SAMPLE_RATE, dtype=np.float32))
    assert pipeline.drain(timeout=15.0) is True
    pipeline.stop()

    item = pipeline.store.transcript()[0]
    assert item.text.startswith("la fecha")
    assert item.translation == "remind me to send the report before Friday"
    assert pipeline.health().speech_translations == 1
    assert pipeline.snapshot()["transcript"][0]["translation"]
    pipeline.close()


def test_pipeline_leaves_translation_off_by_default(tmp_path):
    config = LexiFlowConfig()
    config.state.database_path = tmp_path / "plain.db"
    config.asr.warmup = False
    pipeline = LexiFlowPipeline(config, backend=ScriptedBackend([], config.asr))
    assert pipeline.analytics.translator is None
    assert pipeline.health().speech_translations == 0
    pipeline.close()


def test_translation_survives_a_round_trip_through_sqlite(tmp_path):
    config = LexiFlowConfig()
    config.state.database_path = tmp_path / "round.db"
    config.translation.enabled = True
    config.asr.warmup = False
    pipeline = LexiFlowPipeline(config, backend=ScriptedBackend([], config.asr))
    pipeline.analytics.translator = TranslationEngine(enabled(), DictionaryTranslator())

    pipeline.submit_text("La fecha límite es el viernes")
    time.sleep(0.05)
    rows = pipeline.store.load_session(pipeline.store.session_id)
    assert rows[0]["translation"] == "the deadline is Friday"
    pipeline.close()


class TranslatedRow:
    def __init__(self, text, translation, started_at, ended_at, speaker=None):
        self.text = text
        self.translation = translation
        self.started_at = started_at
        self.ended_at = ended_at
        self.speaker = speaker
        self.spans = []


def test_subtitles_can_carry_the_translation():
    from lexiflow import export

    rows = [TranslatedRow("hola mundo", "hello world", 100.0, 102.0, "Speaker A")]
    original = export.to_srt(rows)
    translated = export.to_srt(rows, translated=True)
    assert "hola mundo" in original and "hello world" not in original
    assert "hello world" in translated
    assert "00:00:00,000 --> 00:00:02,000" in translated


def test_subtitles_fall_back_when_a_line_was_not_translated():
    from lexiflow import export

    rows = [TranslatedRow("solo esto", None, 10.0, 12.0)]
    assert "solo esto" in export.to_srt(rows, translated=True)


def test_markdown_shows_the_translation_under_the_original():
    from lexiflow import export

    payload = {
        "session": {"name": "s"},
        "transcript": [{"text": "hola mundo", "translation": "hello world"}],
        "actions": [],
    }
    body = export.to_markdown(payload)
    assert "- hola mundo" in body
    assert "  - _hello world_" in body
