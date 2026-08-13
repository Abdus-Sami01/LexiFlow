import json

import pytest

from lexiflow.cli import main
from lexiflow.config import LexiFlowConfig


def test_defaults_are_valid():
    assert LexiFlowConfig().validate() == []


def test_sample_rate_must_suit_whisper():
    config = LexiFlowConfig()
    config.audio.target_sample_rate = 44_100
    assert any("16000" in problem for problem in config.validate())


@pytest.mark.parametrize(
    "section,field,value,fragment",
    [
        ("audio", "block_duration_ms", 0, "must be positive"),
        ("audio", "ring_buffer_seconds", -1, "must be positive"),
        ("asr", "beam_size", 0, "beam_size"),
        ("asr", "threads", -2, "cannot be negative"),
        ("asr", "max_queue_size", 0, "max_queue_size"),
        ("asr", "max_realtime_factor", 0, "max_realtime_factor"),
        ("nlp", "summary_sentences", 0, "summary_sentences"),
        ("nlp", "topic_threshold", 2.0, "between 0 and 1"),
        ("nlp", "default_language", "zz", "no rule pack"),
        ("diarization", "similarity_threshold", 1.5, "between 0 and 1"),
        ("diarization", "max_speakers", 0, "at least 1"),
        ("diarization", "adaptation_rate", 0.0, "between 0 and 1"),
        ("translation", "cache_size", 0, "cache_size"),
        ("redaction", "mode", "nope", "unknown"),
        ("state", "max_transcript_items", 0, "max_transcript_items"),
        ("segmenter", "min_band_ratio", 3.0, "between 0 and 1"),
    ],
)
def test_each_bad_setting_is_named(section, field, value, fragment):
    config = LexiFlowConfig()
    setattr(getattr(config, section), field, value)
    problems = config.validate()
    assert any(fragment in problem for problem in problems), problems


def test_segment_bounds_must_make_sense():
    config = LexiFlowConfig()
    config.segmenter.min_segment_seconds = 30.0
    assert any("max_segment_seconds" in problem for problem in config.validate())


def test_partials_longer_than_a_segment_are_flagged():
    config = LexiFlowConfig()
    config.segmenter.partial_min_seconds = 99.0
    assert any("partial" in problem for problem in config.validate())


def test_translation_needs_a_target():
    config = LexiFlowConfig()
    config.translation.enabled = True
    config.translation.target_language = ""
    assert any("target_language" in problem for problem in config.validate())


def test_redaction_needs_kinds():
    config = LexiFlowConfig()
    config.redaction.enabled = True
    config.redaction.kinds = ()
    assert any("kinds" in problem for problem in config.validate())


def test_save_and_load_round_trip(tmp_path):
    config = LexiFlowConfig()
    config.translation.enabled = True
    config.redaction.kinds = ("email", "person")
    target = config.save(tmp_path / "nested" / "lexiflow.json")

    restored = LexiFlowConfig.load(target)
    assert restored.translation.enabled is True
    assert restored.redaction.kinds == ("email", "person")
    assert restored.state.database_path == config.state.database_path


def test_loading_an_invalid_file_explains_why(tmp_path):
    target = tmp_path / "broken.json"
    payload = LexiFlowConfig().to_dict()
    payload["asr"]["beam_size"] = 0
    target.write_text(json.dumps(payload))

    with pytest.raises(ValueError) as caught:
        LexiFlowConfig.load(target)
    assert "beam_size" in str(caught.value)
    assert LexiFlowConfig.load(target, strict=False).asr.beam_size == 0


def test_unknown_keys_are_ignored_rather_than_fatal(tmp_path):
    target = tmp_path / "extra.json"
    payload = LexiFlowConfig().to_dict()
    payload["asr"]["invented_setting"] = True
    payload["not_a_section"] = {"x": 1}
    target.write_text(json.dumps(payload))
    assert LexiFlowConfig.load(target).asr.beam_size == 1


def test_init_writes_a_usable_config(tmp_path, capsys):
    target = tmp_path / "lexiflow.json"
    assert main(["init", str(target), "--translate", "--redact"]) == 0
    assert "wrote" in capsys.readouterr().out

    config = LexiFlowConfig.load(target)
    assert config.translation.enabled is True
    assert config.redaction.enabled is True


def test_init_refuses_to_clobber(tmp_path, capsys):
    target = tmp_path / "lexiflow.json"
    target.write_text("{}")
    assert main(["init", str(target)]) == 1
    assert "--force" in capsys.readouterr().err
    assert main(["init", str(target), "--force"]) == 0


def test_validate_command_reports_both_outcomes(tmp_path, capsys):
    good = tmp_path / "good.json"
    LexiFlowConfig().save(good)
    assert main(["validate", str(good)]) == 0
    assert "is valid" in capsys.readouterr().out

    bad = tmp_path / "bad.json"
    payload = LexiFlowConfig().to_dict()
    payload["diarization"]["max_speakers"] = 0
    bad.write_text(json.dumps(payload))
    assert main(["validate", str(bad)]) == 1
    assert "max_speakers" in capsys.readouterr().err


def test_a_bad_config_stops_a_command_early(tmp_path):
    bad = tmp_path / "bad.json"
    payload = LexiFlowConfig().to_dict()
    payload["asr"]["beam_size"] = 0
    bad.write_text(json.dumps(payload))

    with pytest.raises(SystemExit) as caught:
        main(["--config", str(bad), "demo", "--no-persist"])
    assert "beam_size" in str(caught.value)
