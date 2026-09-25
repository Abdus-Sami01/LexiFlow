import json

import pytest

from lexiflow import selftest
from lexiflow.asr.backends import BackendUnavailable, FasterWhisperBackend
from lexiflow.audio.speaker import find_change_point
from lexiflow.cli import main
from lexiflow.config import ASRConfig, LexiFlowConfig


@pytest.fixture()
def settings(tmp_path):
    config = LexiFlowConfig()
    config.state.database_path = tmp_path / "selftest.db"
    config.asr.backend = "null"
    config.asr.warmup = False
    config.segmenter.min_segment_seconds = 0.4
    config.segmenter.silence_hangover_seconds = 0.3
    return config


def test_reference_audio_holds_two_distinct_voices():
    audio = selftest.two_speaker_audio()
    assert audio.dtype.name == "float32"
    assert audio.size == int(16_000 * 5.6)
    assert find_change_point(audio, 16_000) is not None


def test_reference_audio_ends_in_silence():
    audio = selftest.two_speaker_audio()
    assert float(abs(audio[-1_000:]).max()) == 0.0


def test_run_walks_every_stage(settings):
    result = selftest.run(settings)
    names = [check.name for check in result.checks]
    for expected in ("hardware", "whisper backend", "pipeline", "analytics", "exports"):
        assert expected in names
    known = {selftest.PASS, selftest.WARN, selftest.FAIL}
    assert all(check.status in known for check in result.checks)


def test_the_null_backend_passes_the_structural_checks(settings):
    result = selftest.run(settings)
    by_name = {check.name: check for check in result.checks}
    assert by_name["pipeline"].status == selftest.PASS
    assert by_name["analytics"].status == selftest.PASS
    assert by_name["exports"].status == selftest.PASS
    assert result.failed == []


def test_speed_and_transcription_are_not_claimed_without_a_model(settings):
    by_name = {check.name: check for check in selftest.run(settings).checks}
    assert by_name["speed"].status == selftest.WARN
    assert by_name["transcription"].status == selftest.WARN
    assert "null backend" in by_name["transcription"].detail


def test_diarization_is_not_failed_on_the_null_backend(settings):
    by_name = {check.name: check for check in selftest.run(settings).checks}
    assert by_name["diarization"].status in {selftest.PASS, selftest.WARN}
    assert "found 0" not in by_name["diarization"].detail


def test_disabled_diarization_is_reported_as_such(settings):
    settings.diarization.enabled = False
    by_name = {check.name: check for check in selftest.run(settings).checks}
    assert by_name["diarization"].detail == "disabled in config"


def test_checks_stream_as_they_finish(settings):
    seen = []
    result = selftest.run(settings, on_check=seen.append)
    assert [check.name for check in seen] == [check.name for check in result.checks]


def test_the_caller_config_is_not_mutated(settings):
    settings.segmenter.emit_partials = True
    selftest.run(settings)
    assert settings.segmenter.emit_partials is True


def test_a_missing_model_warns_instead_of_failing(settings):
    by_name = {check.name: check for check in selftest.run(settings, model="nope.en").checks}
    assert by_name["model"].status == selftest.WARN
    assert "models get nope.en" in by_name["model"].detail


def test_summary_text_reflects_the_worst_status():
    result = selftest.SelfTest()
    result.add("a", selftest.PASS)
    assert result.as_text(include_checks=False) == "everything works"
    result.add("b", selftest.WARN)
    assert "worth knowing" in result.as_text(include_checks=False)
    result.add("c", selftest.FAIL)
    assert "1 check(s) failed" in result.as_text(include_checks=False)


def test_text_output_can_include_the_checks():
    result = selftest.SelfTest()
    result.add("hardware", selftest.PASS, "fine", seconds=1.5)
    body = result.as_text()
    assert "[ok  ] hardware: fine (1.50s)" in body


def test_as_dict_is_json_serialisable(settings):
    payload = selftest.run(settings).as_dict()
    assert json.loads(json.dumps(payload))["ok"] is True
    assert payload["checks"][0]["name"] == "hardware"


def test_faster_whisper_refuses_to_download_by_default():
    backend = FasterWhisperBackend(ASRConfig(backend="faster_whisper", model_name="base.en"))
    with pytest.raises(BackendUnavailable) as raised:
        backend.reject_download()
    assert "allow_downloads" in str(raised.value)


def test_the_download_guard_allows_a_local_model(tmp_path):
    weights = tmp_path / "ggml-base.en.bin"
    weights.write_bytes(b"\x00")
    backend = FasterWhisperBackend(
        ASRConfig(backend="faster_whisper", model_path=str(weights))
    )
    backend.reject_download()


def test_the_download_guard_can_be_opted_out_of():
    backend = FasterWhisperBackend(
        ASRConfig(backend="faster_whisper", model_name="base.en", allow_downloads=True)
    )
    backend.reject_download()


def test_faster_whisper_wants_ctranslate2_weights():
    assert FasterWhisperBackend.model_format == "ctranslate2"


def test_selftest_command_prints_checks_and_a_summary(settings, tmp_path, capsys):
    config_path = settings.save(tmp_path / "cfg.json")
    assert main(["--config", str(config_path), "selftest", "--backend", "null"]) == 0
    output = capsys.readouterr().out
    assert "[ok  ] hardware" in output
    assert output.count("[ok  ] hardware") == 1


def test_selftest_command_can_emit_json(settings, tmp_path, capsys):
    config_path = settings.save(tmp_path / "cfg.json")
    assert main(["--config", str(config_path), "selftest", "--backend", "null", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out.split("\n", 2)[2])
    assert payload["ok"] is True


def ct2_directory(tmp_path):
    target = tmp_path / "ct2-model"
    target.mkdir()
    (target / "model.bin").write_bytes(b"\x00" * 16)
    (target / "config.json").write_text("{}")
    return target


def test_a_ctranslate2_directory_resolves_like_a_model(tmp_path):
    from lexiflow.asr import models

    target = ct2_directory(tmp_path)
    assert models.resolve(str(target)) == str(target)
    assert models.is_ctranslate2_model(target) is True
    assert models.model_format(str(target)) == "ctranslate2"


def test_a_plain_directory_is_not_a_model(tmp_path):
    from lexiflow.asr import models

    empty = tmp_path / "not-a-model"
    empty.mkdir()
    assert models.resolve(str(empty)) is None
    assert models.is_ctranslate2_model(empty) is False


def test_a_ggml_file_still_resolves_and_reports_its_format(tmp_path):
    from lexiflow.asr import models

    weights = tmp_path / "ggml-base.en.bin"
    weights.write_bytes(b"\x00" * 16)
    assert models.resolve(str(weights)) == str(weights)
    assert models.model_format(str(weights)) == "ggml"


def test_selftest_warns_when_the_weights_do_not_match_the_backend(settings, tmp_path):
    settings.asr.backend = "faster_whisper"
    weights = tmp_path / "ggml-base.en.bin"
    weights.write_bytes(b"\x00" * 16)

    by_name = {check.name: check for check in selftest.run(settings, model=str(weights)).checks}
    assert by_name["model"].status == selftest.WARN
    assert "ggml" in by_name["model"].detail


def test_the_reference_audio_holds_two_separable_voices():
    from lexiflow.audio.segmenter import SpeechSegmenter
    from lexiflow.audio.speaker import SpeakerTracker
    from lexiflow.config import SegmenterConfig

    audio = selftest.two_speaker_audio()
    segmenter = SpeechSegmenter(SegmenterConfig(), 16_000)
    segments = [item for item in segmenter.push(audio) if item.is_final]
    tail = segmenter.flush()
    if tail is not None:
        segments.append(tail)

    assert len(segments) == 2
    tracker = SpeakerTracker()
    labels = [tracker.assign(item.audio, 16_000).label for item in segments]
    assert labels[0] != labels[1]
