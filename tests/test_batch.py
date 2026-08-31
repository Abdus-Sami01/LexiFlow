import json
import wave

import numpy as np
import pytest

from lexiflow.asr.backends import ScriptedBackend
from lexiflow.batch import BatchJob, BatchRunner, discover, read_audio, transcribe_file
from lexiflow.cli import main
from lexiflow.config import LexiFlowConfig

SAMPLE_RATE = 16_000
LINES = [
    "Remind me to send the pricing sheet to Sarah Chen before Friday.",
    "I am blocked on the audio driver.",
]


def write_wav(path, seconds=2.0, rate=SAMPLE_RATE, fundamental=150.0, channels=1):
    times = np.arange(int(rate * seconds)) / rate
    harmonics = sum(np.sin(2 * np.pi * fundamental * k * times) / k for k in range(1, 12))
    samples = np.clip(harmonics * 0.2, -1.0, 1.0)
    frames = (samples * 32767).astype(np.int16)
    if channels == 2:
        frames = np.repeat(frames, 2)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames.tobytes())
    return path


@pytest.fixture()
def recordings(tmp_path):
    root = tmp_path / "recordings"
    write_wav(root / "standup.wav")
    write_wav(root / "review.wav", fundamental=180.0)
    write_wav(root / "nested" / "planning.wav", fundamental=210.0)
    (root / "notes.txt").write_text("not audio")
    return root


def runner_for(tmp_path, recordings_root, **overrides):
    config = LexiFlowConfig()
    config.state.database_path = tmp_path / "batch.db"
    config.asr.warmup = False
    config.segmenter.min_segment_seconds = 0.4
    config.segmenter.silence_hangover_seconds = 0.3
    options = {
        "config": config,
        "formats": ("md",),
        "output": tmp_path / "notes",
        "backend_factory": lambda: ScriptedBackend(LINES, config.asr),
    }
    options.update(overrides)
    return BatchRunner(**options)


def test_discover_finds_audio_recursively_and_ignores_the_rest(recordings):
    found = discover(recordings)
    assert [path.name for path in found] == ["planning.wav", "review.wav", "standup.wav"]
    assert all(path.suffix == ".wav" for path in found)


def test_discover_accepts_a_single_file(recordings):
    single = discover(recordings / "standup.wav")
    assert len(single) == 1


def test_discover_returns_nothing_for_a_missing_path(tmp_path):
    assert discover(tmp_path / "absent") == []


def test_read_audio_normalises_to_the_target_rate(tmp_path):
    path = write_wav(tmp_path / "a.wav", seconds=1.0, rate=44_100)
    audio = read_audio(path)
    assert audio.dtype == np.float32
    assert abs(audio.size - SAMPLE_RATE) <= 2


def test_read_audio_downmixes_stereo(tmp_path):
    path = write_wav(tmp_path / "stereo.wav", seconds=1.0, channels=2)
    assert abs(read_audio(path).size - SAMPLE_RATE) <= 2


def test_read_audio_rejects_an_unsupported_width(tmp_path):
    path = tmp_path / "odd.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(3)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(b"\x00" * 300)
    with pytest.raises(ValueError):
        read_audio(path)


def test_every_recording_produces_notes(tmp_path, recordings):
    report = runner_for(tmp_path, recordings).run(recordings)

    assert len(report.done) == 3
    assert report.failed == []
    written = {path.name for path in (tmp_path / "notes").iterdir()}
    assert {"standup.md", "review.md", "planning.md", "manifest.json"} <= written

    body = (tmp_path / "notes" / "standup.md").read_text()
    assert "Sarah Chen" in body
    assert "Action items" in body


def test_each_recording_becomes_its_own_session(tmp_path, recordings):
    report = runner_for(tmp_path, recordings).run(recordings)
    assert len({job.session_id for job in report.done}) == 3
    assert all(job.utterances > 0 for job in report.done)


def test_multiple_formats_are_written(tmp_path, recordings):
    runner_for(tmp_path, recordings, formats=("md", "srt", "json")).run(recordings)
    suffixes = {path.suffix for path in (tmp_path / "notes").glob("standup.*")}
    assert suffixes == {".md", ".srt", ".json"}


def test_a_corrupt_file_fails_alone(tmp_path, recordings):
    (recordings / "broken.wav").write_text("definitely not RIFF")
    report = runner_for(tmp_path, recordings).run(recordings)

    assert len(report.done) == 3
    assert len(report.failed) == 1
    assert "broken.wav" in report.failed[0].source
    assert report.failed[0].error
    assert "failed" in report.as_text()


def test_the_manifest_records_every_job(tmp_path, recordings):
    runner = runner_for(tmp_path, recordings)
    runner.run(recordings)

    payload = json.loads(runner.manifest_path.read_text())
    assert payload["counts"]["done"] == 3
    assert len(payload["jobs"]) == 3
    assert all(job["outputs"] for job in payload["jobs"])
    assert payload["audio_seconds"] > 0


def test_a_second_run_skips_finished_work(tmp_path, recordings):
    runner_for(tmp_path, recordings).run(recordings)
    second = runner_for(tmp_path, recordings).run(recordings)

    assert len(second.skipped) == 3
    assert second.done == []


def test_resume_can_be_turned_off(tmp_path, recordings):
    runner_for(tmp_path, recordings).run(recordings)
    second = runner_for(tmp_path, recordings, resume=False).run(recordings)
    assert len(second.done) == 3
    assert second.skipped == []


def test_a_new_recording_is_picked_up_on_the_next_run(tmp_path, recordings):
    runner_for(tmp_path, recordings).run(recordings)
    write_wav(recordings / "retro.wav", fundamental=240.0)

    second = runner_for(tmp_path, recordings).run(recordings)
    assert [job.source for job in second.done] == [str(recordings / "retro.wav")]
    assert len(second.skipped) == 3


def test_workers_greater_than_one_still_processes_everything(tmp_path, recordings):
    report = runner_for(tmp_path, recordings, workers=3).run(recordings)
    assert len(report.done) == 3
    assert len({job.source for job in report.jobs}) == 3


def test_progress_is_reported_for_each_job(tmp_path, recordings):
    seen = []
    runner_for(tmp_path, recordings, on_progress=seen.append).run(recordings)
    assert len(seen) == 3
    assert all(isinstance(job, BatchJob) for job in seen)


def test_a_broken_progress_callback_cannot_stop_the_batch(tmp_path, recordings):
    def explode(job):
        raise RuntimeError("callback failed")

    report = runner_for(tmp_path, recordings, on_progress=explode).run(recordings)
    assert len(report.done) == 3


def test_redaction_applies_to_batch_output(tmp_path, recordings):
    config = LexiFlowConfig()
    config.state.database_path = tmp_path / "redact.db"
    config.asr.warmup = False
    config.segmenter.min_segment_seconds = 0.4
    config.segmenter.silence_hangover_seconds = 0.3
    config.redaction.enabled = True

    BatchRunner(
        config=config,
        formats=("md",),
        output=tmp_path / "scrubbed",
        backend_factory=lambda: ScriptedBackend(LINES, config.asr),
    ).run(recordings)

    body = (tmp_path / "scrubbed" / "standup.md").read_text()
    assert "Sarah Chen" not in body
    assert "[PERSON_1]" in body


def test_an_empty_directory_is_not_an_error(tmp_path):
    report = runner_for(tmp_path, tmp_path / "empty").run(tmp_path / "empty")
    assert report.jobs == []
    assert report.finished_at > 0


def test_realtime_factor_is_reported(tmp_path, recordings):
    report = runner_for(tmp_path, recordings).run(recordings)
    job = report.done[0]
    assert job.audio_seconds == pytest.approx(2.0, abs=0.1)
    assert job.realtime_factor > 0
    assert job.as_dict()["realtime_factor"] == round(job.realtime_factor, 3)


def test_batch_command_reports_and_exits_nonzero_on_failure(tmp_path, recordings, capsys):
    (recordings / "broken.wav").write_text("nope")
    settings = LexiFlowConfig()
    settings.state.database_path = tmp_path / "cli.db"
    settings.asr.warmup = False
    config_path = settings.save(tmp_path / "cli.json")

    status = main(
        [
            "--config",
            str(config_path),
            "batch",
            str(recordings),
            "--output",
            str(tmp_path / "cli-notes"),
            "--backend",
            "null",
        ]
    )
    output = capsys.readouterr()
    assert status == 1
    assert "recording(s) to process" in output.out
    assert "broken.wav" in output.err
    assert (tmp_path / "cli-notes" / "manifest.json").is_file()


def test_batch_command_rejects_a_missing_path(tmp_path, capsys):
    assert main(["batch", str(tmp_path / "nowhere")]) == 1
    assert "does not exist" in capsys.readouterr().err


def transcription_for(tmp_path, source, **overrides):
    config = LexiFlowConfig()
    config.state.database_path = tmp_path / "one.db"
    config.asr.warmup = False
    config.segmenter.min_segment_seconds = 0.4
    config.segmenter.silence_hangover_seconds = 0.3
    for key, value in overrides.items():
        section, _, field_name = key.partition("__")
        setattr(getattr(config, section), field_name, value)
    return transcribe_file(source, config, backend=ScriptedBackend(LINES, config.asr))


def test_one_call_turns_a_recording_into_notes(tmp_path, recordings):
    result = transcription_for(tmp_path, recordings / "standup.wav")

    assert result.rows
    assert "Sarah Chen" in result.text
    assert result.actions
    assert result.session_id
    assert result.audio_seconds == pytest.approx(2.0, abs=0.1)
    assert result.realtime_factor > 0


def test_the_result_renders_every_format(tmp_path, recordings):
    result = transcription_for(tmp_path, recordings / "standup.wav")
    for fmt in ("md", "srt", "vtt", "txt", "json"):
        assert result.render(fmt).strip()


def test_the_result_writes_files(tmp_path, recordings):
    result = transcription_for(tmp_path, recordings / "standup.wav")
    written = result.write(tmp_path / "out" / "standup", ("md", "srt"))
    assert {path.suffix for path in written} == {".md", ".srt"}
    assert all(path.is_file() for path in written)


def test_the_session_is_named_after_the_file(tmp_path, recordings):
    result = transcription_for(tmp_path, recordings / "review.wav")
    assert result.payload["session"]["name"] == "review"


def test_redaction_applies_to_a_single_file_too(tmp_path, recordings):
    result = transcription_for(tmp_path, recordings / "standup.wav", redaction__enabled=True)
    assert "Sarah Chen" not in result.text
    assert "Sarah Chen" not in result.render("md")


def test_a_corrupt_file_raises_rather_than_returning_nothing(tmp_path, recordings):
    (recordings / "bad.wav").write_text("not a riff header")
    with pytest.raises(Exception):
        transcription_for(tmp_path, recordings / "bad.wav")


def test_the_caller_config_is_not_mutated(tmp_path, recordings):
    config = LexiFlowConfig()
    config.state.database_path = tmp_path / "untouched.db"
    config.asr.warmup = False
    config.segmenter.emit_partials = True

    transcribe_file(
        recordings / "standup.wav", config, backend=ScriptedBackend(LINES, config.asr)
    )
    assert config.segmenter.emit_partials is True
    assert config.state.session_name is None


def test_the_public_api_exposes_what_a_library_user_needs():
    import lexiflow

    for name in ("transcribe_file", "LexiFlowPipeline", "LexiFlowConfig", "serve", "export"):
        assert name in lexiflow.__all__
        assert hasattr(lexiflow, name)
    assert sorted(lexiflow.__all__) == lexiflow.__all__
