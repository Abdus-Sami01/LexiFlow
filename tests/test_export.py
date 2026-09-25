import pytest

from lexiflow import export
from lexiflow.asr import models
from lexiflow.nlp.pipeline import AnalyticsEngine
from lexiflow.state.store import SessionStore, TranscriptItem


def item(seq, text, start, end, speaker=None):
    return TranscriptItem(
        seq=seq, text=text, started_at=start, ended_at=end, speaker=speaker
    )


ROWS = [
    item(1, "Morning everyone.", 1000.0, 1002.5, "Speaker A"),
    item(2, "The deadline is Friday.", 1003.0, 1006.25, "Speaker B"),
    item(3, "  spaced   out   text  ", 1007.0, 1009.0, "Speaker A"),
]


def test_cues_are_relative_to_the_first_utterance():
    cues = export.to_cues(ROWS)
    assert cues[0].start == 0.0
    assert cues[0].end == pytest.approx(2.5)
    assert cues[1].start == pytest.approx(3.0)


def test_cues_collapse_whitespace():
    assert export.to_cues(ROWS)[2].text == "spaced out text"


def test_cues_never_overlap():
    overlapping = [item(1, "one", 100.0, 105.0), item(2, "two", 102.0, 104.0)]
    cues = export.to_cues(overlapping)
    assert cues[1].start >= cues[0].end


def test_short_utterances_get_a_minimum_duration():
    cues = export.to_cues([item(1, "hi", 10.0, 10.05)])
    assert cues[0].end - cues[0].start >= export.SUBTITLE_MIN_SECONDS


def test_cues_skip_blank_text():
    assert export.to_cues([item(1, "   ", 1.0, 2.0)]) == []


def test_srt_timecodes_use_commas():
    body = export.to_srt(ROWS)
    assert body.startswith("1\n00:00:00,000 --> 00:00:02,500\n")
    assert "[Speaker A] Morning everyone." in body


def test_vtt_has_a_header_and_dot_timecodes():
    body = export.to_vtt(ROWS)
    assert body.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:02.500" in body


def test_speaker_labels_can_be_suppressed():
    assert "[Speaker A]" not in export.to_srt(ROWS, speakers=False)


def test_text_export_carries_timestamps():
    body = export.to_text(ROWS)
    assert body.splitlines()[0].startswith("[00:00:00]")


def test_markdown_renders_actions_and_speakers():
    payload = {
        "session": {"name": "standup"},
        "metrics": {"utterances": 3, "total_actions": 1},
        "transcript": [{"text": "Morning everyone.", "speaker": "Speaker A"}],
        "actions": [
            {"text": "email finance", "done": False, "due": "Friday", "kind": "action_item"},
            {"text": "ship the rewrite", "done": True, "kind": "decision"},
        ],
        "speakers": [
            {"label": "Speaker A", "share": 0.6, "lines": 2, "average_sentiment": 0.3},
        ],
        "entities": {"person": {"Sarah": 2}},
    }
    body = export.to_markdown(payload)
    assert "# standup" in body
    assert "- [ ] email finance _(due Friday)_" in body
    assert "- [x] **decision** ship the rewrite" in body
    assert "| Speaker A | 60% | 2 | +0.30 |" in body
    assert "Sarah (2)" in body


def test_markdown_handles_an_empty_session():
    body = export.to_markdown({})
    assert "none captured" in body


def test_render_rejects_unknown_format():
    with pytest.raises(ValueError):
        export.render("docx", ROWS)


def test_write_many_creates_one_file_per_format(tmp_path):
    payload = {"session": {"name": "s"}, "transcript": [], "actions": []}
    written = export.write_many(["srt", "vtt", "md"], tmp_path / "session", ROWS, payload)
    assert sorted(path.suffix for path in written) == [".md", ".srt", ".vtt"]
    assert all(path.read_text().strip() for path in written)


def test_write_respects_an_explicit_suffix(tmp_path):
    target = export.write("srt", tmp_path / "captions.srt", ROWS)
    assert target.name == "captions.srt"


def test_model_catalogue_is_self_consistent():
    for name, spec in models.CATALOGUE.items():
        assert spec.name == name
        assert spec.filename.startswith("ggml-") and spec.filename.endswith(".bin")
        assert spec.url.endswith(spec.filename)
        assert spec.megabytes > 0


def test_model_resolve_finds_an_explicit_path(tmp_path):
    weights = tmp_path / "ggml-custom.bin"
    weights.write_bytes(b"not really a model")
    assert models.resolve(str(weights)) == str(weights)
    assert models.resolve("definitely-missing") is None
    assert models.resolve(None) is None


def test_model_directory_follows_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("LEXIFLOW_MODELS", str(tmp_path))
    assert models.models_directory() == tmp_path
    assert models.local_path("base.en").name == "ggml-base.en.bin"
    assert models.is_installed("base.en") is False

    (tmp_path / "ggml-base.en.bin").write_bytes(b"x" * 32)
    assert models.is_installed("base.en") is True
    assert models.installed_models()[0]["filename"] == "ggml-base.en.bin"
    assert any(row["installed"] for row in models.describe_catalogue())


def test_model_download_rejects_unknown_names():
    with pytest.raises(ValueError):
        models.download("gpt-9")


def test_store_round_trips_actions_for_export(tmp_path):
    from lexiflow.config import StateConfig

    config = StateConfig(database_path=tmp_path / "export.db")
    store = SessionStore(config)
    analytics = AnalyticsEngine()
    store.seed(["Remind me to email finance before Friday."], analytics)

    actions = store.load_actions(store.session_id)
    assert actions and actions[0]["done"] is False
    assert store.session_info(store.session_id)["name"] == store.session_name
    assert store.session_info("nope")["id"] == "nope"
    store.close()


class SpannedRow:
    def __init__(self, text, started_at, ended_at, spans, speaker=None):
        self.text = text
        self.started_at = started_at
        self.ended_at = ended_at
        self.spans = spans
        self.speaker = speaker


def test_cues_use_backend_spans_when_present():
    row = SpannedRow(
        "one two",
        100.0,
        110.0,
        [
            {"start": 100.5, "end": 102.0, "text": "one"},
            {"start": 103.0, "end": 105.5, "text": "two"},
        ],
    )
    cues = export.to_cues([row])
    assert len(cues) == 2
    assert cues[0].start == pytest.approx(0.5)
    assert cues[0].end == pytest.approx(2.0)
    assert cues[1].start == pytest.approx(3.0)
    assert cues[1].end == pytest.approx(5.5)
    assert cues[1].text == "two"


def test_cues_fall_back_to_segment_bounds_without_spans():
    row = SpannedRow("only", 100.0, 103.0, [])
    cues = export.to_cues([row])
    assert len(cues) == 1
    assert cues[0].end == pytest.approx(3.0)


def test_cues_ignore_degenerate_spans():
    row = SpannedRow("x", 10.0, 12.0, [{"start": 5.0, "end": 5.0, "text": "x"}])
    assert len(export.to_cues([row])) == 1


def test_spans_can_be_disabled():
    row = SpannedRow(
        "one two",
        100.0,
        110.0,
        [
            {"start": 100.5, "end": 102.0, "text": "one"},
            {"start": 103.0, "end": 105.5, "text": "two"},
        ],
    )
    assert len(export.to_cues([row], use_spans=False)) == 1


def worded_row(text, speaker="Speaker A", start=0.0, step=0.3, gap=0.02):
    words = []
    at = start
    for token in text.split():
        words.append({"start": at, "end": at + step - gap, "text": token})
        at += step
    spans = [{"start": start, "end": at, "text": text, "words": words}]
    return SpannedRow(text, start, at, spans, speaker)


LONG = "remind me to send the pricing sheet to Sarah Chen before Friday and also call the bank"


def test_captions_pack_words_into_readable_lines():
    cues = export.to_cues([worded_row(LONG)], granularity="caption")
    assert 1 < len(cues) < len(LONG.split())
    assert all(len(cue.text) <= export.CAPTION_WIDTH for cue in cues)
    assert " ".join(cue.text for cue in cues) == LONG


def test_caption_width_is_configurable():
    narrow = export.to_cues([worded_row(LONG)], granularity="caption", caption_width=20)
    wide = export.to_cues([worded_row(LONG)], granularity="caption", caption_width=80)
    assert len(narrow) > len(wide)
    assert all(len(cue.text) <= 20 for cue in narrow)


def test_word_granularity_is_untouched():
    cues = export.to_cues([worded_row(LONG)], granularity="word")
    assert len(cues) == len(LONG.split())


def test_captions_break_when_the_speaker_changes():
    words = [
        {"start": 0.0, "end": 0.4, "text": "hello", "speaker": "Speaker A"},
        {"start": 0.4, "end": 0.8, "text": "there", "speaker": "Speaker A"},
        {"start": 0.8, "end": 1.2, "text": "hi", "speaker": "Speaker B"},
    ]
    grouped = export.group_into_captions(words)
    assert [item["speaker"] for item in grouped] == ["Speaker A", "Speaker B"]
    assert grouped[0]["text"] == "hello there"


def test_captions_break_on_a_long_pause():
    words = [
        {"start": 0.0, "end": 0.4, "text": "one"},
        {"start": 5.0, "end": 5.4, "text": "two"},
    ]
    assert len(export.group_into_captions(words)) == 2


def test_captions_break_after_a_sentence_ends():
    words = [
        {"start": 0.0, "end": 0.4, "text": "done."},
        {"start": 0.45, "end": 0.8, "text": "next"},
    ]
    grouped = export.group_into_captions(words)
    assert [item["text"] for item in grouped] == ["done.", "next"]


def test_captions_cap_how_long_one_line_stays_up():
    words = [
        {"start": float(i), "end": float(i) + 0.4, "text": "a"} for i in range(12)
    ]
    grouped = export.group_into_captions(words, max_gap=2.0)
    assert all(item["end"] - item["start"] <= export.CAPTION_MAX_SECONDS for item in grouped)


def test_captions_ignore_blank_words():
    assert export.group_into_captions([{"start": 0.0, "end": 1.0, "text": "  "}]) == []


def test_captions_fall_back_to_segments_without_word_timings():
    plain = SpannedRow("no words here", 0.0, 2.0, [])
    assert len(export.to_cues([plain], granularity="caption")) == 1


def test_srt_renders_grouped_captions():
    body = export.render("srt", [worded_row(LONG)], granularity="caption")
    assert "-->" in body
    assert "Sarah" in body
    assert body.count("-->") == len(export.to_cues([worded_row(LONG)], granularity="caption"))


def test_grouped_caption_lines_stay_within_the_width():
    body = export.to_srt([worded_row(LONG)], granularity="caption")
    lines = [line for line in body.splitlines() if line and "-->" not in line]
    assert all(len(line) <= export.CAPTION_WIDTH + len("[Speaker A] ") for line in lines[1::2])
