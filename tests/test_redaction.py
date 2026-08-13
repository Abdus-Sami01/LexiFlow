import pytest

from lexiflow import export
from lexiflow.asr.backends import ScriptedBackend
from lexiflow.config import LexiFlowConfig, RedactionConfig
from lexiflow.nlp.entities import EntityExtractor
from lexiflow.pipeline import LexiFlowPipeline
from lexiflow.redaction import PATTERNS, Redactor, build

SENSITIVE = (
    "Remind me to email Sarah Chen at sarah.chen@northwind.com, "
    "call +1 555 0142, and charge card 4111 1111 1111 1111."
)


def redactor(**overrides):
    return Redactor(entities=EntityExtractor(), **overrides)


def test_removes_every_default_kind():
    result = redactor().redact(SENSITIVE)
    counts = result.counts()
    assert counts["person"] == 1
    assert counts["email"] == 1
    assert counts["phone"] == 1
    assert counts["card"] == 1
    for leaked in ("sarah.chen@northwind.com", "4111", "555 0142", "Sarah Chen"):
        assert leaked not in result.text


def test_pseudonyms_are_stable_within_a_session():
    tool = redactor()
    first = tool.redact("Sarah Chen owns this.")
    second = tool.redact("Ask Sarah Chen again.")
    assert "[PERSON_1]" in first.text
    assert "[PERSON_1]" in second.text


def test_different_people_get_different_pseudonyms():
    tool = redactor()
    text = tool.redact("Priya Nair briefed Sarah Chen today.").text
    assert "[PERSON_1]" in text and "[PERSON_2]" in text


def test_label_mode_drops_the_counter():
    text = redactor(mode="label").redact("Sarah Chen and Priya Nair").text
    assert text == "[PERSON] and [PERSON]"


def test_mask_mode_hides_the_length_roughly():
    text = redactor(mode="mask").redact("Sarah Chen called").text
    assert "█" in text and "Sarah" not in text


def test_hash_mode_is_deterministic_for_the_same_salt():
    one = Redactor(entities=EntityExtractor(), mode="hash", salt="pepper")
    two = Redactor(entities=EntityExtractor(), mode="hash", salt="pepper")
    assert one.redact("Sarah Chen").text == two.redact("Sarah Chen").text

    other = Redactor(entities=EntityExtractor(), mode="hash", salt="different")
    assert other.redact("Sarah Chen").text != one.redact("Sarah Chen").text


def test_only_requested_kinds_are_touched():
    tool = Redactor(kinds=("email",), entities=EntityExtractor())
    text = tool.redact(SENSITIVE).text
    assert "sarah.chen@northwind.com" not in text
    assert "Sarah Chen" in text
    assert "4111 1111 1111 1111" in text


def test_untouched_text_is_returned_unchanged():
    result = redactor().redact("the ring buffer holds two minutes of audio")
    assert result.changed is False
    assert result.text == "the ring buffer holds two minutes of audio"
    assert redactor().redact("").text == ""


def test_overlapping_matches_are_resolved_once():
    result = redactor().redact("Contact sarah.chen@northwind.com now")
    assert result.text.count("[EMAIL_1]") == 1
    assert len(result.removals) == 1


def test_a_common_word_before_a_name_is_not_mistaken_for_one():
    """Regression: 'Email Sarah Chen' used to mask 'Email Sarah' and leak 'Chen'."""
    text = redactor().redact("Email Sarah Chen about it").text
    assert text == "Email [PERSON_1] about it"


def test_capitalised_phrases_that_are_not_names_survive():
    for phrase in ("Great News everyone", "Monday Morning works", "The Deadline moved"):
        assert redactor().redact(phrase).text == phrase


def test_removal_records_carry_positions():
    removal = redactor().redact("write to sarah.chen@northwind.com").removals[0]
    payload = removal.as_dict()
    assert payload["kind"] == "email"
    assert payload["start"] < payload["end"]
    assert payload["length"] == len("sarah.chen@northwind.com")


@pytest.mark.parametrize(
    "kind,sample",
    [
        ("email", "reach me at a.b@c.io"),
        ("phone", "ring 555 0142 later"),
        ("card", "card 4111 1111 1111 1111"),
        ("iban", "IBAN GB82WEST12345698765432 please"),
        ("ip", "the box at 192.168.1.14"),
        ("ssn", "ssn 123-45-6789"),
        ("url", "see https://internal.example.com/secret"),
    ],
)
def test_each_pattern_matches_its_own_sample(kind, sample):
    assert PATTERNS[kind].search(sample)
    tool = Redactor(kinds=(kind,), entities=None)
    assert tool.redact(sample).counts().get(kind) == 1


def test_build_reads_the_config():
    tool = build(RedactionConfig(mode="label", kinds=("email",)), EntityExtractor())
    assert tool.mode == "label"
    assert tool.kinds == ("email",)


def test_rows_keep_their_timing_when_scrubbed():
    from lexiflow.state.store import TranscriptItem

    rows = [
        TranscriptItem(
            seq=1,
            text="Sarah Chen will email sarah.chen@northwind.com",
            started_at=100.0,
            ended_at=103.0,
            speaker="Speaker A",
            translation="Sarah Chen enviará el correo",
        )
    ]
    scrubbed = redactor().redact_rows(rows)
    assert scrubbed[0].started_at == 100.0
    assert scrubbed[0].ended_at == 103.0
    assert scrubbed[0].speaker == "Speaker A"
    assert "Sarah Chen" not in scrubbed[0].text
    assert "Sarah Chen" not in scrubbed[0].translation


def test_scrubbed_rows_still_export_as_subtitles():
    from lexiflow.state.store import TranscriptItem

    rows = redactor().redact_rows(
        [TranscriptItem(seq=1, text="Sarah Chen called", started_at=10.0, ended_at=12.0)]
    )
    body = export.to_srt(rows)
    assert "00:00:00,000 --> 00:00:02,000" in body
    assert "Sarah Chen" not in body


def test_payload_actions_and_entities_are_scrubbed():
    payload = {
        "transcript": [{"text": "Sarah Chen will call", "translation": None}],
        "actions": [{"text": "email sarah.chen@northwind.com", "kind": "action_item"}],
        "entities": {"person": {"Sarah Chen": 2}, "percent": {"40 percent": 1}},
    }
    cleaned = redactor().redact_payload(payload)
    assert "Sarah Chen" not in cleaned["transcript"][0]["text"]
    assert "sarah.chen@northwind.com" not in cleaned["actions"][0]["text"]
    assert "Sarah Chen" not in str(cleaned["entities"]["person"])
    assert cleaned["entities"]["percent"] == {"40 percent": 1}


def test_pipeline_keeps_the_original_unless_asked(tmp_path):
    config = LexiFlowConfig()
    config.state.database_path = tmp_path / "keep.db"
    config.redaction.enabled = True
    config.asr.warmup = False
    pipeline = LexiFlowPipeline(config, backend=ScriptedBackend([], config.asr))

    pipeline.submit_text("Remind me to email Sarah Chen tomorrow.")
    assert "Sarah Chen" in pipeline.store.transcript()[0].text

    rows, payload, _ = pipeline.redacted()
    assert "Sarah Chen" not in rows[0].text
    assert "Sarah Chen" not in payload["transcript"][0]["text"]
    pipeline.close()


def test_pipeline_can_redact_at_source(tmp_path):
    config = LexiFlowConfig()
    config.state.database_path = tmp_path / "source.db"
    config.redaction.enabled = True
    config.redaction.redact_at_source = True
    config.asr.warmup = False
    pipeline = LexiFlowPipeline(config, backend=ScriptedBackend([], config.asr))

    pipeline.submit_text("Remind me to email Sarah Chen tomorrow.")
    stored = pipeline.store.transcript()[0].text
    assert "Sarah Chen" not in stored
    assert "[PERSON_1]" in stored
    pipeline.close()


def test_redaction_is_off_by_default(tmp_path):
    config = LexiFlowConfig()
    config.state.database_path = tmp_path / "off.db"
    config.asr.warmup = False
    pipeline = LexiFlowPipeline(config, backend=ScriptedBackend([], config.asr))
    assert pipeline.redactor is None
    pipeline.submit_text("Remind me to email Sarah Chen tomorrow.")
    assert "Sarah Chen" in pipeline.store.transcript()[0].text
    pipeline.close()


def test_a_redacted_digest_does_not_leak_the_original(tmp_path):
    """Regression: the summary was built from the store, not the scrubbed rows."""
    config = LexiFlowConfig()
    config.state.database_path = tmp_path / "digest.db"
    config.redaction.enabled = True
    config.asr.warmup = False
    pipeline = LexiFlowPipeline(config, backend=ScriptedBackend([], config.asr))

    for line in [
        "Remind me to email Sarah Chen tomorrow.",
        "Sarah Chen owns the pricing sheet at sarah.chen@northwind.com.",
    ]:
        pipeline.submit_text(line)

    rows, payload, _ = pipeline.redacted()
    digest = pipeline.digest(rows=rows)
    body = export.to_markdown(payload, digest)

    assert "Sarah Chen" not in body
    assert "sarah.chen@northwind.com" not in body
    assert "[PERSON_1]" in body
    pipeline.close()
