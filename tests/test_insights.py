import time

import pytest

from lexiflow import insights
from lexiflow.cli import main
from lexiflow.config import StateConfig
from lexiflow.nlp.pipeline import AnalyticsEngine
from lexiflow.state.store import SessionStore

MEETINGS = {
    "monday": [
        "Remind me to send the pricing sheet to Sarah Chen.",
        "I am blocked on the audio driver, it crashes constantly.",
        "Great news, the rewrite landed and it feels fantastic.",
    ],
    "tuesday": [
        "Remind me to send the pricing sheet to Sarah Chen.",
        "I am blocked on the audio driver, it crashes constantly.",
        "The deadline is Friday and I am worried about the delay.",
    ],
    "wednesday": [
        "We decided to ship the ingestion rewrite first.",
        "I am blocked on the audio driver, it crashes constantly.",
    ],
}


@pytest.fixture()
def populated(tmp_path):
    database = tmp_path / "history.db"
    analytics = AnalyticsEngine()
    for name, lines in MEETINGS.items():
        store = SessionStore(StateConfig(database_path=database, session_name=name))
        store.seed(lines, analytics)
        store.close()
    return SessionStore(StateConfig(database_path=database, session_name="today"))


def test_store_lists_actions_across_sessions(populated):
    rows = populated.all_actions()
    assert rows
    assert len({row["session_name"] for row in rows}) == 3
    assert all(row["done"] is False for row in rows)


def test_store_can_filter_to_open_actions(populated):
    every = populated.all_actions()
    identifier = every[0]["id"]
    populated._connection().execute(
        "UPDATE action_items SET done = 1 WHERE id = ?", (identifier,)
    )
    populated._connection().commit()

    open_rows = populated.all_actions(open_only=True)
    assert identifier not in {row["id"] for row in open_rows}
    assert len(open_rows) == len(every) - 1


def test_session_summaries_skip_empty_sessions(populated):
    summaries = populated.session_summaries()
    names = [row["name"] for row in summaries]
    assert set(names) == {"monday", "tuesday", "wednesday"}
    assert "today" not in names
    assert all(row["utterances"] > 0 for row in summaries)


def test_entity_totals_count_sessions_not_just_mentions(populated):
    people = populated.entity_totals(kind="person")
    assert people
    sarah = next(row for row in people if "Sarah" in row["text"])
    assert sarah["sessions"] == 2
    assert sarah["mentions"] >= 2


def test_open_items_collapse_repeats(populated):
    collapsed = insights.open_items(populated)
    blockers = [item for item in collapsed if item.kind == "blocker"]
    assert len(blockers) == 1
    assert blockers[0].occurrences == 3

    expanded = insights.open_items(populated, collapse=False)
    assert len([item for item in expanded if item.kind == "blocker"]) == 3


def test_open_items_keep_the_earliest_timestamp(populated):
    item = next(item for item in insights.open_items(populated) if item.kind == "blocker")
    assert item.age_days >= 0.0
    assert item.stale is False


def test_stale_items_are_flagged(populated):
    old = time.time() - 30 * 86_400
    populated._connection().execute("UPDATE action_items SET created_at = ?", (old,))
    populated._connection().commit()

    review = insights.build(populated)
    assert review.open_items
    assert all(item.stale for item in review.open_items)
    assert len(review.stale_items) == len(review.open_items)
    assert "stale" in review.as_markdown()


def test_recurring_needs_more_than_one_session(populated):
    recurring = insights.recurring_items(populated)
    texts = [item.text.lower() for item in recurring]
    assert any("audio driver" in text for text in texts)
    assert not any("ingestion rewrite" in text for text in texts)


def test_recurring_groups_near_identical_wording(tmp_path):
    database = tmp_path / "similar.db"
    analytics = AnalyticsEngine()
    for index, line in enumerate(
        [
            "I am blocked on the audio driver crashing.",
            "I am blocked on the audio driver crashing badly.",
        ]
    ):
        store = SessionStore(StateConfig(database_path=database, session_name=f"s{index}"))
        store.seed([line], analytics)
        store.close()

    store = SessionStore(StateConfig(database_path=database, session_name="now"))
    grouped = insights.recurring_items(store)
    assert len(grouped) == 1
    assert grouped[0].occurrences == 2
    assert len(grouped[0].variants) == 2
    store.close()


def test_review_assembles_every_section(populated):
    review = insights.build(populated)
    assert len(review.sessions) == 3
    assert review.open_items
    assert review.recurring
    assert review.people
    assert [row["name"] for row in review.sentiment_trend] == ["monday", "tuesday", "wednesday"]

    body = review.as_markdown()
    for heading in ("Still open", "Keeps coming up", "Named most often", "Mood by session"):
        assert heading in body


def test_review_of_an_empty_history(tmp_path):
    store = SessionStore(StateConfig(database_path=tmp_path / "empty.db"))
    review = insights.build(store)
    assert review.sessions == []
    assert "nothing outstanding" in review.as_markdown()
    store.close()


def test_review_needs_no_database_when_not_persisting(tmp_path):
    store = SessionStore(StateConfig(database_path=tmp_path / "none.db", persist=False))
    assert store.all_actions() == []
    assert store.session_summaries() == []
    assert store.entity_totals() == []
    store.close()


def test_review_command_prints_markdown_and_json(populated, tmp_path, capsys):
    from lexiflow.config import LexiFlowConfig

    settings = LexiFlowConfig()
    settings.state.database_path = populated.database_path
    config_path = settings.save(tmp_path / "review.json")

    assert main(["--config", str(config_path), "review"]) == 0
    assert "Across your sessions" in capsys.readouterr().out

    assert main(["--config", str(config_path), "review", "--json"]) == 0
    assert '"open_items"' in capsys.readouterr().out


def test_review_command_says_when_there_is_no_history(tmp_path, capsys):
    from lexiflow.config import LexiFlowConfig

    settings = LexiFlowConfig()
    settings.state.database_path = tmp_path / "fresh.db"
    config_path = settings.save(tmp_path / "fresh.json")

    assert main(["--config", str(config_path), "review"]) == 1
    assert "no recorded sessions" in capsys.readouterr().err
