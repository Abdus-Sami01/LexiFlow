import pytest

from lexiflow.config import StateConfig
from lexiflow.nlp.pipeline import AnalyticsEngine
from lexiflow.state.store import SNIPPET_CLOSE, SNIPPET_OPEN, SessionStore

LINES = [
    "The ring buffer holds two minutes of audio in memory.",
    "The audio driver crashes when the buffer overruns.",
    "We fixed the buffer and latency dropped by 40 percent.",
    "Nothing related, purely a hiring discussion about headcount.",
]


@pytest.fixture()
def store(tmp_path):
    handle = SessionStore(StateConfig(database_path=tmp_path / "search.db", session_name="one"))
    handle.seed(LINES, AnalyticsEngine())
    yield handle
    handle.close()


def test_results_are_ranked_best_first(store):
    hits = store.search("buffer")
    assert len(hits) == 3
    assert hits[0]["score"] >= hits[-1]["score"]
    assert hits[0]["score"] == 1.0


def test_results_carry_a_highlighted_snippet(store):
    hit = store.search("buffer")[0]
    assert SNIPPET_OPEN in hit["snippet"] and SNIPPET_CLOSE in hit["snippet"]
    assert "buffer" in hit["snippet"].lower()


def test_unrelated_lines_are_not_returned(store):
    assert all("hiring" not in hit["text"] for hit in store.search("buffer"))
    assert store.search("nonexistentword") == []


def test_multiple_terms_rank_above_one(store):
    hits = store.search("buffer audio")
    assert "audio" in hits[0]["text"].lower() and "buffer" in hits[0]["text"].lower()


def test_relevance_is_normalised(store):
    for hit in store.search("buffer"):
        assert 0.0 < hit["score"] <= 1.0


def test_the_fallback_ranks_without_fts(store):
    store._fts_enabled = False
    hits = store.search("buffer audio")
    assert hits
    assert hits[0]["score"] == 1.0
    assert SNIPPET_OPEN in hits[0]["snippet"]
    assert "audio" in hits[0]["text"].lower()


def test_the_fallback_ignores_pure_stopword_queries(store):
    store._fts_enabled = False
    assert store.search("   ") == []


def test_cross_session_search_is_ranked(tmp_path):
    database = tmp_path / "many.db"
    analytics = AnalyticsEngine()
    for index, line in enumerate(LINES):
        handle = SessionStore(StateConfig(database_path=database, session_name=f"s{index}"))
        handle.seed([line], analytics)
        handle.close()

    handle = SessionStore(StateConfig(database_path=database, session_name="now"))
    hits = handle.search_all_sessions("buffer audio")
    assert hits
    assert hits[0]["score"] == 1.0
    assert hits[0]["session_name"].startswith("s")
    assert SNIPPET_OPEN in hits[0]["snippet"]
    handle.close()


def test_in_memory_search_still_works_without_persistence(tmp_path):
    handle = SessionStore(
        StateConfig(database_path=tmp_path / "mem.db", persist=False, session_name="mem")
    )
    handle.seed(LINES, AnalyticsEngine())
    hits = handle.search("buffer")
    assert hits and hits[0]["score"] == 1.0
    handle.close()
