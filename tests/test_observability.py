import logging

import pytest

from lexiflow.asr.backends import ScriptedBackend
from lexiflow.config import LexiFlowConfig
from lexiflow.observability import (
    FAILURES,
    FailureLog,
    configure_logging,
    get_logger,
    record_failure,
)
from lexiflow.pipeline import LexiFlowPipeline


@pytest.fixture(autouse=True)
def clean_log():
    FAILURES.clear()
    yield
    FAILURES.clear()


def test_records_an_exception_with_its_type():
    log = FailureLog()
    failure = log.record("store.persist", sqlite_error())
    assert failure.component == "store.persist"
    assert failure.kind == "OperationalError"
    assert "disk" in failure.message
    assert log.total == 1
    assert log.counts() == {"store.persist": 1}


def sqlite_error():
    import sqlite3

    return sqlite3.OperationalError("disk I/O error")


def test_records_a_plain_string():
    log = FailureLog()
    entry = log.record("audio", "device vanished")
    assert entry.kind == "error"
    assert str(entry) == "audio: error: device vanished"


def test_history_is_bounded():
    log = FailureLog(history=3)
    for index in range(10):
        log.record("store", f"failure {index}")
    assert len(log.recent()) == 3
    assert log.total == 10
    assert log.recent()[-1].message == "failure 9"


def test_summary_shape():
    log = FailureLog()
    log.record("store", "one")
    log.record("translate", "two")
    summary = log.summary()
    assert summary["total"] == 2
    assert summary["by_component"] == {"store": 1, "translate": 1}
    assert {item["component"] for item in summary["recent"]} == {"store", "translate"}


def test_clear_resets_everything():
    log = FailureLog()
    log.record("store", "one")
    log.clear()
    assert log.total == 0
    assert log.recent() == []


def test_module_level_helper_uses_the_shared_log():
    record_failure("translate", "no model", pair="es->en")
    assert FAILURES.total == 1
    assert FAILURES.recent()[0].detail == {"pair": "es->en"}


def test_logger_is_namespaced_and_not_shouting_by_default():
    assert get_logger().name == "lexiflow"
    assert get_logger("store").name == "lexiflow.store"
    logger = configure_logging()
    assert logger.level == logging.WARNING
    assert configure_logging(verbose=True).level == logging.DEBUG
    assert configure_logging(quiet=True).level == logging.ERROR
    assert len([h for h in logger.handlers if isinstance(h, logging.StreamHandler)]) == 1


def test_a_broken_store_write_is_recorded_not_raised(tmp_path, monkeypatch):
    import sqlite3

    config = LexiFlowConfig()
    config.state.database_path = tmp_path / "broken.db"
    config.asr.warmup = False
    pipeline = LexiFlowPipeline(config, backend=ScriptedBackend([], config.asr))

    def explode(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(pipeline.store, "_connection", explode)
    pipeline.submit_text("Remind me to check the backups tonight.")

    assert pipeline.store.transcript()
    assert FAILURES.total >= 1
    assert "store.persist" in FAILURES.counts()
    assert pipeline.health().failures >= 1
    pipeline.close()


def test_a_broken_listener_cannot_stop_a_recording(tmp_path):
    config = LexiFlowConfig()
    config.state.database_path = tmp_path / "listener.db"
    config.state.persist = False
    config.asr.warmup = False
    pipeline = LexiFlowPipeline(config, backend=ScriptedBackend([], config.asr))

    def bad_listener(event, payload):
        raise RuntimeError("subscriber blew up")

    pipeline.store.subscribe(bad_listener)
    pipeline.submit_text("Remind me to rotate the keys on Friday.")

    assert pipeline.store.transcript()
    assert "store.listener" in FAILURES.counts()
    pipeline.close()


def test_health_and_snapshot_expose_failures(tmp_path):
    config = LexiFlowConfig()
    config.state.database_path = tmp_path / "health.db"
    config.state.persist = False
    config.asr.warmup = False
    pipeline = LexiFlowPipeline(config, backend=ScriptedBackend([], config.asr))

    record_failure("translate", "no model for nl->en")
    health = pipeline.health()
    assert health.failures == 1
    assert health.failures_by_component == {"translate": 1}
    assert pipeline.snapshot()["failures"]["total"] == 1
    pipeline.close()


def test_an_unopenable_database_downgrades_to_memory(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("this is a file, not a folder")

    config = LexiFlowConfig()
    config.state.database_path = blocker / "nested" / "session.db"
    config.asr.warmup = False
    pipeline = LexiFlowPipeline(config, backend=ScriptedBackend([], config.asr))

    pipeline.submit_text("Remind me to fix the disk layout tomorrow.")
    assert pipeline.store.transcript()
    assert pipeline.store.config.persist is False
    assert "store.open" in FAILURES.counts()
    pipeline.close()


def test_a_failing_search_returns_empty_and_is_recorded(tmp_path, monkeypatch):
    import sqlite3

    config = LexiFlowConfig()
    config.state.database_path = tmp_path / "search.db"
    config.asr.warmup = False
    pipeline = LexiFlowPipeline(config, backend=ScriptedBackend([], config.asr))
    pipeline.submit_text("Remind me to renew the certificate on Friday.")

    def explode(*args, **kwargs):
        raise sqlite3.OperationalError("no such table")

    monkeypatch.setattr(pipeline.store, "_connection", explode)
    assert pipeline.store.search_all_sessions("certificate") == []
    assert "store.search_all_sessions" in FAILURES.counts()
    pipeline.close()
