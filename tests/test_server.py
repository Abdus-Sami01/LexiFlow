import json
import threading
import urllib.error
import urllib.request

import pytest

from lexiflow.asr.backends import ScriptedBackend
from lexiflow.config import LexiFlowConfig, ServerConfig
from lexiflow.pipeline import LexiFlowPipeline
from lexiflow.server import BadRequest, LexiFlowAPI, LexiFlowServer, NotFound, Unauthorized

LINES = [
    "Remind me to send the pricing sheet to Sarah Chen before Friday.",
    "I am blocked on the audio driver and it is terrible.",
]


def settings(tmp_path, **server):
    config = LexiFlowConfig()
    config.state.database_path = tmp_path / "server.db"
    config.asr.warmup = False
    config.server = ServerConfig(port=0, **server)
    return config


@pytest.fixture()
def pipeline(tmp_path):
    config = settings(tmp_path)
    engine = LexiFlowPipeline(config, backend=ScriptedBackend(LINES, config.asr))
    engine.start(open_microphone=False)
    for line in LINES:
        engine.submit_text(line)
    yield engine
    engine.stop()
    engine.close()


@pytest.fixture()
def api(pipeline):
    built = LexiFlowAPI(pipeline, pipeline.config.server)
    yield built
    built.close()


def call(api, method="GET", path="/", query=None, body=None):
    payload = json.dumps(body).encode("utf-8") if body is not None else b""
    return api.dispatch(method, path, query or {}, payload)


def decoded(response):
    return json.loads(response.body.decode("utf-8"))


def test_the_index_lists_every_route(api):
    payload = decoded(call(api))
    assert payload["name"] == "lexiflow"
    assert any("/transcript" in route for route in payload["routes"])
    assert any("/events" in route for route in payload["routes"])


def test_health_reports_the_running_pipeline(api):
    payload = decoded(call(api, path="/health"))
    assert payload["running"] is True
    assert payload["errors"] == []


def test_transcript_returns_what_was_said(api):
    rows = decoded(call(api, path="/transcript"))["transcript"]
    assert len(rows) == 2
    assert "Sarah Chen" in rows[0]["text"]


def test_transcript_honours_a_limit(api):
    assert len(decoded(call(api, path="/transcript", query={"limit": ["1"]}))["transcript"]) == 1


def test_a_bad_limit_is_a_client_error(api):
    with pytest.raises(BadRequest):
        call(api, path="/transcript", query={"limit": ["soon"]})


def test_actions_can_be_listed_and_ticked_off(api):
    actions = decoded(call(api, path="/actions"))["actions"]
    assert actions

    identifier = actions[0]["id"]
    toggled = decoded(call(api, "POST", f"/actions/{identifier}/toggle", body={"done": True}))
    assert toggled["done"] is True

    still_open = decoded(call(api, path="/actions", query={"open": ["true"]}))["actions"]
    assert identifier not in {row["id"] for row in still_open}


def test_toggling_something_that_does_not_exist_is_a_404(api):
    with pytest.raises(NotFound):
        call(api, "POST", "/actions/nope/toggle", body={})


def test_the_digest_renders_as_json_or_markdown(api):
    assert "summary" in decoded(call(api, path="/digest"))
    markdown = call(api, path="/digest", query={"format": ["md"]})
    assert markdown.content_type.startswith("text/markdown")
    assert markdown.body


def test_search_finds_a_line(api):
    payload = decoded(call(api, path="/search", query={"q": ["pricing"]}))
    assert payload["matches"]
    assert payload["query"] == "pricing"


def test_search_without_a_term_is_rejected(api):
    with pytest.raises(BadRequest):
        call(api, path="/search")


def test_search_can_span_every_session(api):
    assert "matches" in decoded(
        call(api, path="/search", query={"q": ["pricing"], "scope": ["all"]})
    )


def test_sessions_and_review_are_reachable(api):
    assert "sessions" in decoded(call(api, path="/sessions"))
    assert "open_items" in decoded(call(api, path="/review"))


@pytest.mark.parametrize("fmt", ["md", "srt", "vtt", "txt", "json"])
def test_every_export_format_is_served(api, fmt):
    response = call(api, path="/export", query={"format": [fmt]})
    assert response.body.strip()
    assert response.content_type


def test_an_unknown_export_format_is_rejected(api):
    with pytest.raises(BadRequest):
        call(api, path="/export", query={"format": ["docx"]})


def test_text_can_be_injected_without_a_microphone(api, pipeline):
    response = call(api, "POST", "/text", body={"text": "We decided to ship on Tuesday."})
    assert response.status == 202
    assert any("Tuesday" in row.text for row in pipeline.store.transcript())


def test_empty_text_is_rejected(api):
    with pytest.raises(BadRequest):
        call(api, "POST", "/text", body={"text": "   "})


def test_a_body_that_is_not_json_is_rejected(api):
    with pytest.raises(BadRequest):
        api.dispatch("POST", "/text", {}, b"not json at all")


def test_a_body_that_is_not_an_object_is_rejected(api):
    with pytest.raises(BadRequest):
        api.dispatch("POST", "/text", {}, b"[1, 2, 3]")


def test_speakers_can_be_renamed(api, pipeline):
    pipeline.store.record("hello", speaker="Speaker A")
    assert decoded(call(api, path="/speakers"))["speakers"]

    renamed = decoded(call(api, "POST", "/speakers/Speaker A/rename", body={"name": "Priya"}))
    assert renamed["name"] == "Priya"


def test_renaming_without_a_name_is_rejected(api):
    with pytest.raises(BadRequest):
        call(api, "POST", "/speakers/Speaker A/rename", body={})


def test_an_unknown_route_is_a_404(api):
    with pytest.raises(NotFound):
        call(api, path="/nothing-here")


def test_the_wrong_verb_on_a_known_route_is_a_404(api):
    with pytest.raises(NotFound):
        call(api, "POST", "/transcript")


def test_no_token_means_no_check(api):
    api.authorise(None)


def test_a_token_is_enforced_when_configured(pipeline):
    api = LexiFlowAPI(pipeline, ServerConfig(token="secret"))
    try:
        api.authorise("secret")
        with pytest.raises(Unauthorized):
            api.authorise("wrong")
        with pytest.raises(Unauthorized):
            api.authorise(None)
    finally:
        api.close()


def test_events_reach_every_attached_listener(api, pipeline):
    first, second = api.attach(), api.attach()
    pipeline.submit_text("Remind me to email finance on Friday.")

    for stream in (first, second):
        body = json.loads(stream.queue.get(timeout=5.0))
        assert body["event"]
        assert body["data"]

    api.detach(first)
    api.detach(second)


def test_a_listener_that_never_reads_is_dropped_not_blocking(api, pipeline):
    stream = api.attach()
    for index in range(stream.queue.maxsize + 20):
        stream.push("transcript", {"index": index})
    assert stream.dropped > 0
    api.detach(stream)


def test_closing_the_api_releases_the_listeners(api):
    stream = api.attach()
    api.close()
    assert api.streams == []
    assert stream.queue.get(timeout=1.0) is None


def test_redaction_applies_to_the_api_too(tmp_path):
    config = settings(tmp_path)
    config.redaction.enabled = True
    engine = LexiFlowPipeline(config, backend=ScriptedBackend(LINES, config.asr))
    engine.start(open_microphone=False)
    engine.submit_text(LINES[0])

    api = LexiFlowAPI(engine, config.server)
    try:
        body = decoded(call(api, path="/transcript"))["transcript"]
        assert not any("Sarah Chen" in row["text"] for row in body)
    finally:
        api.close()
        engine.stop()
        engine.close()


@pytest.fixture()
def live(pipeline):
    server = LexiFlowServer(pipeline, pipeline.config.server).start()
    yield server
    server.stop()


def fetch(server, path, method="GET", body=None, token=None):
    request = urllib.request.Request(
        f"{server.url}{path}",
        method=method,
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers={"Content-Type": "application/json"}
        | ({"Authorization": f"Bearer {token}"} if token else {}),
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, response.read().decode("utf-8")


def test_the_server_answers_over_a_real_socket(live):
    status, body = fetch(live, "/health")
    assert status == 200
    assert json.loads(body)["running"] is True


def test_the_server_binds_loopback_and_reports_its_url(live):
    assert live.url.startswith("http://127.0.0.1:")
    assert live.port > 0


def test_a_post_travels_end_to_end(live, pipeline):
    status, _ = fetch(live, "/text", "POST", {"text": "Remind me to call the bank tomorrow."})
    assert status == 202
    assert any("bank" in row.text for row in pipeline.store.transcript())


def test_an_unknown_path_returns_404_over_http(live):
    with pytest.raises(urllib.error.HTTPError) as raised:
        fetch(live, "/absolutely-not")
    assert raised.value.code == 404


def test_a_missing_token_returns_401_over_http(pipeline):
    pipeline.config.server.token = "secret"
    server = LexiFlowServer(pipeline, pipeline.config.server).start()
    try:
        with pytest.raises(urllib.error.HTTPError) as raised:
            fetch(server, "/health")
        assert raised.value.code == 401
        assert fetch(server, "/health", token="secret")[0] == 200
    finally:
        server.stop()


def test_the_event_stream_delivers_over_http(live, pipeline):
    received = []

    def read() -> None:
        with urllib.request.urlopen(f"{live.url}/events", timeout=10) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if line.startswith("data:"):
                    received.append(json.loads(line[5:]))
                    return

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    for _ in range(20):
        pipeline.submit_text("Remind me to review the invoice before Friday.")
        reader.join(timeout=0.25)
        if not reader.is_alive():
            break

    assert received, "no server-sent event arrived"
    assert received[0]["event"]


def test_binding_off_loopback_without_a_token_fails_validation():
    config = LexiFlowConfig()
    config.server.host = "0.0.0.0"
    assert any("server.token" in problem for problem in config.validate())

    config.server.token = "secret"
    assert not any(problem.startswith("server.") for problem in config.validate())


def test_an_impossible_port_fails_validation():
    config = LexiFlowConfig()
    config.server.port = 99_999
    assert any("server.port" in problem for problem in config.validate())


def test_renaming_a_forgotten_cluster_still_relabels_the_transcript(api, pipeline):
    pipeline.store.record("hello", speaker="Speaker Q")
    assert pipeline.transcription.speakers.rename("Speaker Q", "x") is False

    assert decoded(call(api, "POST", "/speakers/Speaker Q/rename", body={"name": "Dana"}))
    assert any(row.speaker == "Dana" for row in pipeline.store.transcript())


def test_renaming_a_label_nobody_has_used_is_a_404(api):
    with pytest.raises(NotFound):
        call(api, "POST", "/speakers/Speaker Z/rename", body={"name": "Dana"})
