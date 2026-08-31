"""A loopback HTTP API over a running pipeline, so other tools can use the engine.

The dashboards are one way to read a session. This is the other: plain JSON over
localhost with no dependency beyond the standard library, so an editor plugin, a
shell script or a browser extension can read the live transcript, tick off action
items and follow events without importing any of this package.
"""

from __future__ import annotations

import json
import queue
import re
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from . import __version__, export, insights
from .config import LexiFlowConfig, ServerConfig
from .observability import record_failure
from .pipeline import LexiFlowPipeline

JSON = "application/json; charset=utf-8"
MAX_BODY_BYTES = 256 * 1024
CONTENT_TYPES = {
    "json": JSON,
    "md": "text/markdown; charset=utf-8",
    "txt": "text/plain; charset=utf-8",
    "srt": "application/x-subrip; charset=utf-8",
    "vtt": "text/vtt; charset=utf-8",
}


class Unauthorized(Exception):
    """Raised when a request carries the wrong token, or none at all."""


class NotFound(Exception):
    """Raised for an unknown route or a missing identifier."""


class BadRequest(Exception):
    """Raised when the caller sent something the route cannot work with."""


@dataclass
class Response:
    body: bytes
    status: int = 200
    content_type: str = JSON

    @classmethod
    def of(cls, payload: Any, status: int = 200) -> "Response":
        return cls(json.dumps(payload, default=str).encode("utf-8"), status)

    @classmethod
    def text(cls, body: str, content_type: str, status: int = 200) -> "Response":
        return cls(body.encode("utf-8"), status, content_type)


class EventStream:
    """One connected listener; a slow reader is dropped rather than allowed to block."""

    def __init__(self, maxsize: int = 256) -> None:
        self.queue: "queue.Queue[Optional[str]]" = queue.Queue(maxsize=maxsize)
        self.dropped = 0

    def push(self, event: str, payload: Any) -> None:
        body = json.dumps({"event": event, "data": payload}, default=str)
        try:
            self.queue.put_nowait(body)
        except queue.Full:
            self.dropped += 1

    def close(self) -> None:
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass


class LexiFlowAPI:
    """Routing and payload building, kept free of anything HTTP so it can be tested directly."""

    def __init__(
        self,
        pipeline: LexiFlowPipeline,
        config: Optional[ServerConfig] = None,
    ) -> None:
        self.pipeline = pipeline
        self.config = config or ServerConfig()
        self.streams: List[EventStream] = []
        self.requests = 0
        self._lock = threading.RLock()
        self._unsubscribe: Optional[Callable[[], None]] = pipeline.subscribe(self._fanout)
        self._routes: List[Tuple[str, re.Pattern[str], Callable[..., Response]]] = [
            ("GET", re.compile(r"^/$"), self.index),
            ("GET", re.compile(r"^/health$"), self.health),
            ("GET", re.compile(r"^/snapshot$"), self.snapshot),
            ("GET", re.compile(r"^/transcript$"), self.transcript),
            ("GET", re.compile(r"^/actions$"), self.actions),
            ("POST", re.compile(r"^/actions/(?P<identifier>[^/]+)/toggle$"), self.toggle_action),
            ("GET", re.compile(r"^/digest$"), self.digest),
            ("GET", re.compile(r"^/speakers$"), self.speakers),
            ("POST", re.compile(r"^/speakers/(?P<label>[^/]+)/rename$"), self.rename_speaker),
            ("GET", re.compile(r"^/search$"), self.search),
            ("GET", re.compile(r"^/sessions$"), self.sessions),
            ("GET", re.compile(r"^/review$"), self.review),
            ("GET", re.compile(r"^/export$"), self.export),
            ("POST", re.compile(r"^/text$"), self.submit_text),
        ]

    def close(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        with self._lock:
            for stream in self.streams:
                stream.close()
            self.streams.clear()

    def _fanout(self, event: str, payload: Any) -> None:
        with self._lock:
            listeners = list(self.streams)
        body = payload.as_dict() if hasattr(payload, "as_dict") else payload
        for stream in listeners:
            stream.push(event, body)

    def attach(self) -> EventStream:
        stream = EventStream(self.config.event_queue)
        with self._lock:
            self.streams.append(stream)
        return stream

    def detach(self, stream: EventStream) -> None:
        with self._lock:
            if stream in self.streams:
                self.streams.remove(stream)

    def authorise(self, token: Optional[str]) -> None:
        expected = self.config.token
        if expected and token != expected:
            raise Unauthorized("a valid token is required")

    def dispatch(
        self, method: str, path: str, query: Dict[str, List[str]], body: bytes
    ) -> Response:
        self.requests += 1
        for verb, pattern, handler in self._routes:
            match = pattern.match(path)
            if match is None:
                continue
            if verb != method:
                raise NotFound(f"{path} does not accept {method}")
            return handler(query=query, body=body, **match.groupdict())
        raise NotFound(f"no route for {path}")

    @staticmethod
    def _one(query: Dict[str, List[str]], name: str, default: str = "") -> str:
        values = query.get(name) or []
        return values[0] if values else default

    def _limit(self, query: Dict[str, List[str]]) -> Optional[int]:
        raw = self._one(query, "limit")
        if not raw:
            return None
        try:
            return max(1, int(raw))
        except ValueError as error:
            raise BadRequest(f"limit must be a number, got {raw!r}") from error

    @staticmethod
    def _payload(body: bytes) -> Dict[str, Any]:
        if not body:
            return {}
        try:
            parsed = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BadRequest(f"body is not valid JSON: {error}") from error
        if not isinstance(parsed, dict):
            raise BadRequest("body must be a JSON object")
        return parsed

    def index(self, **_: Any) -> Response:
        return Response.of(
            {
                "name": "lexiflow",
                "version": __version__,
                "session": self.pipeline.store.session_id,
                "routes": sorted(
                    f"{verb} {pattern.pattern}" for verb, pattern, _ in self._routes
                )
                + ["GET ^/events$"],
            }
        )

    def health(self, **_: Any) -> Response:
        return Response.of(self.pipeline.health().__dict__)

    def snapshot(self, query: Dict[str, List[str]], **_: Any) -> Response:
        limit = self._limit(query) or self.config.transcript_limit
        return Response.of(self.pipeline.snapshot(transcript_limit=limit))

    def transcript(self, query: Dict[str, List[str]], **_: Any) -> Response:
        rows = self._rows(self._limit(query))
        return Response.of({"transcript": [row.as_dict() for row in rows]})

    def _rows(self, limit: Optional[int] = None) -> List[Any]:
        """Honour the redaction setting here too; the API is an export like any other."""
        if self.pipeline.config.redaction.enabled:
            rows, _, _ = self.pipeline.redacted(limit)
            return rows
        return self.pipeline.store.transcript(limit)

    def actions(self, query: Dict[str, List[str]], **_: Any) -> Response:
        include_done = self._one(query, "open") not in {"1", "true", "yes"}
        rows = self.pipeline.store.actions(include_done=include_done)
        return Response.of({"actions": [row.as_dict() for row in rows]})

    def toggle_action(self, identifier: str, body: bytes, **_: Any) -> Response:
        payload = self._payload(body)
        done = payload.get("done")
        item = self.pipeline.store.toggle_action(identifier, done)
        if item is None:
            raise NotFound(f"no action item {identifier!r}")
        return Response.of(item.as_dict())

    def digest(self, query: Dict[str, List[str]], **_: Any) -> Response:
        digest = self.pipeline.digest(rows=self._rows(self._limit(query)))
        if self._one(query, "format") == "md":
            return Response.text(digest.as_markdown(), CONTENT_TYPES["md"])
        return Response.of(digest.as_dict())

    def speakers(self, **_: Any) -> Response:
        return Response.of({"speakers": self.pipeline.store.speakers()})

    def rename_speaker(self, label: str, body: bytes, **_: Any) -> Response:
        name = str(self._payload(body).get("name") or "").strip()
        if not name:
            raise BadRequest("give a name to rename this speaker to")
        if not self.pipeline.rename_speaker(label, name):
            raise NotFound(f"no speaker {label!r}")
        return Response.of({"label": label, "name": name})

    def search(self, query: Dict[str, List[str]], **_: Any) -> Response:
        term = self._one(query, "q").strip()
        if not term:
            raise BadRequest("pass ?q= to search for something")
        limit = self._limit(query) or 50
        if self._one(query, "scope") == "all":
            rows = self.pipeline.store.search_all_sessions(term, limit=limit)
        else:
            rows = self.pipeline.store.search(term, limit=limit)
        return Response.of({"query": term, "matches": rows})

    def sessions(self, query: Dict[str, List[str]], **_: Any) -> Response:
        limit = self._limit(query) or 50
        return Response.of({"sessions": self.pipeline.store.session_summaries(limit=limit)})

    def review(self, **_: Any) -> Response:
        return Response.of(insights.build(self.pipeline.store).as_dict())

    def export(self, query: Dict[str, List[str]], **_: Any) -> Response:
        fmt = self._one(query, "format", "md").lower()
        if fmt not in export.FORMATS:
            known = ", ".join(sorted(export.FORMATS))
            raise BadRequest(f"unknown format {fmt!r}, try one of {known}")
        rows = self._rows(self._limit(query))
        payload = self.pipeline.store.export()
        payload["transcript"] = [row.as_dict() for row in rows]
        body = export.render(fmt, rows, payload, self.pipeline.digest(rows=rows))
        return Response.text(body, CONTENT_TYPES.get(fmt, "text/plain; charset=utf-8"))

    def submit_text(self, body: bytes, **_: Any) -> Response:
        text = str(self._payload(body).get("text") or "").strip()
        if not text:
            raise BadRequest("give some text to analyse")
        self.pipeline.submit_text(text)
        return Response.of({"accepted": text}, status=202)


def _handler_for(api: LexiFlowAPI, on_log: Optional[Callable[[str], None]] = None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = f"LexiFlow/{__version__}"

        def log_message(self, fmt: str, *args: Any) -> None:
            if on_log is not None:
                on_log(fmt % args)

        def _token(self) -> Optional[str]:
            header = self.headers.get("Authorization", "")
            if header.lower().startswith("bearer "):
                return header[7:].strip()
            return parse_qs(urlparse(self.path).query).get("token", [None])[0]

        def _cors(self) -> None:
            origin = api.config.allow_origin
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _send(self, response: Response) -> None:
            self.send_response(response.status)
            self.send_header("Content-Type", response.content_type)
            self.send_header("Content-Length", str(len(response.body)))
            self._cors()
            self.end_headers()
            self.wfile.write(response.body)

        def _fail(self, status: int, message: str) -> None:
            self._send(Response.of({"error": message}, status=status))

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                raise BadRequest("request body is too large")
            return self.rfile.read(length) if length > 0 else b""

        def do_OPTIONS(self) -> None:  # noqa: N802 - the stdlib names the hook
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802 - the stdlib names the hook
            parsed = urlparse(self.path)
            if parsed.path == "/events":
                self._stream_events()
                return
            self._route("GET", parsed, b"")

        def do_POST(self) -> None:  # noqa: N802 - the stdlib names the hook
            try:
                body = self._read_body()
            except BadRequest as error:
                self._fail(413, str(error))
                return
            self._route("POST", urlparse(self.path), body)

        def _route(self, method: str, parsed, body: bytes) -> None:
            try:
                api.authorise(self._token())
                self._send(api.dispatch(method, parsed.path, parse_qs(parsed.query), body))
            except Unauthorized as error:
                self._fail(401, str(error))
            except NotFound as error:
                self._fail(404, str(error))
            except BadRequest as error:
                self._fail(400, str(error))
            except Exception as error:
                record_failure("server.request", error, path=parsed.path)
                self._fail(500, f"{type(error).__name__}: {error}")

        def _stream_events(self) -> None:
            try:
                api.authorise(self._token())
            except Unauthorized as error:
                self._fail(401, str(error))
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self._cors()
            self.end_headers()

            stream = api.attach()
            try:
                self.wfile.write(b": connected\n\n")
                self.wfile.flush()
                while True:
                    try:
                        message = stream.queue.get(timeout=15.0)
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    if message is None:
                        break
                    self.wfile.write(f"data: {message}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ValueError):
                pass
            finally:
                api.detach(stream)

    return Handler


class LexiFlowServer:
    """Owns the socket and the background thread; the API object owns the behaviour."""

    def __init__(
        self,
        pipeline: LexiFlowPipeline,
        config: Optional[ServerConfig] = None,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.config = config or ServerConfig()
        self.api = LexiFlowAPI(pipeline, self.config)
        self._httpd = ThreadingHTTPServer(
            (self.config.host, self.config.port), _handler_for(self.api, on_log)
        )
        self._httpd.daemon_threads = True
        self._thread: Optional[threading.Thread] = None

    @property
    def port(self) -> int:
        return int(self._httpd.server_address[1])

    @property
    def url(self) -> str:
        host = self.config.host if ":" not in self.config.host else f"[{self.config.host}]"
        return f"http://{host}:{self.port}"

    def start(self) -> "LexiFlowServer":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="lexiflow-http", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self.api.close()
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def serve_forever(self) -> None:
        self.start()
        try:
            while self._thread is not None and self._thread.is_alive():
                self._thread.join(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def __enter__(self) -> "LexiFlowServer":
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.stop()


def serve(
    config: Optional[LexiFlowConfig] = None,
    pipeline: Optional[LexiFlowPipeline] = None,
    open_microphone: bool = True,
) -> LexiFlowServer:
    settings = config or LexiFlowConfig()
    engine = pipeline or LexiFlowPipeline(settings)
    if not engine.is_running:
        engine.start(open_microphone=open_microphone)
    return LexiFlowServer(engine, settings.server).start()
