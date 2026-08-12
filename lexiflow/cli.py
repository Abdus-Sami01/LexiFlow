"""Command line entry point: ``python -m lexiflow <command>``."""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import __version__, export
from .asr import backend_report, hardware, models
from .asr.backends import ScriptedBackend
from .audio.capture import AudioBackendUnavailable, list_input_devices
from .audio.conversion import prepare_for_whisper
from .config import LexiFlowConfig
from .nlp.pipeline import AnalyticsEngine
from .pipeline import LexiFlowPipeline
from .state.store import SessionStore


@dataclass
class _Row:
    """Adapter so exporters can consume plain SQLite rows."""

    seq: int
    text: str
    started_at: float
    ended_at: float
    compound: float = 0.0
    label: str = "neutral"
    speaker: Optional[str] = None
    spans: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "seq": self.seq,
            "text": self.text,
            "speaker": self.speaker,
            "compound": self.compound,
            "label": self.label,
        }


DEMO_LINES = [
    "Morning everyone, thanks for jumping on so quickly.",
    "Remind me to send the revised pricing sheet to Sarah Chen at Northwind Systems.",
    "The deadline is Friday and honestly I am worried we are going to slip again.",
    "We decided to ship the ingestion rewrite first because the disk I O was terrible.",
    "I am blocked on the audio driver, it crashes every time the buffer overruns.",
    "Great news, the new ring buffer cut latency by 40 percent and it feels fantastic.",
    "Can you follow up with legal before end of week?",
    "Let's meet again next Tuesday at 3 pm to review the dashboard.",
]


def _load_config(path: Optional[str]) -> LexiFlowConfig:
    return LexiFlowConfig.load(path) if path else LexiFlowConfig()


def _apply_model(config: LexiFlowConfig, requested: Optional[str]) -> Optional[str]:
    """Accept a catalogue name or a path, and say so clearly when it is missing."""
    if not requested:
        return None
    resolved = models.resolve(requested)
    if resolved is None:
        return (
            f"model '{requested}' not found; run 'python -m lexiflow models get {requested}'"
            if requested in models.CATALOGUE
            else f"model '{requested}' not found on disk"
        )
    config.asr.model_path = resolved
    return None


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        rate = handle.getframerate()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    dtype = {1: "uint8", 2: "int16", 4: "int32"}.get(width)
    if dtype is None:
        raise ValueError(f"unsupported sample width: {width} bytes")
    return prepare_for_whisper(frames, rate, channels=channels, dtype=dtype), 16_000


def command_doctor(args: argparse.Namespace) -> int:
    print(f"LexiFlow {__version__}\n")
    print(hardware.describe())
    print()
    report = backend_report()
    print("whisper backends :", ", ".join(report["available"]) or "none")
    engine = AnalyticsEngine(_load_config(args.config).nlp)
    for name, value in engine.backends.items():
        print(f"{name:<17}: {value}")
    try:
        devices = list_input_devices()
        print(f"input devices    : {len(devices)}")
    except AudioBackendUnavailable as exc:
        print(f"input devices    : unavailable ({exc})")
    return 0


def command_devices(args: argparse.Namespace) -> int:
    try:
        devices = list_input_devices()
    except AudioBackendUnavailable as exc:
        print(exc, file=sys.stderr)
        return 1
    for device in devices:
        print(
            f"[{device['index']:>2}] {device['name']} "
            f"({device['channels']}ch @ {device['default_samplerate']} Hz)"
        )
    return 0


def command_build(args: argparse.Namespace) -> int:
    print(hardware.build_command(args.source))
    return 0


def _print_stream(pipeline: LexiFlowPipeline, interval: float) -> None:
    last_seq = 0
    while pipeline.is_running:
        for item in pipeline.store.transcript():
            if item.seq <= last_seq:
                continue
            last_seq = item.seq
            marker = {"positive": "+", "negative": "-", "neutral": "="}.get(item.label, "=")
            who = f"{item.speaker}: " if item.speaker else ""
            print(f"[{marker}{item.compound:+.2f}] {who}{item.text}")
            for extraction in item.extractions:
                if extraction["kind"] in {"action_item", "deadline", "blocker", "decision"}:
                    due = f" (due {extraction['due']})" if extraction["due"] else ""
                    print(f"        -> {extraction['kind']}: {extraction['text']}{due}")
        time.sleep(interval)


def command_run(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    if args.device is not None:
        config.audio.device = args.device
    problem = _apply_model(config, args.model)
    if problem:
        print(problem, file=sys.stderr)
        return 1
    if args.backend:
        config.asr.backend = args.backend

    pipeline = LexiFlowPipeline(config)
    try:
        pipeline.start()
    except AudioBackendUnavailable as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"listening on {config.audio.target_sample_rate} Hz mono, ctrl-c to stop")
    try:
        _print_stream(pipeline, args.interval)
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        pipeline.stop()
        health = pipeline.health()
        print(json.dumps(health.__dict__, indent=2))
        if args.digest:
            print()
            print(pipeline.digest().as_markdown())
        if args.export:
            pipeline.store.export_json(Path(args.export))
            print(f"exported session to {args.export}")
        pipeline.store.close()
    return 0


def command_replay(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    problem = _apply_model(config, args.model)
    if problem:
        print(problem, file=sys.stderr)
        return 1
    if args.backend:
        config.asr.backend = args.backend

    audio, rate = _read_wav(Path(args.path))
    pipeline = LexiFlowPipeline(config)
    pipeline.start(open_microphone=False)

    chunk = config.audio.block_size
    for offset in range(0, audio.size, chunk):
        pipeline.feed(audio[offset : offset + chunk], rate)
        if args.realtime:
            time.sleep(chunk / float(rate))

    if not pipeline.drain(timeout=args.drain_timeout):
        print("warning: pipeline still busy at shutdown", file=sys.stderr)
    pipeline.stop()
    print(json.dumps(pipeline.snapshot(), indent=2, default=str))
    pipeline.store.close()
    return 0


def command_demo(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    config.state.persist = not args.no_persist
    store = SessionStore(config.state)
    analytics = AnalyticsEngine(config.nlp)

    lines: List[str] = DEMO_LINES if not args.file else Path(args.file).read_text().splitlines()
    for line in lines:
        if not line.strip():
            continue
        insight = analytics.analyse(line)
        store.record(line, insight)
        marker = insight.sentiment.label if insight.sentiment else "neutral"
        print(f"[{marker:>8}] {line}")
        for extraction in insight.extractions:
            print(f"           {extraction.kind}: {extraction.text}")

    print("\nopen actions:")
    for action in store.actions(include_done=False):
        due = f" (due {action.due})" if action.due else ""
        print(f"  [p{action.priority}] {action.text}{due}")

    digest = store.digest(analytics)
    print()
    print(digest.as_markdown())
    print(f"\naverage analytics latency: {analytics.stats.average_ms:.2f} ms")
    if args.export:
        store.export_json(Path(args.export))
        print(f"exported session to {args.export}")
    store.close()
    return 0


def command_search(args: argparse.Namespace) -> int:
    store = SessionStore(_load_config(args.config).state)
    hits = store.search_all_sessions(args.query, limit=args.limit)
    if not hits:
        print("no matches")
    for hit in hits:
        label = hit["session_name"] or hit["session_id"]
        print(f"[{label} #{hit['seq']}] {hit['text']}")
    store.close()
    return 0


def command_sessions(args: argparse.Namespace) -> int:
    store = SessionStore(_load_config(args.config).state)
    rows = store.past_sessions(limit=args.limit)
    for row in rows:
        started = time.strftime("%Y-%m-%d %H:%M", time.localtime(row["started_at"] or 0))
        print(f"{row['id']}  {started}  {row['name']}")
    if not rows:
        print("no recorded sessions")
    store.close()
    return 0


def command_digest(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    store = SessionStore(config.state)
    analytics = AnalyticsEngine(config.nlp)
    session_id = args.session or store.latest_session_with_transcript()
    if not session_id:
        print("no recorded session has a transcript")
        store.close()
        return 1

    rows = store.load_session(session_id)
    if not rows:
        print(f"session {session_id} has no transcript")
        store.close()
        return 1
    digest = analytics.digest([row["text"] for row in rows])
    print(digest.as_markdown())
    store.close()
    return 0


def _progress(written: int, total: int) -> None:
    if not total:
        return
    share = written / total
    filled = int(share * 30)
    sys.stderr.write(
        f"\r  [{'#' * filled}{'.' * (30 - filled)}] {share * 100:5.1f}% "
        f"({written / (1 << 20):.0f}/{total / (1 << 20):.0f} MB)"
    )
    sys.stderr.flush()


def command_models(args: argparse.Namespace) -> int:
    if args.action == "list":
        print(f"models directory: {models.models_directory()}\n")
        print(f"{'name':<16}{'size':>8}  {'state':<12}note")
        for row in models.describe_catalogue():
            state = "installed" if row["installed"] else "-"
            print(f"{row['name']:<16}{row['megabytes']:>6} MB  {state:<12}{row['note']}")
        extras = [
            item
            for item in models.installed_models()
            if item["filename"] not in {spec.filename for spec in models.CATALOGUE.values()}
        ]
        for item in extras:
            print(f"{item['filename']:<16}{item['megabytes']:>6} MB  local")
        return 0

    if args.action == "path":
        resolved = models.resolve(args.name)
        if resolved is None:
            print(f"{args.name} is not installed", file=sys.stderr)
            return 1
        print(resolved)
        return 0

    if args.name not in models.CATALOGUE:
        print(
            f"unknown model '{args.name}', choose from {', '.join(sorted(models.CATALOGUE))}",
            file=sys.stderr,
        )
        return 1

    spec = models.CATALOGUE[args.name]
    if models.is_installed(args.name) and not args.force:
        print(f"{args.name} already at {models.local_path(args.name)}")
        return 0

    print(f"downloading {spec.filename} ({spec.megabytes} MB) from {spec.url}")
    try:
        path = models.download(args.name, force=args.force, progress=_progress)
    except RuntimeError as exc:
        sys.stderr.write("\n")
        print(exc, file=sys.stderr)
        return 1
    sys.stderr.write("\n")
    print(f"saved to {path}")
    return 0


def _speaker_rows(rows: List["_Row"]) -> List[dict]:
    totals: dict = {}
    for row in rows:
        if not row.speaker:
            continue
        bucket = totals.setdefault(row.speaker, {"lines": 0, "seconds": 0.0, "compound": 0.0})
        bucket["lines"] += 1
        bucket["seconds"] += max(0.0, row.ended_at - row.started_at)
        bucket["compound"] += row.compound
    if not totals:
        return []
    grand = sum(item["seconds"] for item in totals.values()) or 1.0
    return [
        {
            "label": label,
            "lines": item["lines"],
            "seconds": round(item["seconds"], 2),
            "share": round(item["seconds"] / grand, 4),
            "average_sentiment": round(item["compound"] / max(1, item["lines"]), 4),
        }
        for label, item in sorted(totals.items())
    ]


def command_export(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    store = SessionStore(config.state)
    analytics = AnalyticsEngine(config.nlp)

    session_id = args.session or store.latest_session_with_transcript()
    if not session_id:
        print("no recorded session has a transcript", file=sys.stderr)
        store.close()
        return 1

    rows = [_Row(**row) for row in store.load_session(session_id)]
    if not rows:
        print(f"session {session_id} has no transcript", file=sys.stderr)
        store.close()
        return 1

    actions = store.load_actions(session_id)
    info = store.session_info(session_id)
    payload = {
        "session": info,
        "metrics": {"utterances": len(rows), "total_actions": len(actions)},
        "transcript": [row.as_dict() for row in rows],
        "actions": actions,
        "speakers": _speaker_rows(rows),
        "entities": {},
    }
    digest = analytics.digest([row.text for row in rows])

    formats = args.format or ["md"]
    granularity = "word" if args.words else "segment"
    if args.output:
        written = export.write_many(
            formats, Path(args.output), rows, payload, digest, granularity=granularity
        )
        for path in written:
            print(f"wrote {path}")
    else:
        print(export.render(formats[0], rows, payload, digest, granularity=granularity))
    store.close()
    return 0


def command_dashboard(args: argparse.Namespace) -> int:
    try:
        from streamlit.web import cli as streamlit_cli
    except Exception:
        print(
            "streamlit is not installed: pip install 'lexiflow[ui]'",
            file=sys.stderr,
        )
        return 1
    target = str(Path(__file__).resolve().parent / "ui" / "dashboard.py")
    sys.argv = [
        "streamlit",
        "run",
        target,
        "--server.port",
        str(args.port),
        "--server.address",
        args.address,
    ]
    return int(streamlit_cli.main() or 0)


def command_tui(args: argparse.Namespace) -> int:
    try:
        from .ui.tui import LexiFlowTUI
    except Exception:
        print("textual is not installed: pip install 'lexiflow[tui]'", file=sys.stderr)
        return 1
    config = _load_config(args.config)
    problem = _apply_model(config, args.model)
    if problem:
        print(problem, file=sys.stderr)
        return 1
    if args.backend:
        config.asr.backend = args.backend
    LexiFlowTUI(LexiFlowPipeline(config), refresh_seconds=args.interval).run()
    return 0


def command_bench(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    analytics = AnalyticsEngine(config.nlp)
    corpus = DEMO_LINES * max(1, args.iterations)
    started = time.perf_counter()
    for line in corpus:
        analytics.analyse(line)
    elapsed = time.perf_counter() - started
    print(f"lines            : {len(corpus)}")
    print(f"total            : {elapsed * 1000:.1f} ms")
    print(f"per line         : {elapsed / len(corpus) * 1000:.3f} ms")
    print(f"entity backend   : {analytics.entities.backend}")
    print(f"sentiment engine : {analytics.backends['sentiment']}")

    if args.asr:
        backend = ScriptedBackend(["benchmark"], config.asr).load()
        noise = np.random.default_rng(0).normal(0, 0.05, 16_000 * 5).astype(np.float32)
        started = time.perf_counter()
        backend.transcribe(noise)
        print(f"asr scripted pass: {(time.perf_counter() - started) * 1000:.2f} ms")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lexiflow", description=__doc__)
    parser.add_argument("--version", action="version", version=f"lexiflow {__version__}")
    parser.add_argument("--config", help="path to a JSON config file")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="report hardware, backends and devices")
    doctor.set_defaults(handler=command_doctor)

    devices = subparsers.add_parser("devices", help="list input devices")
    devices.set_defaults(handler=command_devices)

    build = subparsers.add_parser("build", help="print the tuned whisper.cpp build command")
    build.add_argument("--source", default="whisper.cpp")
    build.set_defaults(handler=command_build)

    run = subparsers.add_parser("run", help="live capture, transcription and analytics")
    run.add_argument("--device", type=int)
    run.add_argument("--model", help="path to a ggml/bin Whisper model")
    run.add_argument("--backend", help="force a specific backend")
    run.add_argument("--interval", type=float, default=0.25)
    run.add_argument("--export", help="write the session to a JSON file on exit")
    run.add_argument("--digest", action="store_true", help="print a summary on exit")
    run.set_defaults(handler=command_run)

    replay = subparsers.add_parser("replay", help="push a wav file through the live pipeline")
    replay.add_argument("path")
    replay.add_argument("--model")
    replay.add_argument("--backend")
    replay.add_argument("--realtime", action="store_true")
    replay.add_argument("--drain-timeout", type=float, default=120.0)
    replay.set_defaults(handler=command_replay)

    demo = subparsers.add_parser("demo", help="run the analytics engine over sample text")
    demo.add_argument("--file", help="read lines from a text file instead of the built-in demo")
    demo.add_argument("--export")
    demo.add_argument("--no-persist", action="store_true")
    demo.set_defaults(handler=command_demo)

    search = subparsers.add_parser("search", help="search the transcript of every session")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=25)
    search.set_defaults(handler=command_search)

    sessions = subparsers.add_parser("sessions", help="list recorded sessions")
    sessions.add_argument("--limit", type=int, default=25)
    sessions.set_defaults(handler=command_sessions)

    digest = subparsers.add_parser("digest", help="summarise a recorded session")
    digest.add_argument("--session", help="session id, defaults to the most recent one")
    digest.set_defaults(handler=command_digest)

    model_parser = subparsers.add_parser("models", help="list or download ggml Whisper models")
    model_parser.add_argument("action", choices=["list", "get", "path"], nargs="?", default="list")
    model_parser.add_argument("name", nargs="?", default="base.en")
    model_parser.add_argument("--force", action="store_true")
    model_parser.set_defaults(handler=command_models)

    export_parser = subparsers.add_parser("export", help="export a session as srt/vtt/txt/md/json")
    export_parser.add_argument("--session", help="session id, defaults to the most recent one")
    export_parser.add_argument(
        "--format", action="append", choices=sorted(export.FORMATS), help="repeatable"
    )
    export_parser.add_argument("--output", help="path stem; prints to stdout when omitted")
    export_parser.add_argument(
        "--words", action="store_true", help="one subtitle cue per word where the backend gave us"
    )
    export_parser.set_defaults(handler=command_export)

    dashboard = subparsers.add_parser("dashboard", help="launch the Streamlit dashboard")
    dashboard.add_argument("--port", type=int, default=8501)
    dashboard.add_argument("--address", default="localhost", help="0.0.0.0 inside a container")
    dashboard.set_defaults(handler=command_dashboard)

    tui = subparsers.add_parser("tui", help="terminal dashboard, no browser needed")
    tui.add_argument("--model")
    tui.add_argument("--backend")
    tui.add_argument("--interval", type=float, default=1.0)
    tui.set_defaults(handler=command_tui)

    bench = subparsers.add_parser("bench", help="measure analytics latency")
    bench.add_argument("--iterations", type=int, default=25)
    bench.add_argument("--asr", action="store_true")
    bench.set_defaults(handler=command_bench)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
