"""Command line entry point: ``python -m lexiflow <command>``."""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import __version__
from .asr import backend_report, hardware
from .asr.backends import ScriptedBackend
from .audio.capture import AudioBackendUnavailable, list_input_devices
from .audio.conversion import prepare_for_whisper
from .config import LexiFlowConfig
from .nlp.pipeline import AnalyticsEngine
from .pipeline import LexiFlowPipeline
from .state.store import SessionStore

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
            print(f"[{marker}{item.compound:+.2f}] {item.text}")
            for extraction in item.extractions:
                if extraction["kind"] in {"action_item", "deadline", "blocker", "decision"}:
                    due = f" (due {extraction['due']})" if extraction["due"] else ""
                    print(f"        -> {extraction['kind']}: {extraction['text']}{due}")
        time.sleep(interval)


def command_run(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    if args.device is not None:
        config.audio.device = args.device
    if args.model:
        config.asr.model_path = args.model
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
        if args.export:
            pipeline.store.export_json(Path(args.export))
            print(f"exported session to {args.export}")
        pipeline.store.close()
    return 0


def command_replay(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    if args.model:
        config.asr.model_path = args.model
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

    time.sleep(1.0)
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

    print(f"\naverage analytics latency: {analytics.stats.average_ms:.2f} ms")
    if args.export:
        store.export_json(Path(args.export))
        print(f"exported session to {args.export}")
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
    sys.argv = ["streamlit", "run", target, "--server.port", str(args.port)]
    return int(streamlit_cli.main() or 0)


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
    run.set_defaults(handler=command_run)

    replay = subparsers.add_parser("replay", help="push a wav file through the live pipeline")
    replay.add_argument("path")
    replay.add_argument("--model")
    replay.add_argument("--backend")
    replay.add_argument("--realtime", action="store_true")
    replay.set_defaults(handler=command_replay)

    demo = subparsers.add_parser("demo", help="run the analytics engine over sample text")
    demo.add_argument("--file", help="read lines from a text file instead of the built-in demo")
    demo.add_argument("--export")
    demo.add_argument("--no-persist", action="store_true")
    demo.set_defaults(handler=command_demo)

    dashboard = subparsers.add_parser("dashboard", help="launch the Streamlit dashboard")
    dashboard.add_argument("--port", type=int, default=8501)
    dashboard.set_defaults(handler=command_dashboard)

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
