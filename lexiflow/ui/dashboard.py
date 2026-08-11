"""Phase 5: the Streamlit control room for the local engine.

Run it with ``python -m lexiflow dashboard``. The pipeline object lives in
Streamlit's session state so that reruns of the script never restart the audio
threads; the script only ever reads a snapshot.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import streamlit as st

from lexiflow.asr.backends import backend_report
from lexiflow.audio.capture import AudioBackendUnavailable, list_input_devices
from lexiflow.cli import DEMO_LINES
from lexiflow.config import LexiFlowConfig
from lexiflow.pipeline import LexiFlowPipeline

REFRESH_SECONDS = 1.0
SENTIMENT_COLORS = {"positive": "#1f9d55", "negative": "#c53030", "neutral": "#4a5568"}


def _ensure_pipeline() -> LexiFlowPipeline:
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = LexiFlowPipeline(LexiFlowConfig())
    return st.session_state.pipeline


def _devices() -> List[Dict[str, Any]]:
    try:
        return list_input_devices()
    except AudioBackendUnavailable:
        return []


def _sidebar(pipeline: LexiFlowPipeline) -> None:
    st.sidebar.title("LexiFlow")
    st.sidebar.caption("local speech to insight, no cloud calls")

    devices = _devices()
    if devices:
        labels = {device["index"]: f"[{device['index']}] {device['name']}" for device in devices}
        selected = st.sidebar.selectbox(
            "input device", list(labels), format_func=lambda key: labels[key]
        )
        pipeline.config.audio.device = selected
    else:
        st.sidebar.info("no capture device found, use the transcript injector below")

    report = backend_report()
    st.sidebar.write("**whisper backends**")
    st.sidebar.code("\n".join(report["available"]) or "none", language="text")

    columns = st.sidebar.columns(2)
    if columns[0].button("start", use_container_width=True, disabled=pipeline.is_running):
        try:
            pipeline.start(open_microphone=bool(devices))
        except Exception as exc:
            st.sidebar.error(str(exc))
    if columns[1].button("stop", use_container_width=True, disabled=not pipeline.is_running):
        pipeline.stop()

    st.sidebar.divider()
    st.sidebar.write("**action items**")
    actions = pipeline.store.actions()
    if not actions:
        st.sidebar.caption("nothing captured yet")
    for action in actions:
        due = f" · due {action.due}" if action.due else ""
        checked = st.sidebar.checkbox(
            f"{action.text}{due}", value=action.done, key=f"action-{action.identifier}"
        )
        if checked != action.done:
            pipeline.store.toggle_action(action.identifier, checked)

    st.sidebar.divider()
    injected = st.sidebar.text_input("inject transcript line", key="injector")
    if st.sidebar.button("submit line", use_container_width=True) and injected.strip():
        pipeline.submit_text(injected.strip())
    if st.sidebar.button("load demo conversation", use_container_width=True):
        for line in DEMO_LINES:
            pipeline.submit_text(line)


def _metrics_row(snapshot: Dict[str, Any]) -> None:
    health = snapshot["health"]
    metrics = snapshot["metrics"]
    columns = st.columns(5)
    columns[0].metric("utterances", metrics.get("utterances", 0))
    columns[1].metric("open actions", metrics.get("open_actions", 0))
    columns[2].metric("audio captured", f"{health['captured_seconds']:.0f}s")
    columns[3].metric("asr rtf", f"{health['asr_realtime_factor']:.2f}x")
    columns[4].metric("nlp latency", f"{health['analytics_average_ms']:.1f} ms")


def _transcript_panel(snapshot: Dict[str, Any], query: str) -> None:
    st.subheader("live transcript")
    items = snapshot["transcript"]
    if query:
        lowered = query.lower()
        items = [item for item in items if lowered in item["text"].lower()]
        st.caption(f"{len(items)} matching lines")
    if not items:
        st.info("waiting for speech")
        return
    for item in reversed(items[-80:]):
        color = SENTIMENT_COLORS.get(item["label"], "#4a5568")
        stamp = time.strftime("%H:%M:%S", time.localtime(item["ended_at"]))
        st.markdown(
            f"<div style='border-left:3px solid {color};padding:2px 10px;margin:4px 0'>"
            f"<span style='color:#718096;font-size:0.8em'>{stamp} · "
            f"{item['compound']:+.2f}</span><br>{item['text']}</div>",
            unsafe_allow_html=True,
        )


def _sentiment_panel(snapshot: Dict[str, Any]) -> None:
    st.subheader("sentiment timeline")
    points = snapshot["sentiment"]
    if not points:
        st.caption("no data yet")
        return
    st.line_chart(
        {
            "compound": [point["compound"] for point in points],
            "rolling": [point["rolling"] for point in points],
        }
    )
    latest = points[-1]
    st.caption(
        f"latest {latest['compound']:+.2f} · rolling {latest['rolling']:+.2f} "
        f"· momentum {latest['momentum']:+.2f}"
    )


def _entity_panel(snapshot: Dict[str, Any]) -> None:
    st.subheader("entities")
    entities = snapshot["entities"]
    if not entities:
        st.caption("no entities extracted yet")
        return
    for kind, counts in sorted(entities.items()):
        ranked = sorted(counts.items(), key=lambda pair: -pair[1])[:8]
        st.write(f"**{kind}** · " + ", ".join(f"{name} ({count})" for name, count in ranked))


def main() -> None:
    st.set_page_config(page_title="LexiFlow", page_icon="🎙", layout="wide")
    pipeline = _ensure_pipeline()
    _sidebar(pipeline)

    st.title("LexiFlow control room")
    query = st.text_input("search this session", placeholder="type to filter the transcript")

    snapshot = pipeline.snapshot()
    _metrics_row(snapshot)

    left, right = st.columns([3, 2])
    with left:
        _transcript_panel(snapshot, query)
    with right:
        _sentiment_panel(snapshot)
        _entity_panel(snapshot)

    errors = snapshot["health"]["errors"]
    if errors:
        st.error("\n".join(errors))

    if pipeline.is_running:
        time.sleep(REFRESH_SECONDS)
        st.rerun()


main()
