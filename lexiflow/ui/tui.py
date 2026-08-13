"""Phase 5, terminal edition: the same engine driven from a Textual dashboard."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    RichLog,
    Static,
)

from ..config import LexiFlowConfig
from ..pipeline import LexiFlowPipeline

SENTIMENT_MARKS = {"positive": "+", "negative": "-", "neutral": "="}
SENTIMENT_STYLES = {"positive": "green", "negative": "red", "neutral": "grey62"}


class LexiFlowTUI(App):
    """A keyboard-driven view of the live pipeline, no browser required."""

    CSS = """
    Screen { layout: vertical; }
    #body { height: 1fr; }
    #left { width: 3fr; }
    #right { width: 2fr; }
    #transcript {
        height: 1fr;
        border: round $primary;
        padding: 0 1;
        overflow-x: hidden;
        scrollbar-size-horizontal: 0;
        scrollbar-size-vertical: 0;
    }
    #metrics { height: 12; border: round $secondary; }
    #actions { height: 1fr; border: round $warning; }
    #partial { height: auto; color: $text-muted; padding: 0 1; }
    Input { dock: bottom; }
    """

    BINDINGS = [
        Binding("slash", "focus_search", "search"),
        Binding("escape", "leave_search", "leave search", show=False),
        Binding("s", "toggle_capture", "start/stop"),
        Binding("d", "load_demo", "demo"),
        Binding("e", "export_session", "export"),
        Binding("r", "export_redacted", "export redacted"),
        Binding("h", "show_history", "history"),
        Binding("c", "clear_search", "clear filter"),
        Binding("q", "quit", "quit"),
    ]

    def __init__(
        self, pipeline: Optional[LexiFlowPipeline] = None, refresh_seconds: float = 1.0
    ) -> None:
        super().__init__()
        self.pipeline = pipeline or LexiFlowPipeline(LexiFlowConfig())
        self.refresh_seconds = refresh_seconds
        self.query_text = ""
        self._rendered_seq = 0
        self._action_rows: List[str] = []
        self._metric_column = None
        self._panels_ready = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield RichLog(
                    id="transcript", wrap=True, markup=True, highlight=False, auto_scroll=True
                )
                yield Static("", id="partial")
            with Vertical(id="right"):
                yield DataTable(id="metrics", cursor_type="none", show_header=False)
                yield ListView(id="actions")
        yield Input(placeholder="filter the transcript, enter to apply", id="search")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#metrics", DataTable)
        self._metric_column = table.add_columns("metric", "value")[1]
        for name in (
            "utterances",
            "open actions",
            "speakers",
            "language",
            "audio captured",
            "asr rtf",
            "nlp latency",
            "keeping up",
            "failures",
        ):
            table.add_row(name, "-", key=name)
        self.query_one("#transcript", RichLog).focus()
        self._panels_ready = True
        self.set_interval(self.refresh_seconds, self.tick)
        self.call_after_refresh(self.tick)

    def on_resize(self) -> None:
        """RichLog caches wrapped strips, so re-render them against the new width."""
        if self._panels_ready:
            self._replay_transcript()

    def tick(self) -> None:
        if not self._panels_ready:
            return
        snapshot = self.pipeline.snapshot()
        self._render_metrics(snapshot)
        self._render_transcript(snapshot)
        self._render_actions(snapshot)
        self._render_partial(snapshot)

    def _render_metrics(self, snapshot: Dict[str, Any]) -> None:
        health = snapshot["health"]
        metrics = snapshot["metrics"]
        backends = metrics.get("analytics_backends") or {}
        values = {
            "utterances": str(metrics.get("utterances", 0)),
            "open actions": str(metrics.get("open_actions", 0)),
            "speakers": str(metrics.get("speakers", 0)),
            "language": str(backends.get("language", "en")),
            "audio captured": f"{health['captured_seconds']:.0f}s",
            "asr rtf": f"{health['asr_realtime_factor']:.2f}x",
            "nlp latency": f"{health['analytics_average_ms']:.2f} ms",
            "keeping up": "yes" if health.get("keeping_up", True) else "falling behind",
            "failures": str(health.get("failures", 0)),
        }
        table = self.query_one("#metrics", DataTable)
        for name, value in values.items():
            table.update_cell(name, self._metric_column, value)

    def _render_transcript(self, snapshot: Dict[str, Any]) -> None:
        log = self.query_one("#transcript", RichLog)
        needle = self.query_text.lower()
        width = log.content_size.width or None
        for item in snapshot["transcript"]:
            if item["seq"] <= self._rendered_seq:
                continue
            self._rendered_seq = item["seq"]
            if needle and needle not in item["text"].lower():
                continue
            log.write(self._format_line(item), width=width)
            if item.get("translation"):
                log.write(f"      [italic grey62]{item['translation']}[/]", width=width)
            for extraction in item["extractions"]:
                if extraction["kind"] in {"action_item", "deadline", "blocker", "decision"}:
                    due = f" (due {extraction['due']})" if extraction["due"] else ""
                    log.write(
                        f"      [italic cyan]{extraction['kind']}[/]: {extraction['text']}{due}",
                        width=width,
                    )

    @staticmethod
    def _format_line(item: Dict[str, Any]) -> str:
        style = SENTIMENT_STYLES.get(item["label"], "white")
        mark = SENTIMENT_MARKS.get(item["label"], "=")
        stamp = time.strftime("%H:%M:%S", time.localtime(item["ended_at"]))
        who = f" {item['speaker']}" if item.get("speaker") else ""
        return (
            f"[dim]{stamp}[/][{style}]{who} {mark}{item['compound']:+.2f}[/] "
            f"{item['text']}"
        )

    def _render_actions(self, snapshot: Dict[str, Any]) -> None:
        listing = self.query_one("#actions", ListView)
        rows: List[str] = []
        for action in snapshot["actions"]:
            box = "x" if action["done"] else " "
            due = f" · due {action['due']}" if action.get("due") else ""
            prefix = "" if action["kind"] == "action_item" else f"{action['kind']}: "
            rows.append(f"[{box}] {prefix}{action['text']}{due}")
        if rows == getattr(self, "_action_rows", None):
            return
        self._action_rows = rows
        listing.clear()
        for row in rows:
            listing.append(ListItem(Label(row)))

    def _render_partial(self, snapshot: Dict[str, Any]) -> None:
        partial = snapshot.get("partial")
        target = self.query_one("#partial", Static)
        if not partial:
            target.update("")
            return
        who = f"{partial['speaker']} · " if partial.get("speaker") else ""
        target.update(f"[italic]{who}listening… {partial['text']}[/]")

    @on(Input.Submitted, "#search")
    def apply_filter(self, event: Input.Submitted) -> None:
        self.query_text = event.value.strip()
        self._replay_transcript()
        self.action_leave_search()

    def action_focus_search(self) -> None:
        self.query_one("#search", Input).focus()

    def action_leave_search(self) -> None:
        """Hand focus back so the single-key bindings work again."""
        self.query_one("#transcript", RichLog).focus()

    def _replay_transcript(self) -> None:
        log = self.query_one("#transcript", RichLog)
        log.clear()
        self._rendered_seq = 0
        self.tick()

    def action_clear_search(self) -> None:
        self.query_one("#search", Input).value = ""
        self.query_text = ""
        self._replay_transcript()

    def action_toggle_capture(self) -> None:
        if self.pipeline.is_running:
            self.pipeline.stop()
            self.notify("capture stopped")
            return
        try:
            self.pipeline.start()
            self.notify("listening")
        except Exception as error:
            self.notify(str(error), severity="error")

    def action_load_demo(self) -> None:
        from ..cli import DEMO_LINES

        for line in DEMO_LINES:
            self.pipeline.submit_text(line)
        self.tick()

    def action_show_history(self) -> None:
        from .. import insights

        log = self.query_one("#transcript", RichLog)
        review = insights.build(self.pipeline.store)
        if not review.sessions:
            self.notify("no earlier sessions recorded yet", severity="warning")
            return
        width = log.content_size.width or None
        log.write("", width=width)
        for line in review.as_markdown().splitlines():
            log.write(f"[grey62]{line}[/]" if line else "", width=width)

    def action_export_session(self) -> None:
        self._export(redacted=False)

    def action_export_redacted(self) -> None:
        self._export(redacted=True)

    def _export(self, redacted: bool) -> None:
        from .. import export

        if not self.pipeline.store.transcript():
            self.notify("nothing to export yet", severity="warning")
            return

        if redacted:
            rows, payload, _ = self.pipeline.redacted()
            stem = f"{self.pipeline.store.session_name}-redacted"
            digest = self.pipeline.digest(rows=rows)
        else:
            rows = self.pipeline.store.transcript()
            payload = self.pipeline.store.export()
            stem = self.pipeline.store.session_name
            digest = self.pipeline.digest()

        target = export.write(
            "md",
            self.pipeline.store.database_path.parent / f"{stem}.md",
            rows,
            payload,
            digest,
        )
        self.notify(f"wrote {target}")

    def on_unmount(self) -> None:
        self.pipeline.close()


def run(config: Optional[LexiFlowConfig] = None) -> None:
    LexiFlowTUI(LexiFlowPipeline(config or LexiFlowConfig())).run()
