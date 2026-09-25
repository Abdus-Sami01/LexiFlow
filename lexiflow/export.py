"""Turn a recorded session into the formats other tools actually accept."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .audio.speaker import word_turns

SUBTITLE_MIN_SECONDS = 0.4
CAPTION_WIDTH = 42
CAPTION_MAX_SECONDS = 6.0
CAPTION_MAX_GAP = 0.8


@dataclass
class Cue:
    """One subtitle entry, timed relative to the start of the session."""

    index: int
    start: float
    end: float
    text: str
    speaker: Optional[str] = None

    def labelled(self, include_speaker: bool = True) -> str:
        if include_speaker and self.speaker:
            return f"[{self.speaker}] {self.text}"
        return self.text


def _clock(seconds: float, separator: str) -> str:
    total = max(0.0, seconds)
    hours, remainder = divmod(int(total), 3600)
    minutes, whole = divmod(remainder, 60)
    milliseconds = int(round((total - int(total)) * 1000))
    if milliseconds == 1000:
        whole, milliseconds = whole + 1, 0
    return f"{hours:02d}:{minutes:02d}:{whole:02d}{separator}{milliseconds:03d}"


def _row_text(row: Any, translated: bool) -> str:
    if translated:
        rendered = getattr(row, "translation", None)
        if rendered:
            return rendered
    return row.text


def _row_words(row: Any) -> List[Dict[str, Any]]:
    """Every word the backend timed for this row, in order."""
    return [
        word
        for span in (getattr(row, "spans", None) or [])
        for word in (span.get("words") or [])
        if (word.get("text") or "").strip() and word.get("end", 0) > word.get("start", 0)
    ]


def _row_spans(row: Any, granularity: str = "segment") -> List[Dict[str, Any]]:
    """Prefer the backend's own timings, falling back to the segment boundary."""
    spans = [
        span
        for span in (getattr(row, "spans", None) or [])
        if (span.get("text") or "").strip() and span.get("end", 0) > span.get("start", 0)
    ]

    if granularity in {"word", "caption"}:
        words = _row_words(row)
        if words:
            return words

    if spans:
        return spans
    return [{"start": row.started_at, "end": row.ended_at, "text": row.text}]


def group_into_captions(
    words: List[Dict[str, Any]],
    width: int = CAPTION_WIDTH,
    max_seconds: float = CAPTION_MAX_SECONDS,
    max_gap: float = CAPTION_MAX_GAP,
) -> List[Dict[str, Any]]:
    """One cue per word is unreadable, so pack words into lines a viewer can follow."""
    captions: List[Dict[str, Any]] = []
    for word in words:
        text = str(word.get("text") or "").strip()
        if not text:
            continue
        start = float(word.get("start") or 0.0)
        end = float(word.get("end") or start)
        current = captions[-1] if captions else None
        fits = (
            current is not None
            and current["speaker"] == word.get("speaker")
            and len(current["text"]) + 1 + len(text) <= width
            and end - current["start"] <= max_seconds
            and start - current["end"] <= max_gap
            and not current["text"].endswith((".", "?", "!"))
        )
        if fits:
            current["text"] = f"{current['text']} {text}"
            current["end"] = end
            continue
        captions.append(
            {"start": start, "end": end, "text": text, "speaker": word.get("speaker")}
        )
    return captions


def to_cues(
    items: Sequence[Any],
    origin: Optional[float] = None,
    use_spans: bool = True,
    granularity: str = "segment",
    translated: bool = False,
    caption_width: int = CAPTION_WIDTH,
) -> List[Cue]:
    """Normalise transcript rows into monotonic, non-overlapping subtitle cues."""
    rows = [item for item in items if getattr(item, "text", "").strip()]
    if not rows:
        return []

    base = origin if origin is not None else min(row.started_at for row in rows)
    units: List[Dict[str, Any]] = []
    for row in rows:
        speaker = getattr(row, "speaker", None)
        if translated and getattr(row, "translation", None):
            pieces = [
                {"start": row.started_at, "end": row.ended_at, "text": _row_text(row, True)}
            ]
        elif granularity == "caption" and use_spans and _row_words(row):
            pieces = group_into_captions(
                [{**word, "speaker": word.get("speaker") or speaker} for word in _row_words(row)],
                caption_width,
            )
        elif use_spans:
            pieces = _row_spans(row, granularity)
        else:
            pieces = [{"start": row.started_at, "end": row.ended_at, "text": row.text}]
        for piece in pieces:
            units.append({**piece, "speaker": piece.get("speaker") or speaker})

    cues: List[Cue] = []
    previous_end = 0.0

    for position, unit in enumerate(units, start=1):
        start = max(0.0, unit["start"] - base)
        end = max(start, unit["end"] - base)
        if end - start < SUBTITLE_MIN_SECONDS:
            end = start + SUBTITLE_MIN_SECONDS
        if start < previous_end:
            shift = previous_end - start
            start, end = previous_end, end + shift
        previous_end = end
        cues.append(
            Cue(
                index=position,
                start=start,
                end=end,
                text=" ".join(str(unit["text"]).split()),
                speaker=unit.get("speaker"),
            )
        )
    return cues


def to_srt(
    items: Sequence[Any],
    origin: Optional[float] = None,
    speakers: bool = True,
    granularity: str = "segment",
    translated: bool = False,
    caption_width: int = CAPTION_WIDTH,
) -> str:
    blocks = []
    for cue in to_cues(
        items, origin, granularity=granularity, translated=translated, caption_width=caption_width
    ):
        blocks.append(
            f"{cue.index}\n"
            f"{_clock(cue.start, ',')} --> {_clock(cue.end, ',')}\n"
            f"{cue.labelled(speakers)}\n"
        )
    return "\n".join(blocks)


def to_vtt(
    items: Sequence[Any],
    origin: Optional[float] = None,
    speakers: bool = True,
    granularity: str = "segment",
    translated: bool = False,
    caption_width: int = CAPTION_WIDTH,
) -> str:
    blocks = ["WEBVTT\n"]
    for cue in to_cues(
        items, origin, granularity=granularity, translated=translated, caption_width=caption_width
    ):
        blocks.append(
            f"{_clock(cue.start, '.')} --> {_clock(cue.end, '.')}\n"
            f"{cue.labelled(speakers)}\n"
        )
    return "\n".join(blocks)


def to_text(items: Sequence[Any], speakers: bool = True) -> str:
    lines = []
    for cue in to_cues(items):
        stamp = _clock(cue.start, ".")[:8]
        lines.append(f"[{stamp}] {cue.labelled(speakers)}")
    return "\n".join(lines) + ("\n" if lines else "")


def _mixed_turns(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Only split a transcript line when word labels actually disagree within it."""
    words = [word for span in (row.get("spans") or []) for word in (span.get("words") or [])]
    if not words or not any(word.get("speaker") for word in words):
        return []
    turns = word_turns(words)
    return turns if len({turn.get("speaker") for turn in turns}) > 1 else []


def to_markdown(payload: Dict[str, Any], digest: Optional[Any] = None) -> str:
    """A meeting-note style document: summary, actions, speakers, transcript."""
    session = payload.get("session", {})
    metrics = payload.get("metrics", {})
    lines = [f"# {session.get('name', 'LexiFlow session')}", ""]

    if digest is not None:
        lines.append(digest.as_markdown())
        lines.append("")

    actions = payload.get("actions") or []
    lines.extend(["## Action items", ""])
    if actions:
        for action in actions:
            box = "x" if action.get("done") else " "
            due = f" _(due {action['due']})_" if action.get("due") else ""
            kind = action.get("kind", "action_item")
            prefix = "" if kind == "action_item" else f"**{kind.replace('_', ' ')}** "
            lines.append(f"- [{box}] {prefix}{action['text']}{due}")
    else:
        lines.append("- none captured")

    speakers = payload.get("speakers") or []
    if speakers:
        lines.extend(
            [
                "",
                "## Speakers",
                "",
                "| speaker | share | lines | sentiment |",
                "| --- | --- | --- | --- |",
            ]
        )
        for row in speakers:
            lines.append(
                f"| {row['label']} | {row['share'] * 100:.0f}% | {row['lines']} "
                f"| {row['average_sentiment']:+.2f} |"
            )

    entities = payload.get("entities") or {}
    if entities:
        lines.extend(["", "## Entities", ""])
        for kind, counts in sorted(entities.items()):
            ranked = sorted(counts.items(), key=lambda pair: -pair[1])
            joined = ", ".join(f"{name} ({count})" for name, count in ranked)
            lines.append(f"- **{kind}**: {joined}")

    transcript = payload.get("transcript") or []
    lines.extend(["", "## Transcript", ""])
    for row in transcript:
        turns = _mixed_turns(row)
        if turns:
            for turn in turns:
                who = f"**{turn['speaker']}** · " if turn.get("speaker") else ""
                lines.append(f"- {who}{turn['text']}")
            continue
        who = f"**{row['speaker']}** · " if row.get("speaker") else ""
        lines.append(f"- {who}{row['text']}")
        if row.get("translation"):
            lines.append(f"  - _{row['translation']}_")

    lines.extend(
        [
            "",
            "---",
            "",
            f"{metrics.get('utterances', len(transcript))} utterances · "
            f"{metrics.get('total_actions', len(actions))} extracted items · "
            f"generated locally by LexiFlow",
            "",
        ]
    )
    return "\n".join(lines)


def to_json(payload: Dict[str, Any], indent: int = 2) -> str:
    return json.dumps(payload, indent=indent, default=str)


FORMATS: Dict[str, str] = {
    "srt": ".srt",
    "vtt": ".vtt",
    "txt": ".txt",
    "md": ".md",
    "json": ".json",
}


def render(
    fmt: str,
    items: Sequence[Any],
    payload: Optional[Dict[str, Any]] = None,
    digest: Optional[Any] = None,
    speakers: bool = True,
    granularity: str = "segment",
    translated: bool = False,
    caption_width: int = CAPTION_WIDTH,
) -> str:
    """Single entry point used by the CLI and the dashboard download buttons."""
    renderers: Dict[str, Callable[[], str]] = {
        "srt": lambda: to_srt(
            items, speakers=speakers, granularity=granularity, translated=translated,
            caption_width=caption_width,
        ),
        "vtt": lambda: to_vtt(
            items, speakers=speakers, granularity=granularity, translated=translated,
            caption_width=caption_width,
        ),
        "txt": lambda: to_text(items, speakers=speakers),
        "md": lambda: to_markdown(payload or {}, digest),
        "json": lambda: to_json(payload or {}),
    }
    if fmt not in renderers:
        raise ValueError(f"unsupported format '{fmt}', choose from {', '.join(sorted(FORMATS))}")
    return renderers[fmt]()


def write(
    fmt: str,
    destination: Path,
    items: Sequence[Any],
    payload: Optional[Dict[str, Any]] = None,
    digest: Optional[Any] = None,
    speakers: bool = True,
    granularity: str = "segment",
    translated: bool = False,
    caption_width: int = CAPTION_WIDTH,
) -> Path:
    target = Path(destination)
    if target.suffix == "":
        target = target.with_suffix(FORMATS[fmt])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        render(fmt, items, payload, digest, speakers, granularity, translated, caption_width),
        encoding="utf-8",
    )
    return target


def write_many(
    formats: Iterable[str],
    stem: Path,
    items: Sequence[Any],
    payload: Optional[Dict[str, Any]] = None,
    digest: Optional[Any] = None,
    speakers: bool = True,
    granularity: str = "segment",
    translated: bool = False,
    caption_width: int = CAPTION_WIDTH,
) -> List[Path]:
    base = Path(stem)
    return [
        write(
            fmt,
            base.with_suffix(FORMATS[fmt]),
            items,
            payload,
            digest,
            speakers,
            granularity,
            translated,
            caption_width,
        )
        for fmt in formats
    ]
