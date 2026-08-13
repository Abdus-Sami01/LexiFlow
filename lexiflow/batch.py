"""Run the whole pipeline over a folder of recordings instead of a microphone.

Live capture is the demo; a backlog of recordings is the job. This turns any
directory of audio into one set of notes per file, keeps a manifest so an
interrupted run resumes instead of starting over, and isolates failures so one
corrupt file cannot end a batch of two hundred.
"""

from __future__ import annotations

import json
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from . import export
from .audio.conversion import prepare_for_whisper
from .config import LexiFlowConfig
from .observability import record_failure
from .pipeline import LexiFlowPipeline
from .redaction import build as build_redactor

AUDIO_SUFFIXES = (".wav",)
MANIFEST_NAME = "manifest.json"
SAMPLE_WIDTHS = {1: "uint8", 2: "int16", 4: "int32"}


@dataclass
class BatchJob:
    """One recording, and everything that happened to it."""

    source: str
    status: str = "pending"
    session_id: str = ""
    utterances: int = 0
    actions: int = 0
    audio_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    outputs: List[str] = field(default_factory=list)
    error: str = ""

    @property
    def realtime_factor(self) -> float:
        if self.audio_seconds <= 0.0:
            return 0.0
        return self.elapsed_seconds / self.audio_seconds

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["realtime_factor"] = round(self.realtime_factor, 3)
        return payload


@dataclass
class BatchReport:
    jobs: List[BatchJob] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0

    @property
    def done(self) -> List[BatchJob]:
        return [job for job in self.jobs if job.status == "done"]

    @property
    def failed(self) -> List[BatchJob]:
        return [job for job in self.jobs if job.status == "failed"]

    @property
    def skipped(self) -> List[BatchJob]:
        return [job for job in self.jobs if job.status == "skipped"]

    @property
    def audio_seconds(self) -> float:
        return sum(job.audio_seconds for job in self.jobs)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "counts": {
                "total": len(self.jobs),
                "done": len(self.done),
                "failed": len(self.failed),
                "skipped": len(self.skipped),
            },
            "audio_seconds": round(self.audio_seconds, 2),
            "wall_seconds": round(max(0.0, self.finished_at - self.started_at), 2),
            "jobs": [job.as_dict() for job in self.jobs],
        }

    def as_text(self) -> str:
        lines = [
            f"{len(self.done)} done · {len(self.failed)} failed · {len(self.skipped)} skipped",
            f"{self.audio_seconds / 60:.1f} minutes of audio in "
            f"{max(0.0, self.finished_at - self.started_at):.1f}s wall time",
        ]
        for job in self.failed:
            lines.append(f"  failed: {Path(job.source).name}: {job.error}")
        return "\n".join(lines)


def _job_from(payload: Dict[str, Any]) -> BatchJob:
    """Rebuild a job from a manifest entry, ignoring keys we no longer carry."""
    allowed = {item.name for item in fields(BatchJob)}
    return BatchJob(**{key: value for key, value in payload.items() if key in allowed})


def discover(target: Path, suffixes: Sequence[str] = AUDIO_SUFFIXES) -> List[Path]:
    """A file gives one job, a directory gives every recording under it."""
    root = Path(target)
    if root.is_file():
        return [root]
    if not root.is_dir():
        return []
    wanted = {suffix.lower() for suffix in suffixes}
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in wanted
    )


def read_audio(path: Path, target_rate: int = 16_000) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        rate = handle.getframerate()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())

    dtype = SAMPLE_WIDTHS.get(width)
    if dtype is None:
        raise ValueError(f"unsupported sample width: {width * 8} bit")
    return prepare_for_whisper(
        frames, rate, channels=channels, dtype=dtype, target_rate=target_rate
    )


class BatchRunner:
    """Each recording gets its own pipeline, its own session and its own files."""

    def __init__(
        self,
        config: Optional[LexiFlowConfig] = None,
        formats: Sequence[str] = ("md",),
        output: Optional[Path] = None,
        workers: int = 1,
        resume: bool = True,
        backend_factory: Optional[Callable[[], Any]] = None,
        on_progress: Optional[Callable[[BatchJob], None]] = None,
    ) -> None:
        self.config = config or LexiFlowConfig()
        self.formats = tuple(formats)
        self.output = Path(output) if output else Path("lexiflow-notes")
        self.workers = max(1, workers)
        self.resume = resume
        self.backend_factory = backend_factory
        self.on_progress = on_progress

    @property
    def manifest_path(self) -> Path:
        return self.output / MANIFEST_NAME

    def previous(self) -> Dict[str, Dict[str, Any]]:
        if not self.resume or not self.manifest_path.is_file():
            return {}
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            record_failure("batch.manifest", error, path=str(self.manifest_path))
            return {}
        return {
            job["source"]: job
            for job in payload.get("jobs", [])
            if job.get("status") == "done"
        }

    def run(self, target: Path) -> BatchReport:
        sources = discover(Path(target))
        report = BatchReport()
        if not sources:
            report.finished_at = time.time()
            return report

        self.output.mkdir(parents=True, exist_ok=True)
        completed = self.previous()

        pending: List[Path] = []
        for source in sources:
            key = str(source)
            if key in completed:
                job = _job_from(completed[key])
                job.status = "skipped"
                report.jobs.append(self._announce(job))
                continue
            pending.append(source)

        if self.workers == 1:
            for source in pending:
                report.jobs.append(self._announce(self._process(source)))
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = {pool.submit(self._process, source): source for source in pending}
                for future in as_completed(futures):
                    report.jobs.append(self._announce(future.result()))

        report.jobs.sort(key=lambda job: job.source)
        report.finished_at = time.time()
        self._write_manifest(report)
        return report

    def _announce(self, job: BatchJob) -> BatchJob:
        if self.on_progress is not None:
            try:
                self.on_progress(job)
            except Exception as error:
                record_failure("batch.progress", error)
        return job

    def _process(self, source: Path) -> BatchJob:
        job = BatchJob(source=str(source))
        started = time.perf_counter()
        pipeline: Optional[LexiFlowPipeline] = None

        try:
            audio = read_audio(source, self.config.audio.target_sample_rate)
            job.audio_seconds = audio.size / float(self.config.audio.target_sample_rate)

            pipeline = self._pipeline_for(source)
            pipeline.start(open_microphone=False)

            block = self.config.audio.block_size
            for offset in range(0, audio.size, block):
                pipeline.feed(audio[offset : offset + block])

            if not pipeline.drain(timeout=max(60.0, job.audio_seconds * 4)):
                job.error = "pipeline did not finish in time"
            pipeline.stop()

            job.session_id = pipeline.store.session_id
            job.utterances = len(pipeline.store.transcript())
            job.actions = len(pipeline.store.actions())
            job.outputs = [str(path) for path in self._write_outputs(source, pipeline)]
            job.status = "failed" if job.error else "done"
        except Exception as error:
            record_failure("batch.job", error, source=str(source))
            job.status = "failed"
            job.error = f"{type(error).__name__}: {error}"
        finally:
            if pipeline is not None:
                pipeline.close()
            job.elapsed_seconds = time.perf_counter() - started

        return job

    def _pipeline_for(self, source: Path) -> LexiFlowPipeline:
        settings = LexiFlowConfig.from_dict(self.config.to_dict())
        settings.state.session_name = source.stem
        settings.segmenter.emit_partials = False
        backend = self.backend_factory() if self.backend_factory else None
        return LexiFlowPipeline(settings, backend=backend)

    def _write_outputs(self, source: Path, pipeline: LexiFlowPipeline) -> List[Path]:
        rows = pipeline.store.transcript()
        payload = pipeline.store.export()
        digest = pipeline.digest()

        if self.config.redaction.enabled:
            redactor = build_redactor(self.config.redaction, pipeline.analytics.entities)
            rows = redactor.redact_rows(rows)
            payload = redactor.redact_payload(payload)
            payload["transcript"] = [row.as_dict() for row in rows]
            digest = pipeline.digest(rows=rows)

        return export.write_many(self.formats, self.output / source.stem, rows, payload, digest)

    def _write_manifest(self, report: BatchReport) -> Path:
        try:
            self.manifest_path.write_text(
                json.dumps(report.as_dict(), indent=2, default=str), encoding="utf-8"
            )
        except OSError as error:
            record_failure("batch.manifest", error, path=str(self.manifest_path))
        return self.manifest_path


def run(target: Path, **options: Any) -> BatchReport:
    return BatchRunner(**options).run(Path(target))


def summarise(reports: Iterable[BatchJob]) -> Dict[str, int]:
    tally: Dict[str, int] = {}
    for job in reports:
        tally[job.status] = tally.get(job.status, 0) + 1
    return tally
