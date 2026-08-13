"""Prove the installation works on this machine, with this model, right now.

Everything else in the project is tested against a scripted backend, because a
test suite cannot depend on multi-gigabyte weights. That leaves one question a
test suite cannot answer: does the real model, on your hardware, actually
transcribe fast enough to keep up. This runs the whole pipeline end to end and
answers it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from . import export
from .asr import hardware
from .asr.backends import available_backends, create_backend
from .asr.models import resolve
from .audio.speaker import find_change_point
from .config import LexiFlowConfig
from .observability import FAILURES
from .pipeline import LexiFlowPipeline

PASS = "pass"
WARN = "warn"
FAIL = "fail"
MARKS = {PASS: "ok  ", WARN: "warn", FAIL: "FAIL"}


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    seconds: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "seconds": round(self.seconds, 3),
        }

    def __str__(self) -> str:
        timing = f" ({self.seconds:.2f}s)" if self.seconds >= 0.01 else ""
        return f"[{MARKS[self.status]}] {self.name}: {self.detail}{timing}"


@dataclass
class SelfTest:
    checks: List[Check] = field(default_factory=list)

    @property
    def failed(self) -> List[Check]:
        return [check for check in self.checks if check.status == FAIL]

    @property
    def warned(self) -> List[Check]:
        return [check for check in self.checks if check.status == WARN]

    def add(self, name: str, status: str, detail: str = "", seconds: float = 0.0) -> Check:
        check = Check(name, status, detail, seconds)
        self.checks.append(check)
        return check

    def as_dict(self) -> Dict[str, Any]:
        return {
            "checks": [check.as_dict() for check in self.checks],
            "failed": len(self.failed),
            "warned": len(self.warned),
            "ok": not self.failed,
        }

    def as_text(self, include_checks: bool = True) -> str:
        lines = [str(check) for check in self.checks] if include_checks else []
        if lines:
            lines.append("")
        if self.failed:
            lines.append(f"{len(self.failed)} check(s) failed")
        elif self.warned:
            lines.append(f"everything works, {len(self.warned)} thing(s) worth knowing")
        else:
            lines.append("everything works")
        return "\n".join(lines)


def two_speaker_audio(sample_rate: int = 16_000) -> np.ndarray:
    """Two clearly different synthetic voices with a pause, then trailing silence."""
    generator = np.random.default_rng(11)

    def voice(fundamental: float, seconds: float) -> np.ndarray:
        times = np.arange(int(sample_rate * seconds)) / sample_rate
        harmonics = sum(
            np.sin(2 * np.pi * fundamental * k * times) / k for k in range(1, 12)
        )
        envelope = 1.0 + 0.25 * np.sin(2 * np.pi * 3.0 * times)
        noise = generator.normal(0, 0.01, times.size)
        return (harmonics * envelope * 0.2 + noise).astype(np.float32)

    silence = np.zeros(int(sample_rate * 0.8), dtype=np.float32)
    return np.concatenate([voice(115.0, 2.0), silence, voice(235.0, 2.0), silence])


def _preferred_format(settings: LexiFlowConfig) -> str:
    """Which weights the backend that will actually be chosen needs."""
    try:
        return create_backend(settings.asr).model_format
    except Exception:
        return "ggml"


def run(
    config: Optional[LexiFlowConfig] = None,
    model: Optional[str] = None,
    on_check: Optional[Callable[[Check], None]] = None,
) -> SelfTest:
    """Walk the whole pipeline and report what works, what is slow and what is missing."""
    settings = LexiFlowConfig.from_dict((config or LexiFlowConfig()).to_dict())
    settings.segmenter.emit_partials = False
    settings.asr.warmup = False
    result = SelfTest()
    before_failures = FAILURES.total

    def report(check: Check) -> Check:
        if on_check is not None:
            on_check(check)
        return check

    profile = hardware.detect_hardware()
    report(
        result.add(
            "hardware",
            PASS,
            f"{profile.system}/{profile.machine}, {profile.physical_cores} physical cores",
        )
    )

    backends = available_backends()
    real_backends = [name for name in backends if name != "null"]
    report(
        result.add(
            "whisper backend",
            PASS if real_backends else WARN,
            ", ".join(backends)
            if real_backends
            else "only the null backend; install pywhispercpp or faster-whisper",
        )
    )

    requested = model or settings.asr.model_path or settings.asr.model_name
    resolved = resolve(requested)
    if resolved:
        settings.asr.model_path = resolved
        wanted = _preferred_format(settings)
        suffix = Path(resolved).suffix.lower()
        if wanted == "ctranslate2" and suffix == ".bin":
            report(
                result.add(
                    "model",
                    WARN,
                    f"{resolved} is a ggml file, which faster-whisper cannot read; "
                    "install pywhispercpp or point at a CTranslate2 directory",
                )
            )
        else:
            report(result.add("model", PASS, resolved))
    else:
        report(
            result.add(
                "model",
                WARN,
                f"'{requested}' is not on disk; run 'lexiflow models get {requested}'",
            )
        )

    started = time.perf_counter()
    try:
        pipeline = LexiFlowPipeline(settings)
        pipeline.transcription.ensure_loaded()
        load_error = ""
    except Exception as error:
        load_error = f"{type(error).__name__}: {error}"
        report(result.add("model load", FAIL, load_error, time.perf_counter() - started))
        settings.asr.backend = "null"
        pipeline = LexiFlowPipeline(settings)
        pipeline.transcription.ensure_loaded()

    backend_name = pipeline.transcription.backend.name
    if not load_error:
        report(
            result.add(
                "model load",
                WARN if backend_name == "null" else PASS,
                f"{backend_name} ready",
                time.perf_counter() - started,
            )
        )
    else:
        report(
            result.add(
                "fallback",
                WARN,
                "continuing on the null backend so the rest can still be checked",
            )
        )

    audio = two_speaker_audio(settings.audio.target_sample_rate)
    audio_seconds = audio.size / settings.audio.target_sample_rate

    started = time.perf_counter()
    try:
        pipeline.start(open_microphone=False)
        block = settings.audio.block_size
        for offset in range(0, audio.size, block):
            pipeline.feed(audio[offset : offset + block])
        drained = pipeline.drain(timeout=max(60.0, audio_seconds * 8))
        pipeline.stop()
    except Exception as error:
        report(
            result.add("pipeline", FAIL, f"{type(error).__name__}: {error}",
                       time.perf_counter() - started)
        )
        pipeline.close()
        return result

    elapsed = time.perf_counter() - started
    health = pipeline.health()
    report(
        result.add(
            "pipeline",
            PASS if drained and not health.errors else FAIL,
            f"{health.segments_in} segment(s) through three threads"
            + ("" if drained else ", did not drain in time"),
            elapsed,
        )
    )

    transcript = pipeline.store.transcript()
    if backend_name == "null":
        report(
            result.add(
                "transcription",
                WARN,
                "null backend produces no text; install a model to test this properly",
            )
        )
    elif transcript:
        sample = transcript[0].text[:60]
        report(result.add("transcription", PASS, f"{len(transcript)} utterance(s): {sample!r}"))
    else:
        report(
            result.add(
                "transcription",
                FAIL,
                "the model produced no text for clearly voiced audio",
            )
        )

    factor = elapsed / audio_seconds if audio_seconds else 0.0
    if backend_name == "null":
        speed_status = WARN
        speed_detail = "not measurable without a real model"
    elif factor <= settings.asr.max_realtime_factor:
        speed_status = PASS
        speed_detail = f"{factor:.2f}x realtime, comfortably keeping up"
    elif factor <= 1.0:
        speed_status = WARN
        speed_detail = f"{factor:.2f}x realtime, keeping up but with little headroom"
    else:
        speed_status = FAIL
        speed_detail = f"{factor:.2f}x realtime, slower than the audio; pick a smaller model"
    report(result.add("speed", speed_status, speed_detail))

    change = find_change_point(audio, settings.audio.target_sample_rate)
    speakers = health.speakers
    if not settings.diarization.enabled:
        report(result.add("diarization", WARN, "disabled in config"))
    elif backend_name == "null":
        report(
            result.add(
                "diarization",
                WARN if change is None else PASS,
                "speaker change detected in the audio, but attribution needs a real model"
                if change is not None
                else "no speaker change found in the reference audio",
            )
        )
    elif change is not None and speakers >= 2:
        report(result.add("diarization", PASS, f"{speakers} voices separated"))
    else:
        report(
            result.add(
                "diarization",
                WARN,
                f"expected two voices, found {speakers}",
            )
        )

    started = time.perf_counter()
    pipeline.submit_text("Remind me to email finance before Friday, the deadline is Friday.")
    insight_rows = [row for row in pipeline.store.transcript() if "finance" in row.text]
    actions = pipeline.store.actions()
    report(
        result.add(
            "analytics",
            PASS if insight_rows and actions else FAIL,
            f"{len(actions)} item(s) extracted",
            time.perf_counter() - started,
        )
    )

    try:
        payload = pipeline.store.export()
        rendered = {
            fmt: export.render(fmt, pipeline.store.transcript(), payload, pipeline.digest())
            for fmt in sorted(export.FORMATS)
        }
        empty = [fmt for fmt, body in rendered.items() if not body.strip()]
        report(
            result.add(
                "exports",
                PASS if not empty else FAIL,
                "all five formats render" if not empty else f"empty output: {', '.join(empty)}",
            )
        )
    except Exception as error:
        report(result.add("exports", FAIL, f"{type(error).__name__}: {error}"))

    recovered = FAILURES.total - before_failures
    report(
        result.add(
            "recovered failures",
            PASS if recovered == 0 else WARN,
            "none" if recovered == 0 else f"{recovered} during this run, see --verbose",
        )
    )

    pipeline.close()
    return result
