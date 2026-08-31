"""Central configuration for every stage of the LexiFlow pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_STATE_DIR = Path(os.environ.get("LEXIFLOW_HOME", Path.home() / ".lexiflow"))
SUPPORTED_LANGUAGES = frozenset({"en", "es", "fr", "de", "it", "pt"})
REDACTION_MODES = frozenset({"pseudonym", "label", "mask", "hash"})


@dataclass
class AudioConfig:
    """Everything the microphone producer needs to know."""

    target_sample_rate: int = 16_000
    target_channels: int = 1
    capture_sample_rate: Optional[int] = None
    capture_channels: Optional[int] = None
    device: Optional[Any] = None
    block_duration_ms: int = 30
    ring_buffer_seconds: float = 120.0
    dtype: str = "float32"

    @property
    def block_size(self) -> int:
        return max(1, int(self.target_sample_rate * self.block_duration_ms / 1000))

    @property
    def ring_buffer_frames(self) -> int:
        return int(self.target_sample_rate * self.ring_buffer_seconds)


@dataclass
class SegmenterConfig:
    """Energy gated speech segmentation, tuned for conversational speech."""

    frame_duration_ms: int = 30
    min_segment_seconds: float = 1.2
    max_segment_seconds: float = 12.0
    silence_hangover_seconds: float = 0.55
    speech_start_frames: int = 3
    pre_roll_seconds: float = 0.35
    noise_floor_alpha: float = 0.965
    speech_trigger_ratio: float = 3.2
    absolute_silence_rms: float = 1.5e-3
    spectral_gate: bool = True
    min_band_ratio: float = 0.30
    max_spectral_flatness: float = 0.30
    max_zero_crossing_rate: float = 0.35
    emit_partials: bool = True
    partial_interval_seconds: float = 2.0
    partial_min_seconds: float = 1.0


@dataclass
class DiarizationConfig:
    """Speaker attribution by online MFCC clustering; no pretrained model."""

    enabled: bool = True
    similarity_threshold: float = 0.72
    max_speakers: int = 8
    min_seconds: float = 0.6
    adaptation_rate: float = 0.25
    split_on_change: bool = True
    change_threshold: float = 0.35
    change_window_seconds: float = 0.8
    word_level: bool = False
    word_window_seconds: float = 0.6
    word_min_confidence: float = 0.04
    profile_path: Optional[Path] = None


@dataclass
class ASRConfig:
    """Which local Whisper build to use and how hard to drive it."""

    backend: str = "auto"
    model_path: Optional[str] = None
    model_name: str = "base.en"
    language: str = "en"
    translate: bool = False
    threads: int = 0
    beam_size: int = 1
    no_context: bool = True
    word_timestamps: bool = True
    single_segment: bool = False
    max_queue_size: int = 32
    max_realtime_factor: float = 0.75
    allow_downloads: bool = False
    warmup: bool = True

    def resolved_threads(self) -> int:
        if self.threads > 0:
            return self.threads
        return max(1, (os.cpu_count() or 2) - 1)


@dataclass
class NLPConfig:
    """Analytics layer: rules first, tiny models second, never a cloud call."""

    spacy_model: str = "en_core_web_sm"
    enable_spacy: bool = True
    enable_sentiment: bool = True
    enable_rules: bool = True
    enable_topics: bool = True
    detect_language: bool = True
    default_language: str = "en"
    sentiment_window: int = 12
    max_queue_size: int = 256
    topic_window: int = 6
    topic_threshold: float = 0.12
    summary_sentences: int = 5
    keyphrase_limit: int = 12


@dataclass
class TranslationConfig:
    """Local translation: Argos for text, Whisper's translate task for speech."""

    enabled: bool = False
    target_language: str = "en"
    backend: str = "auto"
    speech_translation: bool = True
    analyse_translation: bool = True
    cache_size: int = 512


@dataclass
class RedactionConfig:
    """Remove identifying detail before a transcript leaves the machine."""

    enabled: bool = False
    mode: str = "pseudonym"
    kinds: tuple = ("email", "phone", "card", "iban", "ssn", "person")
    salt: str = ""
    redact_at_source: bool = False


@dataclass
class StateConfig:
    """Persistence and in-memory retention for the shared application state."""

    database_path: Path = field(default_factory=lambda: DEFAULT_STATE_DIR / "lexiflow.db")
    persist: bool = True
    max_transcript_items: int = 5_000
    max_events: int = 2_000
    session_name: Optional[str] = None


@dataclass
class LexiFlowConfig:
    """The single object handed to :class:`lexiflow.pipeline.LexiFlowPipeline`."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    redaction: RedactionConfig = field(default_factory=RedactionConfig)
    state: StateConfig = field(default_factory=StateConfig)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["state"]["database_path"] = str(self.state.database_path)
        if self.diarization.profile_path is not None:
            payload["diarization"]["profile_path"] = str(self.diarization.profile_path)
        return payload

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LexiFlowConfig":
        sections = {
            "audio": AudioConfig,
            "segmenter": SegmenterConfig,
            "diarization": DiarizationConfig,
            "asr": ASRConfig,
            "nlp": NLPConfig,
            "translation": TranslationConfig,
            "redaction": RedactionConfig,
            "state": StateConfig,
        }
        kwargs: Dict[str, Any] = {}
        for key, klass in sections.items():
            raw = dict(payload.get(key) or {})
            allowed = {f.name for f in fields(klass)}
            unknown = set(raw) - allowed
            for name in unknown:
                raw.pop(name)
            if klass is StateConfig and "database_path" in raw:
                raw["database_path"] = Path(raw["database_path"])
            if klass is DiarizationConfig and raw.get("profile_path"):
                raw["profile_path"] = Path(raw["profile_path"])
            if klass is RedactionConfig and "kinds" in raw:
                raw["kinds"] = tuple(raw["kinds"])
            kwargs[key] = klass(**raw)
        return cls(**kwargs)

    @classmethod
    def load(cls, path: os.PathLike | str, strict: bool = True) -> "LexiFlowConfig":
        with open(path, "r", encoding="utf-8") as handle:
            config = cls.from_dict(json.load(handle))
        problems = config.validate()
        if problems and strict:
            raise ValueError(
                f"{path} has {len(problems)} invalid setting(s):\n  "
                + "\n  ".join(problems)
            )
        return config

    def save(self, path: os.PathLike | str, indent: int = 2) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(indent), encoding="utf-8")
        return target

    def validate(self) -> List[str]:
        """Catch the settings that would otherwise fail quietly at 3am."""
        problems: List[str] = []

        if self.audio.target_sample_rate != 16_000:
            problems.append(
                f"audio.target_sample_rate is {self.audio.target_sample_rate}; "
                "Whisper only accepts 16000"
            )
        for name, value in (
            ("audio.block_duration_ms", self.audio.block_duration_ms),
            ("audio.ring_buffer_seconds", self.audio.ring_buffer_seconds),
            ("audio.target_channels", self.audio.target_channels),
        ):
            if value <= 0:
                problems.append(f"{name} must be positive, got {value}")

        if self.segmenter.min_segment_seconds >= self.segmenter.max_segment_seconds:
            problems.append(
                "segmenter.min_segment_seconds must be below max_segment_seconds "
                f"({self.segmenter.min_segment_seconds} >= "
                f"{self.segmenter.max_segment_seconds})"
            )
        if self.segmenter.emit_partials and (
            self.segmenter.partial_min_seconds > self.segmenter.max_segment_seconds
        ):
            problems.append(
                "segmenter.partial_min_seconds is longer than max_segment_seconds, "
                "so no partial can ever be emitted"
            )
        for name, value in (
            ("segmenter.min_band_ratio", self.segmenter.min_band_ratio),
            ("segmenter.max_spectral_flatness", self.segmenter.max_spectral_flatness),
            ("segmenter.max_zero_crossing_rate", self.segmenter.max_zero_crossing_rate),
            ("segmenter.noise_floor_alpha", self.segmenter.noise_floor_alpha),
        ):
            if not 0.0 <= value <= 1.0:
                problems.append(f"{name} must be between 0 and 1, got {value}")

        if not 0.0 < self.diarization.similarity_threshold < 1.0:
            problems.append(
                "diarization.similarity_threshold must be between 0 and 1, got "
                f"{self.diarization.similarity_threshold}"
            )
        if self.diarization.max_speakers < 1:
            problems.append("diarization.max_speakers must be at least 1")
        if not 0.0 < self.diarization.adaptation_rate <= 1.0:
            problems.append("diarization.adaptation_rate must be between 0 and 1")
        if self.diarization.word_window_seconds <= 0.0:
            problems.append("diarization.word_window_seconds must be positive")
        if self.diarization.word_level and not self.asr.word_timestamps:
            problems.append(
                "diarization.word_level needs asr.word_timestamps = true to have words to label"
            )

        if self.asr.threads < 0:
            problems.append("asr.threads cannot be negative; use 0 to autodetect")
        if self.asr.beam_size < 1:
            problems.append("asr.beam_size must be at least 1")
        if self.asr.max_queue_size < 1:
            problems.append("asr.max_queue_size must be at least 1")
        if self.asr.max_realtime_factor <= 0:
            problems.append("asr.max_realtime_factor must be positive")

        if self.nlp.summary_sentences < 1:
            problems.append("nlp.summary_sentences must be at least 1")
        if not 0.0 <= self.nlp.topic_threshold <= 1.0:
            problems.append("nlp.topic_threshold must be between 0 and 1")
        if self.nlp.default_language not in SUPPORTED_LANGUAGES:
            problems.append(
                f"nlp.default_language '{self.nlp.default_language}' has no rule pack; "
                f"choose from {', '.join(sorted(SUPPORTED_LANGUAGES))}"
            )

        if self.translation.enabled and not self.translation.target_language:
            problems.append("translation.enabled needs a translation.target_language")
        if self.translation.cache_size < 1:
            problems.append("translation.cache_size must be at least 1")

        if self.redaction.mode not in REDACTION_MODES:
            problems.append(
                f"redaction.mode '{self.redaction.mode}' is unknown; "
                f"choose from {', '.join(sorted(REDACTION_MODES))}"
            )
        if self.redaction.enabled and not self.redaction.kinds:
            problems.append("redaction.enabled needs at least one entry in redaction.kinds")

        if self.state.max_transcript_items < 1:
            problems.append("state.max_transcript_items must be at least 1")

        return problems
