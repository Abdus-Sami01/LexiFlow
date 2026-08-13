"""Central configuration for every stage of the LexiFlow pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_STATE_DIR = Path(os.environ.get("LEXIFLOW_HOME", Path.home() / ".lexiflow"))


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
            kwargs[key] = klass(**raw)
        return cls(**kwargs)

    @classmethod
    def load(cls, path: os.PathLike | str) -> "LexiFlowConfig":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
