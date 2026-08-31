"""LexiFlow: a fully local, real-time speech-to-insight engine."""

__version__ = "0.2.0"

from . import export, insights, redaction, selftest  # noqa: E402
from .batch import BatchRunner, Transcription, transcribe_file  # noqa: E402
from .config import (  # noqa: E402
    ASRConfig,
    AudioConfig,
    DiarizationConfig,
    LexiFlowConfig,
    NLPConfig,
    RedactionConfig,
    SegmenterConfig,
    ServerConfig,
    StateConfig,
    TranslationConfig,
)
from .nlp.pipeline import AnalyticsEngine, Insight  # noqa: E402
from .observability import FAILURES  # noqa: E402
from .pipeline import LexiFlowPipeline, PipelineHealth  # noqa: E402
from .server import LexiFlowAPI, LexiFlowServer, serve  # noqa: E402
from .state.store import SessionStore  # noqa: E402

__all__ = [
    "ASRConfig",
    "AnalyticsEngine",
    "AudioConfig",
    "BatchRunner",
    "DiarizationConfig",
    "FAILURES",
    "Insight",
    "LexiFlowAPI",
    "LexiFlowConfig",
    "LexiFlowPipeline",
    "LexiFlowServer",
    "NLPConfig",
    "PipelineHealth",
    "RedactionConfig",
    "SegmenterConfig",
    "ServerConfig",
    "SessionStore",
    "StateConfig",
    "Transcription",
    "TranslationConfig",
    "__version__",
    "export",
    "insights",
    "redaction",
    "selftest",
    "serve",
    "transcribe_file",
]
