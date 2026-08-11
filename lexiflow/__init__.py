"""LexiFlow: a fully local, real-time speech-to-insight engine."""

from .config import (
    ASRConfig,
    AudioConfig,
    LexiFlowConfig,
    NLPConfig,
    SegmenterConfig,
    StateConfig,
)
from .pipeline import LexiFlowPipeline, PipelineHealth

__version__ = "0.1.0"

__all__ = [
    "ASRConfig",
    "AudioConfig",
    "LexiFlowConfig",
    "LexiFlowPipeline",
    "NLPConfig",
    "PipelineHealth",
    "SegmenterConfig",
    "StateConfig",
    "__version__",
]
