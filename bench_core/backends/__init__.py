"""Backend interfaces used by benchmark tasks."""

from .data import DataBackend, DataReadResult, DataStatsResult
from .model import ModelBackend, ModelCallResult, ModelEmbedResult
from .scorer import ScoreResult, ScorerBackend
from .training import TrainEvalBackend, TrainEvalResult

__all__ = [
    "DataBackend",
    "DataReadResult",
    "DataStatsResult",
    "ModelBackend",
    "ModelCallResult",
    "ModelEmbedResult",
    "ScoreResult",
    "ScorerBackend",
    "TrainEvalBackend",
    "TrainEvalResult",
]

