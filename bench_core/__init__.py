"""Framework-independent contracts for AutoDataBench."""

from .quota import QuotaConfig, QuotaState
from .snapshot import BestSoFarSnapshot
from .task_def import TaskDefinition

__all__ = ["BestSoFarSnapshot", "QuotaConfig", "QuotaState", "TaskDefinition"]

