"""Task-defined resource budgets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QuotaAxis:
    """Configuration for one resource budget."""

    name: str
    limit: float
    terminal: bool = False
    description: str = ""


@dataclass(frozen=True)
class QuotaConfig:
    """Resource budgets declared by a benchmark task."""

    axes: dict[str, QuotaAxis]

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> QuotaConfig:
        axes: dict[str, QuotaAxis] = {}
        for name, value in values.items():
            if not isinstance(value, dict):
                raise TypeError(f"quota {name!r} must be a mapping")
            if "limit" not in value:
                raise ValueError(f"quota {name!r} is missing 'limit'")

            limit = float(value["limit"])
            if limit < 0:
                raise ValueError(f"quota {name!r} must have a non-negative limit")

            axes[name] = QuotaAxis(
                name=name,
                limit=limit,
                terminal=bool(value.get("terminal", False)),
                description=str(value.get("description", "")),
            )
        return cls(axes)


class QuotaState:
    """Mutable resource usage for one evaluation run."""

    def __init__(self, config: QuotaConfig) -> None:
        self.config = config
        self._spent = {name: 0.0 for name in config.axes}

    def has_axis(self, name: str) -> bool:
        return name in self.config.axes

    def spent(self, name: str) -> float:
        self._axis(name)
        return self._spent[name]

    def remaining(self, name: str) -> float:
        axis = self._axis(name)
        return max(axis.limit - self._spent[name], 0.0)

    def exhausted(self, name: str) -> bool:
        axis = self._axis(name)
        return self._spent[name] >= axis.limit

    def debit(self, name: str, amount: float) -> None:
        if amount < 0:
            raise ValueError("debit amount must be non-negative")
        axis = self._axis(name)
        self._spent[name] = min(self._spent[name] + amount, axis.limit)

    def refund(self, name: str, amount: float) -> None:
        if amount < 0:
            raise ValueError("refund amount must be non-negative")
        self._axis(name)
        self._spent[name] = max(self._spent[name] - amount, 0.0)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "limit": axis.limit,
                "spent": self._spent[name],
                "remaining": self.remaining(name),
                "exhausted": self.exhausted(name),
                "terminal": axis.terminal,
            }
            for name, axis in self.config.axes.items()
        }

    def _axis(self, name: str) -> QuotaAxis:
        try:
            return self.config.axes[name]
        except KeyError as error:
            raise KeyError(f"unknown quota: {name!r}") from error

