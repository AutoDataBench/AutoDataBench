"""Best validation result produced during an evaluation run."""

from dataclasses import dataclass


@dataclass
class BestSoFarSnapshot:
    dataset_path: str | None = None
    score: float | None = None
    dataset_hash: str | None = None
    updates: int = 0
    minimize: bool = False

    def consider(self, *, dataset_path: str, score: float, dataset_hash: str) -> bool:
        """Store a candidate when it strictly improves the current score."""
        improved = (
            self.score is None
            or (self.minimize and score < self.score)
            or (not self.minimize and score > self.score)
        )
        if not improved:
            return False

        self.dataset_path = dataset_path
        self.score = score
        self.dataset_hash = dataset_hash
        self.updates += 1
        return True

    @property
    def empty(self) -> bool:
        return self.dataset_path is None

