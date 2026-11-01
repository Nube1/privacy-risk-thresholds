"""Measurement rows of Chapter 18: integer counts that name their experiment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

FINAL_AUDIT = "final-audit"


@dataclass(frozen=True)
class MeasurementRecord:
    scenario: str
    mechanism: str
    target_law: str
    conditioning: str
    split: str
    seed_stream: str
    data_digest: str
    evaluator_digest: str
    successes: int
    trials: int
    plan_hash: str | None = None
    error_share: float | None = None

    def __post_init__(self):
        for name in ("successes", "trials"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int):
                raise TypeError(f"{name} must be an integer count")
        if self.trials <= 0 or not 0 <= self.successes <= self.trials:
            raise ValueError("need 0 <= successes <= trials and trials > 0")
        if self.split == FINAL_AUDIT and (self.plan_hash is None or self.error_share is None):
            raise ValueError("a final-audit record must name its plan hash and error share")


def training_rows(rows: Iterable[MeasurementRecord], splits: Iterable[str] = ("development",)):
    """Rows usable as development labels. A request for the final-audit split is refused."""
    splits = tuple(splits)
    if FINAL_AUDIT in splits:
        raise PermissionError("final-audit outcomes are never used as development labels")
    return [r for r in rows if r.split in splits]
