"""Base class for soft rule checks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from pipeline.data.base import Dataset


def _to_tier(score: float) -> int:
    """Map a continuous score [0.0, 1.0] to a discrete tier (0, 10, 20, ..., 100)."""
    return int(round(score * 10)) * 10


@dataclass(frozen=True)
class SoftRuleResult:
    """Result of a soft rule check."""

    rule: str
    score: float  # continuous value in [0.0, 1.0] — used by CRITIC
    details: dict[str, Any]  # intermediate diagnostics for rule_details.json

    @property
    def tier(self) -> int:
        """Discrete tier (0, 10, 20, ..., 100) for human-readable output."""
        return _to_tier(self.score)


class SoftRule(Protocol):
    """Protocol for soft rule checks."""

    @property
    def name(self) -> str:
        """Rule identifier, e.g. 'S1', 'S3'."""
        ...

    @property
    def max_points(self) -> int:
        """Maximum point allocation for this rule."""
        ...

    def score(
        self,
        dataset: Dataset,
    ) -> SoftRuleResult:
        """Score the dataset. Returns continuous score [0.0, 1.0] and diagnostic details."""
        ...
