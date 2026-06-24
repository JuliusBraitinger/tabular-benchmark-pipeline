"""Base class for soft rule checks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from pipeline.data.base import Dataset


@dataclass(frozen=True)
class SoftRuleResult:
    """Result of a soft rule check."""

    rule: str
    score: float  # continuous value in [0.0, 1.0] — used by CRITIC
    details: dict[str, Any]  # intermediate diagnostics for rule_details.json


class SoftRule(Protocol):
    """Protocol for soft rule checks."""

    def score(
        self,
        dataset: Dataset,
    ) -> SoftRuleResult:
        """Score the dataset. Returns continuous score [0.0, 1.0] and diagnostic details."""
        ...
