"""Base classes for hard rule checks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class RuleResult:
    """Result of a single hard rule check."""

    rule: str
    passed: bool
    reason: str = ""
    details: dict = field(default_factory=dict)


class HardRule(Protocol):
    """Protocol for hard rule checks.

    Some rules only need metadata (licence, dimensions).
    Others need the actual data (signal check).
    """

    @property
    def name(self) -> str:
        """Rule identifier, e.g. 'A1', 'A3'."""
        ...

    def check_metadata(
        self,
        n_samples: int | None,
        n_features: int | None,
        task_type: str,
        licence: str,
        metadata: dict,
    ) -> RuleResult | None: #rules can carry structured output
        """Check using metadata only. Return None if this rule needs actual data."""
        ...

    def check_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str,
        metadata: dict,
    ) -> RuleResult:
        """Check using the actual data. Called only if check_metadata returned None."""
        ...
