"""Base classes for hard rule checks."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RuleResult:
    """Result of a single hard rule check."""

    rule: str
    passed: bool
    reason: str = ""
    details: dict = field(default_factory=dict)
