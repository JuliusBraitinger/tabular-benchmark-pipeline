"""A3: dataset must contain predictive signal (not pure noise).

MOCKUP: this rule is currently a stub that always passes.
The real check (RandomForest cross-validated AUC / R2 against a threshold)
is planned but not activated yet, so the runner can call A3 without failing
any dataset.
TODO: implement actual signal check later (RF with 3-fold CV, AUC>0.55 / R2>0.02).
"""
from __future__ import annotations

from pipeline.hard_rules.base import RuleResult


def check_metadata(**_kwargs: object) -> None:
    # can't check signal from metadata alone, defer to data check
    return None


def check_data(**_kwargs: object) -> RuleResult:
    """Mockup: always passes. Real signal check will be added later."""
    return RuleResult(rule="A3", passed=True)
