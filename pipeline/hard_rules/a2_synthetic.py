"""A2: dataset must be real (not artificial/synthetic).

MOCKUP: this rule is currently a stub that always passes.
The real check (regex on name/tags for synthetic patterns) is planned but
not activated yet, so the runner can call A2 without failing any dataset.
TODO: implement actual synthetic-data detection later.
"""
from __future__ import annotations

from pipeline.hard_rules.base import RuleResult


def check_metadata(
    **_kwargs: object,
) -> RuleResult:
    # MOCKUP: always passes for now. real check coming later.
    return RuleResult(rule="A2", passed=True)
