"""A6: dataset must have an open licence."""
from __future__ import annotations

import re

from pipeline.config import ALLOWED_LICENCES
from pipeline.hard_rules.base import RuleResult


def normalise_licence(raw: str | None) -> str:
    """Normalize a licence string for comparison."""
    if not raw:
        return "unknown"
    return raw.strip().lower().replace(" ", "-")


def check_metadata(
    licence: str = "",
    **_kwargs: object,
) -> RuleResult:
    """Check licence against allow-list."""
    norm = normalise_licence(licence)

    if norm == "unknown":
        return RuleResult(rule="A6", passed=True, reason="licence unknown, assuming open")

    # Reject NC or ND explicitly
    if re.search(r"\bnc\b|\bnd\b|non.commercial|no.deriv", norm):
        return RuleResult(rule="A6", passed=False, reason=f"restrictive licence: '{licence}'")

    for allowed in ALLOWED_LICENCES:
        if allowed in norm:
            return RuleResult(rule="A6", passed=True)

    return RuleResult(rule="A6", passed=False, reason=f"licence '{licence}' not in allow-list")
