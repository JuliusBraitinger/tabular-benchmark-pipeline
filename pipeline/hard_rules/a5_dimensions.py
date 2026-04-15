"""A5: dataset must meet minimum dimension requirements."""
from __future__ import annotations

from pipeline.config import MIN_FEATURES, MIN_ROWS
from pipeline.hard_rules.base import RuleResult


def check_metadata(
    n_samples: int | None = None,
    n_features: int | None = None,
    **_kwargs: object,
) -> RuleResult | None:
    """Check dimensions from metadata. Returns None if dimensions unknown."""
    if n_samples is None or n_features is None:
        return None  # defer to data check

    reasons: list[str] = []
    if n_samples < MIN_ROWS:
        reasons.append(f"N={n_samples} < {MIN_ROWS}")
    if n_features < MIN_FEATURES:
        reasons.append(f"P={n_features} < {MIN_FEATURES}")

    if reasons:
        return RuleResult(rule="A5", passed=False, reason="; ".join(reasons))
    return RuleResult(rule="A5", passed=True)


def check_data(
    X: "pd.DataFrame",
    **_kwargs: object,
) -> RuleResult:
    """Check dimensions from actual data."""
    n_samples, n_features = X.shape

    reasons: list[str] = []
    if n_samples < MIN_ROWS:
        reasons.append(f"N={n_samples} < {MIN_ROWS}")
    if n_features < MIN_FEATURES:
        reasons.append(f"P={n_features} < {MIN_FEATURES}")

    if reasons:
        return RuleResult(rule="A5", passed=False, reason="; ".join(reasons))
    return RuleResult(rule="A5", passed=True)
