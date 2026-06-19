# A1: task type must be supervised classification or regression.
# Checked via the metadata if task type is provided, otherwise via the actual data.
from __future__ import annotations

import pandas as pd

from pipeline.config import ACCEPTED_TASKS
from pipeline.hard_rules.base import RuleResult

# Each rule takes **_kwargs so the runner can pass the whole metadata bundle and
# every rule picks out only the fields it needs.
def normalise(task_type: str) -> str:
    return task_type.lower().replace("-", "_").strip()

def check_metadata(task_type: str, **_kwargs: object) -> RuleResult | None:
    normalized = normalise(task_type)
    if any(acc in normalized for acc in ACCEPTED_TASKS):  # substring match against allow-list
        return RuleResult(rule="A1", passed=True)
    if normalized == "unknown":
        return None  # defer, need actual data to figure this out
    return RuleResult(rule="A1", passed=False, reason=f"task_type='{task_type}' not accepted")


def check_data(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str = "unknown",
    **_kwargs: object,
) -> RuleResult | None:
    # Infer task type from the target variable when metadata said 'unknown'.
    # The inferred task is carried in details['inferred_task'] so callers can use it downstream.
    if _normalise(task_type) != "unknown":
        return None  # already resolved at metadata phase -> skip check

    if y is None or y.empty:
        return RuleResult(rule="A1", passed=False, reason="no target variable")

    n_unique = y.nunique()
    if n_unique <= 1:
        return RuleResult(rule="A1", passed=False, reason=f"target has only {n_unique} unique values")

    # Discrete target with few classes -> classification, else regression.
    inferred = "classification" if n_unique <= 50 else "regression"
    return RuleResult(rule="A1", passed=True, details={"inferred_task": inferred})
