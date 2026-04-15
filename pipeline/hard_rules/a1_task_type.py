"""A1: task type must be supervised classification or regression.
Checked via the metadata if task type is provided, otherwise via the actual data."""
from __future__ import annotations

import pandas as pd

from pipeline.config import ACCEPTED_TASKS
from pipeline.hard_rules.base import RuleResult

#the kwargs object is used to pass the metadata information from the candidate 
# info to the check_metadata function. Runner function will call the check_metadata function with the metadata information 
# from the candidate info. **_kwargs lets it accept for each rule and ignore the ones it doesn't need. -> can be reused easily 
def check_metadata( task_type: str, **_kwargs: object,) -> RuleResult | None: #regex chek for task type 
    normalized = task_type.lower().replace("-", "_").strip()
    if any(acc in normalized for acc in ACCEPTED_TASKS): #if any task is in the substring the rules gets marked as passed 
        return RuleResult(rule="A1", passed=True)
    if normalized == "unknown":
        return None  # defer, need actual data to figure this out
    return RuleResult(rule="A1", passed=False, reason=f"task_type='{task_type}' not accepted")


def check_data( #check in actual data for the task type via the target variable
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str = "unknown", 
    **_kwargs: object, #again same pattern to accept any metadata but ignore it if not needed
) -> RuleResult | None: 
    """Infer task type from target variable when metadata said 'unknown'.
    Only runs when task_type is 'unknown' from the metadatacheck. The
    inferred task is carried in details['inferred_task'] so callers can
    use it downstream.
    """
    if task_type.lower().replace("-", "_").strip()!= "unknown":
        return None  # already resolved at metadata phase -> skip check

    if y is None or y.empty:
        return RuleResult(rule="A1", passed=False, reason="no target variable")

    n_unique = y.nunique()
    if n_unique <= 1:
        return RuleResult(rule="A1", passed=False, reason=f"target has only {n_unique} unique values")

    # Discrete target with few classes -> classification, else regression.
    inferred = "classification" if n_unique <= 50 else "regression"
    return RuleResult(rule="A1", passed=True, details={"inferred_task": inferred})
