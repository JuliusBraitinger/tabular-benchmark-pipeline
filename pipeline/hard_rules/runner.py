"""Run all hard rules on a dataset. This is the central place where the hard rules are being executed. 
It contains two main functions: run_metadata_checks and run_data_checks, which run the checks which are possible only with the metadata
and which are only possible with the actual downloaded data."""
from __future__ import annotations

import pandas as pd

from pipeline.hard_rules import a1_task_type, a2_synthetic, a3_signal, a4_dimensions, a5_licence
from pipeline.hard_rules.base import RuleResult

# Metadata-level rules (cheap, run first)
_METADATA_CHECKS = [
    ("A1", a1_task_type.check_metadata),
    ("A2", a2_synthetic.check_metadata),
    ("A4", a4_dimensions.check_metadata),
    ("A5", a5_licence.check_metadata),
]

# Data-level rules (expensive, run only if metadata passes)
_DATA_CHECKS = [
    ("A1", a1_task_type.check_data),
    ("A2", a2_synthetic.check_data),
    ("A3", a3_signal.check_data),
    ("A4", a4_dimensions.check_data),

]


def run_metadata_checks( #different rules need different data inputs -> runner needs to provide them all 
    *, 
    n_samples: int | None = None,
    n_features: int | None = None,
    task_type: str = "unknown",
    licence: str = "",
    source: str = "",
    name: str = "",
    metadata: dict | None = None,
    skip: tuple[str, ...] = (),  # rule names to skip, e.g. ("A4",) for sources exempt from the size filter
) -> list[RuleResult]:
    """Run metadata-level hard rules. Returns list of results.

    A None result means the rule needs actual data to decide.
    """
    kwargs = dict( #bundles all the metadata into a dictionary that can be looped for each rule
        n_samples=n_samples,
        n_features=n_features,
        task_type=task_type,
        licence=licence,
        source=source,
        name=name,
        metadata=metadata or {},
    )
    results: list[RuleResult] = []
    for _rule_name, check_fn in _METADATA_CHECKS: #actual check of the rules
        if _rule_name in skip:
            continue
        result = check_fn(**kwargs) #**kwargs is the dictionary with the metada. It can pick the relevant information for each rule and ignore the rest.
        if result is not None:
            results.append(result)
            if not result.passed:
                break  # no point checking remaining rules if one already failed
    return results


def run_data_checks(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str = "classification", #just a default value
    metadata: dict | None = None, #metadata thats in the dictionary
    skip: tuple[str, ...] = (),  # rule names to skip, e.g. ("A4",) for sources exempt from the size filter
) -> list[RuleResult]:
    """Run data-level hard rules (A1/A2/A3/A4 on actual data)."""
    kwargs = dict(X=X, y=y, task_type=task_type, metadata=metadata or {})
    results: list[RuleResult] = []
    for _rule_name, check_fn in _DATA_CHECKS:
        if _rule_name in skip:
            continue
        result = check_fn(**kwargs)
        if result is not None:
            results.append(result)
            if not result.passed:
                break  # fail fast, skip remaining checks
    return results


def all_passed(results: list[RuleResult]) -> bool:
    """True if every result in the list passed."""
    return all(r.passed for r in results)


def failed_rules(results: list[RuleResult]) -> list[RuleResult]:
    """Return only the failed results. Datasets are discarded logging the reason for failure."""
    return [r for r in results if not r.passed]


def inferred_task_type(results: list[RuleResult]) -> str | None:
    """Extract task type inferred by A1.check_data when metadata said 'unknown'."""
    for r in results:
        if r.rule == "A1" and "inferred_task" in r.details:
            return r.details["inferred_task"]
    return None
