# S6 Class Balance (5 points)
# normalized shannon entropy of class distribution

import numpy as np
from pipeline.soft_rules.base import SoftRuleResult

def score(dataset):
    # "classification" matches both "classification" and "supervised_classification"
    if "classification" not in dataset.task_type:
        return SoftRuleResult(
            rule="S6",
            score=float("nan"),
            details={"skipped": "non-classification task", "task_type": dataset.task_type},
        )

    y = dataset.y
    counts = y.value_counts()
    counts = counts[counts > 0]  # a categorical y lists unobserved levels as 0
    k = len(counts)

    if k < 2:
        return SoftRuleResult(rule="S6", score=0.0, details={"k": k})

    proportions = counts / counts.sum()
    entropy = -np.sum(proportions * np.log(proportions))
    balance = entropy / np.log(k)

    return SoftRuleResult(
        rule="S6",
        score=float(balance),
        details={"k": int(k), "entropy": float(entropy)},
    )
