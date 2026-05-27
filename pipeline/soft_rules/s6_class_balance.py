# S6 Class Balance (5 points)
# normalized shannon entropy of class distribution

import numpy as np
from pipeline.soft_rules.base import SoftRuleResult

def score(dataset):
    if "classifcation_target" not in dataset.data.columns:
        return SoftRuleResult(rule="S6", score=0.0, details={"reason": "No classification target"})
    
    y = dataset.y
    counts = y.value_counts()
    k = len(counts)

    if k < 2:
        return SoftRuleResult(rule="S6", score=0.0, details={"k": k})

    proportions = counts / counts.sum()
    entropy = -np.sum(proportions * np.log(proportions))
    balance = entropy / np.log(k)

    return SoftRuleResult(rule="S6", score=balance, details={"k": k, "entropy": entropy})
