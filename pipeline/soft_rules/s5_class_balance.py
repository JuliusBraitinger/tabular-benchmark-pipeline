# S5 Class Balance (5 points)
# normalized shannon entropy of class distribution

import numpy as np
import pandas as pd
from pipeline.soft_rules.base import SoftRuleResult

REGRESSION_BINS = 10 #bins for regression targets (Yang et al. ICML 2021 "Delving into Deep Imbalanced Regression")


def score(dataset):
    y = dataset.y
    # regression targets get split into equal-interval bins first, as in Yang et al.
    # (ICML 2021) "Delving into Deep Imbalanced Regression"
    if "classification" in dataset.task_type:
        counts = y.value_counts()
        counts = counts[counts > 0]  # a categorical y lists unobserved levels as 0
    else:
        counts = pd.Series(np.histogram(y.astype(float), bins=REGRESSION_BINS)[0])

    counts = counts[counts > 0]
    k = len(counts)

    if k < 2:
        return SoftRuleResult(rule="S5", score=0.0, details={"k": k})

    proportions = counts / counts.sum()
    entropy = -np.sum(proportions * np.log(proportions))
    balance = entropy / np.log(k)

    return SoftRuleResult(
        rule="S5",
        score=float(balance),
        details={"k": int(k), "entropy": float(entropy), "task_type": dataset.task_type},
    )
