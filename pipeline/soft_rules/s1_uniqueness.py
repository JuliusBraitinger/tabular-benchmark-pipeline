# S1 Uniqueness (10 points)
# statistical fingerprint: per-column mean+std+skew+kurtosis
# reference: pymfe paper (Alcobaca et al., 2020)

import numpy as np

from pipeline.soft_rules.base import SoftRuleResult


QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]


def summarize(values):
    quants = [values.quantile(q) for q in QUANTILES]
    return np.array(quants + [values.mean(), values.std()])


def compute_fingerprint(dataset):
    X = dataset.X.select_dtypes(include="number")

    means = X.mean()
    stds = X.std()
    skews = X.skew()
    kurts = X.kurtosis()

    return np.concatenate([
        summarize(means),
        summarize(stds),
        summarize(skews),
        summarize(kurts),
    ])


def score(dataset):
    fingerprint = compute_fingerprint(dataset)
    return SoftRuleResult(rule="S1", score=1.0, details={"fingerprint": fingerprint})
