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


def score(dataset, pool_fingerprints=None):
    own_fp = compute_fingerprint(dataset)

    if pool_fingerprints is None or len(pool_fingerprints) < 2:
        return SoftRuleResult(rule="S1", score=1.0, details={
            "meta_fingerprint": own_fp.tolist(),
            "max_similarity": 0.0,
        })
    max_sim = 0.0
    for other_fp in pool_fingerprints.values():
        sim = np.dot(own_fp, other_fp) / ( #similarity between fingerprint and pool (cosine similarity)
            np.linalg.norm(own_fp) * np.linalg.norm(other_fp)
        )
        if sim > max_sim:
            max_sim = sim

    score_val = max(0.0, min(1.0, 1.0 - max_sim))

    return SoftRuleResult(rule="S1", score=score_val, details={
        "meta_fingerprint": own_fp.tolist(),
        "max_similarity": float(max_sim),
    })

